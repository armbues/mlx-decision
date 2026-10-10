"""The H2O-Lightning backend: Qwen3.5 backbone (causal) with a label readout.

Each question is its own sequence (see ``prompt``), ending in the answer
prefill. The sequences start alike, up to the state and ``question:``; that
prefix is computed once per request and each question continues from a copy
of the backbone's cache there, and the prefix is kept for later requests
with the same state (``prefix``). The last token's hidden state times the tied embedding rows of the
option labels' tokens gives one logit per option, divided by the question
type's temperature and normalized; a yes/no answer is then moved to the
release's floor when it is less sure than that. As the release does it with
vLLM's log-probabilities over the whole vocabulary: the log-softmax shifts
every label's logit by the same amount, so the answer is the same.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.cache import fork_cache
from ...backbones.qwen3_5.load import load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Noul, Request
from ..prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache, prefix_key
from ..qwen import precision
from .prompt import PromptSettings, cut_state, render, render_state, shared_prefix

# vLLM's --max-model-len in the release's serve.sh: a longer prompt is refused.
MAX_INPUT_TOKENS = 40960
# Under a lowered limit, tokens kept free beyond a question's own: the "..."
# line and words that tokenize differently where the state is cut or joined.
CUT_MARGIN = 8
REQUIRED_FILES = ("config.json", "tokenizer.json", "serve_config.json")
QUESTION_TYPES = ("choice", "noul", "score")


@dataclass
class Encoded:
    question_id: str
    question_type: str
    option_ids: list[str]
    input_ids: list[int]


@dataclass(frozen=True)
class ServeConfig:
    """What the release's ``serve_config.json`` sets for answering."""

    labels: list[str]
    temperatures: dict[str, float]
    noul_floor: float
    prompt: PromptSettings

    @classmethod
    def read(cls, path: Path) -> "ServeConfig":
        config = json.loads((path / "serve_config.json").read_text())
        labels = [str(label) for label in config.get("labels") or []]
        labels = labels[: int(config.get("max_options", len(labels)))]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError(f"{path}: serve_config.json needs distinct option labels")
        temperature = config.get("temperature", 1.0)
        if isinstance(temperature, dict):
            if set(temperature) != set(QUESTION_TYPES):
                raise ValueError(f"{path}: a temperature per type needs {QUESTION_TYPES}")
            temperatures = {kind: float(temperature[kind]) for kind in QUESTION_TYPES}
        else:
            temperatures = dict.fromkeys(QUESTION_TYPES, float(temperature))
        if not all(math.isfinite(t) and t > 0 for t in temperatures.values()):
            raise ValueError(f"{path}: temperatures must be positive numbers")
        floor = float(config.get("noul_floor", 0) or 0)
        if floor and not 0.5 < floor < 1:
            raise ValueError(f"{path}: noul_floor must be between 0.5 and 1, or 0 (off)")
        return cls(labels, temperatures, floor, PromptSettings.from_config(config["prompt"]))


def commit_noul(p_true: float, floor: float) -> float:
    """A yes/no probability less sure than ``floor`` moves to it, on its own side.

    The answer never changes (a tie goes to true); 0 turns the floor off.
    """
    if not floor or max(p_true, 1 - p_true) >= floor:
        return p_true
    return floor if p_true >= 0.5 else 1 - floor


class H2OBackend:
    def __init__(
        self,
        name,
        backbone,
        tokenizer,
        config: ServeConfig,
        max_input_tokens,
        path=None,
        label_ids: list[int] | None = None,
        prefix_cache_gb: float = 0.0,
    ):
        """``backbone`` None gives a backend that only encodes; ``label_ids``
        (each label's token) default to what ``tokenizer`` makes of them;
        ``prefix_cache_gb`` 0 keeps no prefixes between requests."""
        self.name = name
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        self.labels = config.labels
        self.noul_floor = config.noul_floor
        self.path = path
        self.label_ids = label_ids or label_token_ids(tokenizer, config.labels)
        self.label_rows = None
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
        if backbone is not None:
            rows = backbone.language_model.output_embedding_rows(mx.array(self.label_ids))
            self.label_rows = rows.astype(mx.float32)
            mx.eval(self.label_rows)
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            max_choice_options=len(config.labels),
            max_score_levels=len(config.labels),
            truncates_input=True,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def state_cap(self, request: Request) -> int:
        """The most tokens of state the request's prompts show.

        At the release's limit, the release's cap (32,000). Under a lowered
        limit, also no more than leaves room for the longest question with
        its options, so that a long state is cut rather than refused.
        """
        cap = self.config.prompt.max_state_tokens
        limit = self.capabilities.max_input_tokens
        if limit >= MAX_INPUT_TOKENS:
            return cap
        longest = max(
            len(self._encode_text(render("", question, self.labels, self.config.prompt)[0]))
            for question in request.questions.values()
        )
        room = limit - longest - CUT_MARGIN
        # Without room the questions alone are over the limit: refused below.
        return min(cap, room) if room > 0 else cap

    def state_text(self, request: Request) -> tuple[str, bool]:
        """The rendered state, cut around a "..." line when over the cap."""
        text = render_state(request.state, self.config.prompt.state_compact)
        return cut_state(text, self.state_cap(request), self._encode_text, self._decode)

    def encode(self, request: Request) -> tuple[list[Encoded], bool]:
        """Every question's token ids and whether the state was cut.

        Refuses the request if a question's prompt is over the input limit.
        """
        encoded, cut, _ = self.encode_with_prefix(request)
        return encoded, cut

    def encode_with_prefix(self, request: Request) -> tuple[list[Encoded], bool, list[int]]:
        """``encode``, plus the token ids every question's prompt starts with.

        The shared ids are empty if a prompt does not start with them (the
        questions then run as whole passes).
        """
        limit = self.capabilities.max_input_tokens
        state_text, cut = self.state_text(request)
        prefix = self._encode_text(shared_prefix(state_text, self.config.prompt))
        encoded = []
        for question_id, question in request.questions.items():
            text, option_ids = render(state_text, question, self.labels, self.config.prompt)
            ids = self._encode_text(text)
            if len(ids) > limit:
                raise DecisionError(
                    f"the question with its state needs {len(ids)} tokens "
                    f"(after cutting the state); the limit is {limit}",
                    param=f"questions.{question_id}",
                )
            encoded.append(Encoded(question_id, question.type, option_ids, ids))
        if not all(e.input_ids[: len(prefix)] == prefix for e in encoded):
            prefix = []
        return encoded, cut, prefix

    def prefix_state(self, prefix: list[int]):
        """The backbone's cache after ``prefix``: kept from an earlier request, or computed."""
        key = prefix_key(prefix) if self.prefix_cache is not None else None
        if key is not None and (entry := self.prefix_cache.get(key)) is not None:
            return entry.cache
        cache = self.backbone.make_cache()
        self.backbone(mx.array(prefix)[None], cache=cache)
        mx.eval([c.state for c in cache])
        if key is not None:
            self.prefix_cache.put(key, cache)
        return cache

    def score(self, request: Request) -> BackendOutput:
        encoded, cut, prefix = self.encode_with_prefix(request)
        # A single question without kept prefixes is one pass; otherwise the
        # prefix runs once (or comes from the cache) and each question
        # continues from it, so a request answers the same whether its prefix
        # is new or kept.
        reuse = bool(prefix) and (len(encoded) > 1 or self.prefix_cache is not None)
        state = (len(prefix), self.prefix_state(prefix)) if reuse else None
        probabilities = {}
        for question in encoded:
            values = self.probabilities(
                question.input_ids, len(question.option_ids), question.question_type, state
            )
            if isinstance(request.questions[question.question_id], Noul):
                p_true = commit_noul(values[0], self.noul_floor)
                values = [p_true, 1 - p_true]
            probabilities[question.question_id] = dict(
                zip(question.option_ids, values, strict=True)
            )
        return BackendOutput(
            probabilities=probabilities,
            input_tokens=sum(len(q.input_ids) for q in encoded),
            truncated=cut,
        )

    def logits(self, input_ids: list[int], count: int, prefix=None) -> mx.array:
        """One pass: the first ``count`` labels' logits at the last token, in float32.

        ``prefix`` (its length and the backbone's cache after it) continues
        from there: only the rest of ``input_ids`` is computed.
        """
        if prefix is None:
            hidden = self.backbone(mx.array(input_ids)[None])
        else:
            length, cache = prefix
            hidden = self.backbone(mx.array(input_ids[length:])[None], cache=fork_cache(cache))
        return self.label_rows[:count] @ hidden[0, -1].astype(mx.float32)

    def probabilities(
        self, input_ids: list[int], count: int, question_type: str, prefix=None
    ) -> list[float]:
        """The first ``count`` options' probabilities at the type's temperature."""
        logits = self.logits(input_ids, count, prefix)
        values = mx.softmax(logits / self.config.temperatures[question_type])
        mx.eval(values)
        return values.tolist()


def label_token_ids(tokenizer, labels: list[str]) -> list[int]:
    """Each label's token at the answer slot: ``" " + label`` must be one token."""
    ids = []
    for label in labels:
        tokens = tokenizer.encode(" " + label, add_special_tokens=False).ids
        if len(tokens) != 1:
            raise ValueError(f"the option label {label!r} is not a single token after a space")
        ids.append(tokens[0])
    if len(set(ids)) != len(ids):
        raise ValueError("two option labels share a token")
    return ids


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weights ``load`` reads (the vision tower shares the file), kept as stored."""
    return [(file, None) for file in sorted(path.glob("model*.safetensors"))]


def load(
    path: Path,
    max_input_tokens: int | None = None,
    noul_floor: float | None = None,
    prefix_cache_gb: float = DEFAULT_PREFIX_CACHE_GB,
) -> H2OBackend:
    """Load H2O-Lightning from a release folder or a converted copy.

    ``max_input_tokens`` lowers the limit per question's prompt (default and
    maximum 40,960, the release's). States over the release's cap (32,000
    tokens) are cut around a "..." line, under a lowered limit also states
    that would not leave room for the questions; prompts still over the
    limit are refused.
    ``noul_floor`` replaces the release's yes/no floor (0.801): a yes/no
    answer less sure than it is reported at it, on its own side; 0 turns it
    off (the raw probability, for routing uncertain answers elsewhere).
    ``prefix_cache_gb`` is the memory for kept prefixes (the computed state
    after a request's state), so a request with a state seen recently and
    other questions computes only its questions; 0 turns keeping off (the
    questions of one request still share their prefix).
    """
    if prefix_cache_gb < 0:
        raise ValueError("prefix_cache_gb must be 0 (off) or more")
    if noul_floor is not None and noul_floor != 0 and not 0.5 < noul_floor < 1:
        raise ValueError("noul_floor must be between 0.5 and 1, or 0 to turn it off")
    if max_input_tokens is None:
        max_input_tokens = MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= MAX_INPUT_TOKENS:
        raise ValueError(
            f"max_input_tokens must be between 1 and {MAX_INPUT_TOKENS} (the release's limit)"
        )
    # Checked before the backbone, which takes a while to load.
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    config = ServeConfig.read(path)
    backbone = load_text_model(path, lm_head=False)
    if not backbone.language_model.args.tie_word_embeddings:
        raise ValueError(f"{path}: the label readout needs tied embeddings")
    tokenizer = load_tokenizer(path)
    backend = H2OBackend(
        path.resolve().name,
        backbone,
        tokenizer,
        config,
        max_input_tokens,
        path,
        prefix_cache_gb=prefix_cache_gb,
    )
    if noul_floor is not None:
        backend.noul_floor = float(noul_floor)
    backend.precision = precision(path, backbone)
    return backend
