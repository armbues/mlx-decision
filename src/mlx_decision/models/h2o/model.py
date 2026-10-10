"""The H2O-Lightning backend: Qwen3.5 backbone (causal) with a label readout.

Each question is its own sequence (see ``prompt``), ending in the answer
prefill. The last token's hidden state times the tied embedding rows of the
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

from ...backbones.qwen3_5.load import load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Noul, Request
from ..qwen import precision
from .prompt import PromptSettings, cut_state, render, render_state

# vLLM's --max-model-len in the release's serve.sh: a longer prompt is refused.
MAX_INPUT_TOKENS = 40960
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
    ):
        """``backbone`` None gives a backend that only encodes; ``label_ids``
        (each label's token) default to what ``tokenizer`` makes of them."""
        self.name = name
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.config = config
        self.labels = config.labels
        self.noul_floor = config.noul_floor
        self.path = path
        self.label_ids = label_ids or label_token_ids(tokenizer, config.labels)
        self.label_rows = None
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

    def state_text(self, request: Request) -> tuple[str, bool]:
        """The rendered state, cut around a "..." line when over the release's cap."""
        settings = self.config.prompt
        text = render_state(request.state, settings.state_compact)
        return cut_state(text, settings.max_state_tokens, self._encode_text, self._decode)

    def encode(self, request: Request) -> tuple[list[Encoded], bool]:
        """Every question's token ids and whether the state was cut.

        Refuses the request if a question's prompt is over the input limit.
        """
        limit = self.capabilities.max_input_tokens
        state_text, cut = self.state_text(request)
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
        return encoded, cut

    def score(self, request: Request) -> BackendOutput:
        encoded, cut = self.encode(request)
        probabilities = {}
        for question in encoded:
            values = self.probabilities(
                question.input_ids, len(question.option_ids), question.question_type
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

    def logits(self, input_ids: list[int], count: int) -> mx.array:
        """One pass: the first ``count`` labels' logits at the last token, in float32."""
        hidden = self.backbone(mx.array(input_ids)[None])[0, -1].astype(mx.float32)
        return self.label_rows[:count] @ hidden

    def probabilities(self, input_ids: list[int], count: int, question_type: str) -> list[float]:
        """The first ``count`` options' probabilities at the type's temperature."""
        logits = self.logits(input_ids, count)
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
    path: Path, max_input_tokens: int | None = None, noul_floor: float | None = None
) -> H2OBackend:
    """Load H2O-Lightning from a release folder or a converted copy.

    ``max_input_tokens`` lowers the limit per question's prompt (default and
    maximum 40,960, the release's); states over the release's cap (32,000
    tokens) are cut around a "..." line, longer prompts refused.
    ``noul_floor`` replaces the release's yes/no floor (0.801): a yes/no
    answer less sure than it is reported at it, on its own side; 0 turns it
    off (the raw probability, for routing uncertain answers elsewhere).
    """
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
    backend = H2OBackend(path.resolve().name, backbone, tokenizer, config, max_input_tokens, path)
    if noul_floor is not None:
        backend.noul_floor = float(noul_floor)
    backend.precision = precision(path, backbone)
    return backend
