"""The H2O-Lightning backend: Qwen3.5 backbone (causal) with a label readout.

Each question is its own sequence (see ``prompt``), ending in the answer
prefill. The sequences start alike, up to the state and ``question:``; that
prefix is computed once per request and each question continues from a copy
of the backbone's cache there, and the prefix is kept for later requests
with the same state and images (``prefix``). Images come first in the user
message, so they are part of the prefix. The last token's hidden state
times the tied embedding rows of the option labels' tokens gives one logit
per option, divided by the question type's temperature (the image
temperature for every question of a request with images) and normalized;
a yes/no answer is then moved to the release's floor when it is less sure
than that. As the release does it with
vLLM's log-probabilities over the whole vocabulary: the log-softmax shifts
every label's logit by the same amount, so the answer is the same.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.cache import fork_cache
from ...backbones.qwen3_5.load import has_vision_weights, load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Noul, Request
from ..prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache, prefix_key
from ..qwen import (
    IMAGE_PAD,
    count_image_tokens,
    image_config,
    image_features,
    image_inputs,
    open_images,
    precision,
    prepare,
)
from .prompt import PromptSettings, cut_state, render, render_state, shared_prefix

# vLLM's --max-model-len in the release's serve.sh: a longer prompt is refused.
MAX_INPUT_TOKENS = 40960
# Under a lowered limit, tokens kept free beyond a question's own: the "..."
# line and words that tokenize differently where the state is cut or joined.
CUT_MARGIN = 8
REQUIRED_FILES = ("config.json", "tokenizer.json", "serve_config.json")
QUESTION_TYPES = ("choice", "noul", "score")
IMAGE_MARKERS = ("<|vision_start|>", IMAGE_PAD, "<|vision_end|>")


@dataclass
class Encoded:
    question_id: str
    question_type: str
    option_ids: list[str]
    input_ids: list[int]


@dataclass
class Images:
    """A request's images: each one's token count, and once decoded, its patches."""

    tokens: list[int] = field(default_factory=list)
    prepared: list = field(default_factory=list)
    merge_size: int = 2


@dataclass(frozen=True)
class ImageSettings:
    """The ``image`` block of ``serve_config.json``: the release reads images with it."""

    max_images: int = 4
    max_pixels: int = 1048576  # 0: the image processor's own maximum
    temperature: float = 0.65


@dataclass(frozen=True)
class ServeConfig:
    """What the release's ``serve_config.json`` sets for answering."""

    labels: list[str]
    temperatures: dict[str, float]
    noul_floor: float
    prompt: PromptSettings
    image: ImageSettings | None = None

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
        image = None
        if isinstance(block := config.get("image"), dict):
            image = ImageSettings(
                max_images=int(block.get("max_images", 4)),
                max_pixels=int(block.get("max_pixels") or 0),
                temperature=float(block.get("temperature", temperatures["choice"])),
            )
            if not 1 <= image.max_images <= 16 or not image.temperature > 0:
                raise ValueError(f"{path}: the image block's limits are out of range")
        prompt = PromptSettings.from_config(config["prompt"])
        return cls(labels, temperatures, floor, prompt, image)


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
        self._label_rows = None
        self._rows_layer = None  # the output layer the rows were read from
        self.vision = None
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
        if backbone is not None:
            self.label_rows()
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            max_choice_options=len(config.labels),
            max_score_levels=len(config.labels),
            truncates_input=True,
            max_state_tokens=config.prompt.max_state_tokens,
            supports_images=(
                config.image is not None and path is not None and has_vision_weights(path)
            ),
        )

    def load_vision(self):
        """The vision tower, loaded on first use (it stays in its stored precision)."""
        if self.vision is None:
            # Imported here: the vision code needs numpy (the images extra).
            from ...backbones.qwen3_5.vision import load_vision_model

            self.vision = load_vision_model(self.path)
        return self.vision

    def label_rows(self) -> mx.array:
        """The label tokens' output embedding rows (float32), kept between requests
        and read again when the output layer is replaced (the mixed-precision
        sweep swaps in a quantized one, then the original)."""
        model = self.backbone.language_model
        layer = model.output_embedding_layer()
        if layer is not self._rows_layer:
            rows = model.output_embedding_rows(mx.array(self.label_ids))
            self._label_rows = rows.astype(mx.float32)
            mx.eval(self._label_rows)
            self._rows_layer = layer
        return self._label_rows

    def image_config(self):
        """The release's processor settings at the serve config's pixel limit."""
        return image_config(self.path, self.config.image.max_pixels or None)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def state_cap(self, request: Request, image_tokens: list[int] = ()) -> int:
        """The most tokens of state the request's prompts show.

        At the release's limit, the release's cap (32,000). Under a lowered
        limit, also no more than leaves room for the images and the longest
        question with its options, so that a long state is cut rather than
        refused.
        """
        cap = self.config.prompt.max_state_tokens
        limit = self.capabilities.max_input_tokens
        if limit >= MAX_INPUT_TOKENS:
            return cap
        settings, count = self.config.prompt, len(image_tokens)
        longest = max(
            len(self._encode_text(render("", question, self.labels, settings, count)[0]))
            for question in request.questions.values()
        )
        # Each image's pad becomes its tokens.
        longest += sum(image_tokens) - count
        room = limit - longest - CUT_MARGIN
        # Without room the questions alone are over the limit: refused below.
        return min(cap, room) if room > 0 else cap

    def state_text(self, request: Request, image_tokens: list[int] = ()) -> tuple[str, bool]:
        """The rendered state, cut around a "..." line when over the cap."""
        text = render_state(request.state, self.config.prompt.state_compact)
        cap = self.state_cap(request, image_tokens)
        return cut_state(text, cap, self._encode_text, self._decode)

    def encode(self, request: Request, image_tokens: list[int] = ()) -> tuple[list[Encoded], bool]:
        """Every question's token ids and whether the state was cut.

        ``image_tokens`` holds each image's token count: its pad in the
        prompt becomes that many pad tokens, as vLLM's processor does.
        Refuses the request if a question's prompt is over the input limit.
        """
        encoded, cut, _ = self.encode_with_prefix(request, image_tokens)
        return encoded, cut

    def encode_with_prefix(
        self, request: Request, image_tokens: list[int] = ()
    ) -> tuple[list[Encoded], bool, list[int]]:
        """``encode``, plus the token ids every question's prompt starts with.

        The shared ids are empty if a prompt does not start with them (the
        questions then run as whole passes).
        """
        limit, settings, count = (
            self.capabilities.max_input_tokens,
            self.config.prompt,
            len(image_tokens),
        )
        state_text, cut = self.state_text(request, image_tokens)
        prefix = self._encode_text(shared_prefix(state_text, settings, count))
        if image_tokens:
            prefix = self._expand_images(prefix, image_tokens, request.state, "state")
        encoded = []
        for question_id, question in request.questions.items():
            where = f"questions.{question_id}"
            text, option_ids = render(state_text, question, self.labels, settings, count)
            ids = self._encode_text(text)
            if image_tokens:
                ids = self._expand_images(ids, image_tokens, request.state, where)
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

    def _expand_images(self, ids, image_tokens, state, where) -> list[int]:
        start, pad, end = (self.tokenizer.token_to_id(t) for t in IMAGE_MARKERS)
        count = len(image_tokens)
        if (ids.count(start), ids.count(pad), ids.count(end)) != (count, count, count):
            # The tokenizer reads the markers as special tokens even in user
            # text, where they would be taken for image slots.
            state_ids = self._encode_text(render_state(state, self.config.prompt.state_compact))
            raise DecisionError(
                f"the text contains image markers ({', '.join(IMAGE_MARKERS)}), "
                "which cannot be used in a request with images",
                param="state" if {start, pad, end} & set(state_ids) else where,
            )
        expanded, tokens = [], iter(image_tokens)
        for token in ids:
            expanded += [pad] * next(tokens) if token == pad else [token]
        return expanded

    def open_images(self, request: Request) -> tuple[list, Images]:
        """The request's images opened (headers only) and their token counts."""
        if not request.images:
            return [], Images()
        limit = self.config.image.max_images
        if len(request.images) > limit:
            raise DecisionError(
                f"{len(request.images)} images; this model reads at most {limit} per request",
                param="images",
            )
        config = self.image_config()
        opened = open_images(request.images)
        tokens = [count_image_tokens(image, index, config) for index, image in opened]
        return opened, Images(tokens=tokens, merge_size=config.merge_size)

    def prefix_state(self, prefix: list[int], images: Images | None = None):
        """The backbone's cache after ``prefix``: kept from an earlier request, or computed."""
        prepared = images.prepared if images else []
        key = None
        if self.prefix_cache is not None:
            key = prefix_key(prefix, prepared)
            if (entry := self.prefix_cache.get(key)) is not None:
                return entry.cache
        cache = self.backbone.make_cache()
        if prepared:
            embeddings, positions = image_inputs(
                self.backbone, self.load_vision(), self.tokenizer, prefix, prepared
            )
            self.backbone(mx.array(prefix)[None], embeddings[None], positions, cache=cache)
        else:
            self.backbone(mx.array(prefix)[None], cache=cache)
        mx.eval([c.state for c in cache])
        if key is not None:
            self.prefix_cache.put(key, cache)
        return cache

    def score(self, request: Request) -> BackendOutput:
        # Image sizes come from the file headers, so the input limit is checked
        # before any image is decoded or cut into patches.
        opened, images = self.open_images(request)
        encoded, cut, prefix = self.encode_with_prefix(request, images.tokens)
        config = self.image_config() if opened else None
        images.prepared = [prepare(image, index, config) for index, image in opened]
        # A single question without kept prefixes is one pass; otherwise the
        # prefix (the images included) runs once or comes from the cache, and
        # each question continues from it, so a request answers the same
        # whether its prefix is new or kept.
        reuse = bool(prefix) and (len(encoded) > 1 or self.prefix_cache is not None)
        state = (len(prefix), self.prefix_state(prefix, images)) if reuse else None
        features = None
        if images.prepared and state is None:
            features = image_features(self.load_vision(), images.prepared)
        probabilities = {}
        for question in encoded:
            values = self.probabilities(
                question.input_ids,
                len(question.option_ids),
                question.question_type,
                state,
                images if images.prepared else None,
                features,
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

    def logits(
        self, input_ids: list[int], count: int, prefix=None, images=None, features=None
    ) -> mx.array:
        """One pass: the first ``count`` labels' logits at the last token, in float32.

        ``prefix`` (its length and the backbone's cache after it) continues
        from there: only the rest of ``input_ids`` is computed. ``images``
        (an ``Images`` with prepared patches) fill the image pad tokens;
        ``features`` are their vision outputs when already computed.
        """
        ids = mx.array(input_ids)[None]
        if prefix is None and images is None:
            hidden = self.backbone(ids)
        elif prefix is None:
            embeddings, positions = image_inputs(
                self.backbone,
                self.load_vision(),
                self.tokenizer,
                input_ids,
                images.prepared,
                features,
            )
            hidden = self.backbone(ids, embeddings[None], positions)
        else:
            length, cache = prefix
            positions = None
            if images is not None:
                # The rest is text; its positions continue after the images'.
                from ...backbones.qwen3_5.positions import position_ids

                pad = self.tokenizer.token_to_id(IMAGE_PAD)
                grids = [image.grid for image in images.prepared]
                positions = position_ids(input_ids, pad, grids, images.merge_size)[:, :, length:]
            hidden = self.backbone(ids[:, length:], None, positions, cache=fork_cache(cache))
        return self.label_rows()[:count] @ hidden[0, -1].astype(mx.float32)

    def probabilities(
        self,
        input_ids: list[int],
        count: int,
        question_type: str,
        prefix=None,
        images=None,
        features=None,
    ) -> list[float]:
        """The first ``count`` options' probabilities at the temperature: the
        question type's, or the image temperature with ``images``."""
        logits = self.logits(input_ids, count, prefix, images, features)
        if images is not None:
            temperature = self.config.image.temperature
        else:
            temperature = self.config.temperatures[question_type]
        values = mx.softmax(logits / temperature)
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
    vision: bool = False,
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
    questions of one request still share their prefix). ``vision=True``
    loads the vision tower now, not with the first image. Images are scaled
    to at most the serve config's pixels (1,048,576), at most 4 a request.
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
    if vision:
        if not backend.capabilities.supports_images:
            raise ValueError(f"{path} has no vision weights or image settings")
        backend.load_vision()
    return backend
