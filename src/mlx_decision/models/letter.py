"""The letter-readout backend: Qwen3.5 backbone, options under answer codes.

What the pplx and d3 families share. Each question is its own sequence
(the family's ``Preset`` renders it): the last token's hidden state times
the readout gives one logit per answer code; the codes beyond the
question's options are left out, the rest divided by the release's
temperature and normalized. With non-causal full attention the state
attends to the question, so nothing of one question's pass serves
another; only the images' vision features are computed once per request.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx

from ..backbones.qwen3_5.load import has_vision_weights, load_text_model
from ..backbones.qwen3_5.tokenizer import load_tokenizer
from ..backend import BackendOutput, Capabilities
from ..errors import DecisionError
from ..types import Request
from .qwen import (
    IMAGE_PAD,
    count_image_tokens,
    image_config,
    image_features,
    image_inputs,
    open_images,
    precision,
    prepare,
)
from .readout import Features, Readout, pick, read_out, singletons

REQUIRED_FILES = ("config.json", "tokenizer.json", "decision_config.json", "readout.safetensors")
ATTENTION_MODES = {"noncausal_full_attention": False, "causal": True}  # mode -> causal
IMAGE_MARKERS = ("<|vision_start|>", IMAGE_PAD, "<|vision_end|>")


@dataclass(frozen=True)
class Preset:
    """What a family sets: its prompt and its release's limits.

    ``render(state, question, codes, images, where)`` is the prompt text of
    one question with ``images`` image placeholders; ``options(question)``
    its option ids and the descriptions the prompt lists; ``describe`` how
    the prompt writes a state. ``max_input_tokens`` is the default limit
    per question's sequence, ``input_token_limit`` the most it may be
    raised to. Images are resized to ``min_image_pixels`` to
    ``max_image_pixels``. ``max_score_levels`` None: one per answer code.
    """

    render: Callable
    options: Callable
    describe: Callable
    max_input_tokens: int
    input_token_limit: int
    min_image_pixels: int
    max_image_pixels: int
    max_score_levels: int | None = None


@dataclass
class Encoded:
    question_id: str
    option_ids: list[str]
    input_ids: list[int]


class LetterBackend:
    def __init__(
        self,
        name,
        backbone,
        readout,
        tokenizer,
        codes,
        temperature,
        preset: Preset,
        max_input_tokens: int | None = None,
        path: Path | None = None,
        max_image_pixels: int | None = None,
    ):
        """``max_input_tokens`` and ``max_image_pixels`` lower the preset's
        limits (None keeps them)."""
        self.name = name
        self.backbone = backbone
        self.readout = readout
        self.tokenizer = tokenizer
        self.codes = codes
        self.temperature = temperature
        self.preset = preset
        self.path = path
        self.max_image_pixels = min(
            preset.max_image_pixels, max_image_pixels or preset.max_image_pixels
        )
        self.vision = None
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens or preset.max_input_tokens,
            max_choice_options=len(codes),
            max_score_levels=min(len(codes), preset.max_score_levels or len(codes)),
            supports_images=path is not None and has_vision_weights(path),
            truncates_input=False,
        )

    def load_vision(self):
        """The vision tower, loaded on first use (it stays in its stored precision)."""
        if self.vision is None:
            # Imported here: the vision code needs numpy (the images extra).
            from ..backbones.qwen3_5.vision import load_vision_model

            self.vision = load_vision_model(self.path)
        return self.vision

    def image_config(self):
        return replace(
            image_config(self.path, None),
            min_pixels=min(self.preset.min_image_pixels, self.max_image_pixels),
            max_pixels=self.max_image_pixels,
        )

    def encode(self, request: Request, image_tokens: list[int] = ()) -> list[Encoded]:
        """Every question's token ids; refuses the request if one is over the limit.

        ``image_tokens`` holds each image's token count: its placeholder in
        the prompt becomes that many image pad tokens, as the processor does.
        """
        limit = self.capabilities.max_input_tokens
        encoded = []
        for question_id, question in request.questions.items():
            where = f"questions.{question_id}"
            text = self.preset.render(request.state, question, self.codes, len(image_tokens), where)
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            if image_tokens:
                ids = self._expand_images(ids, image_tokens, request.state, where)
            if len(ids) > limit:
                raise DecisionError(
                    f"the question with its state needs {len(ids)} tokens; "
                    f"the limit is {limit} and input is not cut",
                    param=where,
                )
            encoded.append(Encoded(question_id, self.preset.options(question)[0], ids))
        return encoded

    def _expand_images(self, ids, image_tokens, state, where) -> list[int]:
        start, pad, end = (self.tokenizer.token_to_id(t) for t in IMAGE_MARKERS)
        count = len(image_tokens)
        if (ids.count(start), ids.count(pad), ids.count(end)) != (count, count, count):
            # The tokenizer turns the markers into special tokens even in user
            # text, where they would be taken for image slots.
            state_ids = self.tokenizer.encode(
                self.preset.describe(state), add_special_tokens=False
            ).ids
            raise DecisionError(
                f"the text contains image markers ({', '.join(IMAGE_MARKERS)}), "
                "which cannot be used in a request with images",
                param="state" if {start, pad, end} & set(state_ids) else where,
            )
        expanded, images = [], iter(image_tokens)
        for token in ids:
            expanded += [pad] * next(images) if token == pad else [token]
        return expanded

    def passes(self, request: Request) -> tuple[list[Encoded], list[mx.array]]:
        """Every question's encoding and last hidden state."""
        # Image sizes come from the file headers, so the input limit is checked
        # before any image is decoded or cut into patches.
        opened, config = [], None
        if request.images:
            config = self.image_config()
            opened = open_images(request.images)
        encoded = self.encode(
            request, [count_image_tokens(image, index, config) for index, image in opened]
        )
        images = [prepare(image, index, config) for index, image in opened]
        features = image_features(self.load_vision(), images) if images else None
        hidden = [self.last_hidden(q.input_ids, images, features) for q in encoded]
        return encoded, hidden

    def question_features(self, option_ids: list[str], hidden: mx.array) -> Features:
        count = len(option_ids)
        rows = self.readout[:count].astype(mx.float32)
        return Features(option_ids, hidden, rows, singletons(count), self.temperature)

    def features(self, request: Request) -> dict[str, Features]:
        """What each question's readout reads, before the readout."""
        encoded, hidden = self.passes(request)
        return {
            q.question_id: self.question_features(q.option_ids, last)
            for q, last in zip(encoded, hidden, strict=True)
        }

    def finish(self, question, probabilities: dict[str, float]) -> dict[str, float]:
        """Nothing follows the readout."""
        return probabilities

    def score(
        self,
        request: Request,
        readouts: dict[str, Readout] | None = None,
        finish: bool = True,
    ) -> BackendOutput:
        """``readouts`` replace the readout rows of the questions (by id) with the
        same options; ``finish`` has no step to leave out here."""
        encoded, hidden = self.passes(request)
        probabilities = {}
        for question, last in zip(encoded, hidden, strict=True):
            features = self.question_features(question.option_ids, last)
            readout = pick(readouts, question.question_id, question.option_ids)
            values = read_out(features, readout)
            probabilities[question.question_id] = dict(
                zip(question.option_ids, values, strict=True)
            )
        return BackendOutput(
            probabilities=probabilities,
            input_tokens=sum(len(q.input_ids) for q in encoded),
        )

    def last_hidden(self, input_ids: list[int], images=(), features=None) -> mx.array:
        """One pass: the final hidden state at the last token, in float32.

        ``images`` (prepared) and their vision ``features`` fill the image
        pad tokens of ``input_ids``.
        """
        ids = mx.array(input_ids)[None]
        if images:
            embeddings, positions = image_inputs(
                self.backbone, self.load_vision(), self.tokenizer, input_ids, images, features
            )
            hidden = self.backbone(ids, embeddings[None], positions)
        else:
            hidden = self.backbone(ids)
        last = hidden[0, -1].astype(mx.float32)
        mx.eval(last)
        return last

    def probabilities(
        self, input_ids: list[int], count: int, images=(), features=None
    ) -> list[float]:
        """One sequence's first ``count`` answer codes' probabilities."""
        hidden = self.last_hidden(input_ids, images, features)
        rows = self.readout[:count].astype(mx.float32)
        return read_out(Features([], hidden, rows, singletons(count), self.temperature))


def read_decision_config(path: Path) -> tuple[list[str], float, bool]:
    """Answer codes, temperature and whether attention is causal, from ``decision_config.json``."""
    config = json.loads((path / "decision_config.json").read_text())
    if config.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported decision config format")
    mode = config.get("attention_mode", "causal")
    if mode not in ATTENTION_MODES:
        raise ValueError(f"{path}: unknown attention mode {mode!r}")
    if config.get("pooling", "last") != "last":
        raise ValueError(f"{path}: only last-token pooling is supported")
    temperature = config.get("temperature", 1.0)
    number = isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
    if not number or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{path}: the temperature must be a positive number")
    codes = config.get("codes")
    if not isinstance(codes, list) or not codes or not all(isinstance(c, str) for c in codes):
        raise ValueError(f"{path}: decision_config.json has no answer codes")
    return codes, float(temperature), ATTENTION_MODES[mode]


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weights ``load`` reads (the vision tower shares a shard), kept as stored."""
    files = sorted(path.glob("model*.safetensors")) + [path / "readout.safetensors"]
    return [(file, None) for file in files if file.exists()]


def load(
    path: Path,
    preset: Preset,
    max_input_tokens: int | None = None,
    vision: bool = False,
    max_image_pixels: int | None = None,
) -> LetterBackend:
    """Load a letter-readout model from a release folder or a converted copy.

    ``max_input_tokens`` lowers the limit per question's sequence (at most
    the preset's ``input_token_limit``); longer input is refused, not cut.
    ``vision=True`` loads the vision tower now, not with the first image.
    ``max_image_pixels`` lowers the preset's image maximum.
    """
    limit = preset.input_token_limit
    if max_input_tokens is not None and not 1 <= max_input_tokens <= limit:
        raise ValueError(f"max_input_tokens must be between 1 and {limit}")
    # Checked before the backbone, which takes a while to load.
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    codes, temperature, causal = read_decision_config(path)
    readout = mx.load(str(path / "readout.safetensors"))["weight"]
    if readout.shape[0] < len(codes):
        raise ValueError(f"{path}: the readout has fewer rows than answer codes")
    backbone = load_text_model(path, lm_head=False, causal=causal)
    if readout.shape[1] != backbone.language_model.args.hidden_size:
        raise ValueError(f"{path}: the readout does not match the backbone's width")
    mx.eval(readout)
    tokenizer = load_tokenizer(path)
    backend = LetterBackend(
        path.resolve().name,
        backbone,
        readout,
        tokenizer,
        codes,
        temperature,
        preset,
        max_input_tokens,
        path,
        max_image_pixels,
    )
    backend.precision = precision(path, backbone)
    if vision:
        if not backend.capabilities.supports_images:
            raise ValueError(f"{path} has no vision weights")
        backend.load_vision()
    return backend
