"""The d1 LFM3 backend (d1-2B): LFM3 text model and vision tower, questions read in sets.

A request's questions go in sets of the release's size (16), in their order;
each set is one sequence (the pictures and state, then its questions; the
token ids, state length and question spans from ``prompt.packed_sets``) in
which, on the full-attention layers, the state's tokens see the whole set and
each question the state and itself (``backbones/lfm3``'s ``SetLayout``). The
state's hidden states depend on its set's questions, so nothing is kept
between sets or requests. Answers depend on which questions share a set and
on their order, as in the release.

The sets run as rows of one right-padded batch of at most ``BATCH_TOKENS``
tokens (rows times the widest; at least one row). The answer is read at each
question's last token from the rows of the tied output embeddings that stand
for its options (``readout.option_token_ids``), in float32: each option's
best token, divided by the calibration's temperature, then a softmax over
the options, computed by the readout the families share.

Pictures (``images``, and images lifted out of the state before they get
here) are scaled to the release's cap (``prompt.picture_size``, Pillow
bicubic), then cut into patches by the Qwen preprocessing with the release's
processor settings. The vision tower and projector run once per request;
their rows replace the image tokens in every set's row.

A state too long for the input limit with the request's largest set is cut:
its first and last tokens are kept around a ``CUT_MARK`` line, and the answer
is marked truncated. ``prompt.state_text`` gives the state's text, and the
cut text goes back in as a string state.
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.lfm3.language import SetLayout
from ...backbones.lfm3.load import load_text_model, load_vision
from ...backend import BackendOutput, Capabilities, TrainingPass
from ...errors import DecisionError
from ...types import Choice, Noul, Request, Score
from ..qwen import open_images, precision
from ..readout import Features, Readout, pick, read_out
from .config import Settings, settings
from .prompt import PICTURE_TOKEN, packed_sets, picture_size, prefix, state_text, suffix
from .readout import option_token_ids

# Tokens per set by default (the state and the set's questions); the
# context is the most.
MAX_INPUT_TOKENS = 32768
INPUT_TOKEN_LIMIT = 128000
# Every option code up to this many options resolves to a token (the
# release's fallback codes included, measured with its tokenizer); score
# questions take 2 to 10 levels.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
# Pictures are enlarged to at least this many pixels by the processor, so a
# lower cap would change nothing.
MIN_IMAGE_PIXELS = 65536
REQUIRED_FILES = ("config.json", "tokenizer.json", "calibration.json")
VISION_PREFIX = "model.vision_tower."
# Tokens per batch of set rows (rows times the widest row).
BATCH_TOKENS = 32768
# Where a cut state's head and tail meet, and the tokens left for it and for
# how the cut text tokenizes at its edges.
CUT_MARK = "\n...\n"
CUT_MARGIN = 8


def has_vision_weights(path: Path) -> bool:
    """Whether the folder's weights hold the vision tower (read from the headers)."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = json.loads(index.read_text()).get("weight_map", {})
        return any(name.startswith(VISION_PREFIX) for name in names)
    for file in sorted(path.glob("model*.safetensors")):
        with file.open("rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            if any(name.startswith(VISION_PREFIX) for name in json.loads(handle.read(size))):
                return True
    return False


def question_dict(question_id: str, question) -> dict[str, Any]:
    """A request's question as the prompt code takes it. Missing, empty or blank
    instructions become the question id; score levels that are objects or lists
    become their text, as d1-3B shows them (the release refuses both)."""
    data = question.model_dump()
    instructions = data.get("instructions")
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        data["instructions"] = question_id
    if isinstance(question, Score):
        data["criteria"] = [
            level if isinstance(level, str) else str(level) for level in data["criteria"]
        ]
    return data


def option_keys(question) -> list[str]:
    if isinstance(question, Noul):
        return ["true", "false"]
    if isinstance(question, Choice):
        return list(question.criteria)
    return [str(i) for i in range(len(question.criteria))]


@dataclass
class Packed:
    """One set: its token ids, state length, and per question its span and answer slot."""

    question_ids: list[str]
    ids: list[int]
    state_length: int
    spans: list[list[int]]
    ends: list[int]


@dataclass
class Encoded:
    """A request's sets and, per question id, what its readout needs."""

    sets: list[Packed]
    questions: dict[str, dict[str, Any]]  # as the prompt code takes them
    option_ids: dict[str, list[str]]  # the answer's keys, in option order
    truncated: bool = False

    @property
    def input_tokens(self) -> int:
        return sum(len(packed.ids) for packed in self.sets)


@dataclass
class Pictures:
    """A request's pictures: each one's image tokens and, once decoded, its patches."""

    tokens: list[int] = field(default_factory=list)
    prepared: list = field(default_factory=list)


class D1Lfm3Backend:
    def __init__(
        self,
        name: str,
        model,
        tokenizer: Tokenizer,
        settings: Settings,
        capabilities: Capabilities,
        path: Path | None = None,
    ):
        """``model`` None gives a backend that only encodes."""
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.capabilities = capabilities
        self.path = path
        self._vision = None
        self._image_config = None

    @property
    def vision(self):
        """The vision tower and projector, loaded on first use."""
        if self._vision is None:
            self._vision = load_vision(self.path)
        return self._vision

    @property
    def image_config(self):
        """The release's processor settings (patches, merge, normalization)."""
        if self._image_config is None:
            from ...backbones.qwen3_5.preprocess import ImageConfig

            self._image_config = ImageConfig.from_folder(self.path)
        return self._image_config

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    # Pictures

    def open_pictures(self, request: Request) -> tuple[list, Pictures]:
        """The request's pictures opened (headers only) and their image tokens."""
        if not request.images:
            return [], Pictures()
        from ...backbones.qwen3_5.preprocess import image_tokens

        opened = open_images(request.images)
        tokens = []
        for index, image in opened:
            width, height = self.picture_size(image)
            try:
                tokens.append(image_tokens(height, width, self.image_config))
            except ValueError as error:
                raise DecisionError(str(error), param=f"images.{index}") from None
        return opened, Pictures(tokens)

    def picture_size(self, image) -> tuple[int, int]:
        return picture_size(image.width, image.height, self.capabilities.max_image_pixels)

    def prepare_pictures(self, opened: list, pictures: Pictures) -> None:
        """Decode each picture, scale it to the cap and cut it into patches."""
        from PIL import Image

        from ...backbones.qwen3_5.preprocess import prepare_image
        from ...images import decode_image

        for index, image in opened:
            decoded = decode_image(image, index)
            size = self.picture_size(decoded)
            if size != decoded.size:
                decoded = decoded.resize(size, Image.Resampling.BICUBIC)
            pictures.prepared.append(prepare_image(decoded, self.image_config))

    def picture_features(self, pictures: Pictures) -> mx.array | None:
        """The projector's rows for every image token of the request, in order."""
        if not pictures.prepared:
            return None
        import numpy as np

        pixels = np.concatenate([p.pixel_values for p in pictures.prepared])
        features = self.vision(mx.array(pixels), [p.grid for p in pictures.prepared])
        mx.eval(features)
        return features

    # Prompts and sets

    def cut(self, state, room: int) -> str | None:
        """``state`` cut to about ``room`` tokens around ``CUT_MARK``, or None if it fits
        (or cannot be cut)."""
        text = state_text(state)
        if text is None or room <= 0:
            return None
        ids = self._encode_text(text)
        if len(ids) <= room:
            return None
        head = room // 2
        decode = self.tokenizer.decode
        return (
            decode(ids[:head], skip_special_tokens=False)
            + CUT_MARK
            + decode(ids[len(ids) - (room - head) :], skip_special_tokens=False)
        )

    def _packed(self, state, questions: dict, image_tokens) -> list[Packed]:
        size = self.settings.set_size
        ids = list(questions)
        sets = packed_sets(
            state, list(questions.values()), self.tokenizer, size, list(image_tokens)
        )
        return [
            Packed(
                ids[n * size : (n + 1) * size], s["ids"], s["state_length"], s["spans"], s["ends"]
            )
            for n, s in enumerate(sets)
        ]

    def encode(self, request: Request, image_tokens=()) -> Encoded:
        """The request's sets, checked against the input limit (the state cut to fit
        if needed). ``image_tokens``: each picture's image tokens."""
        questions = {
            question_id: question_dict(question_id, question)
            for question_id, question in request.questions.items()
        }
        for question_id, data in questions.items():
            try:
                suffix(data, self.tokenizer)
            except ValueError as error:
                raise DecisionError(str(error), param=f"questions.{question_id}") from None
        limit = self.capabilities.max_input_tokens
        sets = self._packed(request.state, questions, image_tokens)
        cut = False
        if max(len(packed.ids) for packed in sets) > limit:
            asked = max(len(packed.ids) - packed.state_length for packed in sets)
            fixed = len(self._encode_text(prefix("", list(image_tokens))))
            room = limit - fixed - asked - CUT_MARGIN
            if (text := self.cut(request.state, room)) is not None:
                sets, cut = self._packed(text, questions, image_tokens), True
        for packed in sets:
            if len(packed.ids) > limit:
                count = len(packed.question_ids)
                what = (
                    "the question with its state needs"
                    if count == 1
                    else f"the {count} questions read together from this one on need"
                )
                after = " after cutting the state" if cut else ""
                raise DecisionError(
                    f"{what} {len(packed.ids)} tokens{after}; the limit is {limit}",
                    param=f"questions.{packed.question_ids[0]}",
                )
        option_ids = {qid: option_keys(q) for qid, q in request.questions.items()}
        return Encoded(sets, questions, option_ids, cut)

    # The pass

    def hidden_states(self, sets: list[Packed], features: mx.array | None = None) -> list:
        """The final hidden state (float32) at each question's answer slot, set by set
        in question order. ``features`` replace the image tokens of every row."""
        out = []
        start = 0
        while start < len(sets):
            # As many rows as fit the budget; at least one.
            end = start + 1
            widest = len(sets[start].ids)
            while end < len(sets):
                wider = max(widest, len(sets[end].ids))
                if (end + 1 - start) * wider > BATCH_TOKENS:
                    break
                widest = wider
                end += 1
            last = self.rows_hidden(sets[start:end], features)
            mx.eval(last)
            out += [last[i] for i in range(last.shape[0])]
            start = end
        return out

    def rows_hidden(self, chunk: list[Packed], features: mx.array | None = None) -> mx.array:
        """One right-padded batch of sets: the final hidden states (float32,
        ``[answer slots, width]``) at every row's answer slots in order, not
        evaluated. ``features`` replace the image tokens of every row."""
        image = self.tokenizer.token_to_id(PICTURE_TOKEN)
        widest = max(len(packed.ids) for packed in chunk)
        ids = [packed.ids + [0] * (widest - len(packed.ids)) for packed in chunk]
        question = [[-1] * widest for _ in chunk]
        for row, packed in zip(question, chunk, strict=True):
            for index, (first, last) in enumerate(packed.spans):
                row[first:last] = [index] * (last - first)
        layout = SetLayout(
            [len(packed.ids) for packed in chunk],
            [packed.state_length for packed in chunk],
            mx.array(question, dtype=mx.int32),
        )
        tokens = mx.array(ids)
        if features is None:
            hidden = self.model(tokens, layout=layout)
        else:
            embeddings = self.model.model.embed_tokens(tokens)
            # Every row starts with the same pictures, so their slots agree.
            slots = [i for i, token in enumerate(chunk[0].ids) if token == image]
            if len(slots) != features.shape[0]:
                raise ValueError(f"{len(slots)} image tokens for {features.shape[0]} picture rows")
            embeddings[:, mx.array(slots)] = features.astype(embeddings.dtype)
            hidden = self.model(input_embeddings=embeddings, layout=layout)
        rows = mx.array([n for n, packed in enumerate(chunk) for _ in packed.ends])
        slots = mx.array([e for packed in chunk for e in packed.ends])
        return hidden[rows, slots].astype(mx.float32)

    def passes(self, request: Request) -> tuple[Encoded, dict[str, mx.array]]:
        """The request's encoding and each question's last hidden state by id."""
        # Picture sizes come from the file headers, so the input limit is
        # checked before any picture is decoded.
        opened, pictures = self.open_pictures(request)
        encoded = self.encode(request, pictures.tokens)
        self.prepare_pictures(opened, pictures)
        hidden = self.hidden_states(encoded.sets, self.picture_features(pictures))
        ids = [qid for packed in encoded.sets for qid in packed.question_ids]
        return encoded, dict(zip(ids, hidden, strict=True))

    # LoRA training (``backend.Adaptable``)

    @property
    def decoder_layer(self) -> type:
        return type(self.model.layers[0])

    def network(self):
        return self.model

    def adapter_targets(self) -> list[tuple[str, Any, tuple[str, ...]]]:
        """The MLP projections of every decoder layer."""
        return [
            (f"layers.{index}.feed_forward", layer.feed_forward, ("w1", "w3", "w2"))
            for index, layer in enumerate(self.model.layers)
        ]

    def training_passes(self, request: Request) -> list[TrainingPass]:
        """One pass per set; the pictures' rows computed now (and evaluated),
        shared by the request's sets."""
        opened, pictures = self.open_pictures(request)
        encoded = self.encode(request, pictures.tokens)
        self.prepare_pictures(opened, pictures)
        features = self.picture_features(pictures)
        return [
            TrainingPass(
                packed.ids,
                {
                    qid: self.question_features(
                        encoded.questions[qid], encoded.option_ids[qid], None, encoded.truncated
                    )
                    for qid in packed.question_ids
                },
                (packed, features),
                images=features is not None,
                truncated=encoded.truncated,
            )
            for packed in encoded.sets
        ]

    def pass_hidden(self, passes: list[TrainingPass]) -> list[mx.array]:
        """Each pass's hidden states at its questions' answer slots (``[questions,
        width]``, float32): a pass with pictures alone, text passes as one batch."""
        if any(p.images for p in passes) and len(passes) != 1:
            raise ValueError("a pass with images runs alone")
        hidden = self.rows_hidden([p.frozen[0] for p in passes], passes[0].frozen[1])
        out, start = [], 0
        for p in passes:
            out.append(hidden[start : start + len(p.questions)])
            start += len(p.questions)
        return out

    # The readout

    def question_features(
        self, question: dict, option_ids: list[str], hidden: mx.array, truncated: bool = False
    ) -> Features:
        """The rows of the tokens that stand for the question's options, each
        option's among them, and the hidden state at its last token."""
        groups = option_token_ids(question, self.tokenizer)
        tokens = sorted({token for group in groups for token in group})
        rows = self.model.output_embedding_rows(mx.array(tokens)).astype(mx.float32)
        indices = [[tokens.index(token) for token in group] for group in groups]
        return Features(option_ids, hidden, rows, indices, self.settings.temperature, truncated)

    def features(self, request: Request) -> dict[str, Features]:
        """What each question's readout reads, before the readout."""
        encoded, hidden = self.passes(request)
        return {
            qid: self.question_features(
                encoded.questions[qid], encoded.option_ids[qid], hidden[qid], encoded.truncated
            )
            for qid in request.questions
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
        """``readouts`` replace the option tokens' rows of the questions (by id)
        with the same options; ``finish`` has no step to leave out here."""
        encoded, hidden = self.passes(request)
        answers = {}
        for qid in request.questions:
            option_ids = encoded.option_ids[qid]
            features = self.question_features(encoded.questions[qid], option_ids, hidden[qid])
            values = read_out(features, pick(readouts, qid, option_ids))
            answers[qid] = dict(zip(option_ids, values, strict=True))
        return BackendOutput(
            probabilities=answers,
            input_tokens=encoded.input_tokens,
            truncated=encoded.truncated,
        )


def _input_limit(max_input_tokens: int | None) -> int:
    if max_input_tokens is None:
        return MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= INPUT_TOKEN_LIMIT:
        raise ValueError(
            f"max_input_tokens must be between 1 and {INPUT_TOKEN_LIMIT} (the model's context)"
        )
    return max_input_tokens


def _pixel_limit(found: Settings, max_image_pixels: int | None) -> int:
    if max_image_pixels is None:
        return found.max_pixels
    if not MIN_IMAGE_PIXELS <= max_image_pixels <= found.max_pixels:
        raise ValueError(
            f"max_image_pixels must be between {MIN_IMAGE_PIXELS} and {found.max_pixels} "
            "(the release's cap)"
        )
    return max_image_pixels


def _capabilities(max_input_tokens: int, images: bool, max_image_pixels: int) -> Capabilities:
    return Capabilities(
        max_input_tokens=max_input_tokens,
        max_choice_options=MAX_CHOICE_OPTIONS,
        max_score_levels=MAX_SCORE_LEVELS,
        supports_images=images,
        max_image_pixels=max_image_pixels if images else None,
        truncates_input=True,
    )


def capabilities(path: Path, options: dict) -> Capabilities:
    """What ``load(path, **options)`` gives the backend, from the folder's JSON
    files and weight headers."""
    limit = _input_limit(options.get("max_input_tokens"))
    found = settings(path)
    pixels = _pixel_limit(found, options.get("max_image_pixels"))
    return _capabilities(limit, has_vision_weights(path), pixels)


def load(
    path: Path,
    max_input_tokens: int | None = None,
    vision: bool = False,
    max_image_pixels: int | None = None,
    fit_weights: Path | None = None,
) -> D1Lfm3Backend:
    """Load d1-2B from a release folder.

    ``max_input_tokens`` sets the limit per set (the state, its pictures and
    the set's questions): 32,768 by default, up to 128,000, the model's
    context; a state too long for it with the request's largest set is cut
    around a "..." line, keeping its start and end. ``vision=True`` loads the
    vision tower now, not with the first picture. ``max_image_pixels`` lowers
    the release's cap on pictures (1,638,400 pixels), down to 65,536.
    ``fit_weights`` is a LoRA fit's ``fit.safetensors``: its adapters are fused
    into the text model's weights (applied unfused when they are quantized).
    """
    limit = _input_limit(max_input_tokens)
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    found = settings(path)
    pixels = _pixel_limit(found, max_image_pixels)
    images = has_vision_weights(path)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    model = load_text_model(path)
    backend = D1Lfm3Backend(
        path.resolve().name,
        model,
        tokenizer,
        found,
        _capabilities(limit, images, pixels),
        path,
    )
    backend.precision = precision(path, model)
    if fit_weights is not None:
        from ...adapters import load_adapters

        load_adapters(backend.adapter_targets(), fit_weights)
        mx.eval(model.parameters())
    if vision and images:
        backend.vision  # noqa: B018 (loads it)
    return backend
