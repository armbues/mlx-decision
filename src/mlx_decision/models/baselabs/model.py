"""The Base Labs backend (elephant-1, mammoth-1, joey-1): Qwen3.5 with an answer-code readout.

The releases are Clef's layout (``model.language_model.*``, a stored
``lm_head``, a vision tower) with ``modeling_overrides``
in ``config.json`` saying how they were trained: block-bidirectional
attention in the full-attention layers, causal Gated DeltaNet layers, and
the LM head's logits at the answer codes as the readout (``slot_lm``). Only
the answer codes' rows of the LM head are read, in float32 (the release
upcasts the head); the rest of it stays on disk. joey-1's config ties the
head to the embeddings, yet it stores one too: as the release's
transformers, a stored head is read, the embeddings only without one.

Every question's prompt starts with the same tokens up to the end of the
state block (``prompt``). That prefix runs once per request, its
full-attention layers seeing it both ways, or comes from the kept prefixes
(``prefix``); the questions then run off copies of its cache, right-padded
in batches, each seeing the prefix and itself both ways but not the other
questions. A question's answer is the last token's hidden state times the
answer codes' rows of its options, normalized.

Images sit in the state block (``prompt``), so in the prefix: the vision
tower (loaded with the first image) fills each image's pads with its
patches, every token gets its (time, height, width) position, and the
prefix's full-attention layers see the image tokens both ways, as the
models were trained (the releases' training-side scoring; they serve text
only). The questions' positions continue after the images'. A kept prefix
is keyed with the images' pixels.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.cache import repeat_cache
from ...backbones.qwen3_5.load import has_vision_weights, load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Request
from ..prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache, prefix_key
from ..qwen import (
    IMAGE_PAD,
    count_image_tokens,
    image_config,
    image_inputs,
    image_pixel_limit,
    open_images,
    precision,
    prepare,
)
from ..readout import Features, Readout, pick, read_out, singletons
from .config import check_config
from .prompt import CODES, IMAGE_MARKERS, layout, options, render_question

# The card's training length: a question with the state over it is refused.
MAX_INPUT_TOKENS = 16384
INPUT_TOKEN_LIMIT = 262144
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
REQUIRED_FILES = ("config.json", "tokenizer.json", "readout.json")
HEAD = "lm_head.weight"
EMBEDDINGS = "model.language_model.embed_tokens.weight"
# A converted copy's answer-code rows (float32) and their token ids, instead of the head.
ROWS_FILE = "answer_rows.safetensors"
# Tokens one batch of questions holds at most: its rows times the prefix and
# the longest question (the prefix's cache is repeated per row).
BATCH_TOKENS = 32768


@dataclass
class Images:
    """A request's images: each one's token count, and once decoded, its patches."""

    tokens: list[int] = field(default_factory=list)
    prepared: list = field(default_factory=list)
    merge_size: int = 2


@dataclass
class Encoded:
    question_id: str
    option_ids: list[str]
    suffix: list[int]  # the question's tokens after the shared prefix


def code_token_ids(tokenizer) -> list[int]:
    """Each answer code's token: every code must be one token that decodes back to it."""
    ids = []
    for code in CODES:
        tokens = tokenizer.encode(code, add_special_tokens=False).ids
        if len(tokens) != 1 or tokenizer.decode(tokens) != code:
            raise ValueError(f"the answer code {code!r} is not a single token of the tokenizer")
        ids.append(tokens[0])
    if len(set(ids)) != len(ids):
        raise ValueError("two answer codes share a token")
    return ids


# safetensors dtypes and the NumPy dtype their bytes are read as.
_DTYPES = {"BF16": ("uint16", mx.bfloat16), "F16": ("float16", None), "F32": ("float32", None)}


def _header(file: Path) -> tuple[dict, int]:
    """A safetensors file's header and where its data starts."""
    import struct

    with open(file, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length)), 8 + length


def ties_head(path: Path) -> bool:
    """Whether the config ties the LM head to the embeddings. Only the text
    config's flag counts: the release loads the text model with that config
    alone, so a flag at the top level does not tie."""
    text = json.loads((path / "config.json").read_text()).get("text_config")
    return isinstance(text, dict) and text.get("tie_word_embeddings") is True


def _tensor_file(path: Path, name: str) -> Path | None:
    """The weight file holding the tensor ``name``, if any does."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        found = json.loads(index.read_text())["weight_map"].get(name)
        return path / found if found else None
    for file in sorted(path.glob("model*.safetensors")):
        if name in _header(file)[0]:
            return file
    return None


def head_source(path: Path) -> tuple[Path, str]:
    """The weight file and tensor the answer codes' rows come from, as the
    release's transformers picks them: the stored LM head (tied or not; it
    ties only equal tensors and otherwise keeps the stored one), else the
    embeddings when the config ties the head to them."""
    file = _tensor_file(path, HEAD)
    if file is not None:
        return file, HEAD
    if not ties_head(path):
        raise ValueError(
            f"{path}: the weights have no {HEAD}, and config.json does not tie it to the embeddings"
        )
    file = _tensor_file(path, EMBEDDINGS)
    if file is None:
        raise ValueError(f"{path}: the weights have neither {HEAD} nor {EMBEDDINGS}")
    return file, EMBEDDINGS


def read_rows(file: Path, name: str, rows: list[int]) -> mx.array:
    """Rows of a 2-D tensor in a safetensors file, in float32, reading only those rows."""
    import numpy as np

    header, start = _header(file)
    entry = header[name]
    if entry["dtype"] not in _DTYPES:
        raise ValueError(f"{file.name}: {name} is stored as {entry['dtype']}, not a float type")
    stored, view = _DTYPES[entry["dtype"]]
    count, width = entry["shape"]
    if max(rows) >= count:
        raise ValueError(f"{file.name}: {name} has {count} rows, row {max(rows)} asked for")
    first = entry["data_offsets"][0]
    data = np.memmap(file, dtype=stored, mode="r", offset=start + first, shape=(count, width))
    picked = mx.array(np.ascontiguousarray(data[rows]))
    del data
    if view is not None:
        picked = picked.view(view)
    return picked.astype(mx.float32)


class BaseLabsBackend:
    def __init__(
        self,
        name,
        backbone,
        tokenizer,
        rows: mx.array,
        max_input_tokens: int = MAX_INPUT_TOKENS,
        path: Path | None = None,
        prefix_cache_gb: float = 0.0,
        max_image_pixels: int | None = None,
    ):
        """``rows`` are the answer codes' LM-head rows (float32, ``[codes,
        width]``); ``backbone`` None gives a backend that only encodes;
        ``prefix_cache_gb`` 0 keeps no prefixes between requests. Images
        need the vision weights in ``path``; ``max_image_pixels`` lowers the
        releases' maximum."""
        self.name = name
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.rows = rows
        self.path = path
        self.code_ids = code_token_ids(tokenizer)
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
        self.max_image_pixels = max_image_pixels
        self.vision = None
        images = path is not None and has_vision_weights(path)
        pixels = image_pixel_limit(path, max_image_pixels) if images else None
        self.capabilities = _capabilities(max_input_tokens, pixels)

    def load_vision(self):
        """The vision tower, loaded on first use (it stays in its stored precision)."""
        if self.vision is None:
            # Imported here: the vision code needs NumPy, loaded only for images.
            from ...backbones.qwen3_5.vision import load_vision_model

            self.vision = load_vision_model(self.path)
        return self.vision

    def image_config(self):
        """The releases' processor settings, the maximum lowered by ``max_image_pixels``."""
        return image_config(self.path, self.max_image_pixels)

    def open_images(self, request: Request) -> tuple[list, Images]:
        """The request's images opened (headers only) and their token counts."""
        if not request.images:
            return [], Images()
        config = self.image_config()
        opened = open_images(request.images)
        tokens = [count_image_tokens(image, index, config) for index, image in opened]
        return opened, Images(tokens=tokens, merge_size=config.merge_size)

    def encode(
        self, request: Request, image_tokens: list[int] = ()
    ) -> tuple[list[int], list[Encoded]]:
        """The shared prefix's tokens (each image's pad expanded to its
        ``image_tokens``) and each question's tokens after it. Refuses the
        request if a question with the state is over the input limit, or if
        its text holds image markers while it has images."""
        questions = request.questions
        found = layout(
            self.tokenizer,
            request.state,
            [render_question(q) for q in questions.values()],
            images=len(image_tokens),
        )
        prefix = found.prefix
        if image_tokens:
            prefix = self._expand_images(prefix, found.suffixes, list(questions), image_tokens)
        encoded = [
            Encoded(question_id, options(question)[0], suffix)
            for (question_id, question), suffix in zip(
                questions.items(), found.suffixes, strict=True
            )
        ]
        limit = self.capabilities.max_input_tokens
        for question in encoded:
            tokens = len(prefix) + len(question.suffix)
            if tokens > limit:
                raise DecisionError(
                    f"the question with its state needs {tokens} tokens; "
                    f"the limit is {limit} and input is not cut",
                    param=f"questions.{question.question_id}",
                )
        return prefix, encoded

    def _expand_images(self, prefix, suffixes, question_ids, image_tokens) -> list[int]:
        """``prefix`` with each image's pad repeated for its tokens."""
        start, pad, end = (self.tokenizer.token_to_id(t) for t in IMAGE_MARKERS)
        markers = f"image markers ({', '.join(IMAGE_MARKERS)})"
        for question_id, suffix in zip(question_ids, suffixes, strict=True):
            if {start, pad, end} & set(suffix):
                # The tokenizer reads them as special tokens even in user text.
                raise DecisionError(
                    f"the question contains {markers}, which cannot be used with images",
                    param=f"questions.{question_id}",
                )
        count = len(image_tokens)
        if (prefix.count(start), prefix.count(pad), prefix.count(end)) != (count,) * 3:
            raise DecisionError(
                f"the state contains {markers}, which cannot be used in a request with images",
                param="state",
            )
        expanded, tokens = [], iter(image_tokens)
        for token in prefix:
            expanded += [pad] * next(tokens) if token == pad else [token]
        return expanded

    def prefix_state(self, prefix: list[int], images: Images | None = None):
        """The backbone's cache after ``prefix`` (with ``images``' patches in their
        pads): kept from an earlier request, or computed."""
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

    def batches(self, prefix: int, encoded: list[Encoded]) -> list[list[int]]:
        """The questions (by index, in order) in batches of at most ``BATCH_TOKENS``
        tokens (rows times the prefix and the longest question); at least one per batch."""
        batches: list[list[int]] = []
        width = 0
        for index, question in enumerate(encoded):
            wider = max(width, len(question.suffix))
            if batches and (len(batches[-1]) + 1) * (prefix + wider) <= BATCH_TOKENS:
                batches[-1].append(index)
                width = wider
            else:
                batches.append([index])
                width = len(question.suffix)
        return batches

    def last_hidden(
        self, prefix: list[int], encoded: list[Encoded], images: Images | None = None
    ) -> list[mx.array]:
        """Every question's final hidden state at its last token, in float32.
        With ``images`` (prepared) in the prefix, the questions' positions
        continue after the images'."""
        cache = self.prefix_state(prefix, images)
        after = None
        if images is not None and images.prepared:
            from ...backbones.qwen3_5.positions import position_ids

            pad = self.tokenizer.token_to_id(IMAGE_PAD)
            grids = [image.grid for image in images.prepared]
            widest = max(len(question.suffix) for question in encoded)
            # The questions are text: the same positions for every row.
            full = prefix + [0] * widest
            after = position_ids(full, pad, grids, images.merge_size)[:, :, len(prefix) :]
        hidden: list[mx.array] = [None] * len(encoded)
        for batch in self.batches(len(prefix), encoded):
            rows = [encoded[index].suffix for index in batch]
            width = max(len(row) for row in rows)
            ids = mx.array([row + [0] * (width - len(row)) for row in rows])
            padded = any(len(row) < width for row in rows)
            lengths = mx.array([len(row) for row in rows]) if padded else None
            positions = None
            if after is not None:
                positions = mx.broadcast_to(after[:, :, :width], (3, len(rows), width))
            states = self.backbone(
                ids, None, positions, cache=repeat_cache(cache, len(rows)), lengths=lengths
            )
            for row, index in enumerate(batch):
                hidden[index] = states[row, len(rows[row]) - 1].astype(mx.float32)
            mx.eval([hidden[index] for index in batch])
        return hidden

    def passes(self, request: Request) -> tuple[list[int], list[Encoded], list[mx.array]]:
        # Image sizes come from the file headers, so the input limit is checked
        # before any image is decoded or cut into patches.
        opened, images = self.open_images(request)
        prefix, encoded = self.encode(request, images.tokens)
        if not opened:
            return prefix, encoded, self.last_hidden(prefix, encoded)
        config = self.image_config()
        images.prepared = [prepare(image, index, config) for index, image in opened]
        return prefix, encoded, self.last_hidden(prefix, encoded, images)

    def question_features(self, question: Encoded, hidden: mx.array) -> Features:
        count = len(question.option_ids)
        return Features(question.option_ids, hidden, self.rows[:count], singletons(count), 1.0)

    def features(self, request: Request) -> dict[str, Features]:
        """What each question's readout reads, before the readout."""
        _, encoded, hidden = self.passes(request)
        return {
            q.question_id: self.question_features(q, last)
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
        """``readouts`` replace the answer codes' rows of the questions (by id) with
        the same options."""
        prefix, encoded, hidden = self.passes(request)
        probabilities = {}
        for question, last in zip(encoded, hidden, strict=True):
            features = self.question_features(question, last)
            readout = pick(readouts, question.question_id, question.option_ids)
            values = read_out(features, readout)
            probabilities[question.question_id] = dict(
                zip(question.option_ids, values, strict=True)
            )
        tokens = len(prefix) + sum(len(q.suffix) for q in encoded)
        return BackendOutput(probabilities=probabilities, input_tokens=tokens)


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weight files ``load`` reads from, kept as stored (it reads all but a
    few rows of the LM head; the vision tower, which shares them, with the
    first image)."""
    files = sorted(path.glob("model*.safetensors")) + [path / ROWS_FILE]
    return [(file, None) for file in files if file.exists()]


def answer_rows(path: Path, ids: list[int]) -> mx.array:
    """The answer codes' rows (float32): a converted copy's, checked against the
    tokenizer's code ids, or read from the release's LM head (``head_source``)."""
    file = path / ROWS_FILE
    if not file.exists():
        return read_rows(*head_source(path), ids)
    stored = mx.load(str(file))
    if stored["token_ids"].tolist() != ids:
        raise ValueError(f"{path}: {ROWS_FILE} is for other answer-code tokens")
    return stored["rows"].astype(mx.float32)


def _capabilities(max_input_tokens: int, max_image_pixels: int | None = None) -> Capabilities:
    """Images when ``max_image_pixels`` is given (the folder has vision weights)."""
    return Capabilities(
        max_input_tokens=max_input_tokens,
        max_choice_options=MAX_CHOICE_OPTIONS,
        max_score_levels=MAX_SCORE_LEVELS,
        truncates_input=False,
        supports_images=max_image_pixels is not None,
        max_image_pixels=max_image_pixels,
    )


def _input_limit(max_input_tokens: int | None) -> int:
    if max_input_tokens is None:
        return MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= INPUT_TOKEN_LIMIT:
        raise ValueError(f"max_input_tokens must be between 1 and {INPUT_TOKEN_LIMIT}")
    return max_input_tokens


def capabilities(path: Path, options: dict) -> Capabilities:
    """What ``load(path, **options)`` gives the backend, from the folder's config files."""
    limit = _input_limit(options.get("max_input_tokens"))
    check_config(path)
    if not has_vision_weights(path):
        if options.get("vision"):
            raise ValueError(f"{path} has no vision weights")
        return _capabilities(limit)
    return _capabilities(limit, image_pixel_limit(path, options.get("max_image_pixels")))


def load(
    path: Path,
    max_input_tokens: int | None = None,
    prefix_cache_gb: float = DEFAULT_PREFIX_CACHE_GB,
    vision: bool = False,
    max_image_pixels: int | None = None,
) -> BaseLabsBackend:
    """Load elephant-1, mammoth-1 or joey-1 from a release folder or a converted copy.

    ``max_input_tokens`` sets the limit per question (the state and that
    question's tokens): 16,384 by default, the models' training length, up
    to 262,144; longer input is refused, never cut. ``prefix_cache_gb`` is
    the memory for kept prefixes (the computed state after a request's
    state), so a request with a state seen recently computes only its
    questions; 0 turns keeping off (the questions of one request still share
    their prefix). Images: the vision tower loads with the first image
    request, or now with ``vision=True``; ``max_image_pixels`` lowers the
    releases' maximum of 16,777,216 pixels per image (larger ones are
    scaled down).
    """
    if prefix_cache_gb < 0:
        raise ValueError("prefix_cache_gb must be 0 (off) or more")
    max_input_tokens = _input_limit(max_input_tokens)
    # Checked before the backbone, which takes a while to load.
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    check_config(path)
    if vision and not has_vision_weights(path):
        raise ValueError(f"{path} has no vision weights")
    tokenizer = load_tokenizer(path)
    rows = answer_rows(path, code_token_ids(tokenizer))
    backbone = load_text_model(path, lm_head=False, causal=False)
    if rows.shape[1] != backbone.language_model.args.hidden_size:
        raise ValueError(f"{path}: the LM head does not match the backbone's width")
    mx.eval(rows)
    backend = BaseLabsBackend(
        path.resolve().name,
        backbone,
        tokenizer,
        rows,
        max_input_tokens,
        path,
        prefix_cache_gb=prefix_cache_gb,
        max_image_pixels=max_image_pixels,
    )
    backend.precision = precision(path, backbone)
    if vision:
        backend.load_vision()
    return backend
