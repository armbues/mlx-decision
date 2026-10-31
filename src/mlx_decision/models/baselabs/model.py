"""The Base Labs backend (elephant-1, mammoth-1): Qwen3.5 backbone with an answer-code readout.

The releases are Clef's layout (``model.language_model.*``, an untied
``lm_head``, a vision tower that is not loaded) with ``modeling_overrides``
in ``config.json`` saying how they were trained: block-bidirectional
attention in the full-attention layers, causal Gated DeltaNet layers, and
the LM head's logits at the answer codes as the readout (``slot_lm``). Only
the answer codes' rows of the LM head are read, in float32 (the release
upcasts the head); the rest of it stays on disk.
"""

import json
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import Capabilities
from ..prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache
from ..qwen import precision
from .config import check_config
from .prompt import CODES

# The card's training length: a question with the state over it is refused.
MAX_INPUT_TOKENS = 16384
INPUT_TOKEN_LIMIT = 262144
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
REQUIRED_FILES = ("config.json", "tokenizer.json", "readout.json")
HEAD = "lm_head.weight"


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


def head_file(path: Path) -> Path:
    """The weight file holding the LM head."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        name = json.loads(index.read_text())["weight_map"].get(HEAD)
        if name is None:
            raise ValueError(f"{path}: the weights have no {HEAD}")
        return path / name
    for file in sorted(path.glob("model*.safetensors")):
        if HEAD in _header(file)[0]:
            return file
    raise ValueError(f"{path}: the weights have no {HEAD}")


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
    ):
        """``rows`` are the answer codes' LM-head rows (float32, ``[codes,
        width]``); ``backbone`` None gives a backend that only encodes;
        ``prefix_cache_gb`` 0 keeps no prefixes between requests."""
        self.name = name
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.rows = rows
        self.path = path
        self.code_ids = code_token_ids(tokenizer)
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
        self.capabilities = _capabilities(max_input_tokens)


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weight files ``load`` reads from, kept as stored (it skips the vision
    tower and all but a few rows of the LM head)."""
    return [(file, None) for file in sorted(path.glob("model*.safetensors"))]


def _capabilities(max_input_tokens: int) -> Capabilities:
    return Capabilities(
        max_input_tokens=max_input_tokens,
        max_choice_options=MAX_CHOICE_OPTIONS,
        max_score_levels=MAX_SCORE_LEVELS,
        truncates_input=False,
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
    return _capabilities(limit)


def load(
    path: Path,
    max_input_tokens: int | None = None,
    prefix_cache_gb: float = DEFAULT_PREFIX_CACHE_GB,
) -> BaseLabsBackend:
    """Load elephant-1 or mammoth-1 from a release folder.

    ``max_input_tokens`` sets the limit per question (the state and that
    question's tokens): 16,384 by default, the models' training length, up
    to 262,144; longer input is refused, never cut. ``prefix_cache_gb`` is
    the memory for kept prefixes (the computed state after a request's
    state), so a request with a state seen recently computes only its
    questions; 0 turns keeping off (the questions of one request still share
    their prefix). Text only: the vision tower is not loaded.
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
    tokenizer = load_tokenizer(path)
    rows = read_rows(head_file(path), HEAD, code_token_ids(tokenizer))
    backbone = load_text_model(path, lm_head=False)
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
    )
    backend.precision = precision(path, backbone)
    return backend
