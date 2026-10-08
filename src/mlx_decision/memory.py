"""Whether a model fits in memory, checked before its weights are read.

A model fits when the weights it loads, plus 10% for working memory
(activations, the head's buffers), are within the GPU's recommended
working set: past that, macOS starts swapping. Sizes come from the
safetensors headers, so nothing is loaded to find out.
"""

import json
import struct
from collections.abc import Iterable
from pathlib import Path

from .registry import MARKER_FILE, detect_family, resolve

MARGIN = 1.1
# Bytes per element of the safetensors dtypes.
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip
FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
# Load dtypes as the families name them.
LOAD_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class ModelTooLargeError(ValueError):
    """A model that needs more memory than the limit allows."""

    def __init__(self, message: str, required: int, available: int):
        super().__init__(message)
        self.required = required
        self.available = available


def working_set() -> int:
    """The GPU's recommended working set in bytes."""
    import mlx.core as mx

    return int(mx.device_info()["max_recommended_working_set_size"])


def tensor_bytes(file: Path, load_dtype: str | None = None) -> int:
    """Bytes the tensors in ``file`` take once loaded.

    ``load_dtype`` (``"float16"``, ...) is the precision floating-point
    tensors are converted to when loading; None keeps them as stored.
    """
    with open(file, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    total = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for size in entry["shape"]:
            count *= size
        dtype = entry["dtype"]
        if load_dtype is not None and dtype in FLOAT_DTYPES:
            total += count * LOAD_DTYPE_BYTES[load_dtype]
        else:
            total += count * DTYPE_BYTES[dtype]
    return total


def stored_weights(path: Path, options: dict) -> list[tuple[Path, str | None]]:
    """The default weight files: every ``*.safetensors`` at the top of the folder, as stored."""
    return [(file, None) for file in sorted(path.glob("*.safetensors"))]


def weight_files(path: Path, options: dict) -> Iterable[tuple[Path, str | None]]:
    family = detect_family(path)
    lister = resolve(family.weight_files) if family.weight_files else stored_weights
    return lister(path, options)


def weight_bytes(path: Path, options: dict | None = None) -> int:
    """Bytes of the weights the model's family loads from ``path`` with ``options``."""
    return sum(tensor_bytes(file, dtype) for file, dtype in weight_files(path, options or {}))


def required_bytes(path: Path, options: dict | None = None) -> int:
    """The memory a model is taken to need: its weights plus the margin."""
    return round(weight_bytes(path, options) * MARGIN)


def gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def check_fits(
    path: Path,
    options: dict | None = None,
    budget: int | None = None,
    name: str | None = None,
    extra: int = 0,
) -> int:
    """Raise ``ModelTooLargeError`` unless the model fits; returns the bytes it needs.

    ``budget`` defaults to the GPU's recommended working set. ``name`` is
    how the model is named in the message (default: the folder's name).
    ``extra`` is memory the model takes on top of its weights and margin
    (kept prefixes).
    """
    weights = weight_bytes(path, options)
    required = round(weights * MARGIN) + extra
    limit_name = "the memory budget"
    if budget is None:
        budget, limit_name = working_set(), "this Mac's GPU working set"
    if required <= budget:
        return required
    name = name or path.resolve().name
    message = (
        f"{name} needs about {gb(required)} ({gb(weights)} of weights plus 10%"
        + (f", plus {gb(extra)} for kept prefixes" if extra else "")
        + f"); {limit_name} is {gb(budget)}."
    )
    # A converted folder (marker file) cannot be converted again.
    if detect_family(path).converter is not None and not (path / MARKER_FILE).exists():
        message += f" Make a quantized copy first: mlx-decision convert -m {name} -q (8-bit)."
    message += " To load it anyway, turn off the memory check (--no-memory-check)."
    raise ModelTooLargeError(message, required, budget)
