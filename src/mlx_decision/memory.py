"""Whether a model fits in memory, checked before its weights are read.

A model fits when the weights it loads, plus 10% for working memory
(activations, the head's buffers), are within the GPU's recommended
working set: past that, macOS starts swapping. Sizes come from the
safetensors headers, so nothing is loaded to find out.
"""

import json
import shlex
import struct
from collections.abc import Iterable
from pathlib import Path

from .registry import MARKER_FILE, detect_family, resolve

MARGIN = 1.1
# The bit widths a model can be stored quantized with, largest first.
QUANTIZED_BITS = (8, 4)
# Affine quantization with groups of 64 adds a bf16 scale and bias per group.
GROUP_OVERHEAD_BITS = 0.5
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


def read_header(file: Path) -> dict[str, dict]:
    """The tensors of a safetensors file: name -> {"dtype", "shape", ...}."""
    with open(file, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def element_count(shape: list[int]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count


def tensor_bytes(file: Path, load_dtype: str | None = None) -> int:
    """Bytes the tensors in ``file`` take once loaded.

    ``load_dtype`` (``"float16"``, ...) is the precision floating-point
    tensors are converted to when loading; None keeps them as stored.
    """
    total = 0
    for entry in read_header(file).values():
        count = element_count(entry["shape"])
        dtype = entry["dtype"]
        if load_dtype is not None and dtype in FLOAT_DTYPES:
            total += count * LOAD_DTYPE_BYTES[load_dtype]
        else:
            total += count * DTYPE_BYTES[dtype]
    return total


def quantizable_bytes(path: Path) -> int:
    """Bytes of the tensors in the model at ``path`` its family's converter quantizes."""
    family = detect_family(path)
    total = 0
    for file, _ in weight_files(path, {}):
        for name, entry in read_header(file).items():
            if family.quantizable is None or family.quantizable(name, entry["shape"]):
                total += element_count(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]
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


FLOAT_NAMES = {"F64": "fp64", "F32": "fp32", "F16": "fp16", "BF16": "bf16"}
LOAD_DTYPE_NAMES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def weights_summary(path: Path, options: dict | None = None) -> tuple[int, str | None]:
    """``(parameters, precision)`` of the weights a model loads, from the headers.

    A quantized weight (packed ``U32`` beside its ``.scales``) counts the
    parameters it packs, read from ``config.json``'s ``quantization`` (its
    ``bits``, or a module's own entry); its scales and biases are not
    parameters. Precision is "8-bit" or "4-bit" when quantized ("mixed 4 /
    8-bit" for more than one width), else the floating-point type most
    parameters load in ("bf16", "fp16", "fp32"); None when it cannot be told.
    """
    try:
        config = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        config = {}
    quantization = config.get("quantization") if isinstance(config, dict) else None
    quantization = quantization if isinstance(quantization, dict) else {}
    parameters = 0
    widths: set[int] = set()
    floats: dict[str, int] = {}
    for file, load_dtype in weight_files(path, options or {}):
        header = read_header(file)
        for name, entry in header.items():
            prefix, _, kind = name.rpartition(".")
            packed = header.get(f"{prefix}.weight", {}).get("dtype") == "U32"
            if kind in ("scales", "biases") and packed and f"{prefix}.scales" in header:
                continue  # a quantized weight's scales and biases
            count = element_count(entry["shape"])
            if kind == "weight" and packed and f"{prefix}.scales" in header:
                own = quantization.get(prefix)
                bits = own.get("bits") if isinstance(own, dict) else quantization.get("bits")
                if not isinstance(bits, int) or bits <= 0:
                    return parameters, None
                widths.add(bits)
                parameters += count * 32 // bits
                continue
            parameters += count
            if entry["dtype"] in FLOAT_DTYPES:
                kind_name = (
                    LOAD_DTYPE_NAMES[load_dtype] if load_dtype else FLOAT_NAMES[entry["dtype"]]
                )
                floats[kind_name] = floats.get(kind_name, 0) + count
    if widths:
        if len(widths) == 1:
            return parameters, f"{next(iter(widths))}-bit"
        return parameters, "mixed " + " / ".join(str(bits) for bits in sorted(widths)) + "-bit"
    if floats:
        return parameters, max(floats, key=floats.get)
    return parameters, None


def quantized_size(weights: int, bits: int) -> int:
    """About the size of 16-bit ``weights`` quantized to ``bits``, every
    tensor counted as quantized, which the small unquantized parts barely change."""
    return round(weights * (bits + GROUP_OVERHEAD_BITS) / 16)


def gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def check_fits(
    path: Path,
    options: dict | None = None,
    budget: int | None = None,
    name: str | None = None,
    extra: int = 0,
    source: str | None = None,
    extra_for: str = "kept prefixes",
    override: str | None = "To load it anyway, turn off the memory check (--no-memory-check).",
) -> int:
    """Raise ``ModelTooLargeError`` unless the model fits; returns the bytes it needs.

    ``budget`` defaults to the GPU's recommended working set. ``name`` is
    how the model is named in the message (default: the folder's name).
    ``extra`` is memory the model takes on top of its weights and margin
    (``extra_for``: kept prefixes, or a training run's share). ``source`` is
    what the ``convert`` hint passes to ``-m`` (default: ``name`` if given,
    else the path). ``override`` ends the message (None: nothing to turn off).
    """
    weights = weight_bytes(path, options)
    required = round(weights * MARGIN) + extra
    limit_name = "the memory budget"
    if budget is None:
        budget, limit_name = working_set(), "this Mac's GPU working set"
    if required <= budget:
        return required
    source = source or name or str(path)
    name = name or path.resolve().name
    message = (
        f"{name} needs about {gb(required)} ({gb(weights)} of weights plus 10%"
        + (f", plus {gb(extra)} for {extra_for}" if extra else "")
        + f"); {limit_name} is {gb(budget)}."
    )
    # A converted folder (marker file) cannot be converted again.
    if detect_family(path).converter is not None and not (path / MARKER_FILE).exists():
        quantizable = quantizable_bytes(path)
        sizes = {
            bits: round((weights - quantizable + quantized_size(quantizable, bits)) * MARGIN)
            + extra
            for bits in QUANTIZED_BITS
        }
        bits = next((bits for bits, size in sizes.items() if size <= budget), None)
        if bits is not None:
            flags = "-q" if bits == QUANTIZED_BITS[0] else f"-q --bits {bits}"
            command = f"mlx-decision convert -m {shlex.quote(source)} {flags}"
            message += (
                f" Make a quantized copy first: {command}"
                f" ({bits}-bit, needs about {gb(sizes[bits])})."
            )
        else:
            smallest = QUANTIZED_BITS[-1]
            message += f" Even a {smallest}-bit copy would need about {gb(sizes[smallest])}."
    if override:
        message += f" {override}"
    raise ModelTooLargeError(message, required, budget)
