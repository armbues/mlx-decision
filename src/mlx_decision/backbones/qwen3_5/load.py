"""Load, quantize and save a Qwen3.5 text model (Hugging Face or MLX folders)."""

import copy
import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .qwen3_5 import Model, ModelArgs

SHARD_BYTES = 5 * 2**30
VISION_PREFIXES = ("model.visual.", "visual.")


def load_text_model(path: str | Path) -> Model:
    """Build the text model described by ``config.json`` and load its weights.

    Vision weights in the folder are ignored. Weights keep the precision they
    are stored in. A folder written by ``save_text_model`` with quantization
    has a ``quantization`` entry in its config; layers are quantized where
    the saved weights have scales.
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))

    weight_files = sorted(path.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no model*.safetensors files in {path}")
    weights = {}
    for file in weight_files:
        weights.update(mx.load(str(file)))

    weights = model.sanitize(weights)
    if (quantization := config.get("quantization")) is not None:
        quantize_like(model, quantization, weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


VISION_SHARD = "model-vision.safetensors"


def sanitize_vision_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """The vision weights among ``weights``, under the vision model's own names.

    The release's prefixes are removed and the 3-D patch kernel is flattened
    to the linear layer that replaces it; other weights are dropped.
    """
    sanitized = {}
    for key, value in weights.items():
        prefix = next((p for p in VISION_PREFIXES if key.startswith(p)), None)
        if prefix is None:
            continue
        key = key[len(prefix) :]
        if key == "patch_embed.proj.weight" and value.ndim == 5:
            value = value.reshape(value.shape[0], -1)
        sanitized[key] = value
    return sanitized


def vision_shards(path: str | Path) -> set[str]:
    """Names of the weight files that hold vision weights (all of them without an index)."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {f for name, f in weight_map.items() if name.startswith(VISION_PREFIXES)}
    return {file.name for file in path.glob("model*.safetensors")}


def read_vision_weights(path: str | Path) -> dict[str, mx.array]:
    """The vision tower's weights of a folder (see ``sanitize_vision_weights``); no NumPy."""
    path = Path(path)
    weights = {}
    for file in sorted(vision_shards(path)):
        weights.update(sanitize_vision_weights(mx.load(str(path / file))))
    if not weights:
        raise FileNotFoundError(f"no vision weights in {path}")
    return weights


def write_vision_weights(weights: Mapping[str, mx.array], path: str | Path) -> None:
    """Add vision weights to a folder written by ``save_text_model``.

    They go into their own shard, listed in ``model.safetensors.index.json``
    under the release's ``model.visual.`` names.
    """
    path = Path(path)
    named = {f"{VISION_PREFIXES[0]}{k}": v for k, v in weights.items()}
    mx.save_safetensors(str(path / VISION_SHARD), named, metadata={"format": "mlx"})
    index_file = path / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    index["weight_map"] = dict(
        sorted({**index["weight_map"], **dict.fromkeys(named, VISION_SHARD)}.items())
    )
    index["metadata"]["total_size"] += sum(v.nbytes for v in named.values())
    index_file.write_text(json.dumps(index, indent=2) + "\n")


def has_vision_weights(path: str | Path) -> bool:
    """Whether the folder's weight index lists vision tower weights."""
    index = Path(path) / "model.safetensors.index.json"
    if not index.exists():
        return False
    names = json.loads(index.read_text())["weight_map"]
    return any(name.startswith(VISION_PREFIXES) for name in names)


def quantize_like(model: nn.Module, quantization: dict, weights: dict) -> None:
    """Quantize the layers of ``model`` that are quantized in ``weights``.

    ``quantization`` holds the defaults (``group_size``, ``bits``, ``mode``)
    and, keyed by layer path, settings for layers that differ from them.
    """

    def predicate(path: str, module: nn.Module):
        if f"{path}.scales" not in weights:
            return False
        override = quantization.get(path)
        return override if isinstance(override, dict) else True

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization.get("mode", "affine"),
        class_predicate=predicate,
    )


def output_embeddings_path(model: Model) -> str:
    """Path of the output embedding matrix: ``lm_head``, or the input embeddings when tied."""
    if model.language_model.args.tie_word_embeddings:
        return "language_model.model.embed_tokens"
    return "language_model.lm_head"


def quantizable_layers(model: Model, group_size: int = 64) -> dict[str, int]:
    """Layers that can be quantized at ``group_size``, with their parameter counts."""
    return {
        path: module.weight.size
        for path, module in model.named_modules()
        if hasattr(module, "to_quantized") and module.weight.shape[-1] % group_size == 0
    }


def quantize_text_model(
    model: Model,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
    output_embeddings: bool = True,
    layer_bits: Mapping[str, int | None] | None = None,
) -> dict:
    """Quantize ``model`` in place; returns the config's ``quantization`` entry.

    Every linear and embedding layer whose input width is a multiple of
    ``group_size`` is quantized to ``bits``. With ``output_embeddings=False``
    the output embedding matrix keeps its precision. ``layer_bits`` maps
    layer paths to other bit widths, or to None to leave a layer as it is;
    those layers get their own entry in the returned config, the per-layer
    form mlx-lm uses.
    """
    layer_bits = dict(layer_bits or {})
    if not output_embeddings:
        layer_bits[output_embeddings_path(model)] = None
    quantization: dict = {"group_size": group_size, "bits": bits, "mode": mode}
    quantized = []

    def predicate(path: str, module: nn.Module) -> bool | dict:
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % group_size:
            return False
        if path in layer_bits and layer_bits[path] is None:
            return False
        quantized.append(path)
        if path not in layer_bits or layer_bits[path] == bits:
            return True
        quantization[path] = {"group_size": group_size, "bits": layer_bits[path], "mode": mode}
        return quantization[path]

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate)
    if not quantized:
        raise ValueError(f"no layer can be quantized with group size {group_size}")
    return quantization


def source_shards(path: str | Path) -> list[Path]:
    """The weight files of a folder in the order its index lists them."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = dict.fromkeys(json.loads(index.read_text())["weight_map"].values())
        return [path / name for name in names]
    return sorted(path.glob("model*.safetensors"))


def _safetensors_format(file: Path) -> str | None:
    """The ``format`` entry of a safetensors file's metadata ("pt", "mlx", ...)."""
    with open(file, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return (header.get("__metadata__") or {}).get("format")


def convert_text_weights(
    config: dict,
    output: str | Path,
    shards: Iterable[Path],
    bits: int | None = None,
    group_size: int = 64,
    mode: str = "affine",
    output_embeddings: bool = True,
) -> dict | None:
    """Write the text model of ``shards`` to ``output``, quantized when ``bits`` is set.

    Reads one source shard at a time and writes output shards of at most
    ``SHARD_BYTES`` as it goes, so the full model is never in memory; the
    tensors equal those of ``load_text_model`` + ``quantize_text_model`` +
    ``save_text_model``. ``shards`` may be a generator (a shard fetched just
    before it is read). Returns the config's ``quantization`` entry, or None.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # A lazily built skeleton (MLX allocates nothing until evaluated): which
    # layers get quantized, and the names and shapes the output must have.
    # Built from a copy: the model args rewrite parts of the config in place.
    model = Model(ModelArgs.from_dict(copy.deepcopy(config)))
    quantization = None
    quantized: set[str] = set()
    if bits is not None:
        quantization = quantize_text_model(model, bits, group_size, mode, output_embeddings)
        quantized = {path for path, module in model.named_modules() if hasattr(module, "scales")}
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}

    written: dict[str, str] = {}
    pending: dict[str, mx.array] = {}
    pending_bytes = 0
    files: list[Path] = []
    total = 0

    def flush() -> None:
        nonlocal pending_bytes
        file = output / f".model-{len(files) + 1:05d}.safetensors"
        mx.save_safetensors(str(file), pending, metadata={"format": "mlx"})
        files.append(file)
        written.update(dict.fromkeys(pending, file.name))
        pending.clear()
        pending_bytes = 0

    release = None
    for shard in shards:
        if release is None:
            release = _safetensors_format(shard) != "mlx"
        weights = model.sanitize(mx.load(str(shard)), release=release)
        for name, value in weights.items():
            layer = name.removesuffix(".weight")
            if layer in quantized and name != layer:
                parts = mx.quantize(value, group_size, bits, mode=mode)
                tensors = dict(zip(("weight", "scales", "biases"), parts, strict=False))
                tensors = {f"{layer}.{key}": array for key, array in tensors.items()}
            else:
                tensors = {name: value}
            for key, array in tensors.items():
                if key not in expected:
                    raise ValueError(f"{shard.name}: unexpected weight {key}")
                if array.shape != expected[key]:
                    raise ValueError(
                        f"{shard.name}: {key} has shape {array.shape}, expected {expected[key]}"
                    )
                mx.eval(array)
                if pending and pending_bytes + array.nbytes > SHARD_BYTES:
                    flush()
                pending[key] = array
                pending_bytes += array.nbytes
                total += array.nbytes
        del weights
    missing = sorted(set(expected) - set(written) - set(pending))
    if missing:
        raise ValueError(f"weights missing from the source: {', '.join(missing[:5])}")
    if pending:
        flush()

    names = {}
    for index, file in enumerate(files, 1):
        name = f"model-{index:05d}-of-{len(files):05d}.safetensors"
        file.rename(output / name)
        names[file.name] = name
    index = {
        "metadata": {"total_size": total},
        "weight_map": {key: names[file] for key, file in sorted(written.items())},
    }
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    if quantization is not None:
        config = {**config, "quantization": quantization}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return quantization


def save_text_model(model: Model, path: str | Path, config: dict) -> None:
    """Write ``config.json`` and the weights as ``model-*.safetensors`` shards."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, mx.array]] = [{}]
    size = 0
    for name, value in tree_flatten(model.parameters()):
        if size and size + value.nbytes > SHARD_BYTES:
            shards.append({})
            size = 0
        shards[-1][name] = value
        size += value.nbytes
    weight_map = {}
    for index, shard in enumerate(shards, 1):
        name = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(path / name), shard, metadata={"format": "mlx"})
        weight_map.update(dict.fromkeys(shard, name))
    index = {
        "metadata": {"total_size": sum(v.nbytes for _, v in tree_flatten(model.parameters()))},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
