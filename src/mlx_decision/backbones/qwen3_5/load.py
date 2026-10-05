"""Load, quantize and save a Qwen3.5 text model (Hugging Face or MLX folders)."""

import json
from collections.abc import Mapping
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

    def predicate(path: str, module: nn.Module) -> bool | dict:
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % group_size:
            return False
        if path not in layer_bits:
            return True
        if layer_bits[path] is None:
            return False
        if layer_bits[path] == bits:
            return True
        quantization[path] = {"group_size": group_size, "bits": layer_bits[path], "mode": mode}
        return quantization[path]

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate)
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
