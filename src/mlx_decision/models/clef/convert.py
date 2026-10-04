"""Write a Clef model as an MLX folder, with the backbone optionally quantized.

The joint schema head stays as released: it is small and makes the decision.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import (
    load_text_model,
    output_embeddings_path,
    quantizable_layers,
    quantize_text_model,
    save_text_model,
)
from ...calibration import calibration_requests
from ...mixed import (
    allocate,
    average_bits,
    base_bits,
    group_units,
    measure_sensitivity,
)
from ...registry import MARKER_FILE
from ...types import parse_request

FORMAT_VERSION = 1
COPIED = ("joint_head.safetensors", "joint_head_config.json", "tokenizer.json", "LICENSE")
SENSITIVITY_FILE = "quantization_sensitivity.json"
CALIBRATION = "builtin-1"


def convert(
    path: Path,
    output: Path,
    source: str,
    bits: int | None = None,
    group_size: int = 64,
    quantize_output_embeddings: bool = True,
    target_bits: float | None = None,
    progress: Callable[[int, int, object, float], None] | None = None,
) -> None:
    """Convert; ``target_bits`` switches to mixed precision (``bits`` is then ignored)."""
    config = json.loads((path / "config.json").read_text())
    if "quantization" in config:
        raise ValueError(f"{path} is already quantized; convert the original release")
    mixed = None
    if target_bits is not None:
        backbone, mixed = _mixed_plan(
            path, target_bits, group_size, quantize_output_embeddings, progress
        )
        bits = mixed["base_bits"]
    else:
        backbone = load_text_model(path)
    if bits is not None:
        layer_bits = mixed["layer_bits"] if mixed else None
        config["quantization"] = quantize_text_model(
            backbone,
            bits,
            group_size,
            output_embeddings=quantize_output_embeddings,
            layer_bits=layer_bits,
        )
        mx.eval(backbone.parameters())
    save_text_model(backbone, output, config)
    for name in COPIED:
        if (path / name).exists():
            shutil.copy2(path / name, output / name)
    marker = {
        "family": "clef",
        "format": FORMAT_VERSION,
        "source": source,
        "quantization": (
            {k: config["quantization"][k] for k in ("group_size", "bits", "mode")}
            if bits is not None
            else None
        ),
        "quantized_output_embeddings": bits is not None and quantize_output_embeddings,
    }
    if mixed:
        marker["mixed"] = {
            "target_bits": target_bits,
            "average_bits": mixed["report"]["average_bits"],
            "calibration": CALIBRATION,
        }
        (output / SENSITIVITY_FILE).write_text(json.dumps(mixed["report"], indent=2) + "\n")
    (output / MARKER_FILE).write_text(json.dumps(marker, indent=2) + "\n")


def _mixed_plan(path, target_bits, group_size, quantize_output_embeddings, progress):
    """Measure sensitivity with the whole model, then choose bits per layer."""
    from .model import load

    backend = load(path)
    backbone = backend.backbone
    base = base_bits(target_bits)
    layers = quantizable_layers(backbone, group_size)
    if not quantize_output_embeddings:
        layers.pop(output_embeddings_path(backbone), None)
    units, small = group_units(layers)
    requests = [parse_request(request) for request in calibration_requests()]

    def evaluate() -> list[list[float]]:
        vectors = []
        for request in requests:
            for probabilities in backend.score(request).probabilities.values():
                vectors.append(list(probabilities.values()))
        return vectors

    sensitivity = measure_sensitivity(
        backbone, units, base, group_size, evaluate, progress=progress
    )
    unit_bits = allocate(units, sensitivity, base, target_bits)
    layer_bits = {path: unit_bits[unit.name] for unit in units for path in unit.layers}
    layer_bits.update(dict.fromkeys(small, 8))
    report = {
        "calibration": CALIBRATION,
        "base_bits": base,
        "target_bits": target_bits,
        "average_bits": round(average_bits(units, unit_bits), 3),
        "small_layers": {"bits": 8, "layers": small},
        "units": [
            {
                "name": unit.name,
                "params": unit.params,
                "sensitivity": sensitivity[unit.name],
                "bits": unit_bits[unit.name],
            }
            for unit in units
        ],
    }
    return backbone, {"base_bits": base, "layer_bits": layer_bits, "report": report}
