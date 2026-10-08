"""Write a pplx decider as an MLX folder, with the backbone optionally quantized.

The readout stays as released: it is small and makes the decision. The
vision tower is kept in its stored precision (bf16), as for Clef. There is
no mixed precision: its sensitivity sweep needs the whole model in bf16.
"""

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import (
    convert_text_weights,
    has_vision_weights,
    keeping_vision_weights,
    source_shards,
    vision_shards,
    write_vision_weights,
)
from ...registry import MARKER_FILE

FORMAT_VERSION = 1
COPIED = (
    "readout.safetensors",
    "decision_config.json",
    "tokenizer.json",
    "processor_config.json",
    "LICENSE",
    "NOTICE",
)


def convert(
    path: Path,
    output: Path,
    source: str,
    bits: int | None = None,
    group_size: int = 64,
    quantize_output_embeddings: bool = True,
    target_bits: float | None = None,
    progress=None,
    shards: Iterable[Path] | None = None,
    revision: str | None = None,
) -> None:
    """Convert, one source shard at a time (see the Clef converter for ``shards``
    and ``revision``).

    The release has no output embeddings, so ``quantize_output_embeddings``
    must stay on; ``target_bits`` (mixed precision) is refused.
    """
    if target_bits is not None:
        raise ValueError("pplx models have no mixed precision; use --bits")
    if not quantize_output_embeddings:
        raise ValueError("pplx models have no output embeddings to keep")
    config = json.loads((path / "config.json").read_text())
    if "quantization" in config:
        raise ValueError(f"{path} is already quantized; convert the original release")
    vision = has_vision_weights(path)
    if not vision:
        config.pop("vision_config", None)
    vision_weights: dict[str, mx.array] = {}
    quantization = convert_text_weights(
        config,
        output,
        keeping_vision_weights(
            source_shards(path) if shards is None else shards,
            vision_shards(path) if vision else set(),
            vision_weights,
        ),
        bits,
        group_size,
        lm_head=False,
    )
    if vision:
        write_vision_weights(vision_weights, output)
    for name in COPIED:
        if (path / name).exists():
            shutil.copy2(path / name, output / name)
    marker = {
        "family": "pplx",
        "format": FORMAT_VERSION,
        "source": source,
        **({"revision": revision} if revision else {}),
        "quantization": (
            {k: quantization[k] for k in ("group_size", "bits", "mode")}
            if quantization is not None
            else None
        ),
        "vision": vision,
    }
    (output / MARKER_FILE).write_text(json.dumps(marker, indent=2) + "\n")
