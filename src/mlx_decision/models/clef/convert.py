"""Write a Clef model as an MLX folder, with the backbone optionally quantized.

The joint schema head stays as released: it is small and makes the decision.
See ``qwen_convert`` for the backbone and the vision tower.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from ..qwen_convert import convert_qwen

COPIED = (
    "joint_head.safetensors",
    "joint_head_config.json",
    "tokenizer.json",
    "processor_config.json",
    "LICENSE",
)


def _load_for_sweep(path: Path):
    from .model import load

    # No prefix reuse: the sweep answers the same requests with other weights.
    return load(path, prefix_cache_gb=0)


def convert(
    path: Path,
    output: Path,
    source: str,
    bits: int | None = None,
    group_size: int = 64,
    quantize_output_embeddings: bool = True,
    target_bits: float | None = None,
    progress: Callable[[int, int, object, float], None] | None = None,
    shards: Iterable[Path] | None = None,
    revision: str | None = None,
) -> None:
    """Convert (see ``convert_qwen``); ``target_bits`` switches to mixed precision."""
    convert_qwen(
        "clef",
        COPIED,
        _load_for_sweep,
        path,
        output,
        source,
        bits,
        group_size,
        quantize_output_embeddings,
        target_bits,
        progress,
        shards,
        revision,
    )
