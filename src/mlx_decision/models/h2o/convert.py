"""Write an H2O-Lightning model as an MLX folder, with the backbone optionally quantized.

The serve config, tokenizer and image processor settings are copied as
released. The output embeddings are the tied input embeddings, which also
give the label rows of the readout. See ``qwen_convert`` for the backbone
and the vision tower.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

from ..qwen_convert import convert_qwen

COPIED = (
    "serve_config.json",
    "tokenizer.json",
    "preprocessor_config.json",
    "LICENSE",
)


def _load_for_sweep(path: Path):
    from .model import load

    # No prefix reuse: the sweep answers the same requests with other weights;
    # no floor, which would hide changes of unsure yes/no answers.
    return load(path, prefix_cache_gb=0, noul_floor=0)


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
        "h2o",
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
