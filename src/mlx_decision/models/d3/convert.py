"""Write a d3 model as an MLX folder, with the backbone optionally quantized.

The letter readout, decision config, tokenizer and image processor
settings are copied as released; see ``qwen_convert`` for the backbone and
the vision tower.
"""

from pathlib import Path

from ..pplx.convert import COPIED
from ..qwen_convert import convert_qwen


def _load_for_sweep(path: Path):
    from .model import load

    return load(path)


def convert(path: Path, output: Path, source: str, **options) -> None:
    """Convert (see ``convert_qwen``); ``target_bits`` switches to mixed precision."""
    convert_qwen("d3", COPIED, _load_for_sweep, path, output, source, lm_head=False, **options)
