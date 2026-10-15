"""Write a pplx model as an MLX folder, with the backbone optionally quantized.

The letter readout, decision config and tokenizer are copied as released;
see ``qwen_convert`` for the backbone and the vision tower. No mixed
precision: its sensitivity sweep needs the whole 27B model in bf16.
"""

from pathlib import Path

from ..qwen_convert import convert_qwen

COPIED = (
    "readout.safetensors",
    "decision_config.json",
    "tokenizer.json",
    "processor_config.json",
    "preprocessor_config.json",
    "LICENSE",
    "NOTICE",
)


def convert(path: Path, output: Path, source: str, **options) -> None:
    """Convert (see ``convert_qwen``); ``target_bits`` is refused."""
    convert_qwen("pplx", COPIED, None, path, output, source, lm_head=False, **options)
