"""Write a pplx model as an MLX folder, with the backbone optionally quantized.

The letter readout, decision config and tokenizer are copied as released,
for v1.2 also the joint request format's files; see ``qwen_convert`` for
the backbone and the vision tower. Mixed precision for v1.2 only: its
sensitivity sweep loads the whole model in bf16, which v1.1 (27B only)
never fits here.
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
JOINT_COPIED = COPIED + ("joint_config.json", "tokenizer_config.json")


def _load_for_sweep(path: Path):
    from .model import load

    return load(path)


def convert(path: Path, output: Path, source: str, **options) -> None:
    """Convert (see ``convert_qwen``); ``target_bits`` switches to mixed
    precision for v1.2 and is refused for v1.1."""
    if (path / "joint_config.json").exists():
        convert_qwen(
            "pplx", JOINT_COPIED, _load_for_sweep, path, output, source, lm_head=False, **options
        )
    else:
        if options.get("target_bits") is not None:
            raise ValueError("pplx v1.1 models have no mixed precision (v1.2 has); use --bits")
        convert_qwen("pplx", COPIED, None, path, output, source, lm_head=False, **options)
