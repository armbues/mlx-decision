"""The pplx backend: the letter-readout backend with pplx-decider's prompt and limits.

v1.1 asks each question in its own sequence; v1.2 (a folder with
``joint_config.json``) answers a whole request in one sequence, the joint
request format (``backend.JointBackend``).
"""

import json
from dataclasses import replace
from pathlib import Path

from ...backend import Capabilities
from ..letter import LetterBackend, Preset
from ..letter import capabilities as letter_capabilities
from ..letter import load as load_letter
from .backend import JointBackend
from .joint import escaped_tokens
from .prompt import describe, options, render

# The release's limit per question's sequence; longer input is refused, as
# its code does, never cut.
MAX_INPUT_TOKENS = 8192
# The pixel bounds the release's code sets on its image processor (whatever
# processor_config.json says): at least 256 x 256, at most 512 x 512 pixels
# worth, so an image becomes 64 to 256 tokens.
MIN_IMAGE_PIXELS = 65536
MAX_IMAGE_PIXELS = 262144

PRESET = Preset(
    render=render,
    options=options,
    describe=describe,
    max_input_tokens=MAX_INPUT_TOKENS,
    input_token_limit=MAX_INPUT_TOKENS,
    min_image_pixels=MIN_IMAGE_PIXELS,
    max_image_pixels=MAX_IMAGE_PIXELS,
)

# v1.2: the limit per sequence. The release's default is the backbone's
# context, 262,144 tokens; ours is 32,768, the length it was trained on and
# what its card advises (memory grows with it). A request over the limit is
# answered in chunks, as the release does. Images up to the decision
# config's ``max_pixels`` (262,144 without one, as the release).
JOINT_MAX_INPUT_TOKENS = 32768
JOINT_INPUT_TOKEN_LIMIT = 262144
JOINT_FORMATS = (2,)
JOINT_FILES = ("joint_config.json", "tokenizer_config.json")
JOINT_PRESET = replace(
    PRESET,
    render=None,
    max_input_tokens=JOINT_MAX_INPUT_TOKENS,
    input_token_limit=JOINT_INPUT_TOKEN_LIMIT,
)


def is_joint(path: Path) -> bool:
    return (path / "joint_config.json").exists()


def joint_preset(path: Path) -> Preset:
    """The v1.2 preset with the folder's image maximum."""
    config = json.loads((path / "decision_config.json").read_text())
    max_pixels = config.get("max_pixels") or MAX_IMAGE_PIXELS
    if not isinstance(max_pixels, int) or max_pixels < MIN_IMAGE_PIXELS:
        raise ValueError(
            f"{path}: max_pixels must be a whole number of at least {MIN_IMAGE_PIXELS}"
        )
    return replace(JOINT_PRESET, max_image_pixels=max_pixels)


def check_codes(path: Path) -> None:
    """The answer codes must be single tokens with the ids the decision config
    names, as the release checks."""
    from ...backbones.qwen3_5.tokenizer import load_tokenizer

    config = json.loads((path / "decision_config.json").read_text())
    codes, ids = config.get("codes"), config.get("token_ids")
    tokenizer = load_tokenizer(path)
    if (
        not isinstance(ids, list)
        or len(ids) != len(codes or ())
        or any(
            tokenizer.encode(code, add_special_tokens=False).ids != [token_id]
            for code, token_id in zip(codes, ids, strict=True)
        )
    ):
        raise ValueError(f"{path}: decision_config.json's answer codes do not match the tokenizer")


def capabilities(path: Path, options: dict) -> Capabilities:
    """What ``load(path, **options)`` gives the backend, from the folder's config files."""
    if is_joint(path):
        return letter_capabilities(path, joint_preset(path), options, JOINT_FORMATS)
    return letter_capabilities(path, PRESET, options)


def load(
    path: Path,
    max_input_tokens: int | None = None,
    vision: bool = False,
    max_image_pixels: int | None = None,
) -> LetterBackend:
    """Load pplx-decider from a release folder or a converted copy.

    v1.1: ``max_input_tokens`` lowers the limit per question's sequence
    (default and maximum 8,192, the release's); images become 64 to 256
    tokens each. v1.2: the limit per sequence (default 32,768, at most
    262,144); a request over it is answered in chunks of its questions, a
    question that does not fit alone is refused; images up to the release's
    ``max_pixels`` (1.6 MP), ``max_image_pixels`` lowers it. Input is never
    cut. ``vision=True`` loads the vision tower now, not with the first image.
    """
    if not is_joint(path):
        return load_letter(path, PRESET, max_input_tokens, vision, max_image_pixels)
    for name in JOINT_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    preset = joint_preset(path)
    check_codes(path)
    return load_letter(
        path,
        preset,
        max_input_tokens,
        vision,
        max_image_pixels,
        backend=JointBackend,
        formats=JOINT_FORMATS,
        required=JOINT_FILES,
        escapes=escaped_tokens(path),
    )
