"""The d3 backend: the letter-readout backend with d3's prompt and limits."""

from pathlib import Path

from ..letter import LetterBackend, Preset
from ..letter import load as load_letter
from .prompt import describe, options, render

# The release sets no limit; longer input is refused, never cut. By default
# 32,768 tokens per question: memory grows by about 136 KB a token (d3-flash),
# so d3-flash 4-bit fits a 16 GB Mac up to about there. It can be raised to
# the backbone's context.
MAX_INPUT_TOKENS = 32768
INPUT_TOKEN_LIMIT = 262144
# The bounds the release's runtime sets on its image processor: at least
# 65,536 and at most 1,638,400 pixels (1.6 MP) per image.
MIN_IMAGE_PIXELS = 65536
MAX_IMAGE_PIXELS = 1638400
# The release takes 2 to 10 score levels.
MAX_SCORE_LEVELS = 10

PRESET = Preset(
    render=render,
    options=options,
    describe=describe,
    max_input_tokens=MAX_INPUT_TOKENS,
    input_token_limit=INPUT_TOKEN_LIMIT,
    min_image_pixels=MIN_IMAGE_PIXELS,
    max_image_pixels=MAX_IMAGE_PIXELS,
    max_score_levels=MAX_SCORE_LEVELS,
)


def load(
    path: Path,
    max_input_tokens: int | None = None,
    vision: bool = False,
    max_image_pixels: int | None = None,
) -> LetterBackend:
    """Load a d3 model from a release folder or a converted copy.

    ``max_input_tokens`` sets the limit per question's sequence (default
    32,768, at most 262,144, the backbone's context); longer input is
    refused, not cut. ``vision=True`` loads the vision tower now, not with the first
    image. Images are resized to the release's bounds (at most 1.6 MP);
    ``max_image_pixels`` lowers the maximum.
    """
    return load_letter(path, PRESET, max_input_tokens, vision, max_image_pixels)
