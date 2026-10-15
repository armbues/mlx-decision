"""The pplx backend: the letter-readout backend with pplx-decider's prompt and limits."""

from pathlib import Path

from ..letter import LetterBackend, Preset
from ..letter import load as load_letter
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


def load(path: Path, max_input_tokens: int | None = None, vision: bool = False) -> LetterBackend:
    """Load pplx-decider from a release folder or a converted copy.

    ``max_input_tokens`` lowers the limit per question's sequence (default
    and maximum 8,192, the release's); longer input is refused, not cut.
    ``vision=True`` loads the vision tower now, not with the first image.
    Images are resized to the release's bounds (64 to 256 tokens each).
    """
    return load_letter(path, PRESET, max_input_tokens, vision)
