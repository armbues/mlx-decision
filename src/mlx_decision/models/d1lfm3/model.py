"""The d1 LFM3 backend (d1-2B): LFM3 text model and vision tower, questions read in sets.

A request's questions go in sets of the release's size (16), in their order;
each set is one sequence (the pictures and state, then its questions) in
which, on the full-attention layers, the state's tokens see the whole set and
each question the state and itself (``backbones/lfm3``'s ``SetLayout``). The
state's hidden states depend on its set's questions, so nothing is kept
between sets or requests. Answers depend on which questions share a set and
on their order, as in the release.
"""

import json
import struct
from pathlib import Path

from tokenizers import Tokenizer

from ...backbones.lfm3.load import load_text_model, load_vision
from ...backend import Capabilities
from ..qwen import precision
from .config import Settings, settings

# Tokens per set by default (the state and the set's questions); the
# context is the most.
MAX_INPUT_TOKENS = 32768
INPUT_TOKEN_LIMIT = 128000
# Every option code up to this many options resolves to a token (the
# release's fallback codes included, measured with its tokenizer); score
# questions take 2 to 10 levels.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
# Pictures are enlarged to at least this many pixels by the processor, so a
# lower cap would change nothing.
MIN_IMAGE_PIXELS = 65536
REQUIRED_FILES = ("config.json", "tokenizer.json", "calibration.json")
VISION_PREFIX = "model.vision_tower."


def has_vision_weights(path: Path) -> bool:
    """Whether the folder's weights hold the vision tower (read from the headers)."""
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = json.loads(index.read_text()).get("weight_map", {})
        return any(name.startswith(VISION_PREFIX) for name in names)
    for file in sorted(path.glob("model*.safetensors")):
        with file.open("rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            if any(name.startswith(VISION_PREFIX) for name in json.loads(handle.read(size))):
                return True
    return False


class D1Lfm3Backend:
    def __init__(
        self,
        name: str,
        model,
        tokenizer: Tokenizer,
        settings: Settings,
        capabilities: Capabilities,
        path: Path | None = None,
    ):
        """``model`` None gives a backend that only encodes."""
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.capabilities = capabilities
        self.path = path
        self._vision = None

    @property
    def vision(self):
        """The vision tower and projector, loaded on first use."""
        if self._vision is None:
            self._vision = load_vision(self.path)
        return self._vision

    def score(self, request):
        raise NotImplementedError("d1 LFM3 answers come with its prompt and readout")


def _input_limit(max_input_tokens: int | None) -> int:
    if max_input_tokens is None:
        return MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= INPUT_TOKEN_LIMIT:
        raise ValueError(
            f"max_input_tokens must be between 1 and {INPUT_TOKEN_LIMIT} (the model's context)"
        )
    return max_input_tokens


def _pixel_limit(found: Settings, max_image_pixels: int | None) -> int:
    if max_image_pixels is None:
        return found.max_pixels
    if not MIN_IMAGE_PIXELS <= max_image_pixels <= found.max_pixels:
        raise ValueError(
            f"max_image_pixels must be between {MIN_IMAGE_PIXELS} and {found.max_pixels} "
            "(the release's cap)"
        )
    return max_image_pixels


def _capabilities(max_input_tokens: int, images: bool, max_image_pixels: int) -> Capabilities:
    return Capabilities(
        max_input_tokens=max_input_tokens,
        max_choice_options=MAX_CHOICE_OPTIONS,
        max_score_levels=MAX_SCORE_LEVELS,
        supports_images=images,
        max_image_pixels=max_image_pixels if images else None,
        truncates_input=True,
    )


def capabilities(path: Path, options: dict) -> Capabilities:
    """What ``load(path, **options)`` gives the backend, from the folder's JSON
    files and weight headers."""
    limit = _input_limit(options.get("max_input_tokens"))
    found = settings(path)
    pixels = _pixel_limit(found, options.get("max_image_pixels"))
    return _capabilities(limit, has_vision_weights(path), pixels)


def load(
    path: Path,
    max_input_tokens: int | None = None,
    vision: bool = False,
    max_image_pixels: int | None = None,
) -> D1Lfm3Backend:
    """Load d1-2B from a release folder.

    ``max_input_tokens`` sets the limit per set (the state, its pictures and
    the set's questions): 32,768 by default, up to 128,000, the model's
    context; a state too long for it with the request's largest set is cut
    around a "..." line, keeping its start and end. ``vision=True`` loads the
    vision tower now, not with the first picture. ``max_image_pixels`` lowers
    the release's cap on pictures (1,638,400 pixels), down to 65,536.
    """
    limit = _input_limit(max_input_tokens)
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    found = settings(path)
    pixels = _pixel_limit(found, max_image_pixels)
    images = has_vision_weights(path)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    model = load_text_model(path)
    backend = D1Lfm3Backend(
        path.resolve().name,
        model,
        tokenizer,
        found,
        _capabilities(limit, images, pixels),
        path,
    )
    if vision and images:
        backend.vision  # noqa: B018 (loads it)
    backend.precision = precision(path, model)
    return backend
