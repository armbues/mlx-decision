"""What a run will do: the kinds for a family, and its memory, estimated.

The estimate counts the model's weights and what the kind keeps beside them
(optimizer state, activations, kept features, stored states); token counts
are guessed from the requests' length (four characters a token) and their
images (a head fit: from the image headers and the release's processor
settings; else a fixed guess), so no model is loaded to estimate. Time is
not estimated: it depends on the Mac's chip far more than on anything known
before loading.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.format import Example, Splits
from ..errors import DecisionError
from ..memory import required_bytes
from ..state_images import lift_images

KINDS = ("calibrate", "readout", "head", "finetune", "lora")
# The closing calibration's choices (``fitting.CALIBRATIONS`` says what they do).
CALIBRATIONS = ("auto", "full", "temperature", "none")
# The share of the data that scores a fit's report when nothing else does.
DEFAULT_HOLDOUT = 0.2
# The kind ``auto`` picks per family, followed by a calibration.
MAIN_KIND = {
    "clef": "head",
    "pplx": "readout",
    "d3": "readout",
    "baselabs": "readout",
    "h2o": "readout",
    "d1lfm3": "readout",
    "d1": "readout",
    "laya": "finetune",
    "julia": "finetune",
}
# The families a ``lora`` fit adapts (offered beside their main kind, never
# ``auto``'s pick: it takes minutes where the readout takes seconds).
LORA_FAMILIES = ("d3", "d1lfm3")
# The kinds that train in epochs, and their default number of epochs.
EPOCHS = {"head": 3, "finetune": 3, "lora": 2}
# What each kind does, as the summary before a fit and the menus say it.
DOES = {
    "readout": "trains the answer rows on the model's frozen features, seconds",
    "head": "trains the decision head on the model's stored states",
    "finetune": "trains the model's own weights",
    "calibrate": "fits the confidence only; the answers stay",
    "lora": "adapts the network itself: minutes",
}

CHARS_PER_TOKEN = 4
# Tokens an image is guessed at where its model's processor is not read: about
# a megapixel, at 32 x 32 pixels a token as the Qwen-style towers cut them.
IMAGE_TOKENS = 1024


def kinds(family: str) -> list[str]:
    """The kinds a family can be fitted with: its main kind, if any, ``lora``
    where it adapts, and ``calibrate``."""
    main = MAIN_KIND.get(family)
    found = [main] if main else []
    if family in LORA_FAMILIES:
        found.append("lora")
    return [*found, "calibrate"]


def methods(family: str) -> list[tuple[str, bool]]:
    """What a fit of the family can be asked to do, as ``(method, head_only)``,
    in the order a menu offers them: the family's main kind (``auto``'s pick;
    finetune also with the head alone), ``lora`` where it adapts, then
    ``calibrate``."""
    main = MAIN_KIND.get(family)
    entries = []
    if main is not None:
        entries.append((main, False))
        if main == "finetune":
            entries.append((main, True))
    if family in LORA_FAMILIES:
        entries.append(("lora", False))
    entries.append(("calibrate", False))
    return entries


def resolve(family: str, method: str) -> list[str]:
    """The kinds a run goes through, in order, for ``method`` (a kind or ``auto``).

    The kinds end with calibrate (``calibration="none"`` drops it later,
    ``check_calibration``). Raises ``ValueError`` for a kind the
    family does not take (naming the ones it does).
    """
    offered = kinds(family)
    if method == "auto":
        method = MAIN_KIND.get(family, "calibrate")
    if method not in offered:
        takes = ", ".join(["auto", *offered])
        if method in KINDS:
            raise ValueError(f"{family} models cannot be fitted with {method}; they take {takes}")
        raise ValueError(f"no fitting method is called {method}; {family} models take {takes}")
    return [method] if method == "calibrate" else [method, "calibrate"]


def _without_images(request: dict[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    """The request without its images (those in the state left as their
    ``[image N]`` placeholders), and the images."""
    images = list(request.get("images") or [])
    rest = {key: value for key, value in request.items() if key != "images"}
    try:
        state, lifted = lift_images(rest.get("state"), len(images))
    except DecisionError:
        lifted = []  # the model's check names it
    if lifted:
        rest["state"] = state
        images += [raw for raw, _ in lifted]
    return rest, images


def request_tokens(example: Example, image_tokens: Callable[[Any], int] | None = None) -> int:
    """About the tokens of an example's request: its JSON length over four,
    without the images, plus each image's tokens (``image_tokens``, else
    ``IMAGE_TOKENS``)."""
    rest, images = _without_images(example.request)
    text = max(1, len(json.dumps(rest, ensure_ascii=False, default=str)) // CHARS_PER_TOKEN)
    if image_tokens is None:
        return text + IMAGE_TOKENS * len(images)
    return text + sum(image_tokens(image) for image in images)


@dataclass(frozen=True)
class Estimate:
    """The memory a run needs at its peak and the disk it uses while it runs (a
    head fit's stored states), estimated."""

    peak_bytes: int
    disk_bytes: int = 0

    def as_json(self) -> dict:
        result = {"peak_gb": round(self.peak_bytes / 1e9, 1)}
        if self.disk_bytes:
            result["disk_gb"] = round(self.disk_bytes / 1e9, 1)
        return result


# Bytes per trained parameter: float32 copy and two moments, bfloat16 gradients
# and the kept epoch's bfloat16 copy.
TRAINED_PARAMETER_BYTES = 16
# On top, the larger of two passing peaks: updating the largest tensor (bytes per
# element), and the backward pass (bytes per token, hidden unit and layer, with
# checkpointing). Measured on Julia-1 and laya at 4,096-token batches.
UPDATE_BYTES = 22
BACKWARD_BYTES = 10
HEAD_LAYERS = 2


# The readout fit keeps each labelled question's features: a float32 hidden
# state (pplx 20 KB) and, for d1, its option rows; on the safe side.
READOUT_FEATURE_BYTES = 64_000


# The head fit (Clef): bytes per head parameter (float32 weights, gradients of a
# request and their sum, two moments, the kept epoch's copy); bytes per token of
# the longest request's graph; the prompt's tokens beyond the request's own; bytes
# per hidden unit stored (bfloat16).
HEAD_PARAMETER_BYTES = 24
HEAD_TOKEN_BYTES = 64_000
CLEF_PROMPT_TOKENS = 130
STATE_BYTES = 2


# The lora fit (d3, d1 LFM3; measured on d3-edge and d1-2B, 125 to 3,300
# tokens): 24 bytes per trained parameter (float32 weights, gradients and
# their sum over a step, two moments, the kept epoch's copy); a checkpointed
# step's peak, LORA_STEP_BYTES plus LORA_TOKEN_BYTES per token, hidden unit
# and layer of the longest pass (d3-edge 19.5, d1-2B 17.8; the step's own
# fixed part 0.45 to 0.93 GB); the vision tower's rows of every training
# image, kept for the run, at 2 bytes per image token and hidden unit.
LORA_PARAMETER_BYTES = 24
LORA_STEP_BYTES = 800_000_000
LORA_TOKEN_BYTES = 20
FEATURE_BYTES = 2
LORA_RANK = 8  # training.lora.RANK, the default


def lora_sizes(path: Path) -> tuple[int, int, int]:
    """The text model's hidden size, MLP width and layers, from ``config.json``."""
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    return text["hidden_size"], text["intermediate_size"], text["num_hidden_layers"]


def lora_parameters(path: Path, rank: int = LORA_RANK) -> int:
    """Adapter parameters on the MLP projections of every layer (three a layer,
    each ``rank`` by inputs plus outputs)."""
    hidden, width, layers = lora_sizes(path)
    return layers * 3 * rank * (hidden + width)


def lora_image_tokens(family: str, path: Path, options: dict) -> Callable[[Any], int]:
    """An image's tokens in a d3 or d1 LFM3 prompt, from its header and the
    release's processor settings (``max_image_pixels`` as the loader takes it);
    ``IMAGE_TOKENS`` for one that does not open."""
    from dataclasses import replace

    from ..backbones.qwen3_5.preprocess import ImageConfig, image_tokens
    from ..images import open_image

    if family == "d3":
        from ..models.d3.model import MAX_IMAGE_PIXELS, MIN_IMAGE_PIXELS

        top = min(options.get("max_image_pixels") or MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS)
        config = replace(
            ImageConfig.from_folder(path), min_pixels=min(MIN_IMAGE_PIXELS, top), max_pixels=top
        )

        def size(width: int, height: int) -> tuple[int, int]:
            return width, height
    else:
        from ..models.d1lfm3.config import settings
        from ..models.d1lfm3.prompt import picture_size

        config = ImageConfig.from_folder(path)
        top = options.get("max_image_pixels") or settings(path).max_pixels

        def size(width: int, height: int) -> tuple[int, int]:
            return picture_size(width, height, top)

    def count(image: Any) -> int:
        try:
            opened = open_image(image, 0)
            width, height = size(opened.width, opened.height)
            return image_tokens(height, width, config)
        except (DecisionError, ValueError):
            return IMAGE_TOKENS

    return count


def lora_tokens(examples: list[Example], path: Path, options: dict) -> tuple[int, int]:
    """The longest example's tokens and all examples' image tokens (the rows the
    run keeps), images counted from their headers."""
    from ..registry import detect_family

    longest = kept = 0
    count = None
    for example in examples:
        _, images = _without_images(example.request)
        if images and count is None:
            count = lora_image_tokens(detect_family(path).name, path, options)
        pictures = sum(count(image) for image in images) if images else 0
        longest = max(longest, request_tokens(example, lambda _: 0) + pictures)
        kept += pictures
    return longest, kept


def clef_image_tokens(path: Path, options: dict) -> Callable[[Any], int]:
    """An image's tokens in Clef's prompt, from its header and the release's
    processor settings (``max_image_pixels`` as the loader takes it), the
    start and end markers included; ``IMAGE_TOKENS`` for one that does not open."""
    from ..backbones.qwen3_5.preprocess import image_tokens
    from ..images import open_image
    from ..models.clef.model import DEFAULT_MAX_IMAGE_PIXELS
    from ..models.qwen import image_config

    config = image_config(path, options.get("max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS))

    def count(image: Any) -> int:
        try:
            opened = open_image(image, 0)
            return image_tokens(opened.height, opened.width, config) + 2
        except (DecisionError, ValueError):
            return IMAGE_TOKENS

    return count


def clef_tokens(example: Example, image_tokens: Callable[[Any], int] | None = None) -> int:
    """About the tokens of an example in Clef's prompt (images as
    ``image_tokens`` counts them, else guessed)."""
    return request_tokens(example, image_tokens) + CLEF_PROMPT_TOKENS


def head_tokens(examples: list[Example], path: Path, options: dict) -> list[int]:
    """Each example's tokens in Clef's prompt, its images counted from their
    headers (opened only when there are images)."""
    count = None
    if any(_without_images(example.request)[1] for example in examples):
        count = clef_image_tokens(path, options)
    return [clef_tokens(example, count) for example in examples]


def head_parameters(path: Path) -> tuple[int, int]:
    """A Clef release's head parameters and the backbone's hidden size."""
    from ..memory import element_count, read_header

    header = read_header(path / "joint_head.safetensors")
    count = sum(element_count(entry["shape"]) for entry in header.values())
    config = json.loads((path / "joint_head_config.json").read_text())
    return count, config["hidden_size"]


def labelled(examples: list[Example]) -> int:
    """Labelled questions in ``examples``."""
    return sum(len(example.labels) for example in examples)


def kind_bytes(
    kind: str,
    path: Path,
    options: dict,
    head_only: bool = False,
    questions: int = 0,
    tokens: int = 0,
    kept: int = 0,
) -> int:
    """Memory a kind needs beyond the model's weights and their margin; ``questions``
    is the labelled questions of the whole data (readout), ``tokens`` its longest
    request's (head, lora), ``kept`` the training part's image tokens (lora)."""
    from ..models.marker.model import sizes
    from .finetune import BATCH_TOKENS
    from .head import SHARD_BYTES

    if kind == "readout":
        return questions * READOUT_FEATURE_BYTES
    if kind == "head":
        parameters, _ = head_parameters(path)
        return parameters * HEAD_PARAMETER_BYTES + tokens * HEAD_TOKEN_BYTES + SHARD_BYTES
    if kind == "lora":
        hidden, _, layers = lora_sizes(path)
        step = LORA_STEP_BYTES + LORA_TOKEN_BYTES * tokens * hidden * layers
        features = kept * hidden * FEATURE_BYTES
        return lora_parameters(path) * LORA_PARAMETER_BYTES + step + features
    if kind != "finetune":
        return 0  # calibrate keeps a few numbers per example
    size = sizes(path)
    if head_only:
        trained, largest, layers = size.head, size.largest_head, HEAD_LAYERS
    else:
        trained = size.encoder + size.head
        largest = max(size.largest_encoder, size.largest_head)
        layers = size.layers + HEAD_LAYERS
    passing = max(UPDATE_BYTES * largest, BACKWARD_BYTES * BATCH_TOKENS * size.hidden * layers)
    return trained * TRAINED_PARAMETER_BYTES + passing


def estimate(
    kinds: list[str],
    path: Path,
    options: dict,
    splits: Splits,
    head_only: bool = False,
) -> Estimate:
    """Peak memory and disk of a run on ``splits``."""
    every = splits.holdout + splits.fit + splits.dev
    disk = tokens = kept = 0
    if "head" in kinds:
        counts = head_tokens(every, path, options)
        _, hidden = head_parameters(path)
        disk = sum(counts) * hidden * STATE_BYTES
        tokens = max(counts, default=0)
    if "lora" in kinds:
        tokens, _ = lora_tokens(every, path, options)
        _, kept = lora_tokens(splits.fit + splits.dev, path, options)
    questions = labelled(every)
    extra = max(
        kind_bytes(kind, path, options, head_only, questions, tokens, kept) for kind in kinds
    )
    return Estimate(required_bytes(path, options) + extra, disk)
