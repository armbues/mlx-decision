"""What a run will do: the kinds for a family, and its memory, estimated.

The estimate counts the model's weights and what the kind keeps beside them
(optimizer state, activations, kept features, stored states); token counts
are guessed from the requests' length (four characters a token), so nothing
is loaded to estimate. Time is not estimated: it depends on the Mac's chip
far more than on anything known before loading.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ..data.format import Example, Splits
from ..memory import required_bytes

KINDS = ("calibrate", "readout", "head", "finetune")
# The kind ``auto`` picks per family, followed by a calibration.
MAIN_KIND = {
    "clef": "head",
    "pplx": "readout",
    "d3": "readout",
    "h2o": "readout",
    "d1": "readout",
    "laya": "finetune",
    "julia": "finetune",
}

CHARS_PER_TOKEN = 4


def kinds(family: str) -> list[str]:
    """The kinds a family can be fitted with: its main kind, if any, and ``calibrate``."""
    main = MAIN_KIND.get(family)
    return [main, "calibrate"] if main else ["calibrate"]


def resolve(family: str, method: str) -> list[str]:
    """The kinds a run goes through, in order, for ``method`` (a kind or ``auto``).

    Every run ends with a calibration. Raises ``ValueError`` for a kind the
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


def request_tokens(example: Example) -> int:
    """About the tokens of an example's request: its JSON length over four."""
    return max(1, len(json.dumps(example.request, ensure_ascii=False)) // CHARS_PER_TOKEN)


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


def clef_tokens(example: Example) -> int:
    """About the tokens of an example in Clef's prompt."""
    return request_tokens(example) + CLEF_PROMPT_TOKENS


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
) -> int:
    """Memory a kind needs beyond the model's weights and their margin; ``questions``
    is the labelled questions of the whole data (readout), ``tokens`` its longest
    request's (head)."""
    from ..models.marker.model import sizes
    from .finetune import BATCH_TOKENS
    from .head import SHARD_BYTES

    if kind == "readout":
        return questions * READOUT_FEATURE_BYTES
    if kind == "head":
        parameters, _ = head_parameters(path)
        return parameters * HEAD_PARAMETER_BYTES + tokens * HEAD_TOKEN_BYTES + SHARD_BYTES
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
    disk = 0
    if kinds[0] == "head":
        _, hidden = head_parameters(path)
        disk = sum(clef_tokens(example) for example in every) * hidden * STATE_BYTES
    questions = labelled(every)
    tokens = max((clef_tokens(example) for example in every), default=0)
    extra = max(kind_bytes(kind, path, options, head_only, questions, tokens) for kind in kinds)
    return Estimate(required_bytes(path, options) + extra, disk)
