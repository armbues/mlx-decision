"""What a run will do: the kinds for a family, and its time and memory, estimated.

Estimates come from rates measured on an M-series Mac with 64 GB (the
training spikes): seconds per question and per 1,000 tokens of the scoring
pass, per family, for the copy that was measured; a larger copy (a bigger
model of the family, or less quantized) is taken to be slower in proportion
to its weights. Token counts are guessed from the requests' length (four
characters a token), so nothing is loaded to estimate.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ..data.format import Example, Splits
from ..memory import required_bytes, weight_bytes

KINDS = ("calibrate", "readout", "head", "finetune")
# The kind ``auto`` picks per family, followed by a calibration.
MAIN_KIND = {
    "clef": "head",
    "pplx": "readout",
    "h2o": "readout",
    "d1": "readout",
    "laya": "finetune",
    "julia": "finetune",
}

# Scoring: seconds per question, seconds per 1,000 tokens, and the weight bytes
# of the copy measured (clef-flash bf16, pplx 4-bit, d1 bf16, laya and Julia
# float16); H2O-Lightning answers as fast in bf16 as at 4-bit, so its rate
# holds up to its bf16 size. Clef's per question includes its prompt (about 130
# tokens the request's length does not show; 0.27 s per ANLI-R3 request).
SCORING = {
    "clef": (0.19, 0.45, 19e9),
    "pplx": (0.25, 1.5, 15e9),
    "h2o": (0.11, 0.12, 9.1e9),
    "d1": (0.025, 0.08, 6.2e9),
    "laya": (0.008, 0.02, 0.85e9),
    "julia": (0.003, 0.01, 0.3e9),
}
DEFAULT_SCORING = (0.05, 0.5, None)
CHARS_PER_TOKEN = 4
# Reading weights from disk, bytes per second.
LOAD_RATE = 3e9


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
    """Seconds a run takes, the memory it needs at its peak and the disk it uses
    while it runs (a head fit's stored states), estimated."""

    seconds: float
    peak_bytes: int
    disk_bytes: int = 0

    def as_json(self) -> dict:
        result = {"seconds": round(self.seconds), "peak_gb": round(self.peak_bytes / 1e9, 1)}
        if self.disk_bytes:
            result["disk_gb"] = round(self.disk_bytes / 1e9, 1)
        return result


def scoring_seconds(family: str, weights: int, examples: list[Example]) -> float:
    """Seconds for one scoring pass over ``examples``."""
    per_question, per_thousand, measured = SCORING.get(family, DEFAULT_SCORING)
    scale = max(1.0, weights / measured) if measured else 1.0
    questions = sum(len(example.request["questions"]) for example in examples)
    tokens = sum(request_tokens(example) for example in examples)
    return scale * (questions * per_question + tokens / 1000 * per_thousand)


# Fine-tuning, seconds per 1,000 estimated padded tokens (bfloat16,
# checkpointing; the whole model, the head alone): fitted to fine-tunes of
# Julia-1 and laya (ModernBERT-large) on TREC; the head alone scaled alike.
FINETUNE_RATES = {"julia": (0.038, 0.008), "laya": (0.14, 0.033)}
# Padded tokens per token in batches of random order.
PADDING = 1.5
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
# Its search, seconds per fitted question (R17: 1.4-1.8 s).
READOUT_FIT_SECONDS = 2.0


# The head fit (Clef): bytes per head parameter (float32 weights, gradients of a
# request and their sum, two moments, the kept epoch's copy); bytes per token of
# the longest request's graph; seconds per request of a training step and of the
# dev loss (R12: about 50 s an epoch for 2k requests); the prompt's tokens beyond
# the request's own; bytes per hidden unit stored (bfloat16).
HEAD_PARAMETER_BYTES = 24
HEAD_TOKEN_BYTES = 64_000
HEAD_STEP_SECONDS = 0.03
HEAD_DEV_SECONDS = 0.01
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


def training_seconds(
    family: str, fit: list[Example], dev: list[Example], epochs: int, head_only: bool
) -> float:
    """Seconds of a fine-tune: its epochs over ``fit``, and ``dev``'s loss after each."""
    full, head = FINETUNE_RATES.get(family, (0.25, 0.06))
    rate = head if head_only else full

    def tokens(examples: list[Example]) -> float:
        return sum(request_tokens(e) * len(e.labels) for e in examples) * PADDING / 1000

    # The dev loss is a forward pass, about a third of a step.
    return epochs * (tokens(fit) * rate + tokens(dev) * full / 3)


def estimate(
    family: str,
    kinds: list[str],
    path: Path,
    options: dict,
    splits: Splits,
    epochs: int = 3,
    head_only: bool = False,
) -> Estimate:
    """Time and peak memory of a run on ``splits``."""
    weights = weight_bytes(path, options)
    seconds = weights / LOAD_RATE + scoring_seconds(family, weights, splits.holdout) + 1.0
    every = splits.holdout + splits.fit + splits.dev
    disk = 0
    if kinds[0] in ("calibrate", "readout", "head"):
        seconds += scoring_seconds(family, weights, splits.fit + splits.dev)
    if kinds[0] == "readout":
        fitted = {q for example in splits.fit + splits.dev for q in example.labels}
        seconds += READOUT_FIT_SECONDS * len(fitted)
    elif kinds[0] == "head":
        per_epoch = len(splits.fit) * HEAD_STEP_SECONDS + len(splits.dev) * HEAD_DEV_SECONDS
        seconds += epochs * per_epoch
        _, hidden = head_parameters(path)
        disk = sum(clef_tokens(example) for example in every) * hidden * STATE_BYTES
    elif kinds[0] != "calibrate":
        seconds += training_seconds(family, splits.fit, splits.dev, epochs, head_only)
        # The dev part and the holdout scored again by the trained model.
        seconds += scoring_seconds(family, weights, splits.dev + splits.holdout)
    questions = labelled(every)
    tokens = max((clef_tokens(example) for example in every), default=0)
    extra = max(kind_bytes(kind, path, options, head_only, questions, tokens) for kind in kinds)
    return Estimate(seconds, required_bytes(path, options) + extra, disk)
