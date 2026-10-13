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

from ..data.format import Example
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
# The kinds this version trains.
BUILT = frozenset({"calibrate"})

# Scoring: seconds per question, seconds per 1,000 tokens, and the weight bytes
# of the copy measured (clef-flash bf16, pplx 4-bit, H2O-Lightning 4-bit, d1
# bf16, laya and Julia float16).
SCORING = {
    "clef": (0.10, 0.45, 19e9),
    "pplx": (0.25, 1.5, 15e9),
    "h2o": (0.11, 0.12, 3e9),
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
    family does not take (naming the ones it does) or one not built yet.
    """
    offered = kinds(family)
    if method == "auto":
        main = MAIN_KIND.get(family)
        method = main if main in BUILT else "calibrate"
    if method not in offered:
        takes = ", ".join(["auto", *offered])
        if method in KINDS:
            raise ValueError(f"{family} models cannot be fitted with {method}; they take {takes}")
        raise ValueError(f"no fitting method is called {method}; {family} models take {takes}")
    if method not in BUILT:
        raise ValueError(f"{method} fits are not available in this version yet; use calibrate")
    return [method] if method == "calibrate" else [method, "calibrate"]


def request_tokens(example: Example) -> int:
    """About the tokens of an example's request: its JSON length over four."""
    return max(1, len(json.dumps(example.request, ensure_ascii=False)) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Estimate:
    """Seconds a run takes and the memory it needs at its peak, estimated."""

    seconds: float
    peak_bytes: int

    def as_json(self) -> dict:
        return {"seconds": round(self.seconds), "peak_gb": round(self.peak_bytes / 1e9, 1)}


def scoring_seconds(family: str, weights: int, examples: list[Example]) -> float:
    """Seconds for one scoring pass over ``examples``."""
    per_question, per_thousand, measured = SCORING.get(family, DEFAULT_SCORING)
    scale = max(1.0, weights / measured) if measured else 1.0
    questions = sum(len(example.request["questions"]) for example in examples)
    tokens = sum(request_tokens(example) for example in examples)
    return scale * (questions * per_question + tokens / 1000 * per_thousand)


def kind_bytes(kind: str, path: Path, options: dict) -> int:
    """Memory a kind needs beyond the model's weights and their margin."""
    return 0  # calibrate keeps a few numbers per example


def estimate(
    family: str, kinds: list[str], path: Path, options: dict, examples: list[Example]
) -> Estimate:
    """Time and peak memory of a run on ``examples`` (all of them, holdout included)."""
    weights = weight_bytes(path, options)
    seconds = weights / LOAD_RATE + scoring_seconds(family, weights, examples) + 1.0
    extra = max(kind_bytes(kind, path, options) for kind in kinds)
    return Estimate(seconds, required_bytes(path, options) + extra)
