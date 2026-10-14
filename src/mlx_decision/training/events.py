"""Progress of a training run, as events a callback receives."""

from dataclasses import asdict, dataclass
from typing import Any

# The stages of a fit, in order; a run reports those its kinds go through.
STAGES = ("check", "before", "features", "train", "calibrate", "after", "write", "done")
# The stages of an evaluation: each model loads and scores in turn.
EVAL_STAGES = ("load", "score", "done")


class Cancelled(Exception):
    """A run stopped by its cancel flag; nothing was written."""


@dataclass(frozen=True)
class Event:
    """Where a run is: its ``stage``, the ``step`` of ``total`` within it (when the
    stage counts steps), metrics so far, the estimated seconds left of the whole
    run, and a message (a warning, what the stage does)."""

    stage: str
    step: int | None = None
    total: int | None = None
    metrics: dict[str, Any] | None = None
    eta_s: float | None = None
    message: str | None = None

    def as_json(self) -> dict[str, Any]:
        """The event as JSON data, without the fields it leaves empty."""
        return {key: value for key, value in asdict(self).items() if value is not None}
