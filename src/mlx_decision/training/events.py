"""Progress of a training run, as events a callback receives."""

from ..events import Cancelled, Event

# The stages of a fit, in order; a run reports those its kinds go through.
STAGES = ("check", "before", "features", "train", "calibrate", "after", "write", "done")
# The stages of an evaluation: each model loads and scores in turn.
EVAL_STAGES = ("load", "score", "done")

__all__ = ["EVAL_STAGES", "STAGES", "Cancelled", "Event"]
