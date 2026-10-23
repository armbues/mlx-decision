"""Progress of a long run (a fit, a download, a conversion), as events a callback receives."""

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

# The stages a run's events end with (``--events``' last line).
FINAL_STAGES = ("done", "error", "cancelled")


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


class Steps:
    """Events of a stage counted in steps (bytes, tensors), at most one per
    ``interval`` seconds plus the last, each with the seconds left at the pace
    so far. ``add`` may be called from several threads."""

    def __init__(
        self,
        stage: str,
        total: int | None,
        emit: Callable[[Event], None],
        interval: float = 0.25,
    ):
        self.stage, self.total, self.emit, self.interval = stage, total, emit, interval
        self.done = 0
        self.started = time.perf_counter()
        self.last = 0.0
        self.lock = threading.Lock()

    def add(self, count: int) -> None:
        with self.lock:
            self.at(self.done + count)

    def at(self, done: int, total: int | None = None) -> None:
        """``done`` steps so far (of ``total``, when it changed)."""
        self.done = done
        if total is not None:
            self.total = total
        now = time.perf_counter()
        last = self.total is not None and done >= self.total
        if not last and now - self.last < self.interval:
            return
        self.last = now
        self.emit(Event(self.stage, done, self.total, eta_s=self.eta(now)))

    def eta(self, now: float) -> float | None:
        """Seconds left, once a second's pace is known."""
        elapsed = now - self.started
        if not self.done or self.total is None or elapsed < 1.0:
            return None
        return round(elapsed / self.done * max(self.total - self.done, 0), 1)
