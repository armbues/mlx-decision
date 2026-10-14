"""What ``training fit`` and ``training eval`` print: progress, Ctrl-C, the reports.

Progress goes to a stream (stderr): messages on lines of their own, steps
as a bar redrawn in place in a terminal (elsewhere a line every tenth of
the way). Reports are plain text for stdout. Metrics are shown in percent
(ECE too), MAE in levels.
"""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from .events import Event

BAR_WIDTH = 24
STAGE_WIDTH = 9
METRICS = (("accuracy", "accuracy"), ("macro_f1", "macro-F1"), ("ece", "ECE"))


def duration(seconds: float) -> str:
    """``42 s``, ``3 min 5 s``, ``1 h 12 min``."""
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


class Progress:
    """An ``on_event`` callback that shows events on ``stream``."""

    def __init__(self, stream: TextIO, terminal: bool, plan: bool = True):
        self.stream = stream
        self.terminal = terminal
        self.plan = plan  # show the check stage's plan (not when the menus showed it)
        self.drawn = False  # a bar is on the current line
        self.shown: tuple = ()  # stage and tenth last shown without a terminal

    def _line(self, text: str) -> None:
        if self.drawn:
            self.stream.write("\n")
            self.drawn = False
        self.stream.write(text + "\n")
        self.stream.flush()

    def __call__(self, event: Event) -> None:
        if event.step is not None and event.total:
            self._step(event)
        if event.message:
            self._line(f"{event.stage}: {event.message}")
        if self.plan and event.stage == "check" and event.metrics and "estimate" in event.metrics:
            for line in plan_lines(event.metrics):
                self._line(line)

    def _step(self, event: Event) -> None:
        done = min(event.step / event.total, 1.0)
        filled = round(done * BAR_WIDTH)
        text = f"{event.stage:<{STAGE_WIDTH}} {'█' * filled}{'░' * (BAR_WIDTH - filled)}"
        text += f" {event.step:,}/{event.total:,}"
        if event.metrics and "loss" in event.metrics:
            text += f"  loss {event.metrics['loss']:.4f}"
        if event.eta_s is not None and event.step < event.total:
            text += f"  {duration(event.eta_s)} left"
        if self.terminal:
            self.stream.write("\r" + text + "\033[K")
            self.drawn = True
            if event.step >= event.total:
                self.stream.write("\n")
                self.drawn = False
            self.stream.flush()
            return
        shown = (event.stage, int(done * 10))
        if shown != self.shown:
            self.shown = shown
            self._line(text)

    def close(self) -> None:
        if self.drawn:
            self.stream.write("\n")
            self.stream.flush()
            self.drawn = False


def plan_lines(metrics: dict[str, Any]) -> list[str]:
    """The check stage's summary: the data, its labels, the kinds, the estimate."""
    lines = [
        f"data: {metrics['examples']:,} examples: {metrics['holdout']:,} holdout, "
        f"{metrics['fit']:,} fit, {metrics['dev']:,} dev"
    ]
    for question_id, counts in (metrics.get("labels") or {}).items():
        listed = ", ".join(f"{label} {count:,}" for label, count in counts.items())
        lines.append(f"labels: {question_id}: {listed}")
    lines.append("method: " + ", then ".join(metrics["kinds"]))
    estimate = metrics["estimate"]
    text = f"estimate: about {duration(estimate['seconds'])}, peak {estimate['peak_gb']} GB"
    if estimate.get("disk_gb"):
        text += f", {estimate['disk_gb']} GB on disk while it runs"
    lines.append(text + " (from measurements on an M-series Mac with 64 GB)")
    return lines


@contextmanager
def interrupt_cancels(stream: TextIO) -> Iterator[threading.Event]:
    """A cancel flag set by the first Ctrl-C (the run stops at its next step); a
    second Ctrl-C interrupts at once. The previous handler is restored after."""
    flag = threading.Event()

    def handler(signum, frame) -> None:
        if flag.is_set():
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        flag.set()
        stream.write("\nstopping after the current step (Ctrl-C again to quit now)\n")
        stream.flush()

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _rows(columns: list[dict[str, Any]], names: list[str]) -> list[str]:
    """Metric rows, one column per entry of ``columns`` (a question's metrics)."""
    width = max(10, *(len(name) for name in names))
    lines = [" " * 11 + "".join(f"{name:>{width + 2}}" for name in names)]
    keys = list(METRICS)
    if any("mae" in column for column in columns):
        keys.append(("mae", "MAE"))
    for key, label in keys:
        cells = []
        for column in columns:
            value = column.get(key)
            cells.append(f"{value:.3f}" if key == "mae" and value is not None else _percent(value))
        lines.append(f"  {label:<9}" + "".join(f"{cell:>{width + 2}}" for cell in cells))
    for key in ("refused", "truncated"):
        if any(column.get(key) for column in columns):
            cells = [f"{column.get(key, 0):,}" for column in columns]
            lines.append(f"  {key:<9}" + "".join(f"{cell:>{width + 2}}" for cell in cells))
    return lines


def format_fit(report: dict[str, Any], path: str) -> str:
    """A fit's report: per question, the holdout's metrics before and after."""
    lines = []
    for question_id, entry in report["questions"].items():
        n = entry["before"]["n"]
        lines.append(f"{question_id} ({n:,} holdout examples)")
        lines += _rows([entry["before"], entry["after"]], ["before", "after"])
        lines.append("")
    training = report.get("training") or {}
    if "best_epoch" in training:
        lines.append(f"kept epoch {training['best_epoch']} of {len(training['epochs'])}")
    for question in training.get("questions", []):
        if "skipped" in question:
            lines.append(f"readout {question['question']}: none ({question['skipped']})")
        elif question["steps"]:
            lines.append(
                f"readout {question['question']}: {question['steps']} steps at "
                f"{question['learning_rate']:g}, L2 {question['l2']:g}"
            )
        else:
            lines.append(f"readout {question['question']}: the release's rows were best")
    for warning in report.get("warnings", []):
        lines.append(f"warning: {warning}")
    lines.append(f"took {duration(report['seconds']['total'])}; wrote {path}")
    return "\n".join(lines)


def format_confusion(table: dict[str, dict[str, int]]) -> list[str]:
    """Rows gold, columns predicted."""
    columns = list(next(iter(table.values()), {}))
    width = max(5, *(len(label) for label in [*table, *columns]))
    lines = ["  " + " " * width + "".join(f" {c:>{width}}" for c in columns)]
    for gold, counts in table.items():
        lines.append(f"  {gold:<{width}}" + "".join(f" {counts[c]:>{width},}" for c in columns))
    return lines


def format_eval(result) -> str:
    """Per question: the metrics of each model side by side, then their confusion
    matrices (rows the gold label, columns the predicted one)."""
    models = result.models
    names = [model.name for model in models]
    question_ids = list(dict.fromkeys(q for model in models for q in model.questions))
    lines = []
    for question_id in question_ids:
        entries = [model.questions.get(question_id, {}) for model in models]
        first = next(entry for entry in entries if entry)
        lines.append(f"{question_id} ({first['type']}, {first['n']:,} examples)")
        lines += _rows(entries, names)
        for model, entry in zip(models, entries, strict=True):
            if entry:
                lines.append(f"  confusion, {model.name} (rows: gold, columns: predicted)")
                lines += ["  " + line for line in format_confusion(entry["confusion"])]
        lines.append("")
    data = result.data
    source = data["file"] or "examples"
    lines.append(f"{data['examples']:,} examples from {source}")
    return "\n".join(lines)
