"""What ``training fit`` and ``training eval`` print: progress, Ctrl-C, the reports.

Progress goes to a stream (stderr): messages on lines of their own, steps
as a bar redrawn in place in a terminal (elsewhere a line every tenth of
the way). Reports are plain text for stdout. Metrics are shown in percent
(ECE too), MAE in levels, the log loss in nats.
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
    lines = []
    if metrics.get("source"):  # the file, or a dataset folder and its split
        lines.append(f"from {metrics['source']}")
    if metrics.get("validation"):
        lines.append(f"report on {metrics['validation']}")
    lines.append(
        f"data: {metrics['examples']:,} examples: {metrics['holdout']:,} holdout, "
        f"{metrics['fit']:,} fit, {metrics['dev']:,} dev"
    )
    for question_id, counts in (metrics.get("labels") or {}).items():
        listed = ", ".join(f"{label} {count:,}" for label, count in counts.items())
        lines.append(f"labels: {question_id}: {listed}")
    kinds = metrics["kinds"]
    lines.append("method: " + ", then ".join(kinds))
    lines += method_lines(kinds[0], metrics.get("method"), metrics.get("lora_offered", False))
    estimate = metrics["estimate"]
    text = f"memory: about {estimate['peak_gb']} GB at the peak"
    if estimate.get("disk_gb"):
        text += f", {estimate['disk_gb']} GB on disk while it runs"
    lines.append(text)
    return lines


def method_lines(kind: str, asked: str | None, lora_offered: bool) -> list[str]:
    """What the fit's kind does (naming ``auto``'s pick) and, where the family
    takes it and it is not the one asked for, the LoRA path."""
    from .plan import DOES

    does = DOES.get(kind)
    lines = []
    if does is not None:
        lines.append(f"  {'auto picked ' if asked == 'auto' else ''}{kind}: {does}")
    if lora_offered and kind != "lora":
        lines.append(f"  --method lora {DOES['lora']}")
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
    for key, label in (("mae", "MAE"), ("log_loss", "log loss")):
        if any(key in column for column in columns):
            keys.append((key, label))
    for key, label in keys:
        cells = []
        for column in columns:
            value = column.get(key)
            plain = key in ("mae", "log_loss") and value is not None
            cells.append(f"{value:.3f}" if plain else _percent(value))
        lines.append(f"  {label:<9}" + "".join(f"{cell:>{width + 2}}" for cell in cells))
    for key in ("refused", "truncated"):
        if any(column.get(key) for column in columns):
            cells = [f"{column.get(key, 0):,}" for column in columns]
            lines.append(f"  {key:<9}" + "".join(f"{cell:>{width + 2}}" for cell in cells))
    return lines


def _cell(metrics: dict[str, Any]) -> str:
    return f"{_percent(metrics.get('accuracy'))} / {_percent(metrics.get('ece'))}"


def source_lines(rows: list[tuple[str, str, int, list[str]]], names: list[str]) -> list[str]:
    """A table per source dataset and question: ``rows`` of (source, question,
    examples, cells), a column per name."""
    head = [("source", "question", "n")]
    body = [(source, question, f"{n:,}") for source, question, n, _ in rows]
    widths = [max(len(row[i]) for row in head + body) for i in range(3)]
    cell_width = max(15, *(len(name) for name in names), *(len(c) for r in rows for c in r[3]))
    lines = [
        "  "
        + f"{'source':<{widths[0]}}  {'question':<{widths[1]}}  {'n':>{widths[2]}}"
        + "".join(f"  {name:>{cell_width}}" for name in names)
    ]
    for (source, question, n), (*_, cells) in zip(body, rows, strict=True):
        lines.append(
            "  "
            + f"{source:<{widths[0]}}  {question:<{widths[1]}}  {n:>{widths[2]}}"
            + "".join(f"  {cell:>{cell_width}}" for cell in cells)
        )
    return lines


def format_fit(report: dict[str, Any], path: str) -> str:
    """A fit's report: per question, the metrics before and after on the examples
    that score the report (the holdout or the validation data); for a mix, then
    per source dataset."""
    lines = []
    for question_id, entry in report["questions"].items():
        n = entry["before"]["n"]
        lines.append(f"{question_id} ({n:,} report examples)")
        lines += _rows([entry["before"], entry["after"]], ["before", "after"])
        lines.append("")
    sources = report.get("sources") or {}
    if sources:
        rows = [
            (
                name + (" *" if info["held_out"] else ""),
                question_id,
                entry["before"]["n"],
                [_cell(entry["before"]), _cell(entry["after"])],
            )
            for name, info in sources.items()
            for question_id, entry in info["questions"].items()
        ]
        lines.append("per source (accuracy / ECE on the report examples)")
        lines += source_lines(rows, ["before", "after"])
        if any(info["held_out"] for info in sources.values()):
            lines.append("  * held out: scored, not trained on")
        lines.append("")
    training = report.get("training") or {}
    options = report.get("options") or {}
    if options.get("kinds"):
        text = "method: " + ", then ".join(options["kinds"])
        if "rank" in training:
            text += (
                f" (rank {training['rank']}, learning rate {training['learning_rate']:g}; "
                f"{training['trainable_parameters'] / 1e6:.2f} M trained parameters, "
                f"{training['seconds_per_example']:.3g} s per example)"
            )
        lines.append(text)
    if "best_epoch" in training:
        text = f"kept epoch {training['best_epoch']} of {len(training['epochs'])}"
        if training.get("stopped_early"):
            text += " (stopped early: dev loss rose)"
        lines.append(text)
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
    calibration = report.get("calibration") or {}
    if calibration.get("choice") == "none":
        lines.append("calibration: none")
    elif calibration.get("questions"):
        used = ", ".join(f"{q} {kind}" for q, kind in calibration["questions"].items())
        lines.append(f"calibration ({calibration['choice']}): {used}")
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
    names_by_source = list(dict.fromkeys(s for model in models for s in model.sources))
    if names_by_source:
        rows = []
        for source in names_by_source:
            ids = dict.fromkeys(q for model in models for q in model.sources.get(source, {}))
            for question_id in ids:
                entries = [model.sources.get(source, {}).get(question_id, {}) for model in models]
                n = next(entry["n"] for entry in entries if entry)
                rows.append((source, question_id, n, [_cell(entry) for entry in entries]))
        lines.append("per source (accuracy / ECE)")
        lines += source_lines(rows, names)
        lines.append("")
    data = result.data
    source = data["file"] or "examples"
    lines.append(f"{data['examples']:,} examples from {source}")
    return "\n".join(lines)


def format_label(result) -> str:
    """A labelling run's report: what was written, the teacher on the data's
    own labels (when it has them) and the note on its calibration."""
    soft = result.labels == "soft"
    kind = "soft labels (its probabilities)" if soft else "hard labels (its top answers)"
    lines = [f"{result.examples:,} rows labelled by {result.teacher.name}, as {kind}"]
    if result.left_out:
        lines.append(f"left out: {result.left_out:,} rows with no answer as sure as the threshold")
    if result.refused:
        lines.append(f"refused by the teacher: {result.refused:,} rows")
    if result.teacher.base:
        calibration = result.teacher.calibration or "none"
        lines.append(
            f"teacher: fitted on {result.teacher.base} ({', '.join(result.teacher.kinds)}; "
            f"calibration {calibration})"
        )
    for question, measured in result.measured.items():
        lines.append(
            f"teacher on the data's labels, {question}: accuracy {_percent(measured['accuracy'])}, "
            f"ECE {_percent(measured['ece'])}"
        )
    if result.note:
        lines.append(f"note: {result.note}")
    lines.append(f"took {duration(result.seconds)}; wrote {result.path}")
    return "\n".join(lines)
