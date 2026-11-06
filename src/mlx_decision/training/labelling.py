"""``label``: a teacher's answers to a dataset's rows, written as a new dataset.

The teacher (any model ``load`` takes, fitted ones included) answers every
row of a split, as served, and each answer becomes the row's label: its
distribution (a soft label, the default) or its top answer (``labels``
hard). Rows need questions, not labels: an unlabelled pool works. When the
rows have labels, the teacher's accuracy and ECE on them are measured and
kept, and they decide the note on its calibration (``teacher.teacher_note``);
the labels themselves are the teacher's. The new folder has one split,
``train`` (role train); the lines' image files are linked into its
``images/``.
"""

import gc
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..answers import build_answer
from ..data.dataset import Dataset, SplitOut, dataset_name, open_dataset, write_dataset
from ..data.format import Example
from ..errors import DecisionError
from ..fitted import Fit, FittedBackend
from ..hub import resolve_model_path
from ..metrics import metrics, record
from .events import Event
from .fitting import refuse_images
from .plan import request_tokens
from .scoring import Clock, score
from .teacher import LABELS, Teacher, confidence, describe_teacher, target, teacher_note

# The stages of a labelling run.
LABEL_STAGES = ("check", "load", "score", "write", "done")


@dataclass(frozen=True)
class LabelResult:
    """A labelling run: the folder written, the rows labelled and left out
    (refused by the teacher, or no answer as sure as ``threshold``), the
    labels' kind, the teacher, per labelled question its ``measured``
    accuracy and ECE on the data's own labels, and the note."""

    path: Path
    examples: int
    left_out: int
    refused: int
    labels: str
    teacher: Teacher
    measured: dict[str, dict[str, float]]
    note: str | None
    seconds: float

    def as_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "examples": self.examples,
            "left_out": self.left_out,
            "refused": self.refused,
            "labels": self.labels,
            "teacher": self.teacher.as_json(),
            "measured": self.measured,
            "note": self.note,
            "seconds": self.seconds,
        }


def default_output(teacher: str | Path, data: str | Path) -> Path:
    """``<data name>-<teacher name>`` in the current folder."""
    name = str(teacher).rstrip("/").rsplit("/", 1)[-1] or "teacher"
    return Path(f"{dataset_name(data)}-{name}")


def _read(data: str | Path, split: str | None) -> tuple[Dataset, str, list[Example]]:
    dataset, named = open_dataset(data)
    split = split or named or dataset.pick("train")
    if split not in dataset.splits:
        listed = ", ".join(dataset.splits)
        raise DecisionError(f"{data} has no split {split!r} (it has {listed})")
    examples = dataset.read(split, lenient=True)
    unasked = next((e for e in examples if not e.request.get("questions")), None)
    if unasked is not None:
        raise DecisionError(
            f"{dataset.describe(split)}: row {unasked.id} has no questions; a teacher answers "
            "the questions each row asks (write them into the dataset first)"
        )
    if not examples:
        raise DecisionError(f"{dataset.describe(split)} has no rows")
    return dataset, split, examples


def _own_images(
    example: Example, root: Path, files: dict[str, Path], prefix: str
) -> dict[str, Any]:
    """The request with its image files by their path in the new folder
    (``images/...`` under ``root``, after ``prefix``), noted in ``files``; a
    data URL or a file elsewhere stays as it is."""
    images = example.request.get("images")
    if not isinstance(images, list):
        return example.request
    kept = []
    for value in images:
        if not isinstance(value, str) or value.startswith("data:"):
            kept.append(value)
            continue
        path = Path(value)
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            kept.append(value)
            continue
        inside = relative if relative.startswith("images/") else f"images/{relative}"
        name = f"{prefix}{inside}"
        files[name] = path
        kept.append(name)
    return {**example.request, "images": kept}


def label(
    teacher: str | Path,
    data: str | Path,
    *,
    output: str | Path | None = None,
    split: str | None = None,
    labels: str = "soft",
    threshold: float | None = None,
    sample: int | None = None,
    seed: int = 0,
    overwrite: bool = False,
    on_event: Callable[[Event], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    check_memory: bool = True,
    **load_options,
) -> LabelResult:
    """``teacher``'s answers to ``data``'s rows as a new dataset at ``output``
    (default ``<data>-<teacher>``): ``data`` any dataset ``training fit``
    reads (its train split, or ``split``), its rows needing questions, not
    labels; ``sample`` rows drawn with ``seed``; ``labels`` soft (each
    answer's distribution) or hard (its top answer); ``threshold`` leaves out
    a question's answer below that confidence (a row with none left out).

    ``on_event`` receives the progress (``LABEL_STAGES``); ``cancel()`` stops
    between rows with ``Cancelled`` and nothing written. ``load_options`` go
    to the teacher's loader. ``FileExistsError`` when ``output`` exists and
    not ``overwrite``."""
    from ..model import load

    started = time.perf_counter()
    cancelled = getattr(cancel, "is_set", cancel)

    def emit(event: Event) -> None:
        if on_event is not None:
            on_event(event)

    if labels not in LABELS:
        raise ValueError(f"labels are soft or hard, not {labels!r}")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold is between 0 and 1")
    if sample is not None and sample < 1:
        raise ValueError("a sample is at least one row")
    emit(Event("check", message="reading the data"))
    output = Path(output) if output is not None else default_output(teacher, data)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists (pass overwrite to replace it)")
    dataset, split, examples = _read(data, split)
    if sample is not None and sample < len(examples):
        picked = sorted(random.Random(seed).sample(range(len(examples)), sample))
        examples = [examples[i] for i in picked]
    refuse_images(teacher, examples, load_options)
    name = str(teacher).rstrip("/").rsplit("/", 1)[-1]
    about = describe_teacher(resolve_model_path(teacher), name)

    emit(Event("load", message=f"loading {teacher}"))
    loaded = load(teacher, check_memory=check_memory, **load_options)
    try:
        backend = loaded.backend
        adjust = (
            backend.adjust
            if isinstance(backend, FittedBackend)
            else FittedBackend(backend.name, backend, Fit([]), {}).adjust
        )
        clock = Clock(sum(request_tokens(example) for example in examples))

        def on_step(done: int, example: Example) -> None:
            clock.step(request_tokens(example))
            emit(Event("score", done, len(examples), eta_s=clock.eta()))

        clock.start()
        scored = score(loaded, examples, on_step, cancelled)
        clock.stop()
    finally:
        del loaded
        gc.collect()

    emit(Event("write", message=f"writing {output}"))
    made, files, left_out, refused = [], {}, 0, 0
    gold: dict[str, list[dict]] = {}
    for item in scored:
        if item.raw is None:
            refused += 1
            continue
        probabilities = adjust(item.request.questions, item.raw)
        answers = {
            qid: build_answer(question, probabilities[qid]).model_dump()
            for qid, question in item.request.questions.items()
        }
        for qid, wanted in item.example.labels.items():
            if qid in answers:
                targets = (item.example.targets or {}).get(qid)
                gold.setdefault(qid, []).append(record(answers[qid], wanted, targets))
        kept = {
            qid: answer
            for qid, answer in answers.items()
            if threshold is None or confidence(answer) >= threshold
        }
        if not kept:
            left_out += 1
            continue
        tops = {qid: record(answer, "")["predicted"] for qid, answer in kept.items()}
        soft = {qid: target(answer) for qid, answer in kept.items()} if labels == "soft" else None
        root = dataset.image_root(item.example)
        prefix = f"{item.example.source}/" if item.example.source else ""
        request = _own_images(item.example, root, files, prefix)
        made.append(Example(item.example.id, request, tops, targets=soft))
    if not made:
        raise DecisionError("no row has an answer to keep as a label (refused, or below threshold)")
    measured = {
        qid: {key: round(metrics(rows)[key], 4) for key in ("accuracy", "ece")}
        for qid, rows in gold.items()
        if rows
    }
    questions = list(dict.fromkeys(q for e in made for q in e.request.get("questions") or {}))
    note = teacher_note(about, questions, {q: m["ece"] for q, m in measured.items()} or None)
    source = {
        "source": f"teacher:{name}",
        "teacher": about.as_json(),
        "data": dataset.describe(split),
        "split": split,
        "labels": labels,
        "threshold": threshold,
        "sample": sample,
        "seed": seed if sample is not None else None,
        **({"measured": measured} if measured else {}),
        "note": note,
    }
    created = datetime.now(UTC).isoformat(timespec="seconds")
    folder = Dataset(output, source=source, created=created)
    meta = {"role": "train", "examples": len(made), "split": split}
    if output.exists():  # overwrite: the new folder is ready before the old one goes
        replaced = output.with_name(f".{output.name}.replaced")
        shutil.rmtree(replaced, ignore_errors=True)
        output.rename(replaced)
        try:
            write_dataset(output, folder, [SplitOut("train", made, meta)], files=files)
        except BaseException:
            replaced.rename(output)
            raise
        shutil.rmtree(replaced, ignore_errors=True)
    else:
        write_dataset(output, folder, [SplitOut("train", made, meta)], files=files)
    result = LabelResult(
        output,
        len(made),
        left_out,
        refused,
        labels,
        about,
        measured,
        note,
        round(time.perf_counter() - started, 2),
    )
    emit(Event("done", metrics=result.as_json(), message=f"wrote {output}"))
    return result


__all__ = ["LABEL_STAGES", "LabelResult", "default_output", "label"]
