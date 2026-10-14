"""``fit``: a base model and labelled data in, a fitted model folder out.

The run goes through stages, each reported as events: check (data, kind,
estimate, memory), before (the base scores the holdout), features (the base
scores the training part), train (the kind's own training, when it has
one), calibrate, after (the fitted model's holdout answers), write. The
holdout only scores the report; the training part splits into fit and dev
(dev picks settings and fits the closing calibration). Nothing is written
until the end, so a cancelled or failed run leaves no folder behind.
"""

import hashlib
import json
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .. import __version__
from ..data.format import Example, read_examples, training_splits
from ..errors import DecisionError
from ..fitted import (
    Fit,
    FitQuestion,
    FittedBackend,
    base_reference,
    fingerprint,
    open_folder,
    option_ids,
)
from ..fitted import write as write_fitted
from ..hub import resolve_model_path
from ..memory import check_fits
from ..model import DecisionModel, load
from ..registry import FITTED, check_options, detect_family
from ..types import parse_request
from . import plan
from .calibrate import fit_calibration
from .events import Cancelled, Event
from .scoring import Clock, Scored, records, rehead, reread, score, summary

# Disk left free beside the hidden states a head fit stores.
DISK_MARGIN = 2 * 10**9
# Below these, a fit is warned about (it still runs).
MIN_EXAMPLES = 50
MIN_PER_OPTION = 10


@dataclass(frozen=True)
class FitResult:
    """A finished fit: the folder written, its kinds and the report (also in the
    folder as ``training.json``)."""

    path: Path
    kinds: list[str]
    report: dict[str, Any]

    @property
    def metrics(self) -> dict[str, dict]:
        """Per question id, the holdout's metrics before and after the fit."""
        return self.report["questions"]


@dataclass
class Data:
    """Examples with the SHA-256 of their JSON lines and the file they came from."""

    examples: list[Example]
    sha256: str
    source: str | None


def read_data(data: str | Path | list[Example]) -> Data:
    """Examples from a file or as given, with their hash."""
    if isinstance(data, (str, Path)):
        raw = Path(data).read_bytes()
        return Data(read_examples(data), hashlib.sha256(raw).hexdigest(), str(data))
    examples = list(data)
    lines = "".join(json.dumps(e.to_json(), ensure_ascii=False) + "\n" for e in examples)
    return Data(examples, hashlib.sha256(lines.encode()).hexdigest(), None)


@dataclass
class _Base:
    """The model a fit starts from: a fitted model's base when given one."""

    path: Path
    reference: dict[str, str]
    fingerprint: str
    family: Any
    name: str


def _base(model: str | Path) -> _Base:
    path = resolve_model_path(model)
    if detect_family(path) is FITTED:
        folder = open_folder(path)  # its fit is replaced, its base kept
        reference = folder.marker["base"]
        return _Base(folder.base, reference, folder.marker["fingerprint"], folder.family,
                     folder.base_name)  # fmt: skip
    reference = base_reference(model, path)
    name = reference["repo_id"].rsplit("/", 1)[-1] if "repo_id" in reference else path.name
    return _Base(path, reference, fingerprint(path), detect_family(path), name)


def _questions(examples: list[Example]) -> dict[str, list[tuple[Example, Any]]]:
    """Each labelled question id with the examples asking it (parsed questions)."""
    asked: dict[str, list[tuple[Example, Any]]] = {}
    for example in examples:
        parsed = parse_request(example.request).questions
        for question_id in example.labels:
            asked.setdefault(question_id, []).append((example, parsed[question_id]))
    return asked


def _warnings(training: list[Example]) -> list[str]:
    """What makes a fit on ``training`` unreliable: few examples, rare options."""
    warnings = []
    if len(training) < MIN_EXAMPLES:
        warnings.append(
            f"only {len(training)} training examples ({MIN_EXAMPLES} or more recommended)"
        )
    for question_id, asked in _questions(training).items():
        counts = Counter(example.labels[question_id] for example, _ in asked)
        options = dict.fromkeys(o for _, question in asked for o in option_ids(question))
        rare = [f"{o} ({counts[o]})" for o in options if counts[o] < MIN_PER_OPTION]
        if rare:
            listed = ", ".join(rare[:5]) + (f" and {len(rare) - 5} more" if len(rare) > 5 else "")
            warnings.append(
                f"question {question_id!r}: fewer than {MIN_PER_OPTION} training examples "
                f"for {listed}"
            )
    return warnings


def label_counts(examples: list[Example]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {}
    for example in examples:
        for question_id, label in example.labels.items():
            counts.setdefault(question_id, Counter())[label] += 1
    return {qid: dict(sorted(c.items())) for qid, c in counts.items()}


def _calibrations(scored: list[Scored], fit: Fit) -> Fit:
    """A calibration per fitted question (same id, type and options) on ``scored``,
    added to ``fit``; biases where every example asks it the same way."""
    groups: dict[tuple, tuple[FitQuestion, list[tuple[Scored, str]]]] = {}
    for item in scored:
        if item.raw is None:
            continue
        for question_id in item.example.labels:
            fitted_question = FitQuestion.of(question_id, item.request.questions[question_id])
            key = (question_id, fitted_question.type, tuple(fitted_question.options))
            groups.setdefault(key, (fitted_question, []))[1].append((item, question_id))
    for fitted_question, rows in groups.values():
        options = fitted_question.options
        probabilities = [[item.raw[qid][o] for o in options] for item, qid in rows]
        labels = [options.index(item.example.labels[qid]) for item, qid in rows]
        wordings = {
            json.dumps(item.request.questions[qid].model_dump(), sort_keys=True, default=str)
            for item, qid in rows
        }
        fit.calibrations[fit.add(fitted_question)] = fit_calibration(
            probabilities, labels, bias=len(wordings) == 1
        )
    return fit


@dataclass
class _Rows:
    """A fitted question's readout inputs: the first example's features (whose
    rows, groups and temperature every example must share) and each part's
    hidden states and gold option indexes."""

    question: FitQuestion
    features: Any  # the first example's Features
    parts: dict[str, tuple[list, list[int]]]
    varies: bool = False

    def same(self, other) -> bool:
        first = self.features
        return (
            other.option_ids == first.option_ids
            and other.groups == first.groups
            and other.temperature == first.temperature
            and other.rows.shape == first.rows.shape
            and bool(mx.array_equal(other.rows, first.rows))
        )


def _readouts(fit_part: list[Scored], dev_part: list[Scored], emit, cancelled, rest: float):
    """A readout per fitted question, trained on the fit part's features and picked
    on dev's; the fit and a report per question."""
    from .readout import fit_readout, search_steps

    questions: dict[tuple, _Rows] = {}
    for part, scored in (("fit", fit_part), ("dev", dev_part)):
        for item in scored:
            if item.features is None:
                continue  # refused
            for question_id, label in item.example.labels.items():
                fitted_question = FitQuestion.of(question_id, item.request.questions[question_id])
                features = item.features[question_id]
                key = (question_id, fitted_question.type, tuple(fitted_question.options))
                entry = questions.setdefault(
                    key, _Rows(fitted_question, features, {"fit": ([], []), "dev": ([], [])})
                )
                entry.varies = entry.varies or not entry.same(features)
                entry.parts[part][0].append(features.hidden)
                # The rows are in the family's order of the options.
                entry.parts[part][1].append(features.option_ids.index(label))
    fit = Fit([])
    report = []
    total = search_steps() * len(questions)
    started = time.perf_counter()
    offset = 0
    for entry in questions.values():
        question = entry.question
        (fit_hidden, fit_labels), (dev_hidden, dev_labels) = entry.parts.values()
        row: dict[str, Any] = {
            "question": question.id,
            "options": question.options,
            "fit": len(fit_labels),
            "dev": len(dev_labels),
        }
        reason = None
        if entry.varies:
            reason = "its examples score other tokens (the options are worded differently)"
        elif not fit_labels or not dev_labels:
            reason = "it needs examples in both the fit and the dev part"
        if reason is not None:
            emit(Event("train", message=f"warning: no readout for {question.id!r}: {reason}"))
            report.append(row | {"skipped": reason})
            offset += search_steps()
            continue

        def on_step(done: int, offset=offset) -> None:
            step = offset + done
            elapsed = time.perf_counter() - started
            eta = round(elapsed / step * (total - step) + rest, 1)
            emit(Event("train", step, total, eta_s=eta))

        features = entry.features
        readout, picked = fit_readout(
            features.option_ids,
            features.rows,
            features.groups,
            features.temperature,
            (mx.stack(fit_hidden), fit_labels),
            (mx.stack(dev_hidden), dev_labels),
            on_step,
            cancelled,
        )
        offset += search_steps()
        if readout is not None:
            fit.readouts[fit.add(question)] = readout
        report.append(row | picked.as_json())
        emit(Event("train", metrics=row | picked.as_json(), message=f"readout for {question.id!r}"))
    return fit, report


def _finetune(model, splits, epochs, head_only, seed, emit, cancelled, rest: float):
    """Fine-tune the loaded model in place on the fit part (dev picks the epoch)."""
    from ..backend import Trainable
    from .finetune import Item, train

    backend = model.backend
    if not isinstance(backend, Trainable):
        raise ValueError(f"{backend.name} cannot be fine-tuned")

    def items(examples: list[Example]) -> list[Item]:
        result = []
        for example in examples:
            try:
                by_question = backend.items(model.check(example.request))
            except DecisionError:
                continue  # refused, as scoring would
            for question_id, label in example.labels.items():
                item = by_question[question_id]
                result.append(Item(item, item.option_ids.index(label)))
        return result

    emit(Event("train", message="preparing the training examples"))
    fit_items, dev_items = items(splits.fit), items(splits.dev)
    on_step, on_epoch = _reporters(emit, rest)
    return train(
        backend,
        fit_items,
        dev_items,
        epochs=epochs,
        head_only=head_only,
        seed=seed,
        on_step=on_step,
        on_epoch=on_epoch,
        cancelled=cancelled,
    )


def _reporters(emit, rest: float):
    """``on_step`` and ``on_epoch`` callbacks of an epoch-based kind, as events;
    ``rest`` is the seconds left after training."""
    started = time.perf_counter()

    def on_step(step: int, total: int, loss: float) -> None:
        elapsed = time.perf_counter() - started
        eta = round(elapsed / step * (total - step) + rest, 1)
        emit(Event("train", step, total, metrics={"loss": round(loss, 4)}, eta_s=eta))

    def on_epoch(row: dict) -> None:
        message = f"epoch {row['epoch']}: loss {row['loss']}, dev loss {row['dev_loss']}"
        emit(Event("train", metrics=row, message=message))

    return on_step, on_epoch


def _size(count: int) -> str:
    return f"{count / 1e9:.1f} GB" if count >= 1e9 else f"{max(count / 1e6, 0.1):.1f} MB"


def _store(model, output: Path, examples: list[Example], emit):
    """A store for the hidden states of ``examples`` beside ``output``, once their
    size is known (and shown) and the disk has room for it."""
    from .head import StateStore

    size = 0
    for example in examples:
        try:
            size += model.backend.head_size(model.check(example.request))
        except DecisionError:
            continue  # refused: nothing stored
    folder = output.parent / f".{output.name}.states"
    folder.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder.parent).free
    if size + DISK_MARGIN > free:
        raise ValueError(
            f"the hidden states need about {_size(size)} on disk in {folder.parent}, "
            f"where {_size(free)} are free"
        )
    emit(
        Event(
            "check",
            metrics={"states_gb": round(size / 1e9, 2)},
            message=f"storing the hidden states: about {_size(size)} in {folder}, "
            "removed afterwards",
        )
    )
    return StateStore(folder)


def _head(
    backend, store, fit_part: list[Scored], dev_part: list[Scored], epochs, seed, emit, cancelled
):
    """Train the head in place on the fit part's stored states (dev picks the epoch)."""
    from ..backend import HeadTrainable
    from .head import Item, train

    if not isinstance(backend, HeadTrainable):
        raise ValueError(f"{backend.name} has no head to fit")

    def items(scored: list[Scored]) -> list[Item]:
        result = []
        for item in scored:
            if item.state is None:
                continue  # refused
            questions = store.inputs[item.state].questions
            index = {question.question_id: i for i, question in enumerate(questions)}
            labels = {
                index[question_id]: questions[index[question_id]].option_ids.index(label)
                for question_id, label in item.example.labels.items()
            }
            result.append(Item(item.state, labels))
        return result

    on_step, on_epoch = _reporters(emit, 1.0)
    return train(
        backend,
        store,
        items(fit_part),
        items(dev_part),
        epochs=epochs,
        seed=seed,
        on_step=on_step,
        on_epoch=on_epoch,
        cancelled=cancelled,
    )


def _compact(metrics: dict[str, dict]) -> dict[str, dict]:
    keys = ("accuracy", "macro_f1", "ece", "mae")
    return {key: metrics[key] for key in keys if metrics.get(key) is not None}


@dataclass
class Prepared:
    """What a fit starts from, read and checked before any model loads: the
    base, the data (and validation data), the splits and the data's warnings."""

    origin: _Base
    source: Data
    held: Data | None
    splits: Any
    warnings: list[str]
    load_options: dict

    @property
    def family(self) -> str:
        return self.origin.family.name

    def estimate(self, kinds: list[str], epochs: int, head_only: bool) -> plan.Estimate:
        """Time and peak memory of a run with ``kinds``."""
        return plan.estimate(
            self.family, kinds, self.origin.path, self.load_options, self.splits, epochs, head_only
        )

    def extra(self, kinds: list[str], head_only: bool) -> int:
        """Memory the kinds need beyond the model's weights."""
        every = self.splits.holdout + self.splits.fit + self.splits.dev
        questions = plan.labelled(every)
        tokens = max((plan.clef_tokens(example) for example in every), default=0)
        return max(
            plan.kind_bytes(kind, self.origin.path, self.load_options, head_only, questions, tokens)
            for kind in kinds
        )


def prepare(
    base: str | Path,
    data: str | Path | list[Example],
    *,
    validation: str | Path | list[Example] | None = None,
    holdout: float = 0.2,
    seed: int = 0,
    load_options: dict | None = None,
) -> Prepared:
    """Read the data and find the base, as a fit's check stage does (no model loads)."""
    load_options = dict(load_options or {})
    source = read_data(data)
    held = read_data(validation) if validation is not None else None
    origin = _base(base)
    check_options(origin.family, load_options)
    splits = training_splits(
        source.examples,
        holdout=holdout,
        validation=held.examples if held is not None else None,
        seed=seed,
    )
    warnings = _warnings(splits.fit + splits.dev)
    return Prepared(origin, source, held, splits, warnings, load_options)


def fit(
    base: str | Path,
    data: str | Path | list[Example],
    *,
    output: str | Path,
    method: str = "auto",
    validation: str | Path | list[Example] | None = None,
    holdout: float = 0.2,
    seed: int = 0,
    epochs: int = 3,
    head_only: bool = False,
    on_event: Callable[[Event], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    overwrite: bool = False,
    **load_options,
) -> FitResult:
    """Fit ``base`` (a model folder or repo id, or a fitted model, whose fit is
    replaced) on labelled ``data`` (a JSON lines file or examples) and write the
    fitted model to ``output``.

    ``method`` is a kind (``calibrate``, ``readout``, ``head``, ``finetune``)
    or ``auto``, the family's usual kind; every run ends with a calibration.
    ``validation`` scores the report instead of a ``holdout`` share of the
    data (seeded with ``seed``, stratified by label). ``epochs`` is for the
    kinds that train in epochs; ``head_only`` fine-tunes the marker head
    without the encoder. ``on_event`` receives the progress;
    ``cancel()`` (a callable, or a ``threading.Event``) stops the run at the
    next step with ``Cancelled``. ``load_options`` go to the base's loader.

    Raises ``DataError`` for unusable data (naming the line), ``ValueError``
    for a kind the family does not take, ``ModelTooLargeError`` when the run
    would not fit in memory and ``FileExistsError`` for an existing
    ``output`` (unless ``overwrite``).
    """
    started = time.perf_counter()
    seconds: dict[str, float] = {}
    stage_start = started
    cancelled = getattr(cancel, "is_set", cancel)

    def emit(event: Event) -> None:
        if on_event is not None:
            on_event(event)

    def check_cancel() -> None:
        if cancelled is not None and cancelled():
            raise Cancelled()

    def finish_stage(stage: str) -> None:
        nonlocal stage_start
        now = time.perf_counter()
        seconds[stage] = round(seconds.get(stage, 0.0) + now - stage_start, 2)
        stage_start = now

    # Check: everything that can fail before the model loads.
    emit(Event("check", message="reading the data"))
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists (pass overwrite to replace it)")
    prepared = prepare(
        base, data, validation=validation, holdout=holdout, seed=seed, load_options=load_options
    )
    source, held, origin, splits = prepared.source, prepared.held, prepared.origin, prepared.splits
    kinds = plan.resolve(origin.family.name, method)
    if head_only and "finetune" not in kinds:
        raise ValueError("head_only applies to finetune")
    training = splits.fit + splits.dev
    warnings = prepared.warnings
    for warning in warnings:
        emit(Event("check", message=f"warning: {warning}"))
    trainer = kinds[0] if kinds[0] != "calibrate" else None
    estimate = prepared.estimate(kinds, epochs, head_only)
    extra = prepared.extra(kinds, head_only)
    check_fits(origin.path, load_options, name=origin.name, extra=extra, source=str(origin.path))
    emit(
        Event(
            "check",
            metrics={
                "examples": len(source.examples),
                "holdout": len(splits.holdout),
                "fit": len(splits.fit),
                "dev": len(splits.dev),
                "labels": label_counts(training),
                "kinds": kinds,
                "estimate": estimate.as_json(),
            },
            eta_s=round(estimate.seconds, 1),
            message=f"loading {origin.name}",
        )
    )
    check_cancel()
    model: DecisionModel = load(origin.path, check_memory=False, **load_options)
    model.backend.name = origin.name
    finish_stage("check")

    # Scoring passes: the holdout, then the training part (calibrate alone, and
    # the readout and head fits, which read the kept features or stored states
    # again) or, after fine-tuning, dev and the holdout again.
    reads, stores = trainer == "readout", trainer == "head"
    again = trainer == "finetune"
    passes = splits.holdout + (splits.dev + splits.holdout if again else training)
    clock = Clock(sum(plan.request_tokens(example) for example in passes))

    def scoring(stage: str, examples: list[Example]) -> list[Scored]:
        def on_step(done: int, example: Example) -> None:
            clock.step(plan.request_tokens(example))
            emit(Event(stage, done, len(examples), eta_s=clock.eta()))

        clock.start()
        try:
            return score(model, examples, on_step, cancelled, features=reads, store=store)
        finally:
            clock.stop()
            finish_stage(stage)

    store = _store(model, output, passes, emit) if stores else None
    try:
        holdout_scored = scoring("before", splits.holdout)
        unfitted = FittedBackend(origin.name, model.backend, Fit([]), origin.reference)
        before = summary(records(holdout_scored, unfitted.adjust))
        emit(Event("before", metrics=before))
        tensors: dict = {}
        training_report = None
        fit_ = Fit([])
        if not again:
            training_scored = scoring("features", training)
            if all(item.raw is None for item in training_scored):
                reason = training_scored[0].refused if training_scored else "no examples"
                raise ValueError(f"{origin.name} answered none of the training examples ({reason})")
            calibration_scored, after_scored = training_scored, holdout_scored
        if reads:
            fit_part, dev_part = (
                training_scored[: len(splits.fit)],
                training_scored[len(splits.fit) :],
            )
            fit_, questions_report = _readouts(fit_part, dev_part, emit, cancelled, rest=1.0)
            finish_stage("train")
            training_report = {"questions": questions_report}
            # The fitted rows read the kept features: dev fits the calibration, the
            # holdout scores the report.
            reader = FittedBackend(output.name, model.backend, fit_, origin.reference)
            calibration_scored = reread(dev_part, reader.readouts)
            after_scored = reread(holdout_scored, reader.readouts)
        elif trainer == "head":
            fit_part, dev_part = (
                training_scored[: len(splits.fit)],
                training_scored[len(splits.fit) :],
            )
            trained = _head(model.backend, store, fit_part, dev_part, epochs, seed, emit, cancelled)
            finish_stage("train")
            tensors = trained.tensors
            training_report = {"epochs": trained.epochs, "best_epoch": trained.best}
            # The trained head reads the stored states: dev fits the calibration, the
            # holdout scores the report.
            calibration_scored = rehead(dev_part, store, model.backend)
            after_scored = rehead(holdout_scored, store, model.backend)
        elif trainer is not None:
            rest = plan.scoring_seconds(origin.family.name, 0, splits.dev + splits.holdout)
            trained = _finetune(model, splits, epochs, head_only, seed, emit, cancelled, rest)
            finish_stage("train")
            tensors = trained.tensors
            training_report = {
                "head_only": head_only,
                "epochs": trained.epochs,
                "best_epoch": trained.best,
            }
            # The trained model: dev fits the calibration, the holdout scores the report.
            calibration_scored = scoring("calibrate", splits.dev)
            after_scored = scoring("after", splits.holdout)
            training_scored = calibration_scored  # what the report counts: dev, as trained
    finally:
        if store is not None:
            store.remove()

    check_cancel()
    emit(Event("calibrate", message="fitting the calibration"))
    fit_ = _calibrations(calibration_scored, fit_)
    fit_.kinds = kinds
    finish_stage("calibrate")

    fitted = FittedBackend(output.name, model.backend, fit_, origin.reference)
    after = summary(records(after_scored, fitted.adjust))
    questions = {
        question_id: {"before": before[question_id], "after": after[question_id]}
        for question_id in before
    }
    emit(Event("after", metrics=questions))
    finish_stage("after")

    check_cancel()
    emit(Event("write", message=f"writing {output}"))
    options = {
        "method": method,
        "kinds": kinds,
        "seed": seed,
        "holdout": holdout if held is None else None,
        "validation": (held.source or "examples") if held is not None else None,
        "epochs": epochs,
        "head_only": head_only,
        "load_options": {
            key: value if isinstance(value, (bool, int, float, str, type(None))) else str(value)
            for key, value in load_options.items()
        },
    }
    report = {
        "version": __version__,
        "base": {
            **origin.reference,
            "name": origin.name,
            "family": origin.family.name,
            "precision": getattr(model.backend, "precision", None),
        },
        "options": options,
        "data": {
            "file": source.source,
            "sha256": source.sha256,
            "examples": len(source.examples),
            "holdout": len(splits.holdout),
            "fit": len(splits.fit),
            "dev": len(splits.dev),
            "labels": label_counts(training),
            "refused": sum(item.raw is None for item in holdout_scored + training_scored),
            "truncated": sum(item.truncated for item in holdout_scored + training_scored),
        },
        "estimate": estimate.as_json(),
        "warnings": warnings,
        "questions": questions,
        "training": training_report,
        "calibrations": [
            {
                "question": question.id,
                "options": question.options,
                "temperature": round(fit_.calibrations[i].temperature, 6),
                "bias": fit_.calibrations[i].bias,
            }
            for i, question in enumerate(fit_.questions)
        ],
        "seconds": seconds,
    }
    fit_.meta = {
        "metrics": {
            question_id: {
                "n": entry["before"]["n"],
                "before": _compact(entry["before"]),
                "after": _compact(entry["after"]),
            }
            for question_id, entry in questions.items()
        },
        "seed": seed,
        "data": source.sha256,
        "version": __version__,
    }
    finish_stage("write")
    report["seconds"]["total"] = round(time.perf_counter() - started, 2)
    path = write_fitted(
        output,
        family=origin.family.name,
        base=origin.reference,
        base_fingerprint=origin.fingerprint,
        fit=fit_,
        tensors=tensors,
        report=report,
        overwrite=overwrite,
    )
    emit(Event("done", metrics=questions, message=f"wrote {path}"))
    return FitResult(path, kinds, report)
