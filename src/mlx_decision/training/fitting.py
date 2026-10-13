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
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..data.format import Example, read_examples, training_splits
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
from .scoring import Clock, Scored, records, score, summary

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
class _Data:
    examples: list[Example]
    sha256: str
    source: str | None


def _read(data: str | Path | list[Example]) -> _Data:
    """Examples from a file or as given, with their hash."""
    if isinstance(data, (str, Path)):
        raw = Path(data).read_bytes()
        return _Data(read_examples(data), hashlib.sha256(raw).hexdigest(), str(data))
    examples = list(data)
    lines = "".join(json.dumps(e.to_json(), ensure_ascii=False) + "\n" for e in examples)
    return _Data(examples, hashlib.sha256(lines.encode()).hexdigest(), None)


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


def _label_counts(examples: list[Example]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {}
    for example in examples:
        for question_id, label in example.labels.items():
            counts.setdefault(question_id, Counter())[label] += 1
    return {qid: dict(sorted(c.items())) for qid, c in counts.items()}


def _calibrations(scored: list[Scored]) -> Fit:
    """A calibration per fitted question (same id, type and options) on ``scored``;
    biases where every example asks it the same way."""
    groups: dict[tuple, tuple[FitQuestion, list[tuple[Scored, str]]]] = {}
    for item in scored:
        if item.raw is None:
            continue
        for question_id in item.example.labels:
            fitted_question = FitQuestion.of(question_id, item.request.questions[question_id])
            key = (question_id, fitted_question.type, tuple(fitted_question.options))
            groups.setdefault(key, (fitted_question, []))[1].append((item, question_id))
    fit = Fit(["calibrate"])
    for fitted_question, rows in groups.values():
        options = fitted_question.options
        probabilities = [[item.raw[qid][o] for o in options] for item, qid in rows]
        labels = [options.index(item.example.labels[qid]) for item, qid in rows]
        wordings = {
            json.dumps(item.request.questions[qid].model_dump(), sort_keys=True, default=str)
            for item, qid in rows
        }
        fit.calibrations[len(fit.questions)] = fit_calibration(
            probabilities, labels, bias=len(wordings) == 1
        )
        fit.questions.append(fitted_question)
    return fit


def _compact(metrics: dict[str, dict]) -> dict[str, dict]:
    keys = ("accuracy", "macro_f1", "ece", "mae")
    return {key: metrics[key] for key in keys if metrics.get(key) is not None}


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
    kinds that train in epochs. ``on_event`` receives the progress;
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
    source = _read(data)
    held = _read(validation) if validation is not None else None
    origin = _base(base)
    check_options(origin.family, load_options)
    kinds = plan.resolve(origin.family.name, method)
    splits = training_splits(
        source.examples,
        holdout=holdout,
        validation=held.examples if held is not None else None,
        seed=seed,
    )
    training = splits.fit + splits.dev
    warnings = _warnings(training)
    for warning in warnings:
        emit(Event("check", message=f"warning: {warning}"))
    everything = splits.holdout + training
    estimate = plan.estimate(origin.family.name, kinds, origin.path, load_options, everything)
    extra = max(plan.kind_bytes(kind, origin.path, load_options) for kind in kinds)
    check_fits(origin.path, load_options, name=origin.name, extra=extra, source=str(origin.path))
    emit(
        Event(
            "check",
            metrics={
                "examples": len(source.examples),
                "holdout": len(splits.holdout),
                "fit": len(splits.fit),
                "dev": len(splits.dev),
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

    clock = Clock(sum(plan.request_tokens(example) for example in everything))

    def scoring(stage: str, examples: list[Example]) -> list[Scored]:
        def on_step(done: int, example: Example) -> None:
            clock.step(plan.request_tokens(example))
            emit(Event(stage, done, len(examples), eta_s=clock.eta()))

        clock.start()
        try:
            return score(model, examples, on_step, cancelled)
        finally:
            clock.stop()
            finish_stage(stage)

    holdout_scored = scoring("before", splits.holdout)
    unfitted = FittedBackend(origin.name, model.backend, Fit([]), origin.reference)
    before = summary(records(holdout_scored, unfitted.adjust))
    emit(Event("before", metrics=before))
    training_scored = scoring("features", training)
    if all(item.raw is None for item in training_scored):
        reason = training_scored[0].refused if training_scored else "no training examples"
        raise ValueError(f"{origin.name} answered none of the training examples ({reason})")

    # The kinds' own training comes here (later kinds); then the calibration, on
    # dev after a trained kind, on the whole training part alone.
    check_cancel()
    emit(Event("calibrate", message="fitting the calibration"))
    alone = kinds == ["calibrate"]
    calibration_part = training_scored if alone else training_scored[len(splits.fit) :]
    fit_ = _calibrations(calibration_part)
    fit_.kinds = kinds
    finish_stage("calibrate")

    fitted = FittedBackend(output.name, model.backend, fit_, origin.reference)
    after = summary(records(holdout_scored, fitted.adjust))
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
            "labels": _label_counts(training),
            "refused": sum(item.raw is None for item in holdout_scored + training_scored),
            "truncated": sum(item.truncated for item in holdout_scored + training_scored),
        },
        "estimate": estimate.as_json(),
        "warnings": warnings,
        "questions": questions,
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
        report=report,
        overwrite=overwrite,
    )
    emit(Event("done", metrics=questions, message=f"wrote {path}"))
    return FitResult(path, kinds, report)
