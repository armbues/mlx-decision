"""``evaluate``: a model scored on labelled data, and optionally a second one beside it.

Each model answers every example as served (a fitted model with its fit, a
base with its own last steps), and every labelled question gets the
metrics of the coverage suite (accuracy, macro-F1, ECE, MAE for scores),
its refusals and cut inputs, and a confusion matrix. Models are loaded one
at a time: the first is released before the second loads.
"""

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.format import Example
from ..fitted import Fit, FittedBackend, option_ids
from ..metrics import confusion
from ..types import parse_request
from .events import Event
from .fitting import read_data
from .plan import request_tokens
from .scoring import Clock, records, score, summary


@dataclass(frozen=True)
class ModelEval:
    """One model's results: per labelled question id, its metrics (rounded),
    ``type``, ``labels`` (the options in order) and ``confusion`` (gold ->
    predicted -> count)."""

    name: str
    model: str
    questions: dict[str, dict[str, Any]]
    seconds: float

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "questions": self.questions,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class EvalResult:
    """The data scored and each model's results, in the order given."""

    data: dict[str, Any]
    models: list[ModelEval] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {"data": self.data, "models": [m.as_json() for m in self.models]}


def _labelled(examples: list[Example]) -> dict[str, tuple[str, list[str]]]:
    """Each labelled question id with its type and options (from its first example)."""
    found: dict[str, tuple[str, list[str]]] = {}
    for example in examples:
        questions = parse_request(example.request).questions
        for question_id in example.labels:
            if question_id not in found:
                question = questions[question_id]
                found[question_id] = (question.type, option_ids(question))
    return found


def _identity(questions, probabilities):
    return {question_id: dict(values) for question_id, values in probabilities.items()}


def _score_model(
    model: str | Path,
    examples: list[Example],
    labelled: dict[str, tuple[str, list[str]]],
    emit: Callable[[Event], None],
    cancelled: Callable[[], bool] | None,
    check_memory: bool,
    load_options: dict,
) -> ModelEval:
    from ..model import load

    started = time.perf_counter()
    emit(Event("load", message=f"loading {model}"))
    loaded = load(model, check_memory=check_memory, **load_options)
    try:
        backend = loaded.backend
        name = backend.name
        clock = Clock(sum(request_tokens(example) for example in examples))

        def on_step(done: int, example: Example) -> None:
            clock.step(request_tokens(example))
            emit(Event("score", done, len(examples), eta_s=clock.eta()))

        clock.start()
        scored = score(loaded, examples, on_step, cancelled)
        clock.stop()
        # A fitted model answers as served; a base leaves its last steps (H2O's
        # yes/no floor) to the caller.
        adjust = (
            _identity
            if isinstance(backend, FittedBackend)
            else FittedBackend(name, backend, Fit([]), {}).adjust
        )
        by_question = records(scored, adjust)
        metrics = summary(by_question)
        questions = {}
        for question_id, (kind, labels) in labelled.items():
            rows = by_question.get(question_id, [])
            if not rows:
                continue
            questions[question_id] = {
                "type": kind,
                **metrics[question_id],
                "labels": labels,
                "confusion": confusion(rows, labels),
            }
    finally:
        del loaded
        gc.collect()
        _clear_cache()
    return ModelEval(name, str(model), questions, round(time.perf_counter() - started, 2))


def _clear_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()


def evaluate(
    model: str | Path,
    data: str | Path | list[Example],
    *,
    compare: str | Path | None = None,
    on_event: Callable[[Event], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    check_memory: bool = True,
    **load_options,
) -> EvalResult:
    """Score ``model`` (any model ``load`` takes, fitted ones included) on labelled
    ``data`` (a JSON lines file or examples), and ``compare`` after it.

    ``on_event`` receives the progress (stages ``load``, ``score``, ``done``);
    ``cancel()`` stops between examples with ``Cancelled``. ``load_options`` go
    to both models' loaders. Raises ``DataError`` for unusable data and
    ``ValueError`` when no question has a label.
    """
    cancelled = getattr(cancel, "is_set", cancel)

    def emit(event: Event) -> None:
        if on_event is not None:
            on_event(event)

    source = read_data(data)
    labelled = _labelled(source.examples)
    if not labelled:
        raise ValueError("the data has no labels to score against")
    info = {"file": source.source, "sha256": source.sha256, "examples": len(source.examples)}
    result = EvalResult(info)
    for name in [model] + ([compare] if compare is not None else []):
        result.models.append(
            _score_model(
                name, source.examples, labelled, emit, cancelled, check_memory, load_options
            )
        )
    emit(Event("done", metrics=result.as_json()))
    return result
