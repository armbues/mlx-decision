"""One scoring pass over labelled examples: the model's probabilities before the
family's last steps, so a fit can come between them and its answers."""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from ..answers import build_answer
from ..backend import Readable
from ..data.format import Example
from ..errors import DecisionError
from ..metrics import metrics, record
from ..model import DecisionModel
from ..models.readout import Features, pick, read_out
from ..types import Request
from .events import Cancelled


@dataclass
class Scored:
    """An example as the model saw it: the checked request and its probabilities
    per question before the family's last steps (and, from a features pass,
    what each question's readout read); ``refused`` instead when the model
    would not answer it."""

    example: Example
    request: Request | None = None
    raw: dict[str, dict[str, float]] | None = None
    truncated: bool = False
    refused: str | None = None
    features: dict[str, Features] | None = None
    state: int | None = None  # its index in a StateStore


def _read(features: dict[str, Features], readouts=None) -> dict[str, dict[str, float]]:
    return {
        question_id: dict(
            zip(f.option_ids, read_out(f, pick(readouts, question_id, f.option_ids)), strict=True)
        )
        for question_id, f in features.items()
    }


def score(
    model: DecisionModel,
    examples: list[Example],
    on_step: Callable[[int, Example], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    features: bool = False,
    store=None,
) -> list[Scored]:
    """Each example scored by ``model`` in turn; ``on_step(done, example)`` after
    each, and ``Cancelled`` between them once ``cancelled()`` is true. With
    ``features`` (a family that reads out), the readouts' features are kept and
    the probabilities read from them, as ``score`` does; with ``store`` (a
    ``StateStore``, Clef), the head's inputs are stored and read by the head."""
    backend = model.backend
    readable = isinstance(backend, Readable)
    if features and not readable:
        raise ValueError(f"{backend.name} has no readout features")
    results = []
    for done, example in enumerate(examples, 1):
        if cancelled is not None and cancelled():
            raise Cancelled()
        try:
            request = model.check(example.request)
            if features:
                found = backend.features(request)
                truncated = any(f.truncated for f in found.values())
                results.append(Scored(example, request, _read(found), truncated, features=found))
            elif store is not None:
                inputs = backend.head_inputs(request)
                raw = backend.read_head(inputs)
                state = store.add(inputs)
                results.append(Scored(example, request, raw, inputs.truncated, state=state))
            else:
                output = (
                    backend.score(request, finish=False) if readable else backend.score(request)
                )
                results.append(Scored(example, request, output.probabilities, output.truncated))
        except DecisionError as error:
            results.append(Scored(example, refused=str(error)))
        if on_step is not None:
            on_step(done, example)
    return results


def reread(scored: list[Scored], readouts) -> list[Scored]:
    """``scored`` (from a features pass) read out again: ``readouts(questions)``
    gives the readouts by question id (a ``FittedBackend``'s)."""
    return [
        replace(item, raw=_read(item.features, readouts(item.request.questions)))
        if item.features is not None
        else item
        for item in scored
    ]


def rehead(scored: list[Scored], store, backend) -> list[Scored]:
    """``scored`` (with stored head inputs) read again by ``backend``'s head as it
    is now."""
    return [
        replace(item, raw=backend.read_head(store.get(item.state)))
        if item.state is not None
        else item
        for item in scored
    ]


def records(scored: list[Scored], adjust) -> dict[str, list[dict]]:
    """The metrics' records per labelled question id; ``adjust(questions, raw)``
    turns raw probabilities into the answered ones (a ``FittedBackend``'s)."""
    result: dict[str, list[dict]] = {}
    for item in scored:
        labels = item.example.labels
        if item.raw is None:
            for question_id, gold in labels.items():
                result.setdefault(question_id, []).append(
                    {"gold": gold, "predicted": None, "refused": item.refused}
                )
            continue
        probabilities = adjust(item.request.questions, item.raw)
        for question_id, gold in labels.items():
            question = item.request.questions[question_id]
            answer = build_answer(question, probabilities[question_id]).model_dump()
            entry = record(answer, gold)
            entry["truncated"] = item.truncated
            result.setdefault(question_id, []).append(entry)
    return result


def summary(by_question: dict[str, list[dict]]) -> dict[str, dict]:
    """``metrics`` per question id, rounded for reports."""
    return {
        question_id: {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in metrics(rows).items()
        }
        for question_id, rows in by_question.items()
        if rows
    }


class Clock:
    """Seconds left of the scoring passes, from the tokens done so far."""

    def __init__(self, total_tokens: int):
        self.total = total_tokens
        self.done = 0
        self.seconds = 0.0
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def step(self, tokens: int) -> None:
        now = time.perf_counter()
        if self._start is not None:
            self.seconds += now - self._start
        self._start = now
        self.done += tokens

    def stop(self) -> None:
        self._start = None

    def eta(self) -> float | None:
        if not self.done:
            return None
        return round(self.seconds / self.done * max(self.total - self.done, 0), 1)
