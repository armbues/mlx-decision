"""Training data: labelled requests in JSON lines, and seeded splits of them.

Each line is a request body (``state``, ``questions``) plus ``labels``, an
object from question id to the correct answer: an option id (``choice``),
``true`` / ``false`` (``noul``) or a level (``score``, 0 for the first one).
An optional ``id`` (a string, unique in the file) names the example; without
one it is the line number. Questions without a label are asked but not
trained on. Images are not supported.

A file is checked as a whole before any model work: every request passes the
checks every model makes, every label is an answer of its question; the first
error names its line.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import DecisionError
from ..state_images import lift_images
from ..types import Choice, Noul, Score, decode_json, parse_request


class DataError(DecisionError):
    """A line of a training data file that cannot be used; ``line`` counts from 1."""

    def __init__(self, message: str, line: int, param: str | None = None):
        super().__init__(message, param=param)
        self.line = line

    def __str__(self) -> str:
        return f"line {self.line}: {super().__str__()}"


@dataclass(frozen=True)
class Example:
    """One labelled request. ``labels`` holds the answers as option ids: the
    ``choice`` option, "true" / "false" for ``noul``, the level as a string
    ("0", "1", ...) for ``score``, the way answers and metrics name them."""

    id: str
    request: dict[str, Any]  # the request body as given: state, questions
    labels: dict[str, str]

    def to_json(self) -> dict[str, Any]:
        """The line's object, with the labels in the file's form."""
        questions = self.request["questions"]
        labels = {
            question_id: _label_to_json(questions[question_id], label)
            for question_id, label in self.labels.items()
        }
        return {"id": self.id, **self.request, "labels": labels}


def _label_to_json(question: dict, label: str) -> Any:
    kind = question["type"]
    if kind == "noul":
        return label == "true"
    if kind == "score":
        return int(label)
    return label


def _label(question: Noul | Choice | Score, value: Any, param: str, line: int) -> str:
    """A label as an option id, or the error saying what the question takes."""
    if isinstance(question, Noul):
        if isinstance(value, bool):
            return "true" if value else "false"
        if value in ("true", "false"):
            return value
        raise DataError("a noul label is true or false", line, param)
    if isinstance(question, Score):
        levels = len(question.criteria)
        if isinstance(value, str) and value.isdigit() and value.isascii():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < levels:
            return str(value)
        raise DataError(f"a score label is a level from 0 to {levels - 1}", line, param)
    if isinstance(value, str) and value in question.criteria:
        return value
    options = ", ".join(question.criteria)
    raise DataError(f"{json.dumps(value)} is none of the options ({options})", line, param)


def parse_example(data: Any, line: int) -> Example:
    """One line's object as an example; ``DataError`` naming the line and field."""
    if not isinstance(data, dict):
        raise DataError('a line must be an object: {"state": ..., "labels": ...}', line)
    body = {key: value for key, value in data.items() if key not in ("id", "labels")}
    example_id = data.get("id", str(line))
    if not isinstance(example_id, str) or not example_id.strip():
        raise DataError("an id is a non-empty string", line, "id")
    for field in ("images", "videos"):
        if body.get(field):
            raise DataError("training data cannot hold images", line, field)
    try:
        request = parse_request(body)
        _, images = lift_images(request.state)
    except DecisionError as error:
        raise DataError(error.message, line, error.param) from None
    if images:
        raise DataError("training data cannot hold images", line, images[0][1])
    if "labels" not in data:
        raise DataError("no labels: an object from question id to the correct answer", line)
    if not isinstance(data["labels"], dict):
        raise DataError("labels must be an object from question id to the answer", line, "labels")
    labels = {}
    for question_id, value in data["labels"].items():
        param = f"labels.{question_id}"
        if question_id not in request.questions:
            raise DataError("the request has no question with this id", line, param)
        labels[question_id] = _label(request.questions[question_id], value, param, line)
    return Example(example_id, body, labels)


def parse_examples(lines: Iterable[str | bytes]) -> list[Example]:
    """The examples of JSON lines (blank lines skipped), checked as a whole."""
    examples: list[Example] = []
    seen: dict[str, int] = {}
    for number, text in enumerate(lines, 1):
        if not text.strip():
            continue
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8")
            except UnicodeDecodeError as error:
                raise DataError(f"not valid UTF-8: {error}", number) from None
        try:
            data = decode_json(text)
        except DecisionError as error:
            raise DataError(error.message, number) from None
        example = parse_example(data, number)
        if example.id in seen:
            raise DataError(f"the same id as line {seen[example.id]}", number, "id")
        seen[example.id] = number
        examples.append(example)
    if not examples:
        raise DecisionError("the data holds no examples")
    if not any(example.labels for example in examples):
        raise DecisionError("no example has a label")
    return examples


def read_examples(path: str | Path) -> list[Example]:
    """The examples of a JSON lines file, checked as a whole."""
    with open(path, "rb") as file:
        return parse_examples(file)


def write_examples(path: str | Path, examples: Iterable[Example]) -> int:
    """Writes the examples as JSON lines; returns how many."""
    count = 0
    with open(path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass(frozen=True)
class Splits:
    """The parts of a training run: ``holdout`` only scores the report; ``fit``
    trains; ``dev`` picks settings and the best epoch and fits the temperature."""

    holdout: list[Example]
    fit: list[Example]
    dev: list[Example]


DEV_SHARE = 0.1
DEV_MINIMUM = 50


def split(
    examples: list[Example], share: float, seed: int | str = 0, minimum: int = 0
) -> tuple[list[Example], list[Example]]:
    """The examples split in two, stratified by their labels: the rest and a part
    of ``share`` of them (at least ``minimum``, at most all but one). Each keeps the
    examples' order; the same seed gives the same split."""
    count = len(examples)
    size = min(max(round(share * count), minimum), count - 1)
    if size <= 0:
        return list(examples), []
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        strata[tuple(sorted(example.labels.items()))].append(index)
    rng = random.Random(f"{seed}")
    keys = sorted(strata)
    for key in keys:
        rng.shuffle(strata[key])
    # Largest remainders: each label combination gives its share, rounded so the
    # sizes add up; ties go to a seeded draw.
    quotas = {key: size * len(strata[key]) / count for key in keys}
    taken = {key: int(quotas[key]) for key in keys}
    order = sorted(keys, key=lambda key: (taken[key] - quotas[key], rng.random()))
    for key in order[: size - sum(taken.values())]:
        taken[key] += 1
    picked = {index for key in keys for index in strata[key][: taken[key]]}
    rest = [example for index, example in enumerate(examples) if index not in picked]
    part = [example for index, example in enumerate(examples) if index in picked]
    return rest, part


def training_splits(
    examples: list[Example],
    *,
    holdout: float = 0.2,
    validation: list[Example] | None = None,
    seed: int = 0,
) -> Splits:
    """Holdout (``validation`` if given, else a ``holdout`` share of the examples),
    then fit and dev from the rest: dev is a tenth, at least 50 examples but at
    most half of the rest."""
    if validation is None:
        if not 0 < holdout < 1:
            raise ValueError(f"the holdout is a share between 0 and 1, not {holdout}")
        rest, held = split(examples, holdout, f"{seed}:holdout")
    else:
        rest, held = list(examples), list(validation)
    fit, dev = split(rest, DEV_SHARE, f"{seed}:dev", min(DEV_MINIMUM, len(rest) // 2))
    return Splits(held, fit, dev)
