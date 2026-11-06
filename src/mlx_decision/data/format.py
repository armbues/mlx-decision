"""Training data: labelled requests in JSON lines, and seeded splits of them.

Each line is a request body (``state``, ``questions``) plus ``labels``, an
object from question id to the correct answer: an option id (``choice``),
``true`` / ``false`` (``noul``) or a level (``score``, 0 for the first one).
A label may also be soft, a distribution over the answers: an object from
option id (a score's level as a string, "0", "1", ...) to its probability,
the probabilities summing to 1, options left out at 0; for a ``noul``, the
probability of true (a number from 0 to 1). A soft label that puts all of its
weight on one answer is that answer, a hard label. An optional ``id`` (a
string, unique in the file) names the example; without one it is the line
number. Questions without a label are asked but not
trained on. A line may hold images as a request does (``images``, and images
inside the state): data URLs, or file paths relative to the folder the file
is in (a dataset folder: the folder, whose ``images/`` holds them). Paths
are read as absolute paths, so the examples answer from any working folder.
Videos are refused.

A dataset folder (``dataset``) holds such files, one per split, whose lines
may take their questions and labels from the folder's ``dataset.json``.

A file is checked as a whole before any model work: every request passes the
checks every model makes, every label is an answer of its question; the first
error names its line. Every image is opened as a model opens it (its type
and size read, not decoded), so a missing or unreadable image is an error of
its line.
"""

import json
import math
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
    """A line of a training data file that cannot be used; ``line`` counts from 1,
    ``file`` names the split file of a dataset folder (None for a plain file)."""

    def __init__(self, message: str, line: int, param: str | None = None, file: str | None = None):
        super().__init__(message, param=param)
        self.line = line
        self.file = file

    def __str__(self) -> str:
        where = f"{self.file} line {self.line}" if self.file else f"line {self.line}"
        return f"{where}: {super().__str__()}"


@dataclass(frozen=True)
class Example:
    """One labelled request. ``labels`` holds the answers as option ids: the
    ``choice`` option, "true" / "false" for ``noul``, the level as a string
    ("0", "1", ...) for ``score``, the way answers and metrics name them. A
    soft label is in ``targets`` too (question id -> option id -> probability,
    every option of the question, summing to 1), its label the most likely
    answer (the first of the question's options on a tie; a noul's true at
    0.5)."""

    id: str
    request: dict[str, Any]  # the request body as given: state, questions
    labels: dict[str, str]
    # A dataset folder's raw values of the source's label columns (column ->
    # value as text), from which ``dataset.json`` maps the labels; None elsewhere.
    values: dict[str, str] | None = None
    targets: dict[str, dict[str, float]] | None = None
    # The source dataset's name when read from a mix; None elsewhere.
    source: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The line's object, with the labels in the file's form."""
        questions = self.request.get("questions") or {}
        targets = self.targets or {}
        labels = {
            question_id: _target_to_json(questions[question_id], targets[question_id])
            if question_id in targets
            else _label_to_json(questions[question_id], label)
            for question_id, label in self.labels.items()
        }
        return {"id": self.id, **self.request, "labels": labels}

    def gold(self, question_id: str, options: list[str]) -> int | tuple[float, ...]:
        """A labelled question's gold in ``options``' order (a family's order of
        the question's option ids): the label's index, or a soft label's
        probabilities."""
        target = (self.targets or {}).get(question_id)
        if target is None:
            return options.index(self.labels[question_id])
        return tuple(target.get(option, 0.0) for option in options)


def _target_to_json(question: dict, target: dict[str, float]) -> Any:
    if question["type"] == "noul":
        return target["true"]
    return dict(target)


def _label_to_json(question: dict, label: str) -> Any:
    kind = question["type"]
    if kind == "noul":
        return label == "true"
    if kind == "score":
        return int(label)
    return label


# How far a soft label's probabilities may sum from 1 (they are renormalised),
# and up to which distance that is rounding (they are kept as given).
SUM_TOLERANCE = 1e-3
ROUNDING = 1e-5


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _soft(
    question: Noul | Choice | Score, value: Any, param: str, line: int
) -> tuple[str, dict[str, float] | None]:
    """A label as an option id and, when it is soft, its distribution over the
    question's options (None for a hard label, or a soft one on one answer)."""
    if isinstance(question, Noul):
        if not _number(value):
            return _label(question, value, param, line), None
        if not 0 <= value <= 1:
            raise DataError(
                "a soft noul label is the probability of true, from 0 to 1", line, param
            )
        target = {"true": float(value), "false": 1 - float(value)}
    elif isinstance(value, dict):
        options = [str(level) for level in range(len(question.criteria))]
        if isinstance(question, Choice):
            options = list(question.criteria)
        for key, probability in value.items():
            if key not in options:
                listed = ", ".join(options)
                raise DataError(
                    f"a soft label names {json.dumps(key)}, none of the options ({listed})",
                    line,
                    param,
                )
            if not _number(probability) or probability < 0:
                raise DataError(
                    f"a soft label's probability is a number from 0 to 1, not "
                    f"{json.dumps(probability)} for {json.dumps(key)}",
                    line,
                    param,
                )
        total = sum(value.values())
        if abs(total - 1) > SUM_TOLERANCE:
            raise DataError(f"a soft label's probabilities sum to 1, not {total:.6g}", line, param)
        if abs(total - 1) <= ROUNDING:  # rounded as given: kept as it is
            total = 1.0
        target = {option: float(value.get(option, 0)) / total for option in options}
    else:
        return _label(question, value, param, line), None
    label = max(target, key=target.get)  # the first on a tie: true for a noul at 0.5
    if target[label] >= 1 - 1e-9:
        return label, None
    return label, target


def _label(question: Noul | Choice | Score, value: Any, param: str, line: int) -> str:
    """A label as an option id, or the error saying what the question takes."""
    if isinstance(question, Noul):
        if isinstance(value, bool):
            return "true" if value else "false"
        if value in ("true", "false"):
            return value
        raise DataError("a noul label is true or false, or the probability of true", line, param)
    if isinstance(question, Score):
        levels = len(question.criteria)
        if isinstance(value, str) and value.isdigit() and value.isascii():
            value = int(value)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < levels:
            return str(value)
        raise DataError(
            f"a score label is a level from 0 to {levels - 1}, or an object from level "
            "to probability",
            line,
            param,
        )
    if isinstance(value, str) and value in question.criteria:
        return value
    options = ", ".join(question.criteria)
    raise DataError(
        f"{json.dumps(value)} is none of the options ({options}), nor an object from "
        "option to probability",
        line,
        param,
    )


def resolve_image(value: str, root: Path | None) -> str:
    """An image of a line as the model reads it: a data URL as it is, a file
    path made absolute against ``root`` (the folder the line's file is in;
    None: the working folder's)."""
    if value.startswith("data:"):
        return value
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return str(path.absolute())


def image_files(request: dict[str, Any], root: Path | None = None) -> list[Path]:
    """The files a request's ``images`` name (data URLs left out), resolved as
    the reader resolves them; entries that are not text are skipped."""
    images = request.get("images")
    if not isinstance(images, list):
        return []
    return [
        Path(resolve_image(value, root))
        for value in images
        if isinstance(value, str) and not value.startswith("data:")
    ]


def _resolve_images(images: list[Any], root: Path | None, line: int) -> list[str]:
    resolved = []
    for index, value in enumerate(images):
        if not isinstance(value, str) or not value.strip():
            raise DataError("an image is a file path or a data URL", line, f"images.{index}")
        resolved.append(resolve_image(value, root))
    return resolved


def _open_images(given: list[Any], images: list[str], state: Any, line: int) -> None:
    """Open every image of a line (its header, as a model does before
    answering): a missing, unreadable or oversized one is the line's error."""
    from ..images import open_image

    for index, (written, value) in enumerate(zip(given, images, strict=True)):
        where = f"images.{index}"
        named = "" if value.startswith("data:") else f"{written}: "
        try:
            if named:
                path = Path(value)
                if not path.is_file():
                    raise DecisionError("no such image file")
                try:
                    value = path.read_bytes()
                except OSError as error:
                    raise DecisionError(error.strerror or "cannot be read") from None
            open_image(value, index, allow_paths=False)
        except DecisionError as error:
            raise DataError(named + error.message, line, where) from None
    try:
        _, lifted = lift_images(state, len(images))
        for index, (raw, _) in enumerate(lifted, len(images)):
            try:
                open_image(raw, index, allow_paths=False)
            except DecisionError as error:
                raise DecisionError(error.message, param=lifted[index - len(images)][1]) from None
    except DecisionError as error:
        raise DataError(error.message, line, error.param) from None


def parse_example(
    data: Any, line: int, root: Path | None = None, open_images: bool = True
) -> Example:
    """One line's object as an example; ``DataError`` naming the line and field.
    Image paths are relative to ``root`` (None: the working folder);
    ``open_images`` opens every image (off to browse a dataset quickly)."""
    if not isinstance(data, dict):
        raise DataError('a line must be an object: {"state": ..., "labels": ...}', line)
    body = {key: value for key, value in data.items() if key not in ("id", "labels")}
    example_id = data.get("id", str(line))
    if not isinstance(example_id, str) or not example_id.strip():
        raise DataError("an id is a non-empty string", line, "id")
    if body.get("videos"):
        raise DataError("training data cannot hold videos", line, "videos")
    try:
        request = parse_request(body)
    except DecisionError as error:
        raise DataError(error.message, line, error.param) from None
    if request.images:
        given = list(request.images)
        body["images"] = _resolve_images(given, root, line)
        if open_images:
            _open_images(given, body["images"], request.state, line)
    elif open_images:
        _open_images([], [], request.state, line)
    if "labels" not in data:
        raise DataError("no labels: an object from question id to the correct answer", line)
    if not isinstance(data["labels"], dict):
        raise DataError("labels must be an object from question id to the answer", line, "labels")
    labels, targets = {}, {}
    for question_id, value in data["labels"].items():
        param = f"labels.{question_id}"
        if question_id not in request.questions:
            raise DataError("the request has no question with this id", line, param)
        label, target = _soft(request.questions[question_id], value, param, line)
        labels[question_id] = label
        if target is not None:
            targets[question_id] = target
    return Example(example_id, body, labels, targets=targets or None)


def parse_examples(
    lines: Iterable[str | bytes], parse=parse_example, require_labels: bool = True
) -> list[Example]:
    """The examples of JSON lines (blank lines skipped), checked as a whole;
    ``parse`` turns one line's object into an example (a dataset folder's
    lines take what they leave out from its ``dataset.json``)."""
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
        example = parse(data, number)
        if example.id in seen:
            raise DataError(f"the same id as line {seen[example.id]}", number, "id")
        seen[example.id] = number
        examples.append(example)
    if not examples:
        raise DecisionError("the data holds no examples")
    if require_labels and not any(example.labels for example in examples):
        raise DecisionError("no example has a label")
    return examples


def check_examples(
    lines: Iterable[str | bytes], limit: int = 20, parse=parse_example
) -> tuple[list[Example], list[DecisionError]]:
    """``parse_examples`` that goes on past a bad line: the examples of the
    good lines and the errors (each ``DataError`` names its line), at most
    ``limit`` of them, the file's own (no examples, no labels) last. The file
    is usable when there are no errors."""
    examples: list[Example] = []
    errors: list[DecisionError] = []
    seen: dict[str, int] = {}
    for number, text in enumerate(lines, 1):
        if len(errors) >= limit:
            break
        try:
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
            example = parse(data, number)
            if example.id in seen:
                raise DataError(f"the same id as line {seen[example.id]}", number, "id")
        except DataError as error:
            errors.append(error)
            continue
        seen[example.id] = number
        examples.append(example)
    if not errors and not examples:
        errors.append(DecisionError("the data holds no examples"))
    elif not errors and not any(example.labels for example in examples):
        errors.append(DecisionError("no example has a label"))
    return examples, errors


def read_examples(path: str | Path) -> list[Example]:
    """The examples of a JSON lines file, checked as a whole (image paths
    relative to the file's folder)."""
    root = Path(path).parent

    def parse(data: Any, line: int) -> Example:
        return parse_example(data, line, root)

    with open(path, "rb") as file:
        return parse_examples(file, parse)


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


def stratum(example: Example) -> tuple:
    """What a split keeps in proportion: the labels, else the raw values."""
    return tuple(sorted(example.labels.items())) or tuple(sorted((example.values or {}).items()))


def split(
    examples: list[Example], share: float, seed: int | str = 0, minimum: int = 0
) -> tuple[list[Example], list[Example]]:
    """The examples split in two, stratified by their labels (``stratum``): the rest and a part
    of ``share`` of them (at least ``minimum``, at most all but one). Each keeps the
    examples' order; the same seed gives the same split."""
    count = len(examples)
    size = min(max(round(share * count), minimum), count - 1)
    if size <= 0:
        return list(examples), []
    strata: dict[tuple, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        strata[stratum(example)].append(index)
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
