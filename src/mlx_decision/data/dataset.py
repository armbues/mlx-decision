"""Dataset folders: one dataset, its questions and its splits in one place.

A dataset folder holds ``dataset.json`` and one JSON-lines file per split,
named after the split (``train.jsonl``). ``dataset.json`` has:

- ``questions``: the questions every line is asked (as in a request body);
  left out while the dataset has none, or when every line has its own.
- ``answers``: per question id, the ``column`` whose values answer it, its
  ``map`` (each value to an option id, ``true`` / ``false`` or a level) and
  the values ``left_out`` (their rows are not trained on that question).
- ``columns``: per value column, its ``values`` with their ``rows`` (and a
  ``name`` where one is known), in order.
- ``splits``: per split its ``role`` (``train``, ``report``, ``evaluate`` or
  None), the source ``split`` it came from, its ``rows`` and ``examples``,
  the ``sample`` and ``seed`` when sampled, the ``part`` when cut from a
  split.
- ``source`` (where the rows came from) and ``created``.

A line holds ``id``, ``state`` and ``values`` (the raw values of the label
columns, as text), or ``labels`` as a plain file has them, and ``questions``
only where they differ per row. Reading a split fills in the dataset's
questions and maps the values to labels, so everything after the reader sees
the examples a plain file gives. A plain ``.jsonl`` file is a dataset of one
split.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..errors import DecisionError
from .format import DataError, Example, _label_to_json, parse_example, parse_examples

DATASET_FILE = "dataset.json"
SPLIT_SUFFIX = ".jsonl"
# What a split is for: training, the fit's report, Evaluate.
ROLES = ("train", "report", "evaluate")


def is_dataset_folder(path: str | Path) -> bool:
    return (Path(path) / DATASET_FILE).is_file()


def raw_value(value: Any) -> str:
    """A value as ``values`` and the answers' maps hold it: text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class Dataset:
    """A dataset folder as its ``dataset.json`` describes it, or a plain file
    (``folder`` False: one split, named after the file, nothing else known)."""

    path: Path
    folder: bool = True
    questions: dict[str, Any] | None = None
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)
    splits: dict[str, dict[str, Any]] = field(default_factory=dict)
    source: dict[str, Any] | None = None
    created: str | None = None

    @property
    def name(self) -> str:
        return self.path.name if self.folder else self.path.name.removesuffix(SPLIT_SUFFIX)

    def split_path(self, split: str) -> Path:
        return self.path / f"{split}{SPLIT_SUFFIX}" if self.folder else self.path

    def with_role(self, role: str) -> str | None:
        """The split with ``role``, or None."""
        return next((name for name, meta in self.splits.items() if meta.get("role") == role), None)

    def pick(self, purpose: str) -> str:
        """The split a run reads: ``train`` the train split, ``report`` the report
        split, ``evaluate`` the evaluate split, then the report split; else the
        only split."""
        order = {"train": ("train",), "report": ("report",), "evaluate": ("evaluate", "report")}[
            purpose
        ]
        for role in order:
            found = self.with_role(role)
            if found is not None:
                return found
        if len(self.splits) == 1:
            return next(iter(self.splits))
        listed = ", ".join(self.splits)
        roles = " or ".join(order)
        raise DecisionError(
            f"{self.path}: no split has the role {roles}; pass one of its files "
            f"({listed}) as {self.path}/SPLIT{SPLIT_SUFFIX}"
        )

    def missing(self) -> list[str]:
        """What a fit or an evaluation still needs: questions, and a map for
        every value of the columns that answer them."""
        if not self.folder or not self.columns:
            return []  # labels in the lines
        if not self.questions:
            return ["it has no questions yet"]
        if not self.answers:
            return ["no question is answered by one of its columns"]
        found = []
        for question_id, answer in self.answers.items():
            column = answer.get("column")
            values = (self.columns.get(column) or {}).get("values") or {}
            done = set(answer.get("map") or {}) | set(answer.get("left_out") or [])
            open_values = [value for value in values if value not in done]
            if question_id not in self.questions:
                found.append(f"the answers name a question {question_id!r} it does not have")
            elif open_values:
                listed = ", ".join(open_values[:8]) + (", ..." if len(open_values) > 8 else "")
                found.append(
                    f"{question_id}: {column} values not mapped to an answer or left out: {listed}"
                )
        return found

    def require_ready(self) -> None:
        missing = self.missing()
        if missing:
            raise DecisionError(
                f"{self.path} cannot be used for training or evaluation yet: "
                + "; ".join(missing)
                + f" (set questions and answers in its {DATASET_FILE}, or write them in the "
                "Studio)"
            )

    def parse_line(self, split: str):
        """The line parser of ``split``'s file: its questions and labels filled
        in from ``dataset.json``."""
        file = self.split_path(split).name

        def parse(data: Any, line: int) -> Example:
            try:
                return self._parse(data, line)
            except DataError as error:
                error.file = file
                raise

        return parse

    def _parse(self, data: Any, line: int) -> Example:
        if not isinstance(data, dict):
            raise DataError('a line must be an object: {"state": ..., "values": ...}', line)
        values = data.get("values")
        if values is not None and (
            not isinstance(values, dict) or not all(isinstance(v, str) for v in values.values())
        ):
            raise DataError("values must be an object from column to text", line, "values")
        body = {key: value for key, value in data.items() if key != "values"}
        if body.get("images"):
            raise DataError(
                "images are not used for training yet (a dataset folder keeps them as paths "
                "into its images/ folder for a later version)",
                line,
                "images",
            )
        questions = body.get("questions")
        if questions is None:
            if not self.questions:
                raise DataError("no questions: the line has none and the dataset none yet", line)
            questions = body["questions"] = self.questions
        labels = dict(body.get("labels") or {})
        for question_id, answer in self.answers.items():
            if question_id in labels or question_id not in questions or not values:
                continue
            column = answer.get("column")
            value = values.get(column)
            if value is None or value in (answer.get("left_out") or []):
                continue
            mapped = (answer.get("map") or {}).get(value)
            if mapped is None:
                raise DataError(
                    f"{column} value {value!r} is not mapped for {question_id} "
                    f"({DATASET_FILE} answers)",
                    line,
                    f"values.{column}",
                )
            labels[question_id] = mapped
        body["labels"] = labels
        return replace(parse_example(body, line), values=values)

    def read(self, split: str | None = None) -> list[Example]:
        """The examples of ``split`` (a plain file: its only one), checked as a whole."""
        if not self.folder:
            with open(self.path, "rb") as file:
                return parse_examples(file)
        split = split or next(iter(self.splits))
        with open(self.split_path(split), "rb") as file:
            return parse_examples(file, self.parse_line(split))

    def sha256(self, split: str | None = None) -> str:
        """The SHA-256 a fit records for its data: a plain file's bytes; a split's
        bytes and the questions and answers its lines take from ``dataset.json``."""
        digest = hashlib.sha256()
        with open(self.split_path(split or next(iter(self.splits))), "rb") as file:
            for block in iter(lambda: file.read(2**20), b""):
                digest.update(block)
        if self.folder:
            shared = {"questions": self.questions, "answers": self.answers}
            digest.update(json.dumps(shared, sort_keys=True, ensure_ascii=False).encode())
        return digest.hexdigest()

    def describe(self, split: str | None = None) -> str:
        """How runs name what they read: the file, or the dataset and its split."""
        if not self.folder:
            return str(self.path)
        return f"{self.path} ({split} split)"

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if self.questions:
            body["questions"] = self.questions
        if self.answers:
            body["answers"] = self.answers
        if self.columns:
            body["columns"] = self.columns
        body["splits"] = self.splits
        if self.source is not None:
            body["source"] = self.source
        if self.created is not None:
            body["created"] = self.created
        return body


def read_description(folder: Path) -> Dataset:
    """A folder's ``dataset.json``; ``DecisionError`` naming what is wrong."""
    path = folder / DATASET_FILE
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DecisionError(f"cannot read {path}: {error}") from None
    if not isinstance(loaded, dict):
        raise DecisionError(f"{path} must hold an object")
    kinds = {"questions": dict, "answers": dict, "columns": dict, "splits": dict, "source": dict}
    for key, kind in kinds.items():
        if loaded.get(key) is not None and not isinstance(loaded[key], kind):
            raise DecisionError(f"{path}: {key} must be an object")
    splits = loaded.get("splits") or {}
    if not splits:
        found = sorted(p.name.removesuffix(SPLIT_SUFFIX) for p in folder.glob(f"*{SPLIT_SUFFIX}"))
        splits = {name: {} for name in found}
    if not splits:
        raise DecisionError(f"{folder} holds no split files (NAME{SPLIT_SUFFIX})")
    for name, meta in splits.items():
        role = (meta or {}).get("role")
        if role is not None and role not in ROLES:
            raise DecisionError(f"{path}: split {name}'s role is one of {', '.join(ROLES)}")
    for question_id, answer in (loaded.get("answers") or {}).items():
        if not isinstance(answer, dict) or not isinstance(answer.get("column"), str):
            raise DecisionError(f"{path}: answers.{question_id} needs a column")
        if not isinstance(answer.get("map") or {}, dict):
            raise DecisionError(f"{path}: answers.{question_id}.map must be an object")
        answer["map"] = {raw_value(k): v for k, v in (answer.get("map") or {}).items()}
        answer["left_out"] = [raw_value(v) for v in answer.get("left_out") or []]
    return Dataset(
        folder,
        True,
        loaded.get("questions") or None,
        loaded.get("answers") or {},
        loaded.get("columns") or {},
        {name: dict(meta or {}) for name, meta in splits.items()},
        loaded.get("source"),
        loaded.get("created"),
    )


def open_dataset(path: str | Path) -> tuple[Dataset, str | None]:
    """The dataset at ``path`` (a folder, a split file inside one, or a plain
    file) and the split ``path`` names (None for a folder)."""
    path = Path(path)
    if path.is_dir():
        if not is_dataset_folder(path):
            raise DecisionError(f"{path} is no dataset folder (it has no {DATASET_FILE})")
        return read_description(path), None
    if is_dataset_folder(path.parent) and path.name.endswith(SPLIT_SUFFIX):
        dataset = read_description(path.parent)
        split = path.name.removesuffix(SPLIT_SUFFIX)
        dataset.splits.setdefault(split, {})
        return dataset, split
    if not path.exists():
        raise FileNotFoundError(f"no such file or folder: {path}")
    return Dataset(path, False, splits={path.name.removesuffix(SPLIT_SUFFIX): {}}), None


@dataclass
class Read:
    """The examples a run reads, the split, their SHA-256 and how to name them."""

    examples: list[Example]
    split: str
    sha256: str
    description: str
    dataset: Dataset


def read_for(path: str | Path, purpose: str = "train") -> Read:
    """The examples of ``path`` for a fit (``train``) or an evaluation
    (``evaluate``): a folder's split with that role, or the split file named;
    a dataset that is not ready says what it misses."""
    dataset, split = open_dataset(path)
    split = split or dataset.pick(purpose)
    dataset.require_ready()
    examples = dataset.read(split)
    return Read(examples, split, dataset.sha256(split), dataset.describe(split), dataset)


def report_split(path: str | Path) -> Path | None:
    """The report split's file of a dataset folder given as a fit's data (the
    folder itself, or its train split's file), or None."""
    try:
        dataset, split = open_dataset(path)
    except (DecisionError, OSError):
        return None
    if not dataset.folder:
        return None
    report = dataset.with_role("report")
    if report is None or report == (split or dataset.pick("train")):
        return None
    return dataset.split_path(report)


def dataset_name(path: str | Path) -> str:
    """A data path's name for output names: the folder's, a plain file's stem."""
    path = Path(path)
    if path.is_file() and is_dataset_folder(path.parent):
        return path.parent.name
    return path.name.removesuffix(SPLIT_SUFFIX) if path.suffix else path.name


# Writing


def line_json(example: Example, shared_questions: bool, mapped: set[str]) -> dict[str, Any]:
    """A folder line: ``questions`` only when not shared, ``labels`` only for
    questions the values do not answer."""
    obj: dict[str, Any] = {"id": example.id}
    for key, value in example.request.items():
        if key == "questions" and shared_questions:
            continue
        obj[key] = value
    if example.values:
        obj["values"] = example.values
    questions = example.request.get("questions") or {}
    labels = {
        question_id: _label_to_json(questions[question_id], label)
        for question_id, label in example.labels.items()
        if question_id not in mapped
    }
    if labels:
        obj["labels"] = labels
    return obj


@dataclass
class SplitOut:
    """One split file to write: its name, examples and ``splits`` entry."""

    name: str
    examples: list[Example]
    meta: dict[str, Any]


def shared_questions(splits: list[SplitOut]) -> dict[str, Any] | None:
    """The questions every example asks, when they are the same on every line."""
    first = None
    for out in splits:
        for example in out.examples:
            questions = example.request.get("questions")
            if questions is None:
                continue
            if first is None:
                first = questions
            elif questions is not first and questions != first:
                return None
    return first


def partial_folder(folder: str | Path) -> Path:
    folder = Path(folder)
    return folder.with_name(f".{folder.name}.partial")


def write_dataset(folder: str | Path, dataset: Dataset, splits: list[SplitOut], before_rename=None):
    """Write ``dataset.json`` and the split files to a partial folder beside
    ``folder``, then rename it into place: the dataset appears whole or not at
    all. The questions go into ``dataset.json`` when every line asks the same
    ones (else each line keeps its own). ``before_rename`` may raise to stop.
    ``FileExistsError`` when ``folder`` exists."""
    import shutil

    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} exists")
    partial = partial_folder(folder)
    shutil.rmtree(partial, ignore_errors=True)
    shared = shared_questions(splits)
    if shared is not None:
        dataset.questions = shared
    mapped = set(dataset.answers)
    try:
        partial.mkdir(parents=True)
        dataset.splits = {out.name: out.meta for out in splits}
        for out in splits:
            with open(partial / f"{out.name}{SPLIT_SUFFIX}", "w", encoding="utf-8") as file:
                for example in out.examples:
                    obj = line_json(example, shared is not None, mapped)
                    file.write(json.dumps(obj, ensure_ascii=False) + "\n")
        text = json.dumps(dataset.to_json(), indent=2, ensure_ascii=False) + "\n"
        (partial / DATASET_FILE).write_text(text, encoding="utf-8")
        if before_rename is not None:
            before_rename()
        os.rename(partial, folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    dataset.path = folder
    return folder
