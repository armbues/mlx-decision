"""Dataset folders: one dataset, its questions and its splits in one place.

A dataset folder holds ``dataset.json`` and one JSON-lines file per split,
named after the split (``train.jsonl``). ``dataset.json`` has:

- ``format``: 1 (missing means 1); a higher one is from a newer version and
  refused.
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

Keys it does not know are kept as they are when it is written again.

A line holds ``id``, ``state`` and ``values`` (the raw values of the label
columns, as text), or ``labels`` as a plain file has them, and ``questions``
only where they differ per row. Reading a split fills in the dataset's
questions and maps the values to labels, so everything after the reader sees
the examples a plain file gives. A plain ``.jsonl`` file is a dataset of one
split. Image paths in a line are relative to the folder (``images/17.jpg``),
in a plain file to the file's folder; the reader makes them absolute.
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..errors import DecisionError
from .format import DataError, Example, image_files, parse_example, parse_examples, resolve_image

DATASET_FILE = "dataset.json"
# The version of dataset.json this one writes and reads (missing means 1).
DATASET_FORMAT = 1
SPLIT_SUFFIX = ".jsonl"
# What a split is for: training, the fit's report, Evaluate.
ROLES = ("train", "report", "evaluate")


def is_dataset_folder(path: str | Path) -> bool:
    return (Path(path) / DATASET_FILE).is_file()


def roles_by_name(names: list[str]) -> dict[str, str | None]:
    """Each split's role by its name: the first named like train trains, one
    like validation or dev reports, one like test evaluates when another
    reports, else reports; the rest none."""
    lower = {name: name.lower() for name in names}

    def first(words: tuple[str, ...], taken: set[str]) -> str | None:
        return next(
            (n for n in names if n not in taken and any(w in lower[n] for w in words)), None
        )

    train = first(("train",), set())
    report = first(("valid", "dev", "val"), {train})
    test = first(("test",), {train, report})
    roles: dict[str, str | None] = dict.fromkeys(names)
    if train:
        roles[train] = "train"
    if report:
        roles[report] = "report"
    if test:
        roles[test] = "evaluate" if report else "report"
    return roles


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
    # The keys of dataset.json this version does not know, written back as they are.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name if self.folder else self.path.name.removesuffix(SPLIT_SUFFIX)

    @property
    def root(self) -> Path:
        """The folder the lines' image paths are relative to."""
        return self.path if self.folder else self.path.parent

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

    def first_line(self) -> dict[str, Any]:
        """The first line of the first split file (an empty dict when there is
        none or it does not parse): whether lines carry questions and labels."""
        try:
            with open(self.split_path(next(iter(self.splits))), encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        loaded = json.loads(line)
                        return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError, StopIteration):
            pass
        return {}

    def missing(self) -> list[str]:
        """What a fit or an evaluation still needs: questions (the dataset's or
        every line's own), and for the values its lines keep, a map of every
        value of the column that answers them."""
        if not self.folder:
            return []  # a plain file: questions and labels in the lines
        line = self.first_line()
        if not self.questions:
            if self.columns or "questions" not in line:
                return ["it has no questions yet"]
            return []  # each line asks its own
        if not self.answers:
            if self.columns or ("values" in line and "labels" not in line):
                return ["no question is answered by one of its columns"]
            return []
        found = []
        for question_id, answer in self.answers.items():
            column = answer.get("column")
            if self.columns and column not in self.columns:
                found.append(f"{question_id}: answered by {column}, which has no values here")
                continue
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

    def parse_line(self, split: str, lenient: bool = False):
        """The line parser of ``split``'s file: its questions and labels filled
        in from ``dataset.json``. ``lenient`` (to browse a dataset that is not
        ready): a line without questions keeps its state and values, and an
        unmapped value gives no label."""
        file = self.split_path(split).name

        def parse(data: Any, line: int) -> Example:
            try:
                return self._parse(data, line, lenient)
            except DataError as error:
                error.file = file
                raise

        return parse

    def _parse(self, data: Any, line: int, lenient: bool = False) -> Example:
        if not isinstance(data, dict):
            raise DataError('a line must be an object: {"state": ..., "values": ...}', line)
        values = data.get("values")
        if values is not None:
            scalar = (str, int, float, bool)
            if not isinstance(values, dict) or not all(
                v is None or isinstance(v, scalar) for v in values.values()
            ):
                raise DataError("values must be an object from column to a value", line, "values")
            values = {k: raw_value(v) for k, v in values.items() if v is not None}
        body = {key: value for key, value in data.items() if key != "values"}
        questions = body.get("questions")
        if questions is None:
            if not self.questions:
                if lenient:
                    return _unasked(data, line, values, self.root)
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
            if mapped is None and lenient:
                continue
            if mapped is None:
                raise DataError(
                    f"{column} value {value!r} is not mapped for {question_id} "
                    f"({DATASET_FILE} answers)",
                    line,
                    f"values.{column}",
                )
            labels[question_id] = mapped
        body["labels"] = labels
        example = parse_example(body, line, self.root, open_images=not lenient)
        return replace(example, values=values)

    def read(self, split: str | None = None, lenient: bool = False) -> list[Example]:
        """The examples of ``split`` (a plain file: its only one), checked as a
        whole; ``lenient`` as ``parse_line`` (a dataset without labels yet)."""
        if not self.folder:
            root = self.root

            def plain(data: Any, line: int) -> Example:
                return parse_example(data, line, root)

            def browse(data: Any, line: int) -> Example:
                return _plain_lenient(data, line, root)

            with open(self.path, "rb") as file:
                if lenient:
                    return parse_examples(file, browse, require_labels=False)
                return parse_examples(file, plain)
        split = split or next(iter(self.splits))
        path = self.split_path(split)
        with open(path, "rb") as file:
            try:
                return parse_examples(
                    file, self.parse_line(split, lenient), require_labels=not lenient
                )
            except DataError as error:
                error.file = error.file or path.name  # a line that is not JSON, too
                raise

    def sha256(self, split: str | None = None) -> str:
        """The SHA-256 a fit records for its data: a plain file's bytes; a split's
        bytes and the questions and answers its lines take from ``dataset.json``;
        then the bytes of the image files the lines name, in order (data without
        image files hashes as before they were read)."""
        digest = hashlib.sha256()
        path = self.split_path(split or next(iter(self.splits)))
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(2**20), b""):
                digest.update(block)
        if self.folder:
            shared = {"questions": self.questions, "answers": self.answers}
            digest.update(json.dumps(shared, sort_keys=True, ensure_ascii=False).encode())
        for image in self._image_files(path):
            digest_file(digest, image)
        return digest.hexdigest()

    def _image_files(self, path: Path) -> list[Path]:
        """The image files the lines of ``path`` name, in order."""
        found = []
        with open(path, "rb") as file:
            for raw in file:
                if b'"images"' not in raw:
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    found.extend(image_files(data, self.root))
        return found

    def describe(self, split: str | None = None) -> str:
        """How runs name what they read: the file, or the dataset and its split."""
        if not self.folder:
            return str(self.path)
        return f"{self.path} ({split} split)"

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {"format": DATASET_FORMAT}
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
        body.update({key: value for key, value in self.extra.items() if key not in body})
        return body


KNOWN_KEYS = ("format", "questions", "answers", "columns", "splits", "source", "created")


def check_format(path: Path, loaded: dict[str, Any]) -> None:
    """Refuse a ``dataset.json`` whose ``format`` this version does not read."""
    from .. import __version__

    version = loaded.get("format", DATASET_FORMAT)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise DecisionError(f"{path}: format must be a whole number from 1")
    if version > DATASET_FORMAT:
        raise DecisionError(
            f"{path}: its format is {version}, which mlx-decision {__version__} does not "
            f"know (it reads format {DATASET_FORMAT}); a newer mlx-decision wrote this "
            "dataset (upgrade to use it)"
        )


def link_file(source: Path, target: Path) -> None:
    """``target`` as a hard link to ``source`` (no copy on the same disk), else
    a copy; its folders made."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def digest_file(digest, path: Path) -> None:
    """Add a file's bytes to ``digest`` (a missing one adds nothing: reading
    the data fails on it first)."""
    try:
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(2**20), b""):
                digest.update(block)
    except OSError:
        pass


def _unasked(data: dict, line: int, values: dict[str, str] | None, root: Path) -> Example:
    """A line of a dataset without questions: its id, state, images (paths
    made absolute) and values."""
    example_id = data.get("id", str(line))
    if not isinstance(example_id, str) or not example_id.strip():
        raise DataError("an id is a non-empty string", line, "id")
    if "state" not in data:
        raise DataError("a line needs a state", line, "state")
    request = {k: v for k, v in data.items() if k not in ("id", "values", "labels")}
    if isinstance(request.get("images"), list):
        request["images"] = [
            resolve_image(value, root) if isinstance(value, str) else value
            for value in request["images"]
        ]
    return Example(example_id, request, {}, values)


def _plain_lenient(data: Any, line: int, root: Path) -> Example:
    """A plain file's line read to browse or answer it: one without questions
    keeps its state; one without labels is an example without them. Images
    are not opened."""
    if isinstance(data, dict) and not data.get("questions"):
        return _unasked(data, line, None, root)
    if isinstance(data, dict) and "labels" not in data:
        data = {**data, "labels": {}}
    return parse_example(data, line, root, open_images=False)


def read_description(folder: Path) -> Dataset:
    """A folder's ``dataset.json``; ``DecisionError`` naming what is wrong."""
    path = folder / DATASET_FILE
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DecisionError(f"cannot read {path}: {error}") from None
    if not isinstance(loaded, dict):
        raise DecisionError(f"{path} must hold an object")
    check_format(path, loaded)
    kinds = {"questions": dict, "answers": dict, "columns": dict, "splits": dict, "source": dict}
    for key, kind in kinds.items():
        if loaded.get(key) is not None and not isinstance(loaded[key], kind):
            raise DecisionError(f"{path}: {key} must be an object")
    splits = loaded.get("splits") or {}
    if not splits:  # roles by the files' names
        found = sorted(p.name.removesuffix(SPLIT_SUFFIX) for p in folder.glob(f"*{SPLIT_SUFFIX}"))
        splits = {name: {"role": role} for name, role in roles_by_name(found).items()}
    if not splits:
        raise DecisionError(f"{folder} holds no split files (NAME{SPLIT_SUFFIX})")
    for name, meta in splits.items():
        if meta is not None and not isinstance(meta, dict):
            raise DecisionError(f"{path}: splits.{name} must be an object")
    for name, body in (loaded.get("columns") or {}).items():
        values = body.get("values") if isinstance(body, dict) else None
        if (
            not isinstance(body, dict)
            or not isinstance(values or {}, dict)
            or not all(isinstance(info, dict) for info in (values or {}).values())
        ):
            raise DecisionError(
                f'{path}: columns.{name} must be an object: {{"values": {{VALUE: {{"rows": N}}}}}}'
            )
    for question_id, question in (loaded.get("questions") or {}).items():
        if not isinstance(question, dict):
            raise DecisionError(f"{path}: questions.{question_id} must be an object")
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
        {key: value for key, value in loaded.items() if key not in KNOWN_KEYS},
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
    try:
        trained = split or dataset.pick("train")
    except DecisionError:
        return None  # no split to train on: the fit says so
    if report is None or report == trained:
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
    labels = {  # soft labels as distributions
        question_id: label
        for question_id, label in example.to_json()["labels"].items()
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


def write_dataset(
    folder: str | Path,
    dataset: Dataset,
    splits: list[SplitOut],
    before_rename=None,
    files: dict[str, bytes | Path] | None = None,
):
    """Write ``dataset.json`` and the split files to a partial folder beside
    ``folder``, then rename it into place: the dataset appears whole or not at
    all. The questions go into ``dataset.json`` when every line asks the same
    ones (else each line keeps its own). ``files`` are written too, by their
    path in the folder (the lines' images: ``images/NAME``): bytes, or a file
    linked in (``link_file``). ``before_rename``
    may raise to stop. ``FileExistsError`` when ``folder`` exists."""
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
        for name, data in (files or {}).items():
            target = partial / name
            if partial.resolve() not in target.resolve().parents:
                raise ValueError(f"{name}: not a path inside the dataset folder")
            if isinstance(data, Path):
                link_file(data, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
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
