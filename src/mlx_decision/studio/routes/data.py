"""Datasets: the workspace's ``data/``, each a dataset folder (``dataset.json``
and one JSON-lines file per split) or a plain JSON-lines file of labelled
examples (a dataset of one split) (DS-1 to DS-5, TRS-1).

``GET /studio/api/data`` lists them; ``GET /studio/api/data/{name}``
describes one split (``split``; by default the train split, else the first):
the dataset's splits, its questions with their labels and counts, the
split's label values with their counts, the answers map, what a fit still
needs (``missing``), where it came from and a few examples. ``GET
.../examples`` reads a page of a split's examples (``offset``, ``limit``, a
label or value filter, a search in the states), each state cut to a line;
``GET .../examples/{index}`` one example whole. ``PUT .../questions``
saves a folder's questions and answers map into its ``dataset.json``.
``POST .../reveal`` shows it in Finder (a browser on this Mac only).
``POST .../set`` saves a question set made from a split's questions and
first examples. ``DELETE`` deletes a dataset unless a job that is not
over reads it (409).

``POST /studio/api/data`` uploads a file the page read: training JSON lines
(``text``), checked as ``training fit`` checks them and kept as a plain
file; or rows of columns (``text`` with ``format`` csv or jsonl, the state's
``text_column`` and the label's ``label_column``), kept as a dataset folder
without questions. ``POST /studio/api/data/columns`` describes such rows'
columns first.
"""

import asyncio
import json
import shutil
import sys
import tempfile
import threading
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response

from ...data.dataset import DATASET_FILE, Dataset, SplitOut, open_dataset, write_dataset
from ...data.format import DataError, Example, _label, check_examples
from ...data.make import Made
from ...data.shapes import Wording, build, class_names, guess
from ...data.sources import SourceError, table_from_text
from ...errors import DecisionError
from ...types import parse_request
from ..jobs import Jobs
from ..security import is_loopback
from ..workspace import Conflict, NotFound, Workspace, WorkspaceError, check_name
from . import ApiError, field, json_body
from .datasets import describe_table
from .sets import reveal
from .settings import tilde

ERRORS_SHOWN = 20
SAMPLE = 5  # examples the inspector shows
PREVIEW_CHARS = 240
LINE_CHARS = 200  # a state cut to a line, in the example table
PAGE_LIMIT = 200  # the most examples a page holds
CACHED_FILES = 4  # splits kept read, for paging through them
SET_EXAMPLES = 10  # examples a set made from a dataset holds
UPLOAD_KINDS = ("csv", "jsonl")


class Read:
    """A split's examples as read, with each state as text (for the search)."""

    def __init__(self, dataset: Dataset, split: str, examples: list[Example]):
        self.dataset = dataset
        self.split = split
        self.examples = examples
        self.texts = [state_text(example.request.get("state")).casefold() for example in examples]


class ReadFiles:
    """The last few splits read, kept while they do not change on disk (the
    split file's and ``dataset.json``'s size and modification time), so
    paging through a large split does not read it again for every page."""

    def __init__(self, size: int = CACHED_FILES):
        self.size = size
        self.kept: OrderedDict[Path, tuple[tuple, Read]] = OrderedDict()
        self.lock = threading.Lock()

    def read(self, location: Path, split: str | None) -> Read:
        """A split of the dataset at ``location`` (``DecisionError`` when it
        does not read; ``NotFound`` for a split it does not have)."""
        dataset, _ = open_dataset(location)
        if split is None:
            split = dataset.with_role("train") or next(iter(dataset.splits))
        if split not in dataset.splits:
            raise NotFound(f"{dataset.name} has no split {split!r}")
        path = dataset.split_path(split)
        stamps = []
        for one in (path, location / DATASET_FILE) if dataset.folder else (path,):
            stat = one.stat()
            stamps.append((stat.st_size, stat.st_mtime_ns))
        stamp = tuple(stamps)
        with self.lock:
            kept = self.kept.get(path)
            if kept and kept[0] == stamp:
                self.kept.move_to_end(path)
                return kept[1]
        read = Read(dataset, split, dataset.read(split, lenient=True))
        with self.lock:
            self.kept[path] = (stamp, read)
            self.kept.move_to_end(path)
            while len(self.kept) > self.size:
                self.kept.popitem(last=False)
        return read


def data_router(files: Workspace, jobs: Jobs, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/data")
    cache = ReadFiles()

    def location(name: str) -> Path:
        try:
            return files.dataset_location(name)
        except NotFound:
            raise NotFound(f"no dataset named {name!r}") from None

    async def read(name: str, split: str | None) -> tuple[Path, Read]:
        """The dataset and its split read; 404 when there is none, 422 when it
        no longer reads (changed behind the Studio's back)."""
        path = location(name)
        try:
            return path, await asyncio.to_thread(cache.read, path, split)
        except DecisionError as error:
            raise ApiError(422, f"{name}: {error}", param="name") from None

    def readers(name: str) -> None:
        """409 while a job that has not ended reads dataset ``name``."""
        for job in jobs.jobs.values():
            if job.state != "ended" and name in (
                job.details.get("data"),
                job.details.get("report"),
            ):
                raise Conflict(f"{job.title} reads {name}: stop it or wait until it ends")

    @router.get("")
    async def list_data():
        """``{files, folder, display, upload_limit}``: each dataset as
        ``workspace.dataset_summary`` gives it; the ``data/`` folder
        (``display`` with the home folder as ``~``) and the most an upload's
        body can hold (a larger file is copied into the folder instead)."""
        folder = files.root / "data"
        return {
            "files": await asyncio.to_thread(files.list_data),
            "folder": str(folder),
            "display": tilde(str(folder)),
            "upload_limit": max_body_bytes,
        }

    @router.post("/columns")
    async def columns(request: Request):
        """``{text, format}`` (rows of an upload: csv or jsonl) -> its ``rows``,
        ``columns``, ``sample``, ``text_columns`` and ``label_columns`` with
        the guessed ``text`` and ``label``, as a Hub dataset's preview."""
        body = await json_body(request, max_body_bytes)
        text = field(body, "text", str)
        kind = upload_kind(body)
        try:
            table = await asyncio.to_thread(table_from_text, text, kind, "upload")
        except SourceError as error:
            raise ApiError(422, str(error), param="text") from None
        return await asyncio.to_thread(describe_table, table)

    @router.get("/{name}")
    async def describe_data(name: str, split: str | None = None):
        """One split of a dataset, described (see the module); 404 when there
        is none, 422 when it no longer reads."""
        path, found = await read(name, split)
        summary = await asyncio.to_thread(describe, found)
        return {"name": name, **where(path, found), **summary}

    @router.get("/{name}/examples")
    async def examples_page(
        name: str,
        split: str | None = None,
        offset: int = 0,
        limit: int = 50,
        question: str | None = None,
        label: str | None = None,
        column: str | None = None,
        value: str | None = None,
        search: str | None = None,
    ):
        """A page of a split's examples: ``{split, total, matched, offset,
        limit, examples}``, each example's ``index`` (its place in the file,
        from 0), ``id``, ``state`` (cut to a line), ``labels`` and ``values``.
        ``question`` with ``label`` keeps the examples with that label for
        that question, ``column`` with ``value`` those with that value;
        ``search`` the ones whose state holds it (any case; a JSON state as
        its JSON). ``matched`` counts what the filter and the search keep."""
        if offset < 0:
            raise ApiError(400, "offset is 0 or more", param="offset")
        if not 1 <= limit <= PAGE_LIMIT:
            raise ApiError(400, f"limit is 1 to {PAGE_LIMIT}", param="limit")
        if (question is None) != (label is None):
            raise ApiError(400, "a label filter needs both question and label", param="label")
        if (column is None) != (value is None):
            raise ApiError(400, "a value filter needs both column and value", param="value")
        _, found = await read(name, split)
        return await asyncio.to_thread(
            page, found, offset, limit, (question, label), (column, value), search
        )

    @router.get("/{name}/examples/{index}")
    async def example(name: str, index: int, split: str | None = None):
        """One example whole: ``index``, ``id``, ``state`` (as in the file),
        ``questions`` (the request's; none before the dataset has any),
        ``labels`` and ``values``; 404 past the end."""
        _, found = await read(name, split)
        if not 0 <= index < len(found.examples):
            raise NotFound(f"{name} has no example {index} in {found.split}")
        chosen = found.examples[index]
        return {
            "index": index,
            "id": chosen.id,
            "state": chosen.request.get("state"),
            "questions": chosen.request.get("questions") or {},
            "labels": chosen.labels,
            "values": chosen.values or {},
        }

    @router.put("/{name}/questions")
    async def save_questions(name: str, request: Request):
        """``{questions, answers}`` -> the split described again (as ``GET``).
        Checked as the CLI reads them: valid questions (as a request's), each
        answer naming a question and a value column, mapping that column's
        values to the question's answers (a value mapped or left out, not
        both). 400 naming what is wrong, 409 while a job reads the dataset."""
        body = await json_body(request, max_body_bytes)
        path = location(name)
        if not path.is_dir():
            raise ApiError(400, f"{name} is a plain file: its lines hold their own questions")
        readers(name)
        dataset, _ = await asyncio.to_thread(open_dataset, path)
        questions = body.get("questions", {})
        answers = body.get("answers", {})
        checked_questions, checked_answers = check_questions(dataset, questions, answers)
        try:
            await asyncio.to_thread(
                files.save_dataset_questions, name, checked_questions, checked_answers
            )
        except WorkspaceError as error:
            raise ApiError(error.status, error.message) from None
        return await describe_data(name)

    @router.post("/{name}/reveal", status_code=204)
    async def reveal_data(name: str, request: Request):
        """Show the dataset in Finder (only for a browser on this Mac)."""
        client = request.client.host if request.client else None
        if not is_loopback(client):
            raise ApiError(
                403, "Finder opens only for a browser on this Mac", kind="permission_error"
            )
        path = location(name)
        if sys.platform == "darwin":
            await asyncio.to_thread(reveal, path)
        return Response(status_code=204)

    @router.post("/{name}/set")
    async def to_set(name: str, split: str | None = None):
        """``{name, examples}``: a new question set holding the split's questions
        and its first examples, each named by its id and its labels; saved
        under the dataset's name or, when taken, the first free ``NAME-2``,
        .... 422 for a dataset without questions yet."""
        _, found = await read(name, split)
        if not found.examples:
            raise ApiError(422, f"{name} has no examples", param="name")
        if not found.examples[0].request.get("questions"):
            raise ApiError(422, f"{name} has no questions yet", param="name")
        body = set_from(found.examples[:SET_EXAMPLES])
        saved, _ = await asyncio.to_thread(files.save_new_set, name, body)
        return {"name": saved, "examples": len(body["examples"])}

    @router.delete("/{name}", status_code=204)
    async def delete_data(name: str):
        """204 once deleted; 404 when there is none; 409 while a job that has
        not ended (a fit or an evaluation) reads it."""
        check_name(name, "a dataset's name")
        readers(name)
        await asyncio.to_thread(files.delete_data, name)

    @router.post("")
    async def upload(request: Request):
        """``{name, text}`` -> the dataset described (as ``GET``): training JSON
        lines written to ``data/NAME.jsonl``; with ``format`` (csv or jsonl),
        ``text_column`` and ``label_column``, rows of columns kept as the
        dataset folder ``data/NAME/`` (one train split, no questions yet).
        400 with ``errors`` (``line``, ``message``, ``param``) for training
        lines that do not check, 409 for a name taken, 413 above the body
        limit, 422 for rows that cannot be read."""
        body = await json_body(request, max_body_bytes)
        name = check_name(field(body, "name", str), "a dataset's name")
        text = field(body, "text", str)
        if body.get("format") is not None:
            kind = upload_kind(body)
            columns = {
                "text": field(body, "text_column", str),
                "label": field(body, "label_column", str),
            }
            if files.data_taken(name):
                raise Conflict(f"a dataset named {name} exists already")
            await asyncio.to_thread(save_rows, files, name, text, kind, columns)
            return await describe_data(name)
        examples, errors = await asyncio.to_thread(check_examples, text.splitlines(), ERRORS_SHOWN)
        if errors:
            raise BadData(errors)
        await asyncio.to_thread(files.save_data, name, text.encode("utf-8"))
        return await describe_data(name)

    return router


def upload_kind(body: dict) -> str:
    kind = field(body, "format", str)
    if kind not in UPLOAD_KINDS:
        raise ApiError(400, f"format is one of {', '.join(UPLOAD_KINDS)}", param="format")
    return kind


def save_rows(files: Workspace, name: str, text: str, kind: str, columns: dict) -> Path:
    """Rows of an upload kept as a dataset folder without questions: the
    state's column and the label's raw values (one train split)."""
    try:
        table = table_from_text(text, kind, name)
        mapping = guess(table, "text", columns)
        built = build(table, mapping, Wording(id=name), list(range(table.length)), raw=True)
    except SourceError as error:
        raise ApiError(422, str(error), param="text") from None
    if not built.examples:
        raise ApiError(422, "no row has a label value", param="label_column")
    made = Made(built.examples, table, mapping, built.skipped, columns["label"])
    made.classes = class_names(table.columns[columns["label"]], table, None)
    folder = files.dataset_folder(name)
    dataset = Dataset(
        folder,
        columns=made.columns(),
        source={"source": "upload", "format": kind, "columns": dict(columns)},
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    meta = {"role": "train", "rows": table.length, "examples": len(built.examples)}
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix=".", suffix=".partial") as scratch:
        # Written beside data/ first, then moved in under the lock.
        outs = [SplitOut("train", made.examples, meta)]
        staged = write_dataset(Path(scratch) / name, dataset, outs)
        with files.lock:
            if files.data_taken(name):
                raise Conflict(f"a dataset named {name} exists already")
            shutil.move(str(staged), folder)
    files.changed("data", name, "saved")
    return folder


def check_questions(dataset: Dataset, questions: Any, answers: Any) -> tuple[dict, dict]:
    """``questions`` and ``answers`` as ``dataset.json`` keeps them; 400 naming
    what is wrong."""
    if not isinstance(questions, dict):
        raise ApiError(400, "questions is an object from id to question", param="questions")
    if not isinstance(answers, dict):
        raise ApiError(400, "answers is an object from question id to its map", param="answers")
    parsed = None
    if questions:
        try:
            parsed = parse_request({"state": "", "questions": questions})
        except DecisionError as error:
            raise ApiError(400, error.message, param=f"questions.{error.param or ''}") from None
    result: dict[str, dict] = {}
    for question_id, answer in answers.items():
        param = f"answers.{question_id}"
        if parsed is None or question_id not in parsed.questions:
            raise ApiError(400, f"no question {question_id!r} to answer", param=param)
        if not isinstance(answer, dict):
            raise ApiError(400, "an answer is {column, map, left_out}", param=param)
        column = answer.get("column")
        known = None
        if isinstance(column, str):
            known = (dataset.columns.get(column) or {}).get("values")
        if known is None:
            listed = ", ".join(dataset.columns) or "none"
            raise ApiError(400, f"no value column {column!r} (columns: {listed})", param=param)
        mapped = answer.get("map") or {}
        left_out = answer.get("left_out") or []
        if not isinstance(mapped, dict) or not isinstance(left_out, list):
            raise ApiError(400, "map is an object, left_out a list", param=param)
        checked: dict[str, Any] = {}
        for value, target in mapped.items():
            if value not in known:
                raise ApiError(400, f"{column} has no value {value!r}", param=f"{param}.map")
            try:
                _label(parsed.questions[question_id], target, f"{param}.map.{value}", 0)
            except DataError as error:
                raise ApiError(400, error.message, param=error.param) from None
            checked[value] = target
        for value in left_out:
            if value not in known:
                raise ApiError(400, f"{column} has no value {value!r}", param=f"{param}.left_out")
            if value in checked:
                raise ApiError(400, f"{value!r} is mapped and left out", param=f"{param}.left_out")
        result[question_id] = {"column": column, "map": checked, "left_out": list(left_out)}
    return questions, result


class BadData(ApiError):
    """A file that does not check: the first error as the message, all of them
    (up to ``ERRORS_SHOWN``) in ``errors``."""

    def __init__(self, errors: list[DecisionError]):
        listed = [
            {
                "line": error.line if isinstance(error, DataError) else None,
                "message": error.message,
                "param": error.param,
            }
            for error in errors
        ]
        super().__init__(400, str(errors[0]), param="text", extra={"errors": listed})


def describe(found: Read) -> dict[str, Any]:
    """What the page shows about a split: ``split``, ``splits`` (each
    ``name``, ``role``, ``examples``, the source ``split``, ``sample``,
    ``part``), ``examples`` (how many in this split), ``questions`` (per
    question id in first use: ``type``, ``instructions``, ``labelled``,
    ``labels`` as ``[label, count]`` most common first, ``options`` the
    question's own, ``answered_by`` the column whose values answer it),
    ``values`` (per value column its values here as ``[value, count]`` in
    the column's order), ``columns`` (the dataset's, with names), ``answers``,
    ``missing``, ``source``, ``created`` and ``sample`` (the first few
    examples: ``id``, a preview of the state, the labels, the values)."""
    dataset, examples = found.dataset, found.examples
    questions: dict[str, dict[str, Any]] = {}
    counts: dict[str, Counter] = {}
    asked = [dataset.questions] if dataset.questions else []
    asked += [
        own
        for own in (e.request.get("questions") for e in examples)
        if own and own is not dataset.questions
    ]
    for one in asked:
        for question_id, question in one.items():
            if question_id in questions:
                continue
            criteria = question.get("criteria")
            entry = {
                "id": question_id,
                "type": question.get("type"),
                "instructions": question.get("instructions"),
                "options": list(criteria) if isinstance(criteria, dict | list) else None,
                "answered_by": (dataset.answers.get(question_id) or {}).get("column"),
            }
            if question.get("type") == "score" and isinstance(criteria, list):
                entry["options"] = [str(n) for n in range(len(criteria))]
            if question.get("type") == "noul":
                entry["options"] = ["true", "false"]
            questions[question_id] = entry
            counts[question_id] = Counter()
    for example in examples:
        for question_id, label in example.labels.items():
            counts.setdefault(question_id, Counter())[label] += 1
    for question_id, question in questions.items():
        question["labelled"] = sum(counts[question_id].values())
        question["labels"] = [[label, n] for label, n in counts[question_id].most_common()]
    values: dict[str, list] = {}
    for column, body in dataset.columns.items():
        here = Counter((e.values or {}).get(column) for e in examples)
        order = list((body.get("values") or {}).keys())
        order += sorted(v for v in here if v is not None and v not in order)
        values[column] = [[value, here.get(value, 0)] for value in order]
    return {
        "split": found.split,
        "kind": "folder" if dataset.folder else "file",
        "splits": [
            {
                "name": name,
                "role": meta.get("role"),
                "examples": meta.get("examples"),
                "split": meta.get("split"),
                "sample": meta.get("sample"),
                "part": meta.get("part"),
            }
            for name, meta in dataset.splits.items()
        ],
        "examples": len(examples),
        "questions": list(questions.values()),
        "values": values,
        "columns": dataset.columns,
        "answers": dataset.answers,
        "missing": dataset.missing(),
        "source": dataset.source,
        "created": dataset.created,
        "sample": [
            {
                "id": example.id,
                "state": preview(example.request.get("state")),
                "labels": example.labels,
                "values": example.values or {},
            }
            for example in examples[:SAMPLE]
        ],
    }


def set_from(examples: list[Example]) -> dict[str, Any]:
    """A question set from labelled examples, laid out as the Studio saves one:
    the questions as each was first asked, every example by name (its id and
    its labels, so answers can be held against them) with its state and its
    images when they are data URLs, the first also at the top level."""
    questions: dict[str, Any] = {}
    listed = []
    for example in examples:
        asked = example.request["questions"]
        for question_id, question in asked.items():
            questions.setdefault(question_id, question)
        # Ids are unique in a file, so the names are too.
        labels = ", ".join(label_word(asked[q], label) for q, label in example.labels.items())
        named = f"{example.id} · {labels}" if labels else example.id
        item: dict[str, Any] = {"name": named, "state": example.request.get("state", "")}
        images = example.request.get("images")
        if images and all(isinstance(i, str) and i.startswith("data:image/") for i in images):
            item["images"] = images
        listed.append(item)
    body: dict[str, Any] = {"state": listed[0]["state"], "questions": questions}
    if "images" in listed[0]:
        body["images"] = listed[0]["images"]
    body["examples"] = listed
    return body


def label_word(question: dict[str, Any], label: str) -> str:
    """A label as the Playground's answers show it: an option id, yes or no,
    a score's level counted from 1."""
    kind = question.get("type")
    if kind == "noul":
        return "yes" if label == "true" else "no"
    if kind == "score":
        return f"level {int(label) + 1}"
    return label


def where(path: Path, found: Read) -> dict[str, Any]:
    """The dataset's ``path`` (its folder or file), the same with the home folder
    as ``~`` (``display``), the split's file (``file``) and its size
    (``bytes``)."""
    file = found.dataset.split_path(found.split)
    return {
        "path": str(path),
        "display": tilde(str(path)),
        "file": str(file),
        "bytes": file.stat().st_size,
    }


def state_text(state: Any) -> str:
    """A state as text: a JSON state as its JSON."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def preview(state: Any) -> str:
    """The start of a state as text (a JSON state as JSON)."""
    text = state_text(state)
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def one_line(state: Any) -> str:
    """A state cut to a line: its whitespace runs as single spaces, at most
    ``LINE_CHARS`` characters."""
    text = " ".join(state_text(state).split())
    return text if len(text) <= LINE_CHARS else text[: LINE_CHARS - 1] + "…"


def page(
    found: Read,
    offset: int,
    limit: int,
    by_label: tuple[str | None, str | None],
    by_value: tuple[str | None, str | None],
    search: str | None,
) -> dict[str, Any]:
    """The examples ``examples_page`` answers with."""
    needle = search.strip().casefold() if search else ""
    question, label = by_label
    column, value = by_value
    kept = []
    for index, chosen in enumerate(found.examples):
        if question is not None and chosen.labels.get(question) != label:
            continue
        if column is not None and (chosen.values or {}).get(column) != value:
            continue
        if needle and needle not in found.texts[index]:
            continue
        kept.append(index)
    shown = kept[offset : offset + limit]
    return {
        "split": found.split,
        "total": len(found.examples),
        "matched": len(kept),
        "offset": offset,
        "limit": limit,
        "examples": [
            {
                "index": index,
                "id": found.examples[index].id,
                "state": one_line(found.examples[index].request.get("state")),
                "labels": found.examples[index].labels,
                "values": found.examples[index].values or {},
            }
            for index in shown
        ],
    }
