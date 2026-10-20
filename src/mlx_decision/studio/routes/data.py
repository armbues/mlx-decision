"""Datasets: the JSON-lines files of labelled examples in the workspace's
``data/`` (DS-1 to DS-4, TRS-1).

``POST /studio/api/data`` ``{name, text}`` uploads a file the page read: it
is checked as ``training fit`` checks it (``read_examples``' rules), and a
bad file is refused with up to 20 of its errors, each naming its line. A
good one is written under ``name`` (a workspace name; an existing file is
not replaced). ``GET /studio/api/data`` lists the files; ``GET
/studio/api/data/{name}`` describes one: its questions with their labels
and counts, a few examples and its path. ``GET
/studio/api/data/{name}/examples`` reads a page of its examples (``offset``,
``limit``, a label filter, a search in the states), each state cut to a
line; ``GET /studio/api/data/{name}/examples/{index}`` one example whole.
``POST /studio/api/data/{name}/reveal`` shows the file in Finder (a browser
on this Mac only). ``DELETE /studio/api/data/{name}`` deletes one, unless a
fit or an evaluation that is not over reads it (409).
"""

import asyncio
import json
import sys
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response

from ...data.format import DataError, Example, check_examples, read_examples
from ...errors import DecisionError
from ..jobs import Jobs
from ..security import is_loopback
from ..workspace import Conflict, NotFound, Workspace, check_name
from . import ApiError, field, json_body
from .sets import reveal
from .settings import tilde

ERRORS_SHOWN = 20
SAMPLE = 5  # examples the inspector shows
PREVIEW_CHARS = 240
LINE_CHARS = 200  # a state cut to a line, in the example table
PAGE_LIMIT = 200  # the most examples a page holds
CACHED_FILES = 4  # files kept read, for paging through them


class Read:
    """A file's examples as read, with each state as text (for the search)."""

    def __init__(self, examples: list[Example]):
        self.examples = examples
        self.texts = [state_text(example.request.get("state")).casefold() for example in examples]


class ReadFiles:
    """The last few files read, kept while they do not change on disk (by
    size and modification time), so paging through a large file does not
    read it again for every page."""

    def __init__(self, size: int = CACHED_FILES):
        self.size = size
        self.kept: OrderedDict[Path, tuple[tuple[int, int], Read]] = OrderedDict()
        self.lock = threading.Lock()

    def read(self, path: Path) -> Read:
        """The file's examples (``DecisionError`` when it does not read)."""
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        with self.lock:
            kept = self.kept.get(path)
            if kept and kept[0] == stamp:
                self.kept.move_to_end(path)
                return kept[1]
        read = Read(read_examples(path))
        with self.lock:
            self.kept[path] = (stamp, read)
            self.kept.move_to_end(path)
            while len(self.kept) > self.size:
                self.kept.popitem(last=False)
        return read


def data_router(files: Workspace, jobs: Jobs, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/data")
    cache = ReadFiles()

    def existing(name: str) -> Path:
        path = files.data_path(name)
        if not path.is_file():
            raise NotFound(f"no data file named {name!r}")
        return path

    async def read(name: str) -> tuple[Path, Read]:
        """The file and its examples; 404 when there is none, 422 when it no
        longer reads (changed behind the Studio's back)."""
        path = existing(name)
        try:
            return path, await asyncio.to_thread(cache.read, path)
        except DecisionError as error:
            raise ApiError(422, f"{name}: {error}", param="name") from None

    @router.get("")
    async def list_data():
        """``{files, folder, display, upload_limit}``: each file's ``name``,
        ``bytes`` and ``modified``; the ``data/`` folder (``display`` with the
        home folder as ``~``) and the most an upload's body can hold (a larger
        file is copied into the folder instead)."""
        folder = files.root / "data"
        return {
            "files": await asyncio.to_thread(files.list_data),
            "folder": str(folder),
            "display": tilde(str(folder)),
            "upload_limit": max_body_bytes,
        }

    @router.get("/{name}")
    async def describe_data(name: str):
        """``describe``'s summary of one file; 404 when there is none, 422 when
        it no longer reads (changed behind the Studio's back)."""
        path, found = await read(name)
        summary = await asyncio.to_thread(describe, found.examples)
        return {"name": name, **where(path), **summary}

    @router.get("/{name}/examples")
    async def examples_page(
        name: str,
        offset: int = 0,
        limit: int = 50,
        question: str | None = None,
        label: str | None = None,
        search: str | None = None,
    ):
        """A page of the file's examples: ``{total, matched, offset, limit,
        examples}``, each example's ``index`` (its place in the file, from
        0), ``id``, ``state`` (cut to a line) and ``labels``. ``question``
        with ``label`` keeps the examples with that label for that question;
        ``search`` the ones whose state holds it (any case; a JSON state as
        its JSON). ``matched`` counts what the filter and the search keep."""
        if offset < 0:
            raise ApiError(400, "offset is 0 or more", param="offset")
        if not 1 <= limit <= PAGE_LIMIT:
            raise ApiError(400, f"limit is 1 to {PAGE_LIMIT}", param="limit")
        if (question is None) != (label is None):
            raise ApiError(400, "a label filter needs both question and label", param="label")
        _, found = await read(name)
        return await asyncio.to_thread(page, found, offset, limit, question, label, search)

    @router.get("/{name}/examples/{index}")
    async def example(name: str, index: int):
        """One example whole: ``index``, ``id``, ``state`` (as in the file),
        ``questions`` (the request's) and ``labels``; 404 past the end."""
        _, found = await read(name)
        if not 0 <= index < len(found.examples):
            raise NotFound(f"{name} has no example {index}")
        chosen = found.examples[index]
        return {
            "index": index,
            "id": chosen.id,
            "state": chosen.request.get("state"),
            "questions": chosen.request["questions"],
            "labels": chosen.labels,
        }

    @router.post("/{name}/reveal", status_code=204)
    async def reveal_data(name: str, request: Request):
        """Show the file in Finder (only for a browser on this Mac)."""
        client = request.client.host if request.client else None
        if not is_loopback(client):
            raise ApiError(
                403, "Finder opens only for a browser on this Mac", kind="permission_error"
            )
        path = existing(name)
        if sys.platform == "darwin":
            await asyncio.to_thread(reveal, path)
        return Response(status_code=204)

    @router.delete("/{name}", status_code=204)
    async def delete_data(name: str):
        """204 once deleted; 404 when there is none; 409 while a job that has
        not ended (a fit or an evaluation) reads it."""
        check_name(name, "a data file's name")
        for job in jobs.jobs.values():
            if job.state != "ended" and job.details.get("data") == name:
                raise Conflict(f"{job.title} reads {name}: stop it or wait until it ends")
        await asyncio.to_thread(files.delete_data, name)

    @router.post("")
    async def upload(request: Request):
        """``{name, text}`` -> the file's summary (as ``GET``), written to
        ``data/NAME.jsonl``. 400 with ``errors`` (``line``, ``message``,
        ``param``) for a file that does not check, 409 for a name taken, 413
        above the body limit."""
        body = await json_body(request, max_body_bytes)
        name = check_name(field(body, "name", str), "a data file's name")
        text = field(body, "text", str)
        examples, errors = await asyncio.to_thread(check_examples, text.splitlines(), ERRORS_SHOWN)
        if errors:
            raise BadData(errors)
        path = await asyncio.to_thread(files.save_data, name, text.encode("utf-8"))
        return {"name": name, **where(path), **describe(examples)}

    return router


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


def describe(examples: list[Example]) -> dict[str, Any]:
    """What the page shows about a file: ``examples`` (how many), ``questions``
    (per question id in first use: ``type``, ``instructions``, ``labelled``,
    ``labels`` as ``[label, count]`` most common first, ``options`` the
    question's own) and ``sample`` (the first few examples: ``id``, a preview
    of the state, the labels)."""
    questions: dict[str, dict[str, Any]] = {}
    counts: dict[str, Counter] = {}
    for example in examples:
        for question_id, question in example.request["questions"].items():
            if question_id not in questions:
                criteria = question.get("criteria")
                questions[question_id] = {
                    "id": question_id,
                    "type": question.get("type"),
                    "instructions": question.get("instructions"),
                    "options": list(criteria) if isinstance(criteria, dict | list) else None,
                }
                if question.get("type") == "score" and isinstance(criteria, list):
                    questions[question_id]["options"] = [str(n) for n in range(len(criteria))]
                if question.get("type") == "noul":
                    questions[question_id]["options"] = ["true", "false"]
                counts[question_id] = Counter()
        for question_id, label in example.labels.items():
            counts[question_id][label] += 1
    for question_id, question in questions.items():
        question["labelled"] = sum(counts[question_id].values())
        question["labels"] = [[label, n] for label, n in counts[question_id].most_common()]
    return {
        "examples": len(examples),
        "questions": list(questions.values()),
        "sample": [
            {
                "id": example.id,
                "state": preview(example.request.get("state")),
                "labels": example.labels,
            }
            for example in examples[:SAMPLE]
        ],
    }


def where(path: Path) -> dict[str, Any]:
    """A file's ``path``, the same with the home folder as ``~`` (``display``)
    and its size (``bytes``)."""
    return {"path": str(path), "display": tilde(str(path)), "bytes": path.stat().st_size}


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
    question: str | None,
    label: str | None,
    search: str | None,
) -> dict[str, Any]:
    """The examples ``examples_page`` answers with."""
    needle = search.strip().casefold() if search else ""
    kept = []
    for index, chosen in enumerate(found.examples):
        if question is not None and chosen.labels.get(question) != label:
            continue
        if needle and needle not in found.texts[index]:
            continue
        kept.append(index)
    shown = kept[offset : offset + limit]
    return {
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
            }
            for index in shown
        ],
    }
