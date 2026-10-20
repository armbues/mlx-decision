"""Training data: the JSON-lines files in the workspace's ``data/`` (TRS-1, TRS-5).

``POST /studio/api/data`` ``{name, text}`` uploads a file the page read: it
is checked as ``training fit`` checks it (``read_examples``' rules), and a
bad file is refused with up to 20 of its errors, each naming its line. A
good one is written under ``name`` (a workspace name; an existing file is
not replaced). ``GET /studio/api/data`` lists the files; ``GET
/studio/api/data/{name}`` describes one: its questions with their labels
and counts, a few examples and its path.
"""

import asyncio
import json
from collections import Counter
from typing import Any

from fastapi import APIRouter, Request

from ...data.format import DataError, Example, check_examples, read_examples
from ...errors import DecisionError
from ..workspace import NotFound, Workspace, check_name
from . import ApiError, field, json_body

ERRORS_SHOWN = 20
SAMPLE = 5  # examples the inspector shows
PREVIEW_CHARS = 240


def data_router(files: Workspace, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/data")

    @router.get("")
    async def list_data():
        """``{files}``: each file's ``name``, ``bytes`` and ``modified``."""
        return {"files": await asyncio.to_thread(files.list_data)}

    @router.get("/{name}")
    async def describe_data(name: str):
        """``describe``'s summary of one file; 404 when there is none, 422 when
        it no longer reads (changed behind the Studio's back)."""
        path = files.data_path(name)
        if not path.is_file():
            raise NotFound(f"no data file named {name!r}")
        try:
            examples = await asyncio.to_thread(read_examples, path)
        except DecisionError as error:
            raise ApiError(422, f"{name}: {error}", param="name") from None
        return {"name": name, "path": str(path), "bytes": path.stat().st_size, **describe(examples)}

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
        return {"name": name, "path": str(path), "bytes": path.stat().st_size, **describe(examples)}

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


def preview(state: Any) -> str:
    """The start of a state as text (a JSON state as JSON)."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"
