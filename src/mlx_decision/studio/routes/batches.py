"""Batches: start one on a dataset's split, follow it, read its results.

``POST /studio/api/batches`` makes the batch's folder and starts it (201);
its progress comes over the stream (``batch`` on each change of state,
``batch_progress`` as rows are answered, thinned). ``GET .../{name}`` gives
the summary and, per question, the answers' counts and, with labels, the
metrics, confusion matrix and reliability bins; ``.../rows`` the table's
pages, filtered here; ``.../export`` the results as CSV or JSON lines;
``.../dataset`` saves the answers as a labelled dataset in ``data/``.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..batch_results import PAGE, UNSURE, Results
from ..batches import Batch, Batches
from ..workspace import Workspace, check_name
from . import ApiError, field, json_body

# The longest a batch's or dataset's name may be when given.
MAX_NAME = 80


def batches_router(batches: Batches, files: Workspace, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/batches")
    results = Results(batches.root)

    @router.get("")
    async def listed():
        """``{batches}``: every batch's summary, the running one first."""
        return {"batches": await asyncio.to_thread(batches.list)}

    @router.post("", status_code=201)
    async def start(request: Request):
        """``{dataset, split, model, questions, sample, seed, name}`` -> ``{batch}``:
        made and started (``questions`` a set's name, or null for the
        dataset's; ``sample`` a number of rows, or null for all)."""
        body = await json_body(request, max_body_bytes)
        dataset = field(body, "dataset", str)
        split = field(body, "split", str, required=False)
        model = field(body, "model", str)
        questions = field(body, "questions", str, required=False)
        sample = field(body, "sample", int, required=False)
        seed = field(body, "seed", int, required=False)
        name = field(body, "name", str, required=False)
        if name is not None:
            check_name(name, "a batch's name")
        batches.check_free()  # before the rows are read
        folder = await asyncio.to_thread(
            batches.prepare, dataset, split, model, questions, sample, seed or 0, name
        )
        try:
            return {"batch": await batches.start(folder.name)}
        except BaseException:
            await asyncio.to_thread(batches.delete, folder.name)  # the start was refused
            raise

    @router.get("/{name}")
    async def one(name: str, unsure: float | None = None):
        """``{batch, results}``: the summary and ``Results.analyse``."""
        info = await asyncio.to_thread(batches.get, name)
        analysed = await asyncio.to_thread(
            results.analyse, name, UNSURE if unsure is None else unsure
        )
        return {"batch": info, "results": analysed}

    @router.get("/{name}/rows")
    async def rows(
        name: str,
        offset: int = 0,
        limit: int = PAGE,
        question: str | None = None,
        answer: str | None = None,
        wrong: bool = False,
        unsure: float | None = None,
        search: str | None = None,
        errors: bool = False,
        row: int | None = None,
    ):
        """A page of the table (``Results.page``), or with ``row`` that row whole."""
        await asyncio.to_thread(batches.get, name)
        if row is not None:
            return {"row": await asyncio.to_thread(results.row, name, row)}
        return await asyncio.to_thread(
            results.page, name, offset, limit, question, answer, wrong, unsure, search, errors
        )

    @router.get("/{name}/export")
    async def export(name: str, format: str = "csv"):
        """The answered rows as ``csv`` or ``jsonl``, as a download."""
        await asyncio.to_thread(batches.get, name)
        if format == "csv":
            text, kind = await asyncio.to_thread(results.export_csv, name), "text/csv"
        elif format == "jsonl":
            text, kind = await asyncio.to_thread(results.export_jsonl, name), "application/x-ndjson"
        else:
            raise ApiError(400, "format is csv or jsonl", param="format")
        headers = {"Content-Disposition": f'attachment; filename="{name}.{format}"'}
        return Response(text.encode("utf-8"), media_type=f"{kind}; charset=utf-8", headers=headers)

    @router.post("/{name}/stop")
    async def stop(name: str):
        return {"batch": await batches.stop(name)}

    @router.post("/{name}/run")
    async def run_rest(name: str):
        """Run the Rest: the rows not answered yet."""
        return {"batch": await batches.start(name)}

    @router.post("/{name}/again", status_code=201)
    async def again(name: str):
        """Run Again: a new batch of the same rows, questions and model, started."""
        batches.check_free()
        folder = await asyncio.to_thread(batches.again, name)
        try:
            return {"batch": await batches.start(folder.name)}
        except BaseException:
            await asyncio.to_thread(batches.delete, folder.name)
            raise

    @router.delete("/{name}")
    async def delete(name: str):
        await asyncio.to_thread(batches.delete, name)
        results.forget(name)
        return {"deleted": name}

    @router.post("/{name}/dataset", status_code=201)
    async def save_dataset(name: str, request: Request):
        """``{name, threshold}`` -> ``{dataset: {name, examples, left_out, split}}``:
        the answers saved as a dataset in ``data/``."""
        body = await json_body(request, max_body_bytes)
        target = field(body, "name", str)
        threshold = field(body, "threshold", (int, float), required=False)
        if threshold is not None and not 0 <= threshold <= 1:
            raise ApiError(400, "threshold is between 0 and 1", param="threshold")
        if batches.running is not None and batches.running.name == name:
            raise ApiError(409, f"{name} is running: stop it or wait until it is done")
        await asyncio.to_thread(batches.get, name)
        if await asyncio.to_thread(files.data_taken, target):
            raise ApiError(409, f"a dataset named {target} exists already", param="name")
        folder = files.dataset_folder(target)
        saved = await asyncio.to_thread(results.save_dataset, name, folder, threshold)
        files.changed("data", target, "saved")
        return {"dataset": saved}

    return router


class BatchMessages:
    """A batch's changes as stream messages, for ``Batches.on_change``: ``batch``
    with the summary on each change of state, ``batch_progress`` (``name``,
    ``answered``, ``total``, ``errors``, ``seconds_left``) at most once per
    ``interval``, the last one held until its turn. Called on the loop."""

    def __init__(self, publish: Callable[[str, dict[str, Any]], Any], interval: float = 0.25):
        self.publish = publish
        self.interval = interval
        self.sent = float("-inf")
        self.timer: asyncio.TimerHandle | None = None

    def __call__(self, batch: Batch, progress: bool, summary: dict[str, Any]) -> None:
        if not progress:
            self._cancel()
            self.publish("batch", {"batch": summary})
            return
        message = {
            "name": batch.name,
            "answered": batch.info["answered"],
            "total": batch.info["total"],
            "errors": batch.info["errors"],
            "seconds_left": batch.seconds_left(),
        }
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now - self.sent < self.interval:
            self._cancel()
            self.timer = loop.call_later(self.sent + self.interval - now, self._send, message)
            return
        self._send(message)

    def _send(self, message: dict[str, Any]) -> None:
        self.timer = None
        self.sent = asyncio.get_running_loop().time()
        self.publish("batch_progress", message)

    def _cancel(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
