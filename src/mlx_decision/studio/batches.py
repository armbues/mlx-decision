"""Batches: a dataset's rows answered by one model, in the Studio's process.

A batch reads a split of a dataset (all its rows or a seeded sample), puts
the questions on every row (the dataset's own, or a question set's) and
answers the rows one at a time on the shared worker, each the call the
Playground's run makes. The worker takes the Playground's and the API's
requests between two rows, so they wait at most one row.

Its folder, ``batches/NAME/``, stands alone once it is made, whatever
happens to the dataset::

    batch.json      what it runs and how far it got (``summary``)
    questions.json  {"questions": ...}: the questions as asked
    rows.jsonl      the rows to answer: row, id, state, labels (and the
                    row's own questions where they differ from the shared)
    results.jsonl   one line per row answered, in order: row, response or
                    error, latency_ms, input_tokens, truncated

``mlx-decision run -m MODEL -q questions.json --states rows.jsonl --json``
answers the same rows (a request ignores the fields it does not know).
A row's labels are the dataset's for a question asked as the dataset asks
it (the same id, type and options).

While it runs, its model is pinned in the pool: a request whose model
would need its memory is answered busy instead of the two being swapped
row by row, and Unload of it is refused. A GPU job and a batch refuse each
other (``refuse``, the jobs' ``blocked``).

Stop is read between rows: the row on the worker is answered and kept.
Run the Rest answers the rows ``results.jsonl`` does not have yet; a torn
last line (the Studio killed while writing it) is dropped first.

Use it from the event loop; file work runs in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..data.dataset import open_dataset
from ..errors import DecisionError
from ..gpu import Busy, Gpu
from ..memory import ModelTooLargeError
from ..server import answer
from .workspace import NotFound, Workspace, WorkspaceError, check_name, encode, line

logger = logging.getLogger(__name__)

BATCH_FILE = "batch.json"
QUESTIONS_FILE = "questions.json"
ROWS_FILE = "rows.jsonl"
RESULTS_FILE = "results.jsonl"
FOLDER = "batches"

# The rows a batch may ask (a larger split is sampled).
MAX_ROWS = 1_000_000
# The rows the time left is measured on.
TIMING_WINDOW = 50

# Fields of a dataset line that are not part of the request.
NOT_ASKED = ("id", "labels", "values", "questions", "model")


class BatchError(WorkspaceError):
    """A batch that cannot start or be changed; ``status`` is the API's."""

    def __init__(
        self, message: str, status: int = 400, param: str | None = None, code: str | None = None
    ):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def same_question(asked: Any, labelled: Any) -> bool:
    """Whether a question is asked as the dataset's labels answer it: the same
    type and the same options (a choice's ids, a score's number of levels)."""
    if not isinstance(asked, dict) or not isinstance(labelled, dict):
        return False
    kind = asked.get("type", "noul")
    if kind != labelled.get("type", "noul"):
        return False
    if kind == "choice":
        a, b = asked.get("criteria"), labelled.get("criteria")
        return isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b)
    if kind == "score":
        a, b = asked.get("criteria"), labelled.get("criteria")
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
    return True


def default_name(dataset: str, model: str) -> str:
    """``DATASET-MODEL`` made a workspace name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{dataset}-{model}").strip("-.") or "batch"
    return text[:70]


@dataclass
class Batch:
    """A batch as it runs: its summary (``batch.json``) and what the loop needs."""

    folder: Path
    info: dict[str, Any]
    stop_requested: bool = False
    task: asyncio.Task | None = None
    times: deque = field(default_factory=lambda: deque(maxlen=TIMING_WINDOW))

    @property
    def name(self) -> str:
        return self.info["name"]

    def seconds_left(self) -> float | None:
        if not self.times:
            return None
        left = self.info["total"] - self.info["answered"]
        return round(left * sum(self.times) / len(self.times), 1)


class Batches:
    """The workspace's batches and the one that runs.

    ``models`` is the Studio's ``Models`` (``check`` and ``fingerprint``);
    ``record`` (optional) records an answer's speed (blocks: called off the
    loop); ``on_change(batch, progress)`` is called on the loop when a batch
    changes state (``progress`` False) or answered a row (True)."""

    def __init__(
        self,
        gpu: Gpu,
        files: Workspace,
        models,
        record: Callable[[Any], None] | None = None,
    ):
        self.gpu = gpu
        self.files = files
        self.models = models
        self.record = record
        self.on_change: Callable[[Batch, bool], None] | None = None
        self.running: Batch | None = None
        # Called before a batch starts, raising ``BatchError`` to refuse it (the jobs' check).
        self.refuse: Callable[[], None] | None = None

    @property
    def root(self) -> Path:
        return self.files.root / FOLDER

    def folder(self, name: str) -> Path:
        return self.files.path(FOLDER, name, what="a batch's name")

    # Reading

    def open(self) -> None:
        """After a restart: a batch left ``running`` is ``stopped`` (blocks)."""
        self.root.mkdir(exist_ok=True)
        for folder in sorted(self.root.iterdir()):
            if folder.name.startswith(".") or not folder.is_dir():
                continue
            for leftover in folder.glob(".*.tmp"):
                leftover.unlink(missing_ok=True)
            info = self._read_info(folder)
            if info is not None and info.get("state") == "running":
                answered, errors = count_results(folder / RESULTS_FILE)
                info["state"] = "stopped"
                info["message"] = "the Studio stopped while it ran"
                info["answered"], info["errors"] = answered, errors
                if answered >= info.get("total", 0):
                    info["state"], info["message"] = "done", None
                if info.get("passes"):
                    info["passes"][-1].setdefault("ended", None)
                self.files.write(folder / BATCH_FILE, encode(info))

    def list(self) -> list[dict[str, Any]]:
        """Every batch's summary, the running one first, then newest first (blocks)."""
        found = []
        if self.root.is_dir():
            for folder in self.root.iterdir():
                if folder.name.startswith(".") or not folder.is_dir():
                    continue
                try:
                    check_name(folder.name, "a batch's name")
                except WorkspaceError:
                    continue  # not a workspace name (a copy made in Finder): not a batch here
                info = self.summary(folder.name)
                if info is not None:
                    found.append(info)
        found.sort(key=lambda info: info.get("created") or "", reverse=True)
        found.sort(key=lambda info: info.get("state") != "running")
        return found

    def summary(self, name: str) -> dict[str, Any] | None:
        """``batch.json`` of ``name`` (the running batch's as it is now), None if
        there is none (blocks)."""
        if self.running is not None and self.running.name == name:
            return self.live(self.running)
        return self._read_info(self.folder(name))

    def get(self, name: str) -> dict[str, Any]:
        info = self.summary(name)
        if info is None:
            raise NotFound(f"no batch named {name!r}")
        return info

    def live(self, batch: Batch) -> dict[str, Any]:
        return {
            **batch.info,
            "seconds_left": batch.seconds_left(),
            "stop_requested": batch.stop_requested,
        }

    @staticmethod
    def _read_info(folder: Path) -> dict[str, Any] | None:
        """``batch.json``, named after its folder (a copied folder is its own batch)."""
        try:
            info = json.loads((folder / BATCH_FILE).read_bytes())
        except (OSError, ValueError):
            return None
        if not isinstance(info, dict) or not isinstance(info.get("name"), str):
            return None
        return {**info, "name": folder.name}

    def rows(self, name: str) -> list[dict[str, Any]]:
        """The rows of batch ``name`` (blocks)."""
        return read_lines(self.folder(name) / ROWS_FILE)

    def results(self, name: str) -> list[dict[str, Any]]:
        """The rows answered so far, a torn last line left out (blocks)."""
        return read_lines(self.folder(name) / RESULTS_FILE)

    def questions(self, name: str) -> dict[str, Any]:
        data = json.loads((self.folder(name) / QUESTIONS_FILE).read_bytes())
        return data["questions"]

    # Starting

    def prepare(
        self,
        dataset: str,
        split: str | None,
        model: str,
        questions_from: str | None = None,
        sample: int | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> Path:
        """Write a new batch's folder: its rows, questions and summary (blocks).

        ``questions_from`` names a question set whose questions are asked
        (None: the dataset's own); ``sample`` the number of rows to draw (None:
        all). Raises ``BatchError`` (400 or 404 or 409) for what cannot run."""
        location = self.files.dataset_location(dataset)
        try:
            data, _ = open_dataset(location)
        except (DecisionError, OSError) as error:
            raise BatchError(f"{dataset} cannot be read: {error}", param="dataset") from None
        if split is None:
            split = next(iter(data.splits), None)
        if split not in data.splits:
            raise BatchError(f"{dataset} has no split {split!r}", param="split")
        asked: dict[str, Any] | None = None
        if questions_from is not None:
            body, _ = self.files.load_set(questions_from)
            asked = body.get("questions") if isinstance(body, dict) else None
            if not isinstance(asked, dict) or not asked:
                raise BatchError(f"the set {questions_from} has no questions", param="questions")
        elif data.folder and not data.questions and not data.first_line().get("questions"):
            raise BatchError(
                f"{dataset} has no questions yet: write them in the Playground or pick a "
                "question set",
                param="questions",
            )
        try:
            examples = data.read(split, lenient=True)
        except DecisionError as error:
            raise BatchError(f"{dataset}/{split}: {error}", param="dataset") from None
        except OSError as error:
            raise BatchError(
                f"{dataset}/{split} cannot be read: {error}", param="dataset"
            ) from None
        total = len(examples)
        if sample is not None:
            if sample < 1:
                raise BatchError("a sample is at least one row", param="sample")
            if sample < total:
                picked = sorted(random.Random(seed).sample(range(total), sample))
            else:
                picked, sample = list(range(total)), None
        else:
            picked = list(range(total))
        if len(picked) > MAX_ROWS:
            raise BatchError(f"at most {MAX_ROWS:,} rows: take a sample", param="sample")
        if model not in self.gpu.pool.specs:
            raise BatchError(f"no model is named {model}", param="model")
        chosen = model

        shared = asked if asked is not None else first_questions(examples, data.questions)
        lines, labelled = [], set()
        for index in picked:
            example = examples[index]
            own = example.request.get("questions") or {}
            row = {"row": index, "id": example.id}
            row.update({k: v for k, v in example.request.items() if k not in NOT_ASKED})
            if asked is None and own and own != shared:
                row["questions"] = own
            questions = asked if asked is not None else (own or shared)
            labels = {
                qid: label
                for qid, label in example.labels.items()
                if qid in questions and same_question(questions[qid], own.get(qid))
            }
            labelled.update(labels)
            row["labels"] = labels
            lines.append(row)

        folder = self._claim(name, default_name(dataset, chosen))
        name = folder.name
        info = {
            "name": name,
            "dataset": dataset,
            "split": split,
            "rows_in_split": total,
            "sample": sample,
            "seed": seed if sample is not None else None,
            "questions_from": questions_from,
            "model": chosen,
            "fingerprint": self.models.fingerprint(chosen),
            "total": len(lines),
            "answered": 0,
            "errors": 0,
            "labelled": sorted(labelled),
            "state": "ready",
            "message": None,
            "created": now(),
            "passes": [],
        }
        try:
            self.files.write(folder / QUESTIONS_FILE, encode({"questions": shared}))
            self.files.write(folder / ROWS_FILE, b"".join(line(row) for row in lines))
            (folder / RESULTS_FILE).touch()
            self.files.write(folder / BATCH_FILE, encode(info))
        except BaseException:
            remove_folder(folder)
            raise
        return folder

    def _claim(self, given: str | None, default: str) -> Path:
        """A new batch's folder, made: ``given`` (409 when taken), else ``default``,
        ``default-2``, ... (the first free; ``mkdir`` decides, so two starts at
        once get two names)."""
        self.root.mkdir(exist_ok=True)
        if given is not None:
            folder = self.folder(check_name(given, "a batch's name"))
            try:
                folder.mkdir()
            except FileExistsError:
                raise BatchError(
                    f"a batch named {given} exists already", status=409, param="name"
                ) from None
            return folder
        check_name(default, "a batch's name")
        for number in range(1, 10_000):
            name = default if number == 1 else f"{default[:70]}-{number}"
            folder = self.folder(name)
            try:
                folder.mkdir()
            except FileExistsError:
                continue
            return folder
        raise BatchError(f"no free name like {default}", status=409, param="name")

    async def start(self, name: str) -> dict[str, Any]:
        """Run batch ``name`` from where it stopped (its first pass, or Run the
        Rest); refused while another runs or the GPU is held (409)."""
        self.check_free()
        folder = self.folder(name)
        info = await asyncio.to_thread(self._read_info, folder)
        if info is None:
            raise NotFound(f"no batch named {name!r}")
        if info["answered"] >= info["total"]:
            raise BatchError(f"{name} has answered every row", status=409)
        model = info["model"]
        if model not in self.gpu.pool.specs:
            raise BatchError(
                f"{model} is not in the Studio's list any more", status=409, param="model"
            )
        current = await asyncio.to_thread(self.models.fingerprint, model)
        if current != info.get("fingerprint"):
            raise BatchError(
                f"{model} changed since the batch started: Run Again makes a new batch",
                status=409,
                param="model",
            )
        self.check_free()  # again, after the awaits
        batch = Batch(folder, info)
        info["state"] = "running"
        info["message"] = None
        info["passes"] = [*info.get("passes", []), {"started": now(), "ended": None}]
        self.running = batch
        self.gpu.pool.pin(model, f"batch {name}")
        try:
            await asyncio.to_thread(self._save, batch)
        except BaseException:
            self.gpu.pool.unpin(model)
            self.running = None
            raise
        batch.task = asyncio.get_running_loop().create_task(self._run(batch))
        self._changed(batch, False)
        return self.live(batch)

    def check_free(self) -> None:
        if self.running is not None:
            raise BatchError(
                f"{self.running.name} is running; one batch at a time",
                status=409,
                code="batch_running",
            )
        holder = self.gpu.holder
        if holder is not None:
            raise BatchError(
                f"the GPU is busy with {holder}; start the batch when it is done",
                status=409,
                code="gpu_job_running",
            )
        if self.refuse is not None:
            self.refuse()

    async def stop(self, name: str) -> dict[str, Any]:
        """Stop the running batch ``name`` after the row it is on."""
        batch = self.running
        if batch is None or batch.name != name:
            raise BatchError(f"{name} is not running", status=409)
        batch.stop_requested = True
        self._changed(batch, False)
        return self.live(batch)

    async def shutdown(self) -> None:
        """Stop the running batch and wait for it (the Studio stops)."""
        batch = self.running
        if batch is not None and batch.task is not None:
            batch.stop_requested = True
            with suppress(Exception):
                await batch.task

    # Running

    async def _run(self, batch: Batch) -> None:
        info = batch.info
        state, message = "done", None
        handle = None
        try:
            rows, done, errors = await asyncio.to_thread(self._rest, batch.folder)
            info["answered"], info["errors"] = len(done), errors
            shared = await asyncio.to_thread(self.questions, batch.name)
            handle = await asyncio.to_thread(open, batch.folder / RESULTS_FILE, "ab")
            for row in rows:
                if batch.stop_requested:
                    break
                request = {k: v for k, v in row.items() if k not in ("row", "id", "labels")}
                request.setdefault("questions", shared)
                started = time.perf_counter()
                try:
                    answered = await self.gpu.run(
                        answer, self.gpu.pool, info["model"], request, self.models.check
                    )
                except Busy as error:
                    state, message = "stopped", str(error)
                    break
                except ModelTooLargeError as error:
                    state, message = "failed", str(error)
                    break
                except DecisionError as error:
                    if error.param == "model":
                        state, message = "failed", error.message
                        break
                    result = {
                        "row": row["row"],
                        "error": {"message": error.message, "param": error.param},
                    }
                    if error.code:
                        result["error"]["code"] = error.code
                    info["errors"] += 1
                else:
                    result = {
                        "row": row["row"],
                        # Unrounded, so the metrics are training eval's; the export rounds.
                        "response": answered.result.model_dump(),
                        "latency_ms": round(answered.seconds * 1000, 3),
                        "input_tokens": answered.result.usage.input_tokens,
                        "truncated": answered.result.truncated,
                    }
                    if self.record is not None:
                        asyncio.get_running_loop().run_in_executor(None, self._record, answered)
                handle.write(line(result))
                handle.flush()
                batch.times.append(time.perf_counter() - started)
                info["answered"] += 1
                self._changed(batch, True)
            if batch.stop_requested and state == "done" and info["answered"] < info["total"]:
                state = "stopped"
        except asyncio.CancelledError:
            state, message = "stopped", "the batch was cancelled"
            raise
        except Exception as error:
            logger.exception("batch %s failed", batch.name)
            state, message = "failed", f"the batch failed: {error}"
        finally:
            # The pool and the running batch first: nothing below may skip them.
            self.gpu.pool.unpin(info["model"])
            self.running = None
            if handle is not None:
                with suppress(OSError):
                    handle.close()
            if info["answered"] >= info["total"] and state != "failed":
                state, message = "done", None
            info["state"], info["message"] = state, message
            info["passes"][-1]["ended"] = now()
            try:
                self._save(batch)  # small; on the loop, so a cancel cannot skip it
            except Exception:
                logger.exception("saving batch %s failed", batch.name)
            self._changed(batch, False)

    def _rest(self, folder: Path) -> tuple[list[dict[str, Any]], set[int], int]:
        """The rows not answered yet, the rows answered and how many of those are
        errors; a torn last line of ``results.jsonl`` cut off first."""
        results = folder / RESULTS_FILE
        cut_torn_line(results)
        answered = [item for item in read_lines(results) if isinstance(item.get("row"), int)]
        done = {item["row"] for item in answered}
        errors = sum("error" in item for item in answered)
        rows = [row for row in read_lines(folder / ROWS_FILE) if row.get("row") not in done]
        return rows, done, errors

    def _record(self, answered) -> None:
        try:
            self.record(answered)
        except Exception:
            logger.exception("recording the speed failed")

    def _save(self, batch: Batch) -> None:
        self.files.write(batch.folder / BATCH_FILE, encode(batch.info))

    def _changed(self, batch: Batch, progress: bool) -> None:
        if self.on_change is not None:
            try:
                self.on_change(batch, progress)
            except Exception:
                logger.exception("the batch's change callback failed")

    # Changing

    def delete(self, name: str) -> None:
        """Delete batch ``name``; refused while it runs (blocks)."""
        running = self.running
        if running is not None and running.name.casefold() == name.casefold():
            raise BatchError(f"{name} is running: stop it first", status=409)
        folder = self.folder(name)
        if not (folder / BATCH_FILE).is_file():
            raise NotFound(f"no batch named {name!r}")
        remove_folder(folder)

    def again(self, name: str) -> Path:
        """A new batch with ``name``'s rows, questions and model (blocks)."""
        source = self.folder(name)
        info = self._read_info(source)
        if info is None:
            raise NotFound(f"no batch named {name!r}")
        if info["model"] not in self.gpu.pool.specs:
            raise BatchError(f"{info['model']} is not in the Studio's list any more", param="model")
        # A name with a number added (tweets-d3-edge-2) runs again as tweets-d3-edge-3;
        # a name that only ends in digits (a model's Julia-1) keeps them.
        base = default_name(info.get("dataset", ""), info["model"])
        folder = self._claim(
            None, base if re.fullmatch(rf"{re.escape(base)}(-\d+)?", name) else name
        )
        new = folder.name
        try:
            for file in (QUESTIONS_FILE, ROWS_FILE):
                self.files.write(folder / file, (source / file).read_bytes())
            (folder / RESULTS_FILE).touch()
            fresh = {
                **info,
                "name": new,
                "fingerprint": self.models.fingerprint(info["model"]),
                "answered": 0,
                "errors": 0,
                "state": "ready",
                "message": None,
                "created": now(),
                "passes": [],
            }
            self.files.write(folder / BATCH_FILE, encode(fresh))
        except BaseException:
            remove_folder(folder)
            raise
        return folder


def first_questions(examples, questions: dict[str, Any] | None) -> dict[str, Any]:
    """The questions shared by the rows: the dataset's, else the first row's."""
    if questions:
        return questions
    for example in examples:
        own = example.request.get("questions")
        if own:
            return own
    raise BatchError("the rows have no questions: pick a question set", param="questions")


def count_results(path: Path) -> tuple[int, int]:
    """The rows ``results.jsonl`` answered, and how many of them are errors."""
    items = [item for item in read_lines(path) if isinstance(item.get("row"), int)]
    return len({item["row"] for item in items}), sum("error" in item for item in items)


def read_lines(path: Path) -> list[dict[str, Any]]:
    """A JSON-lines file's objects; a last line that does not parse (torn) is
    left out, as are blank lines."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    out = []
    lines = data.split(b"\n")
    for number, text in enumerate(lines):
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except ValueError:
            if number == len(lines) - 1:  # torn: the Studio was killed while writing it
                continue
            raise
        if isinstance(item, dict):
            out.append(item)
    return out


def cut_torn_line(path: Path) -> None:
    """Cut a last line without its newline off ``path``."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    if data and not data.endswith(b"\n"):
        keep = data.rfind(b"\n") + 1
        with open(path, "r+b") as handle:
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())


def remove_folder(folder: Path) -> None:
    """Remove a batch folder (its files, then the folder)."""
    import shutil

    shutil.rmtree(folder, ignore_errors=True)
