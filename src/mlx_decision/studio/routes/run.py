"""The Playground's live check and runs, and what a run leaves behind.

``POST /check`` validates a request for a model against its static
capabilities (read from its files, nothing loaded, no GPU), so the page
shows the API's own errors as the user types. ``POST /run`` answers a
request on the shared worker, like the Jev API, then appends the run to
its set's history and records its speed. The Jev API's answers record
their speed too (``Recorder.answered``).
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ...backend import Capabilities
from ...errors import DecisionError
from ...fitted import fingerprint
from ...gpu import Busy, Gpu
from ...images import check_data_urls
from ...machine import sysctl
from ...memory import ModelTooLargeError
from ...model import check_request
from ...registry import static_capabilities
from ...server import Answered, answer
from ..catalog import model_facts, repo_size
from ..workspace import Workspace, WorkspaceError, check_name, dataset_document
from . import ApiError, field, json_body

logger = logging.getLogger(__name__)

# An example's name, and a Run All's or a Compare's id, as a run may name them.
MAX_TAG = 80
GROUP_ID = re.compile(r"[A-Za-z0-9-]{1,40}")


class Models:
    """The pool's models as the Studio needs them before loading: static
    capabilities, fingerprints and the Models page's facts, each read once
    per model."""

    def __init__(
        self, gpu: Gpu, workspace_models: Path | None = None, files: Workspace | None = None
    ):
        self.gpu = gpu
        self.workspace_models = workspace_models
        self.files = files  # where the speed records are
        self._chip: str | None = None
        self._capabilities: dict[str, Capabilities] = {}
        self._fingerprints: dict[str, str] = {}
        self._facts: dict[str, dict] = {}
        self._freed: dict[str, tuple[str, int]] = {}

    def speed(self, name: str) -> dict[str, Any] | None:
        """The model's measured speed on this Mac's chip (``speed_summary``),
        None before its first answer here (blocks: call it off the loop)."""
        if self.files is None:
            return None
        if self._chip is None:
            self._chip = sysctl("machdep.cpu.brand_string") or ""
        records = self.files.read_speed(self.fingerprint(name), self._chip or None)
        return speed_summary(records)

    def forget(self, name: str) -> None:
        """Drop what was read for ``name`` (the model came or went in a rescan)."""
        for table in (self._capabilities, self._fingerprints, self._facts, self._freed):
            table.pop(name, None)

    def facts(self, name: str) -> dict:
        """``catalog.model_facts`` for the model (blocks: call it off the loop)."""
        if name not in self._facts:
            self._facts[name] = model_facts(self.gpu.pool.specs[name], self.workspace_models)
        return self._facts[name]

    def freed_bytes(self, name: str, kind: str | None) -> int | None:
        """What deleting the model frees on disk (``ModelRemoval.kind``'s
        ``kind``): a cache repo's every revision, nothing for a link, else the
        folder; None when it may not be deleted (blocks: call it off the loop)."""
        if kind is None:
            return None
        known = self._freed.get(name)
        if known is None or known[0] != kind:
            facts = self.facts(name)
            if kind == "cache":
                size = repo_size(self.gpu.pool.specs[name].path)
            elif facts["link"]:
                size = 0
            else:
                size = facts["disk_bytes"]
            known = self._freed[name] = (kind, size)
        return known[1]

    def memory(self, name: str) -> dict:
        """What loading the model takes: ``bytes`` alone (with its base), ``added``
        when the weights it shares (``group``'s) are loaded, and ``over_budget``
        when it alone does not fit the budget."""
        pool = self.gpu.pool
        try:
            alone, added, group = pool.footprint(name)
            return {"bytes": alone, "added": added, "group": group, "over_budget": False}
        except ModelTooLargeError as error:
            return {
                "bytes": error.required,
                "added": error.required,
                "group": pool.footprint_group(name),
                "over_budget": True,
            }

    def capabilities(self, name: str) -> Capabilities:
        if name not in self._capabilities:
            spec = self.gpu.pool.specs[name]
            self._capabilities[name] = static_capabilities(spec.path, **spec.options)
        return self._capabilities[name]

    def fingerprint(self, name: str) -> str:
        if name not in self._fingerprints:
            self._fingerprints[name] = fingerprint(self.gpu.pool.specs[name].path)
        return self._fingerprints[name]

    def check(self, name: str, body: Any) -> None:
        """``check_request`` against the model's static capabilities; raises
        ``DecisionError`` (with ``param`` "model" when they cannot be read)."""
        try:
            capabilities = self.capabilities(name)
        except (ValueError, OSError) as error:
            raise DecisionError(f"cannot read what {name} takes: {error}", param="model") from None
        check_request(body, capabilities, name, open_images=True)


class Recorder:
    """Writes a run's history line and every answer's speed record."""

    def __init__(self, files: Workspace, models: Models):
        self.files = files
        self.models = models
        self.chip = sysctl("machdep.cpu.brand_string")
        self._tasks: set[asyncio.Task] = set()

    def speed(self, answered: Answered, source: str) -> None:
        """Record how long the answer took (blocks: call it off the loop)."""
        self.files.record_speed(
            self.models.fingerprint(answered.model),
            {
                "model": answered.model,
                "chip": self.chip,
                "latency_ms": round(answered.seconds * 1000, 3),
                "input_tokens": answered.result.usage.input_tokens,
                "truncated": answered.result.truncated,
                "source": source,
            },
        )

    def run(
        self,
        set_name: str | None,
        request: dict,
        answered: Answered,
        example: str | None = None,
        batch: str | None = None,
        compare: str | None = None,
    ) -> dict[str, Any]:
        """Record a Playground run: its speed and, for a set, its history line
        (with the example it asked and the Run All or Compare it belongs to,
        when given). Returns what the page gets besides the answers (blocks)."""
        result = answered.result
        run = {
            "model": answered.model,
            "latency_ms": round(answered.seconds * 1000, 3),
            "load_ms": None
            if answered.load_seconds is None
            else round(answered.load_seconds * 1000, 3),
            "input_tokens": result.usage.input_tokens,
            "truncated": result.truncated,
        }
        try:
            self.speed(answered, "studio")
        except Exception:
            logger.exception("recording the speed failed")
        if set_name is None:
            return {**run, "id": None, "time": None}
        tags = {
            key: value
            for key, value in (("example", example), ("batch", batch), ("compare", compare))
            if value
        }
        line = self.files.append_history(
            set_name,
            {
                "model": answered.model,
                "fingerprint": self.models.fingerprint(answered.model),
                **tags,
                "request": request,
                "response": result.to_wire(),
                **{key: value for key, value in run.items() if key != "model"},
            },
        )
        if line is None:  # the set is gone
            return {**run, **tags, "id": None, "time": None}
        return {**run, **tags, "id": line["id"], "time": line["time"]}

    def answered(self, answered: Answered) -> None:
        """The Jev API's hook: record the speed in the background (on the loop)."""
        task = asyncio.get_running_loop().create_task(
            asyncio.to_thread(self._speed_quietly, answered)
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _speed_quietly(self, answered: Answered) -> None:
        try:
            self.speed(answered, "api")
        except Exception:
            logger.exception("recording the speed failed")


def run_router(gpu: Gpu, models: Models, recorder: Recorder, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api")
    pool = gpu.pool

    @router.post("/check")
    async def check(request: Request):
        """``{request, model}`` -> ``{model, errors}``: the errors the API would give
        before answering (none, or the first one, with its ``param``)."""
        body = await json_body(request, max_body_bytes)
        decision = field(body, "request", object)
        name = field(body, "model", str, required=False)

        def run_check() -> tuple[str | None, list[dict]]:
            try:
                chosen = pool.resolve(name)
            except DecisionError as error:
                return None, [describe(error)]
            try:
                check_data_urls(decision)
                models.check(chosen, decision)
            except DecisionError as error:
                return chosen, [describe(error)]
            return chosen, []

        chosen, errors = await asyncio.to_thread(run_check)
        return {"model": chosen, "errors": errors}

    @router.post("/run")
    async def run(request: Request):
        """``{request, model, set, example, batch, compare}`` -> the answers, the
        timings and the history line's id (``set`` names the set whose history
        gets the run; ``example`` the example it asks, ``batch`` the Run All or
        ``compare`` the Compare it belongs to; all optional)."""
        gpu.check()  # before reading the body
        body = await json_body(request, max_body_bytes)
        decision = field(body, "request", dict)
        name = field(body, "model", str, required=False)
        if name is None and isinstance(decision.get("model"), str):
            name = decision["model"]
        set_name = field(body, "set", str, required=False)
        example = field(body, "example", str, required=False)
        if example is not None and not 0 < len(example) <= MAX_TAG:
            raise ApiError(400, f"example must be 1 to {MAX_TAG} characters", param="example")
        batch = group_id(body, "batch")
        compare = group_id(body, "compare")
        if batch is not None and compare is not None:
            raise ApiError(400, "a run is in a Run All or a Compare, not both", param="compare")
        if set_name is not None:
            try:
                if dataset_document(set_name) is None:  # a dataset's: checked there
                    check_name(set_name, "a set's name")
            except WorkspaceError as error:
                raise ApiError(400, error.message, param="set") from None
        check_data_urls(decision)
        try:
            answered = await gpu.run(answer, pool, name, decision, models.check)
        except ModelTooLargeError as error:  # a DecisionError, named by the model
            raise ApiError(422, str(error), param="model") from None
        except (Busy, DecisionError):
            raise
        except Exception:
            logger.exception("the run failed")
            raise ApiError(
                500, "the model failed to answer the request", kind="api_error"
            ) from None
        recorded = await asyncio.to_thread(
            recorder.run, set_name, decision, answered, example, batch, compare
        )
        return {"response": answered.result.to_wire(), **recorded}

    return router


def group_id(body: dict, key: str) -> str | None:
    """The Run All's (``batch``) or Compare's (``compare``) id a run names."""
    value = field(body, key, str, required=False)
    if value is not None and not GROUP_ID.fullmatch(value):
        raise ApiError(400, f"{key} must be 1 to 40 letters, digits or '-'", param=key)
    return value


def describe(error: DecisionError) -> dict[str, Any]:
    return {"message": error.message, "param": error.param, "code": error.code}


def speed_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """What the Models page says about a model's speed (PG-9): the median
    latency of its answers, how many there were and their input tokens' range
    (with the chip they ran on); None before the first. Measured only."""
    latencies = sorted(
        float(record["latency_ms"])
        for record in records
        if isinstance(record.get("latency_ms"), int | float)
    )
    if not latencies:
        return None
    middle = len(latencies) // 2
    median = (
        latencies[middle] if len(latencies) % 2 else (latencies[middle - 1] + latencies[middle]) / 2
    )
    tokens = [r["input_tokens"] for r in records if isinstance(r.get("input_tokens"), int)]
    return {
        "median_ms": round(median, 1),
        "count": len(latencies),
        "tokens_min": min(tokens) if tokens else None,
        "tokens_max": max(tokens) if tokens else None,
        "chip": next((r.get("chip") for r in reversed(records) if r.get("chip")), None),
    }
