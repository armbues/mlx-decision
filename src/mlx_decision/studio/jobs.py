"""Jobs: downloads, conversions, fits and evaluations, each the CLI in a process of its own.

A job runs the environment's own command line (``python -m mlx_decision``)
with ``--events``: progress comes as JSON lines on its stdout, which go to
``jobs/<id>/events.jsonl`` and to ``on_change``; its stderr goes to
``stderr.log``; ``job.json`` holds the command, the times and the outcome.

The outcome is the job's last event: ``done``, ``error`` or ``cancelled``.
A process that ends without one was ``cancelled`` when Stop was pressed,
else it ``failed`` (with the end of its stderr). A Stop that comes after
``done`` does not undo it.

Stop sends SIGINT (the CLI stops at its next step and writes nothing), a
second one after ``stop_grace`` seconds (it stops at once), then SIGKILL
after ``kill_grace`` more, and removes what a killed write leaves beside
the job's output. The process runs in its own session, so a Ctrl-C in the
Studio's terminal reaches only the Studio, which stops its jobs before it
exits. Its stdin is a pipe the Studio never writes to: if the Studio is
killed, the pipe closes and the CLI stops by itself (``watchdog``; the job
gets ``MLX_DECISION_WATCH_STDIN=1``, which turns that on).

A job that loads a model (``holds_gpu``) first takes the GPU from the
Studio's pool (``Gpu.acquire``: the request on the worker finishes, the
pool unloads, model work answers busy naming the job) and gives it back
when it ends. One such job runs at a time; downloads run alongside.

Use it from the event loop. ``job.json`` and the event lines are small
writes made on the loop.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import signal
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..events import FINAL_STAGES
from ..fitted import REPORT_FILE
from ..gpu import Gpu
from ..watchdog import WATCH_VARIABLE, remove
from .workspace import INTERRUPTED, JOB_FILE, Workspace, check_name, encode, now

EVENTS_FILE = "events.jsonl"
# In a job's arguments: its folder, where its ``inputs`` are.
FOLDER = "{folder}"
STDERR_FILE = "stderr.log"
# The CLI as the Studio runs it: the same environment's package.
COMMAND = (sys.executable, "-m", "mlx_decision")
STOP_GRACE_S = 20.0
KILL_GRACE_S = 5.0
# Last events that decide the outcome.
# What a killed write leaves beside an output folder NAME: the fitted or
# converted folder before its rename, a fit's stored hidden states, the
# full-size files of a quantized download.
WORKING_SUFFIXES = (".partial", ".states", ".download")
STDERR_TAIL = 20  # lines of stderr a failed job reports
LINE_LIMIT = 2**24  # the longest event line read (an evaluation's results)

logger = logging.getLogger(__name__)


def report_digest(folder: Path) -> str | None:
    """The SHA-256 of a fitted model's report, which differs from fit to fit."""
    try:
        return hashlib.sha256((Path(folder) / REPORT_FILE).read_bytes()).hexdigest()
    except OSError:
        return None


def with_replaced(summary: dict[str, Any]) -> dict[str, Any]:
    """``summary`` with ``replaced``: a fit that ended done whose folder now holds
    another fit's model (one fitted later under the same name). Unknown for jobs
    recorded before ``wrote``, so False."""
    wrote, output = summary.get("wrote"), summary.get("output")
    current = report_digest(Path(output)) if wrote and output else None
    return {**summary, "replaced": current is not None and current != wrote}


class JobConflict(Exception):
    """A second job that needs the GPU while ``running`` has it."""

    def __init__(self, running: "Job"):
        super().__init__(
            f"{running.title} is running; one job at a time can use the GPU. "
            "Start this one when it has ended."
        )
        self.running = running


class JobBlocked(Exception):
    """A GPU job refused while something else uses the model pool (a batch)."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    argv: list[str]
    holds_gpu: bool
    folder: Path
    output: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)  # what it was started with
    also: list[Path] = field(default_factory=list)  # more files it writes (a Get's splits)
    state: str = "waiting"  # waiting (for the GPU), running, ended
    outcome: str | None = None  # done, error, cancelled, failed, interrupted
    message: str | None = None
    created: str = ""
    started: str | None = None
    ended: str | None = None
    stop_requested: bool = False
    stopped_by_user: bool = False  # Stop pressed, not the Studio's shutdown
    returncode: int | None = None
    wrote: str | None = None  # a fit's report's SHA-256: which fit its folder holds
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    starting: asyncio.Task | None = field(default=None, repr=False)  # submit's start
    task: asyncio.Task | None = field(default=None, repr=False)  # reads its events

    @property
    def last(self) -> dict[str, Any] | None:
        return self.events[-1] if self.events else None

    @property
    def outputs(self) -> list[Path]:
        """Every file or folder it writes: ``output``, then ``also``."""
        return ([self.output] if self.output is not None else []) + self.also

    def summary(self) -> dict[str, Any]:
        """What ``job.json`` holds, and what a list of jobs shows."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "command": ["mlx-decision", *self.argv],
            "holds_gpu": self.holds_gpu,
            "output": str(self.output) if self.output is not None else None,
            "outputs": [str(path) for path in self.outputs],
            "details": self.details,
            "state": self.state,
            "outcome": self.outcome,
            "message": self.message,
            "created": self.created,
            "started": self.started,
            "ended": self.ended,
            "stop_requested": self.stop_requested,
            "returncode": self.returncode,
            "wrote": self.wrote,
            "last": self.last,
        }


class Jobs:
    """The workspace's jobs: those of this process, and the ended ones on disk.

    ``command`` is what runs (the CLI; tests put a stand-in there).
    ``on_change(job, event)`` is called on the loop when a job changes
    (``event`` None) or sends an event. ``blocked``, when set, says why a GPU
    job cannot start now (a batch runs), or None.
    """

    def __init__(
        self,
        gpu: Gpu,
        workspace: Workspace,
        *,
        command: Sequence[str] = COMMAND,
        stop_grace: float = STOP_GRACE_S,
        kill_grace: float = KILL_GRACE_S,
    ):
        self.gpu = gpu
        self.workspace = workspace
        self.command = list(command)
        self.stop_grace = stop_grace
        self.kill_grace = kill_grace
        self.on_change: Callable[[Job, dict[str, Any] | None], None] | None = None
        self.jobs: dict[str, Job] = {}
        self.gpu_job: Job | None = None
        self.blocked: Callable[[], str | None] | None = None
        self.closing = False

    # Starting and running

    async def start(
        self,
        kind: str,
        argv: Sequence[str],
        *,
        holds_gpu: bool,
        output: Path | None = None,
        title: str | None = None,
        details: dict[str, Any] | None = None,
        inputs: dict[str, bytes] | None = None,
        also: Sequence[Path] = (),
    ) -> Job:
        """Run ``mlx-decision ARGV --events`` as a job.

        ``output`` is the folder the job writes (its working folders are
        removed after a kill). A GPU job waits for the request on the worker
        and the unload before its process starts. Raises ``JobConflict``
        while another GPU job is waiting or running, and ``RuntimeError``
        once the Studio is stopping, ``JobBlocked`` while ``blocked`` says so.
        ``details`` are what the job was started
        with (the page's fields), kept in its summary. ``inputs`` are files
        written into the job's folder before it runs (``FOLDER`` in ``argv``
        stands for that folder). ``also`` are more files it writes (cleaned up
        as ``output`` is).
        """
        job = self._create(kind, argv, holds_gpu, output, title, details, inputs, also)
        await self._launch(job)
        return job

    def submit(
        self,
        kind: str,
        argv: Sequence[str],
        *,
        holds_gpu: bool,
        output: Path | None = None,
        title: str | None = None,
        details: dict[str, Any] | None = None,
        inputs: dict[str, bytes] | None = None,
        also: Sequence[Path] = (),
    ) -> Job:
        """``start`` without waiting: the job is returned at once, waiting for
        the GPU or about to run; one that cannot start ends ``failed``. Raises
        ``JobConflict`` and ``RuntimeError`` as ``start`` does, before
        anything is created."""
        job = self._create(kind, argv, holds_gpu, output, title, details, inputs, also)

        async def launch() -> None:
            with suppress(Exception):  # the job's outcome says what went wrong
                await self._launch(job)

        job.starting = asyncio.create_task(launch(), name=f"start {job.id}")
        return job

    def _create(
        self,
        kind: str,
        argv: Sequence[str],
        holds_gpu: bool,
        output: Path | None,
        title: str | None,
        details: dict[str, Any] | None = None,
        inputs: dict[str, bytes] | None = None,
        also: Sequence[Path] = (),
    ) -> Job:
        if self.closing:
            raise RuntimeError("the Studio is stopping")
        if holds_gpu and self.gpu_job is not None:
            raise JobConflict(self.gpu_job)
        if holds_gpu and self.blocked is not None and (reason := self.blocked()):
            raise JobBlocked(reason)
        job = self._new(kind, list(argv), holds_gpu, output, title, details or {}, inputs or {})
        job.also = [Path(path) for path in also]
        if holds_gpu:
            self.gpu_job = job
        self._changed(job)
        return job

    async def _launch(self, job: Job) -> None:
        try:
            if job.holds_gpu:
                await self.gpu.acquire(job.title)
            if job.stop_requested:  # Stop pressed while it waited for the GPU
                self._end(job, "cancelled", "stopped before it started")
                return
            await self._spawn(job)
        except BaseException as error:
            if job.outcome is None:
                self._end(job, "failed", f"could not start: {error}")
            raise
        job.task = asyncio.create_task(self._watch(job), name=f"job {job.id}")
        if job.stop_requested:  # Stop pressed while the process was being started
            asyncio.create_task(self._escalate(job), name=f"stop {job.id}")

    def _new(
        self,
        kind: str,
        argv: list[str],
        holds_gpu: bool,
        output: Path | None,
        title: str | None,
        details: dict[str, Any],
        inputs: dict[str, bytes],
    ) -> Job:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{kind}-{secrets.token_hex(3)}"
        folder = self.workspace.path("jobs", job_id)
        folder.mkdir()
        for name, data in inputs.items():  # a job's own files: names, not paths
            self.workspace.write(self.workspace.inside(folder / check_name(name)), data)
        argv = [word.replace(FOLDER, str(folder)) for word in argv]
        output = Path(output) if output is not None else None
        title = title or (f"{kind} {output.name}" if output is not None else kind)
        job = Job(
            job_id, kind, title, argv, holds_gpu, folder, output, dict(details), created=now()
        )
        self.jobs[job.id] = job
        self._save(job)
        return job

    async def _spawn(self, job: Job) -> None:
        # Append: the Studio adds lines (stray stdout) that later stderr must not overwrite.
        with open(job.folder / STDERR_FILE, "ab") as stderr:
            job.process = await asyncio.create_subprocess_exec(
                *self.command,
                *job.argv,
                "--events",
                stdin=asyncio.subprocess.PIPE,  # never written: closes when the Studio dies
                env={**os.environ, WATCH_VARIABLE: "1"},  # so the CLI watches it
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                cwd=self.workspace.root,
                start_new_session=True,  # the terminal's Ctrl-C reaches only the Studio
                limit=LINE_LIMIT,
            )
        job.state = "running"
        job.started = now()
        self._save(job)
        self._changed(job)

    async def _watch(self, job: Job) -> None:
        """Read the job's events until its process ends, then settle its outcome."""
        process = job.process
        assert process is not None and process.stdout is not None
        try:
            with open(job.folder / EVENTS_FILE, "ab") as log:
                async for raw in process.stdout:
                    event = _parse(raw)
                    if event is None:  # not an event: keep it with the stderr
                        with open(job.folder / STDERR_FILE, "ab") as stderr:
                            stderr.write(b"stdout: " + raw)
                        continue
                    log.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    log.flush()
                    job.events.append(event)
                    self._changed(job, event)
            job.returncode = await process.wait()
            outcome, message = self._outcome(job)
        except Exception as error:  # noqa: BLE001 - the job must still end
            logger.exception("job %s", job.id)
            outcome, message = "failed", f"the Studio lost the job: {error}"
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        if outcome != "done":
            self._remove_leftovers(job)
        self._end(job, outcome, message)

    def _outcome(self, job: Job) -> tuple[str, str | None]:
        last = job.last
        if last is not None and last.get("stage") in FINAL_STAGES:
            return last["stage"], last.get("message")
        if job.stop_requested:
            return "cancelled", "stopped"
        return "failed", self._stderr_tail(job) or f"ended with exit code {job.returncode}"

    def _end(self, job: Job, outcome: str, message: str | None) -> None:
        if self.closing and outcome == "cancelled" and not job.stopped_by_user:
            # Stopped by the Studio's shutdown, not by a Stop pressed.
            outcome, message = INTERRUPTED, "the Studio stopped while it ran"
        if outcome == "done" and job.kind == "fit" and job.output is not None:
            job.wrote = report_digest(job.output)
        job.state = "ended"
        job.outcome = outcome
        job.message = message
        job.ended = now()
        if job.holds_gpu:
            self.gpu.release(job.title)
            if self.gpu_job is job:
                self.gpu_job = None
        self._save(job)
        self._changed(job)
        logger.info("job %s: %s", job.title, outcome)

    # Stopping

    async def stop(self, job_id: str) -> Job:
        """Ask a job to stop: SIGINT, a second one after ``stop_grace``, SIGKILL
        after ``kill_grace`` more. Returns at once; the outcome follows. Raises
        ``KeyError`` for an unknown job."""
        job = self.jobs[job_id]
        if job.state == "ended" or job.stop_requested:
            return job
        job.stop_requested = True
        job.stopped_by_user = not self.closing
        self._save(job)
        self._changed(job)
        if job.process is not None:
            asyncio.create_task(self._escalate(job), name=f"stop {job.id}")
        return job

    async def _escalate(self, job: Job) -> None:
        process = job.process
        assert process is not None
        steps = (
            (signal.SIGINT, self.stop_grace),
            (signal.SIGINT, self.kill_grace),
            (signal.SIGKILL, None),
        )
        for sig, wait in steps:
            if process.returncode is not None:
                return
            with suppress(ProcessLookupError):
                os.killpg(process.pid, sig)  # its own session: the whole group
            if wait is None:
                return
            with suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(process.wait()), wait)
                return
            logger.warning("job %s did not stop after %s s", job.title, wait)

    async def shutdown(self) -> None:
        """Stop every job and wait for them (the Studio is stopping)."""
        self.closing = True
        open_jobs = [job for job in self.jobs.values() if job.state != "ended"]
        for job in open_jobs:
            await self.stop(job.id)
        limit = self.stop_grace + self.kill_grace + 5
        starting = [job.starting for job in open_jobs if job.starting is not None]
        if starting:  # those waiting for the GPU end once they have it
            await asyncio.wait(starting, timeout=limit)
        tasks = [job.task for job in open_jobs if job.task is not None]
        if tasks:
            await asyncio.wait(tasks, timeout=limit)

    def _remove_leftovers(self, job: Job) -> None:
        for output in job.outputs:
            for suffix in WORKING_SUFFIXES:
                remove(output.parent / f".{output.name}{suffix}")

    def delete(self, job_id: str) -> dict[str, Any]:
        """Delete an ended job's folder (its record, events and stderr; not
        what it wrote) and forget it; returns its summary. Raises ``KeyError``
        and ``ValueError`` for a job that has not ended."""
        summary = self.get(job_id)
        if summary.get("state") != "ended":
            raise ValueError(f"{summary.get('title', job_id)} has not ended: stop it first")
        remove(self.workspace.path("jobs", job_id))
        self.jobs.pop(job_id, None)
        self.workspace.changed("jobs", job_id, "deleted")
        return summary

    # Reading

    def get(self, job_id: str) -> dict[str, Any]:
        """A job's summary, also of a job that ran before this process (from
        its ``job.json``). Raises ``KeyError``."""
        if job_id in self.jobs:
            return with_replaced(self.jobs[job_id].summary())
        try:
            summary = json.loads((self.workspace.path("jobs", job_id) / JOB_FILE).read_bytes())
        except (OSError, ValueError) as error:
            raise KeyError(job_id) from error
        if not isinstance(summary, dict):
            raise KeyError(job_id)
        return with_replaced(summary)

    def events(self, job_id: str) -> list[dict[str, Any]]:
        """A job's events so far. Raises ``KeyError``."""
        if job_id in self.jobs:
            return list(self.jobs[job_id].events)
        self.get(job_id)
        path = self.workspace.path("jobs", job_id) / EVENTS_FILE
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return []
        return [event for line in raw.splitlines() if (event := _parse(line)) is not None]

    def current(self) -> list[dict[str, Any]]:
        """This process's jobs, newest first (what a tab opened now shows)."""
        return [job.summary() for job in reversed(self.jobs.values())]

    def writer(self, path: Path) -> str | None:
        """The title of a job that has not ended and writes ``path`` (its letters'
        case ignored, as macOS's disks do), or None."""
        wanted = str(path).casefold()
        for job in self.jobs.values():
            if job.state != "ended" and any(str(out).casefold() == wanted for out in job.outputs):
                return job.title
        return None

    def list(self) -> list[dict[str, Any]]:
        """Every job's summary, newest first: this process's and those on disk."""
        found = {job_id: with_replaced(job.summary()) for job_id, job in self.jobs.items()}
        for folder in sorted((self.workspace.root / "jobs").iterdir(), reverse=True):
            if folder.name in found or folder.name.startswith(".") or not folder.is_dir():
                continue
            with suppress(KeyError):
                found[folder.name] = self.get(folder.name)
        return sorted(found.values(), key=lambda job: job.get("created") or "", reverse=True)

    # Bookkeeping

    def _save(self, job: Job) -> None:
        self.workspace.write(job.folder / JOB_FILE, encode(job.summary()))

    def _stderr_tail(self, job: Job) -> str:
        try:
            text = (job.folder / STDERR_FILE).read_text(errors="replace")
        except OSError:
            return ""
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-STDERR_TAIL:])

    def _changed(self, job: Job, event: dict[str, Any] | None = None) -> None:
        if self.on_change is not None:
            try:
                self.on_change(job, event)
            except Exception:
                logger.exception("the jobs' change callback failed")


def _parse(raw: bytes) -> dict[str, Any] | None:
    """An event line, or None for anything else."""
    try:
        event = json.loads(raw)
    except ValueError:
        return None
    return event if isinstance(event, dict) and isinstance(event.get("stage"), str) else None


__all__ = ["INTERRUPTED", "Job", "JobConflict", "Jobs"]
