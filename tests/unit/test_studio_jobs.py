"""The Studio's job runner, with ``tests/fake_job.py`` standing in for the CLI."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from fake_backend import FakeBackend, write_model
from mlx_decision import DecisionModel
from mlx_decision.gpu import Busy, Gpu
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio.jobs import JobConflict, Jobs
from mlx_decision.studio.workspace import Workspace

FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"
BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}


class Loads:
    """A pool's load function that notes each load and who held the GPU then."""

    def __init__(self):
        self.gpu: Gpu | None = None
        self.seen: list[tuple[str, str | None]] = []

    def __call__(self, spec):
        self.seen.append((spec.name, self.gpu.holder))
        return DecisionModel(FakeBackend())


@pytest.fixture
def workspace(tmp_path):
    files = Workspace(tmp_path / "workspace")
    files.open()
    return files


@pytest.fixture
def gpu(tmp_path):
    for name in ("alpha", "beta"):
        write_model(tmp_path / "pool" / name)
    loads = Loads()
    gpu = Gpu(ModelPool(discover([tmp_path / "pool"]), budget=10**9, load=loads))
    loads.gpu = gpu
    gpu.loads = loads
    yield gpu
    gpu.close()


@pytest.fixture
def jobs(gpu, workspace):
    return Jobs(gpu, workspace, command=[sys.executable, str(FAKE_JOB)],
                stop_grace=0.6, kill_grace=0.6)  # fmt: skip


def run(coroutine):
    return asyncio.run(coroutine)


async def until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def finished(job, timeout: float = 15.0):
    await asyncio.wait_for(job.task, timeout)
    return job


def fit(jobs, *options, output="models/fitted", **kwargs):
    return jobs.start("fit", ["training", "fit", "-o", output, *options],
                      holds_gpu=True, output=jobs.workspace.root / output, **kwargs)  # fmt: skip


# Outcomes


def test_a_job_that_ends_with_done(jobs, workspace):
    async def scenario():
        job = await finished(await fit(jobs, "--steps", "3"))
        assert job.outcome == "done" and job.message == "finished"
        return job

    job = run(scenario())
    # -o is relative: the job runs in the workspace.
    assert (workspace.root / "models" / "fitted" / "weights").exists()
    assert not (workspace.root / "models" / ".fitted.partial").exists()
    lines = (job.folder / "events.jsonl").read_text().splitlines()
    stages = [json.loads(line)["stage"] for line in lines]
    assert stages == ["check", "work", "work", "work", "done"]
    saved = json.loads((job.folder / "job.json").read_text())
    assert saved["outcome"] == "done" and saved["state"] == "ended"
    assert saved["command"] == ["mlx-decision", "training", "fit", "-o", "models/fitted",
                                "--steps", "3"]  # fmt: skip
    assert saved["started"] and saved["ended"] and saved["returncode"] == 0
    assert (job.folder / "stderr.log").read_text() == ""


@pytest.mark.parametrize(
    ("end", "outcome", "message"),
    [
        ("error", "error", "the data has no labels"),  # its last event
        ("crash", "failed", "RuntimeError: it broke"),  # the end of its stderr
        ("quiet", "failed", "ended with exit code 0"),  # no last event, no stderr
    ],
)
def test_without_done_the_last_event_or_stderr_says_why(jobs, end, outcome, message):
    async def scenario():
        return await finished(await fit(jobs, "--steps", "1", "--end", end))

    job = run(scenario())
    assert job.outcome == outcome
    assert message in job.message


def test_lines_that_are_not_events_go_to_stderr(jobs, workspace):
    jobs.command = [sys.executable, "-c", "print('hello'); print('{\"stage\": \"done\"}')"]

    async def scenario():
        return await finished(await jobs.start("eval", ["x"], holds_gpu=True))

    job = run(scenario())
    assert job.outcome == "done" and [event["stage"] for event in job.events] == ["done"]
    assert (job.folder / "stderr.log").read_text() == "stdout: hello\n"


# Stop


def test_stop_cancels_at_the_next_step_and_nothing_is_written(jobs, workspace):
    async def scenario():
        job = await fit(jobs, "--steps", "200")
        await until(lambda: len(job.events) >= 2)
        await jobs.stop(job.id)
        return await finished(job)

    job = run(scenario())
    assert job.outcome == "cancelled" and job.last["stage"] == "cancelled"
    assert job.stop_requested and job.returncode == 130
    assert list((workspace.root / "models").iterdir()) == []


def test_a_stop_after_done_does_not_undo_it(jobs, workspace):
    async def scenario():
        job = await fit(jobs, "--steps", "1", "--hold", "5")
        await until(lambda: job.last is not None and job.last["stage"] == "done")
        await jobs.stop(job.id)  # the process dies of the signal, the fit is written
        return await finished(job)

    job = run(scenario())
    assert job.returncode == -signal.SIGINT
    assert job.outcome == "done"
    assert (workspace.root / "models" / "fitted").exists()


def test_a_second_interrupt_follows_after_the_grace(jobs):
    async def scenario():
        job = await fit(jobs, "--steps", "200", "--stubborn")
        await until(lambda: len(job.events) >= 2)
        asked = time.monotonic()
        await jobs.stop(job.id)
        await finished(job)
        return job, time.monotonic() - asked

    job, took = run(scenario())
    assert job.outcome == "cancelled" and job.last["stage"] == "cancelled"
    assert jobs.stop_grace <= took < jobs.stop_grace + jobs.kill_grace
    assert "ignored SIGINT 1" in (job.folder / "stderr.log").read_text()


def test_a_job_deaf_to_interrupts_is_killed_and_its_leftovers_removed(jobs, workspace):
    async def scenario():
        job = await fit(jobs, "--steps", "200", "--deaf")
        await until(lambda: len(job.events) >= 2)
        assert (workspace.root / "models" / ".fitted.partial").exists()
        asked = time.monotonic()
        await jobs.stop(job.id)
        await finished(job)
        return job, time.monotonic() - asked

    job, took = run(scenario())
    assert job.returncode == -signal.SIGKILL
    assert job.outcome == "cancelled" and job.message == "stopped"
    assert took >= jobs.stop_grace + jobs.kill_grace
    assert list((workspace.root / "models").iterdir()) == []


# The GPU


def test_a_gpu_job_holds_the_gpu_and_names_itself(jobs, gpu):
    async def scenario():
        await gpu.run(gpu.pool.get, "alpha")
        job = await fit(jobs, "--steps", "100", title="Fit support")
        assert gpu.holder == "Fit support"
        assert gpu.pool.loaded() == []  # unloaded before the job's process started
        with pytest.raises(Busy, match="busy with Fit support"):
            await gpu.run(gpu.pool.get, "alpha")
        with pytest.raises(JobConflict, match="Fit support is running") as refused:
            await jobs.start("convert", ["convert"], holds_gpu=True)
        assert refused.value.running is job
        await jobs.stop(job.id)
        await finished(job)
        assert gpu.holder is None
        await gpu.run(gpu.pool.get, "alpha")  # loads again afterwards
        return job

    run(scenario())
    assert gpu.loads.seen == [("alpha", None), ("alpha", None)]


def test_the_request_in_flight_finishes_before_the_job_and_none_loads_while_held(jobs, gpu):
    gate = threading.Event()
    order = []

    def in_flight():
        model = gpu.pool.get("alpha")
        assert gate.wait(5)
        order.append("request answered")
        return model.decide_request(BODY)

    async def scenario():
        running = asyncio.ensure_future(gpu.run(in_flight))
        queued = asyncio.ensure_future(gpu.run(gpu.pool.get, "beta"))
        await asyncio.sleep(0.05)
        starting = asyncio.ensure_future(fit(jobs, "--steps", "50"))
        await until(lambda: gpu.holder is not None)
        late = asyncio.ensure_future(gpu.run(gpu.pool.get, "beta"))
        await asyncio.sleep(0.2)
        assert not starting.done()  # waits for the request on the worker
        gate.set()
        await running
        job = await starting
        order.append("job started")
        for refused in (queued, late):
            with pytest.raises(Busy):
                await refused
        assert gpu.pool.loaded() == []
        await jobs.stop(job.id)
        await finished(job)

    run(scenario())
    assert order == ["request answered", "job started"]
    assert gpu.loads.seen == [("alpha", None)]  # nothing loaded while the job held it


def test_stop_while_waiting_for_the_gpu(jobs, gpu):
    gate = threading.Event()

    async def scenario():
        running = asyncio.ensure_future(gpu.run(gate.wait, 5))
        starting = asyncio.ensure_future(fit(jobs))
        await until(lambda: jobs.gpu_job is not None)
        await jobs.stop(jobs.gpu_job.id)
        gate.set()
        await running
        job = await starting
        assert job.process is None and job.outcome == "cancelled"
        assert gpu.holder is None and jobs.gpu_job is None

    run(scenario())


def test_downloads_run_alongside_each_other_and_a_gpu_job(jobs, gpu):
    async def scenario():
        fitting = await fit(jobs, "--steps", "20")
        downloads = [
            await jobs.start("download", ["download", name, "--steps", "5"],
                             holds_gpu=False, output=jobs.workspace.root / "models" / name)
            for name in ("one", "two")
        ]  # fmt: skip
        assert gpu.holder == fitting.title  # the downloads did not touch it
        for job in (*downloads, fitting):
            await finished(job)
        return downloads

    assert [job.outcome for job in run(scenario())] == ["done", "done"]


# Shutdown, a killed Studio, the jobs on disk


def test_shutdown_stops_the_jobs_first(jobs, gpu):
    async def scenario():
        job = await fit(jobs, "--steps", "500")
        await until(lambda: len(job.events) >= 2)
        await jobs.shutdown()
        assert job.outcome == "cancelled" and gpu.holder is None
        with pytest.raises(RuntimeError, match="stopping"):
            await fit(jobs)

    run(scenario())


def test_the_studio_shutdown_stops_its_jobs(tmp_path, workspace):
    from fastapi.testclient import TestClient

    from mlx_decision.studio import create_studio

    write_model(tmp_path / "pool" / "alpha")
    pool = ModelPool(discover([tmp_path / "pool"]), budget=10**9)
    app = create_studio(pool, workspace=workspace.root, load_default=False)
    jobs = app.state.jobs
    jobs.command = [sys.executable, str(FAKE_JOB)]
    with TestClient(app) as client:

        async def start():
            return await jobs.start("fit", ["--steps", "500"], holds_gpu=True)

        job = client.portal.call(start)
        deadline = time.monotonic() + 10
        while len(job.events) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    assert job.outcome == "cancelled"
    assert json.loads((job.folder / "job.json").read_text())["outcome"] == "cancelled"


@pytest.mark.parametrize("every", ["0.05", "30"])  # stops at its next step / stuck in one
def test_a_killed_studio_ends_its_job_within_5_s(tmp_path, every):
    script = textwrap.dedent(f"""
        import asyncio, sys, time
        sys.path.insert(0, {str(FAKE_JOB.parent)!r})
        from fake_backend import write_model
        from mlx_decision.gpu import Gpu
        from mlx_decision.pool import ModelPool, discover
        from mlx_decision.studio.jobs import Jobs
        from mlx_decision.studio.workspace import Workspace

        async def main():
            files = Workspace({str(tmp_path / "ws")!r}); files.open()
            jobs = Jobs(Gpu(ModelPool([], budget=1)), files,
                        command=[sys.executable, {str(FAKE_JOB)!r}])
            job = await jobs.start("fit", ["--steps", "1000", "--every", "{every}",
                                           "-o", "models/x"], holds_gpu=True,
                                   output=files.root / "models" / "x")
            while not job.events:
                await asyncio.sleep(0.01)
            print(job.process.pid, flush=True)
            await asyncio.sleep(60)

        asyncio.run(main())
    """)
    studio = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    pid = int(studio.stdout.readline())
    time.sleep(0.2)  # into a step
    studio.kill()
    studio.wait()
    killed = time.monotonic()
    while _alive(pid):
        assert time.monotonic() - killed < 5, "the job outlived the Studio by 5 s"
        time.sleep(0.05)
    assert list((tmp_path / "ws" / "models").iterdir()) == []


def _alive(pid: int) -> bool:
    try:
        finished_pid, _ = os.waitpid(pid, os.WNOHANG)  # not our child: raises
    except ChildProcessError:
        pass
    else:
        if finished_pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_jobs_on_disk_are_listed_and_unfinished_ones_marked_interrupted(jobs, gpu, workspace):
    async def scenario():
        return await finished(await fit(jobs, "--steps", "2"))

    done = run(scenario())
    stale = workspace.root / "jobs" / "20260101-000000-fit-abcdef"
    stale.mkdir()
    (stale / "job.json").write_text(json.dumps({"id": stale.name, "state": "running",
                                                "outcome": None, "created": "2026"}))  # fmt: skip
    (workspace.root / "models" / ".lost.states").mkdir()
    workspace.open()  # the Studio starting again
    later = Jobs(gpu, workspace)
    listed = later.list()
    assert [job["id"] for job in listed] == [done.id, stale.name]
    assert listed[1]["outcome"] == "interrupted" and listed[1]["state"] == "ended"
    assert [event["stage"] for event in later.events(done.id)][-1] == "done"
    assert later.get(done.id)["outcome"] == "done"
    with pytest.raises(KeyError):
        later.get("nothing-here")
    assert not (workspace.root / "models" / ".lost.states").exists()
