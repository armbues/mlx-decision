"""The jobs' API and stream: start a job from names, follow it, stop it.

Most tests run ``tests/fake_job.py`` in the CLI's place; the last one runs
the real CLI (``python -m mlx_decision training fit``) on the fake backend.
"""

import asyncio
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from fake_backend import TEAM, write_model
from mlx_decision.config import Config
from mlx_decision.data.format import Example, write_examples
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio, server
from mlx_decision.studio.routes import jobs as jobs_routes

TESTS = Path(__file__).resolve().parents[1]
FAKE_JOB = TESTS / "fake_job.py"
HERE = "http://127.0.0.1:8000"
BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}, "model": "fake"}


def write_data(path: Path, count: int = 100) -> Path:
    labels = ("tech",) * 8 + ("billing", "sales")
    examples = [
        Example(
            f"e{i}",
            {"state": f"ticket {i}", "questions": {"team": TEAM}},
            {"team": labels[i % len(labels)]},
        )
        for i in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_examples(path, examples)
    return path


def fake_command(jobs, grace: float = 0.6) -> None:
    jobs.command = [sys.executable, str(FAKE_JOB)]
    jobs.stop_grace = jobs.kill_grace = grace


@pytest.fixture
def models_folder(tmp_path):
    folder = tmp_path / "models"
    write_model(folder / "fake")
    write_model(folder / "other")
    return folder


@pytest.fixture
def studio(tmp_path, models_folder):
    """A logged-in client of a Studio with two fake models, a data file named
    ``tickets`` and fake jobs."""
    # "other" is served with a lower input limit, a load option fits take along.
    specs = discover([models_folder / "fake"])
    specs += discover([models_folder / "other"], {"max_input_tokens": 512})
    pool = ModelPool(specs, budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    write_data(tmp_path / "ws" / "data" / "tickets.jsonl")
    fake_command(app.state.jobs)
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        client.workspace = tmp_path / "ws"
        yield client


def start(studio, **body):
    return studio.post("/studio/api/jobs", json=body)


def stop(studio, job_id: str):
    return studio.post(f"/studio/api/jobs/{job_id}/stop", json={})


def wait_for(studio, job_id: str, timeout: float = 15.0) -> dict:
    """The job once it has ended, with its events."""
    deadline = time.monotonic() + timeout
    while True:
        found = studio.get(f"/studio/api/jobs/{job_id}").json()
        if found["job"]["state"] == "ended":
            return found
        assert time.monotonic() < deadline, f"still {found['job']['state']}"
        time.sleep(0.02)


def wait_until(condition, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


# Starting from names


def test_a_fit_from_names_runs_to_its_report(studio, models_folder):
    response = start(studio, kind="fit", model="fake", data="tickets", epochs=2, seed=7,
                     holdout=0.25, method="calibrate", early_stop=False)  # fmt: skip
    assert response.status_code == 202, response.text
    job = response.json()["job"]
    output = studio.workspace / "models" / "fake-tickets"
    assert job["kind"] == "fit" and job["title"] == "Fit fake-tickets"
    assert job["holds_gpu"] is True and job["output"] == str(output.resolve())
    assert job["command"] == [
        "mlx-decision", "training", "fit", "-m", str(models_folder / "fake"),
        "--data", str((studio.workspace / "data" / "tickets.jsonl").resolve()),
        "-o", str(output.resolve()), "--method", "calibrate", "--epochs", "2",
        "--holdout", "0.25", "--seed", "7", "--no-early-stop",
    ]  # fmt: skip
    found = wait_for(studio, job["id"])
    assert found["job"]["outcome"] == "done"
    assert [event["stage"] for event in found["events"]] == ["check", *["work"] * 10, "done"]
    assert output.is_dir()
    listed = studio.get("/studio/api/jobs").json()["jobs"]
    assert [entry["id"] for entry in listed] == [job["id"]]
    assert listed[0]["outcome"] == "done"


def test_defaults_and_the_other_kinds(studio, models_folder, tmp_path):
    ws = studio.workspace
    data = str((ws / "data" / "tickets.jsonl").resolve())
    job = start(studio, kind="fit", model="other", data="tickets", name="mine").json()["job"]
    # The model's input limit goes along; nothing else is set.
    assert job["command"][3:] == [
        "-m", str(models_folder / "other"), "--max-input-tokens", "512",
        "--data", data, "-o", str((ws / "models" / "mine").resolve()),
    ]  # fmt: skip
    wait_for(studio, job["id"])
    job = start(studio, kind="convert", model="fake").json()["job"]
    assert job["title"] == "Convert fake-q8"
    assert job["command"][1:] == [
        "convert", "-m", str(models_folder / "fake"),
        "-o", str((ws / "models" / "fake-q8").resolve()), "--bits", "8",
    ]  # fmt: skip
    wait_for(studio, job["id"])
    job = start(studio, kind="convert", model="fake", bits=4, name="small").json()["job"]
    assert job["command"][-3:] == [str((ws / "models" / "small").resolve()), "--bits", "4"]
    wait_for(studio, job["id"])
    job = start(studio, kind="eval", model="fake", data="tickets", compare="other").json()["job"]
    assert job["title"] == "Evaluate fake against other" and job["output"] is None
    assert job["command"][1:] == [
        "training", "eval", "-m", str(models_folder / "fake"), "--data", data,
        "--compare", str(models_folder / "other"),
    ]  # fmt: skip
    wait_for(studio, job["id"])


def test_a_download_goes_where_the_config_says(studio, tmp_path):
    job = start(studio, kind="download", repo="Org/model-x", bits=4).json()["job"]
    # No models_dir: the CLI puts it into the Hugging Face cache.
    assert job["command"] == ["mlx-decision", "download", "Org/model-x", "--bits", "4"]
    assert job["title"] == "Download Org/model-x (4-bit)"
    assert job["holds_gpu"] is True and job["output"] is None  # it converts as it fetches
    wait_for(studio, job["id"])
    # models_dir set (read at each start): the repo's name in it.
    models_dir = tmp_path / "downloads"
    studio.app_state.settings.config = Config(tmp_path / "config.toml", models_dir=models_dir)
    job = start(studio, kind="download", repo="Org/model-x").json()["job"]
    assert job["command"][1:] == [
        "download",
        "Org/model-x",
        "--local-dir",
        str(models_dir / "model-x"),
    ]
    assert job["output"] == str(models_dir / "model-x")
    wait_for(studio, job["id"])


def test_fields_are_checked_and_paths_refused(studio):
    def refused(status, param, **body):
        response = start(studio, **body)
        assert response.status_code == status, (body, response.text)
        error = response.json()["error"]
        assert error["param"] == param, (body, error)
        return error["message"]

    refused(400, "kind", kind="shell")
    refused(400, "kind", repo="Org/x")
    refused(404, "model", kind="fit", model="/etc", data="tickets")
    refused(404, "model", kind="convert", model="../models/fake")
    refused(400, "model", kind="convert", model=3)
    refused(400, "data", kind="fit", model="fake", data="../../etc/passwd")
    refused(400, "data", kind="fit", model="fake", data="/tmp/x")
    refused(404, "data", kind="fit", model="fake", data="missing")
    refused(400, "name", kind="fit", model="fake", data="tickets", name="../out")
    refused(400, "name", kind="convert", model="fake", name=".hidden")
    refused(404, "compare", kind="eval", model="fake", data="tickets", compare="nope")
    refused(400, "bits", kind="convert", model="fake", bits=3)
    refused(400, "bits", kind="convert", model="fake", bits=True)
    refused(400, "method", kind="fit", model="fake", data="tickets", method="magic")
    refused(400, "epochs", kind="fit", model="fake", data="tickets", epochs=0)
    refused(400, "holdout", kind="fit", model="fake", data="tickets", holdout=0.95)
    refused(400, "seed", kind="fit", model="fake", data="tickets", seed="7")
    refused(400, "early_stop", kind="fit", model="fake", data="tickets", early_stop="no")
    for repo in ("org", "../x", "org/../x", "/abs/path", "org/x/y", "org/x --local-dir=/"):
        refused(400, "repo", kind="download", repo=repo)
    # A model's name that is taken, on disk or in the list.
    (studio.workspace / "models" / "taken").mkdir()
    assert "taken exists" in refused(409, "name", kind="convert", model="fake", name="taken")
    refused(409, "name", kind="convert", model="fake", name="other")
    assert studio.post("/studio/api/jobs", content=b"x", headers={
        "Content-Type": "application/json"}).status_code == 400  # fmt: skip
    assert studio.get("/studio/api/jobs").json() == {"jobs": []}  # nothing started


# One GPU job at a time, Stop


def long_fit(studio, **body):
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "1000", "--every", "0.02"]
    job = start(studio, kind="fit", model="fake", data="tickets", **body).json()["job"]
    wait_until(lambda: jobs.jobs[job["id"]].events)  # it has started work
    return job


def test_a_second_gpu_job_is_refused_naming_the_first(studio):
    fit = long_fit(studio)
    response = start(studio, kind="convert", model="fake")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "gpu_job_running"
    assert error["message"].startswith("Fit fake-tickets is running")
    # A download runs alongside; two of one repo are one too many.
    download = start(studio, kind="download", repo="Org/model-x")
    assert download.status_code == 202
    assert start(studio, kind="download", repo="Org/model-x").status_code == 409
    for job in (fit, download.json()["job"]):
        assert stop(studio, job["id"]).status_code == 200
        wait_for(studio, job["id"])
    assert start(studio, kind="convert", model="fake").status_code == 202


def test_a_quantized_download_holds_the_gpu_and_a_full_size_one_does_not(studio):
    jobs = studio.app_state.jobs
    # A GPU job first: a quantized download is refused naming it, full size runs.
    fit = long_fit(studio)
    refused = start(studio, kind="download", repo="Org/model-x", bits=8)
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "gpu_job_running"
    assert error["message"].startswith("Fit fake-tickets is running")
    full = start(studio, kind="download", repo="Org/model-y")
    assert full.status_code == 202 and full.json()["job"]["holds_gpu"] is False
    for job in (fit, full.json()["job"]):
        stop(studio, job["id"])
        wait_for(studio, job["id"])

    # The quantized download first: it has the GPU, as a conversion would.
    download = start(studio, kind="download", repo="Org/model-x", bits=4).json()["job"]
    assert download["holds_gpu"] is True
    wait_until(lambda: jobs.jobs[download["id"]].events)  # it has started work
    busy = studio.post("/v1/systemone", json=BODY)
    assert busy.status_code == 503 and "Download Org/model-x (4-bit)" in busy.text
    refused = start(studio, kind="convert", model="fake")
    assert refused.status_code == 409
    assert refused.json()["error"]["message"].startswith("Download Org/model-x (4-bit) is running")
    assert start(studio, kind="download", repo="Org/model-z", bits=8).status_code == 409
    full = start(studio, kind="download", repo="Org/model-y")  # full size: alongside
    assert full.status_code == 202
    for job in (download, full.json()["job"]):
        stop(studio, job["id"])
        wait_for(studio, job["id"])
    assert studio.post("/v1/systemone", json=BODY).status_code == 200  # the GPU is back


def test_stop_ends_the_job_and_leaves_no_folder(studio):
    job = long_fit(studio)
    assert studio.post("/v1/systemone", json=BODY).status_code == 503  # the GPU is the job's
    stopped = stop(studio, job["id"])
    assert stopped.status_code == 200 and stopped.json()["job"]["stop_requested"] is True
    found = wait_for(studio, job["id"])
    assert found["job"]["outcome"] == "cancelled"
    assert found["events"][-1]["stage"] == "cancelled"
    models = studio.workspace / "models"
    assert sorted(path.name for path in models.iterdir()) == []
    # Stop again, or on a job from an earlier run: it stays as it ended.
    again = stop(studio, job["id"]).json()["job"]
    assert again["outcome"] == "cancelled"
    assert stop(studio, "20260101-000000-fit-abcdef").status_code == 404
    assert studio.get("/studio/api/jobs/20260101-000000-fit-abcdef").status_code == 404
    assert studio.get("/studio/api/jobs/..").status_code == 404
    assert studio.post("/v1/systemone", json=BODY).status_code == 200  # the GPU is back


def test_stop_while_waiting_for_the_gpu(studio):
    gpu = studio.app_state.gpu
    started = threading.Event()
    release = threading.Event()

    def slow(*args):
        started.set()
        release.wait(10)

    future = gpu._worker.submit(slow)  # a request on the worker
    started.wait(5)
    job = start(studio, kind="convert", model="fake")
    assert job.status_code == 202  # answered while the job waits
    job = job.json()["job"]
    assert job["state"] == "waiting"
    stop(studio, job["id"])
    release.set()
    future.result(5)
    found = wait_for(studio, job["id"])
    assert found["job"]["outcome"] == "cancelled" and found["job"]["started"] is None
    assert found["job"]["message"] == "stopped before it started"


def test_a_job_that_cannot_start_ends_failed(studio):
    studio.app_state.jobs.command = ["/nonexistent/python"]
    job = start(studio, kind="convert", model="fake").json()["job"]
    found = wait_for(studio, job["id"])
    assert found["job"]["outcome"] == "failed"
    assert found["job"]["message"].startswith("could not start")
    assert studio.app_state.gpu.holder is None


# The stream, on a real server


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Live:
    """A Studio served by uvicorn in a thread, with a logged-in client."""

    def __init__(self, tmp_path, models_folder):
        self.port = free_port()
        pool = ModelPool(discover([models_folder]), budget=10**9)
        self.workspace = tmp_path / "ws"
        self.app = create_studio(pool, port=self.port, workspace=self.workspace,
                                 config=Config(tmp_path / "config.toml"))  # fmt: skip
        write_data(self.workspace / "data" / "tickets.jsonl")
        fake_command(self.app.state.jobs)
        self.server = server(self.app, port=self.port, log_level="warning", lifespan="on")
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        wait_until(lambda: self.server.started)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=10)
        response = self.client.get(f"/?token={self.app.state.session.token}")
        assert response.status_code == 303, response.text

    def stream(self, last_event_id: str | None = None):
        headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
        return self.client.stream("GET", "/studio/api/stream", headers=headers)

    def start(self, steps: int = 10, every: float = 0.05) -> dict:
        self.app.state.jobs.command = [
            sys.executable, str(FAKE_JOB), "--steps", str(steps), "--every", str(every)
        ]  # fmt: skip
        response = self.client.post("/studio/api/jobs", json={
            "kind": "fit", "model": "fake", "data": "tickets"})  # fmt: skip
        assert response.status_code == 202, response.text
        return response.json()["job"]

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(15)
        self.client.close()


@pytest.fixture
def live(tmp_path, models_folder):
    live = Live(tmp_path, models_folder)
    yield live
    live.close()


def messages(lines, until=lambda message: False):
    """(id, message) pairs from an SSE response's lines, up to the first that
    ``until`` accepts; status messages are left out."""
    message_id = None
    for line in lines:
        if line.startswith("id: "):
            message_id = line[4:]
        elif line.startswith("data: "):
            message = json.loads(line[6:])
            if message["type"] == "status":
                continue
            yield message_id, message
            if until(message):
                return


def ended(message) -> bool:
    return message["type"] == "job" and message["job"]["state"] == "ended"


def test_the_stream_carries_a_job_and_its_events(live):
    with live.stream() as response:
        lines = response.iter_lines()
        [(_, snapshot)] = list(messages(lines, lambda message: True))
        assert snapshot["type"] == "snapshot" and snapshot["jobs"] == []
        job = live.start()
        seen = [message for _, message in messages(lines, ended)]
    changes = [message["job"] for message in seen if message["type"] == "job"]
    assert all(change["id"] == job["id"] for change in changes)
    assert [change["state"] for change in changes] == ["waiting", "running", "ended"]
    assert changes[-1]["outcome"] == "done" and changes[-1]["last"]["stage"] == "done"
    events = [message for message in seen if message["type"] == "event"]
    assert {message["job"] for message in events} == {job["id"]}
    # Steps 20 ms apart are thinned; the first event, the last step and the
    # last event go out, each with its place among the job's events.
    indices = [message["index"] for message in events]
    assert indices == sorted(set(indices)) and len(indices) < 12
    assert indices[0] == 0 and indices[-2:] == [10, 11]
    assert events[-2]["event"] == {"stage": "work", "step": 10, "total": 10}
    assert events[-1]["event"]["stage"] == "done"
    # The job's change to "ended" comes after its last event.
    assert seen[-1]["type"] == "job"


def test_a_reconnect_replays_what_it_missed(live):
    job = live.start(steps=30, every=0.05)
    with live.stream() as response:
        lines = response.iter_lines()
        first = list(messages(lines, lambda m: m["type"] == "event" and m["index"] >= 3))
    last_id, last_seen = first[-1][0], first[-1][1]["index"]
    time.sleep(0.4)  # events go by while the tab is away
    with live.stream(last_id) as response:
        rest = list(messages(response.iter_lines(), ended))
    assert rest[0][1]["type"] != "snapshot"
    # Exactly what was published after the last id seen, in order.
    hub = live.app.state.hub
    after = int(last_id.rsplit("-", 1)[1])
    published = [item for number, item in hub.recent if number > after]
    assert rest == [item for item in published if item[1]["type"] != "status"][: len(rest)]
    indices = [message["index"] for _, message in rest if message["type"] == "event"]
    assert indices[0] > last_seen and indices[-1] == 31 and indices == sorted(set(indices))
    assert rest[-1][1]["job"]["id"] == job["id"]


def test_a_tab_opened_during_a_job_sees_its_progress_so_far(live):
    job = live.start(steps=40, every=0.05)
    jobs = live.app.state.jobs
    wait_until(lambda: len(jobs.jobs[job["id"]].events) >= 5)
    with live.stream() as response:  # a reload: no id
        lines = response.iter_lines()
        [(_, snapshot)] = list(messages(lines, lambda message: True))
        [running] = snapshot["jobs"]
        assert running["id"] == job["id"] and running["state"] == "running"
        assert running["last"]["stage"] == "work"
        # The page loads the events so far, then takes the stream's from there.
        so_far = live.client.get(f"/studio/api/jobs/{job['id']}").json()["events"]
        streamed = [m for _, m in messages(lines, ended) if m["type"] == "event"]
    assert len(so_far) >= 5
    assert so_far == jobs.events(job["id"])[: len(so_far)]
    later = [m for m in streamed if m["index"] >= len(so_far)]
    steps = [event.get("step") for event in so_far + [m["event"] for m in later]]
    assert steps[: len(so_far)] == [None, *range(1, len(so_far))]
    assert steps[-2:] == [40, None]  # the last step, then done


def test_job_messages_thin_pure_progress():
    from mlx_decision.studio.jobs import Job
    from mlx_decision.studio.routes.jobs import JobMessages

    async def main():
        sent = []
        messages = JobMessages(lambda kind, data: sent.append((kind, data)), interval=0.2)
        job = Job("j", "fit", "Fit x", [], True, Path("."))

        def emit(**event):
            job.events.append(event)
            messages(job, event)

        messages(job, None)
        emit(stage="check", message="reading")
        for step in range(1, 6):  # fast steps: the first goes out, the rest are held
            emit(stage="before", step=step, total=50)
        emit(stage="train", step=1, total=9, metrics={"loss": 0.5})  # loss: never held
        emit(stage="train", step=2, total=9, metrics={"loss": 0.4})
        for step in range(1, 4):
            emit(stage="score", step=step, total=3)  # a new stage, then the last step
        emit(stage="after", step=1, total=5)
        emit(stage="after", step=2, total=5)  # held, then sent by the timer
        await asyncio.sleep(0.35)
        job.state = "ended"
        messages(job, None)
        return sent

    sent = asyncio.run(main())
    kinds = [kind for kind, _ in sent]
    assert kinds[0] == "job" and kinds[-1] == "job"
    events = [(data["index"], data["event"]) for kind, data in sent if kind == "event"]
    assert [index for index, _ in events] == [0, 1, 6, 7, 8, 10, 11, 12]
    assert events[-1][1] == {"stage": "after", "step": 2, "total": 5}  # flushed
    assert [event.get("step") for _, event in events[4:6]] == [1, 3]  # score: first, last


def test_a_model_written_by_a_job_reaches_every_tab(live):
    jobs = live.app.state.jobs
    jobs.command = [
        sys.executable,
        str(FAKE_JOB),
        "--steps",
        "3",
        "--every",
        "0.02",
        "--fake-model",
    ]
    with live.stream() as response:
        lines = response.iter_lines()
        list(messages(lines, lambda message: True))  # the snapshot
        started = live.client.post("/studio/api/jobs", json={"kind": "convert", "model": "fake"})
        assert started.status_code == 202, started.text
        seen = [m for _, m in messages(lines, lambda message: message["type"] == "catalog")]
    assert seen[-1] == {"type": "catalog", "added": ["fake-q8"], "removed": []}
    # After the job's end: the page drops its row, then the model comes.
    ended_at = next(
        i for i, m in enumerate(seen) if m["type"] == "job" and m["job"]["state"] == "ended"
    )
    assert ended_at < len(seen) - 1
    assert "fake-q8" in live.app.state.gpu.pool.specs


# The real command line


def test_a_real_fit_through_the_cli(tmp_path, models_folder, monkeypatch):
    """``python -m mlx_decision training fit --events`` on the fake backend, from
    the request to the report. A sitecustomize registers the fake family in the
    job's process."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        f"import sys\nsys.path.insert(0, {str(TESTS)!r})\nimport fake_backend\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(site))
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    assert app.state.jobs.command == [sys.executable, "-m", "mlx_decision"]
    write_data(tmp_path / "ws" / "data" / "tickets.jsonl")
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        response = client.post("/studio/api/jobs", json={
            "kind": "fit", "model": "fake", "data": "tickets", "seed": 3})  # fmt: skip
        assert response.status_code == 202, response.text
        found = wait_for(client, response.json()["job"]["id"], timeout=60)
        job, events = found["job"], found["events"]
        stderr = tmp_path / "ws" / "jobs" / job["id"] / "stderr.log"
        assert job["outcome"] == "done", (job, stderr.read_text())
        assert job["returncode"] == 0
        stages = [event["stage"] for event in events]
        assert stages[0] == "check" and stages[-1] == "done"
        assert events[-1]["metrics"], events[-1]  # the report
        fitted = tmp_path / "ws" / "models" / "fake-tickets"
        assert (fitted / "mlx_decision.json").is_file()
        # The job's files: its command, its events, its stderr.
        folder = tmp_path / "ws" / "jobs" / job["id"]
        assert json.loads((folder / "job.json").read_text())["outcome"] == "done"
        assert len((folder / "events.jsonl").read_text().splitlines()) == len(events)
        assert (folder / "stderr.log").exists()
        # The page's report: the fitted folder's own, as training fit wrote it.
        report = client.get(f"/studio/api/jobs/{job['id']}/report").json()
        assert report["path"] == str(fitted.resolve())
        assert report["report"]["questions"] == events[-1]["metrics"]
        assert report["report"]["options"]["seed"] == 3
        # Show in Finder for what it wrote; nothing for a job that wrote nothing.
        shown = []
        monkeypatch.setattr(jobs_routes, "reveal", shown.append)
        monkeypatch.setattr(jobs_routes.sys, "platform", "darwin")
        assert client.post(f"/studio/api/jobs/{job['id']}/reveal", json={}).status_code == 204
        assert shown == [fitted.resolve()]
        assert client.get("/studio/api/jobs/nope/report").status_code == 404
        # Evaluate the fit against its base, on the file it was fitted on: the
        # warning, then both models' results as the report.
        deadline = time.monotonic() + 30
        while "fake-tickets" not in client.get("/studio/api/models").text:  # the rescan
            assert time.monotonic() < deadline
            time.sleep(0.1)
        body = {"kind": "eval", "model": "fake-tickets", "compare": "fake", "data": "tickets"}
        planned = client.post("/studio/api/eval/plan", json=body).json()
        assert "fake-tickets was fitted on this very file" in planned["warnings"][0]
        response = client.post("/studio/api/jobs", json=body)
        assert response.status_code == 202, response.text
        found = wait_for(client, response.json()["job"]["id"], timeout=60)
        assert found["job"]["outcome"] == "done", found["job"]
        result = client.get(f"/studio/api/jobs/{found['job']['id']}/report").json()["result"]
        assert [model["name"] for model in result["models"]] == ["fake-tickets", "fake"]
        assert result["data"]["examples"] == 100
        assert set(result["models"][0]["questions"]["team"]) >= {"accuracy", "macro_f1", "ece",
                                                                 "confusion", "labels"}  # fmt: skip


def test_a_report_only_for_a_fit_that_ended_done(tmp_path, models_folder):
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    app.state.jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "1", "--every", "0.01",
                              "--end", "error"]  # fmt: skip
    write_data(tmp_path / "ws" / "data" / "tickets.jsonl")
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        started = client.post("/studio/api/jobs", json={"kind": "fit", "model": "fake",
                                                        "data": "tickets"})  # fmt: skip
        job = wait_for(client, started.json()["job"]["id"])["job"]
        assert job["outcome"] == "error"
        missing = client.get(f"/studio/api/jobs/{job['id']}/report")
        assert missing.status_code == 404 and "has no report" in missing.text
        gone = client.post(f"/studio/api/jobs/{job['id']}/reveal", json={})
        assert gone.status_code == 404 and "left nothing to show" in gone.text
