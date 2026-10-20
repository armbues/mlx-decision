"""Public datasets: the presets listed with their question and splits, and a
``data`` job built from a preset's name, never a path."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.presets import PRESETS
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio

HERE = "http://127.0.0.1:8000"
FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"


@pytest.fixture
def studio(tmp_path):
    write_model(tmp_path / "models" / "fake")
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    app.state.jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "500"]
    app.state.jobs.stop_grace = app.state.jobs.kill_grace = 0.5
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.data = (tmp_path / "ws" / "data").resolve()
        yield client


def get(studio, **body):
    return studio.post("/studio/api/jobs", json={"kind": "data", **body})


def stop(studio, job):
    studio.post(f"/studio/api/jobs/{job['id']}/stop", json={})


def test_the_presets(studio):
    presets = studio.get("/studio/api/datasets").json()["presets"]
    assert [p["name"] for p in presets] == ["trec", "offensive", "sst5", "openbookqa", "anli-r3"]
    trec = presets[0]
    assert trec["title"] == "TREC" and trec["repo"] == "CogComp/trec"
    assert trec["revision"] == PRESETS["trec"].source.revision
    assert trec["question"]["id"] == "answer_type" and trec["question"]["type"] == "choice"
    assert [option for option, _ in trec["question"]["options"]] == [
        "abbreviation", "entity", "description", "human", "location", "numeric",
    ]  # fmt: skip
    assert trec["splits"] == [
        {"name": "train", "rows": 5452, "evaluation": False},
        {"name": "test", "rows": 500, "evaluation": True},
    ]
    assert trec["default_split"] == "train"
    sst5 = presets[2]
    assert sst5["question"]["type"] == "score"
    assert [option for option, _ in sst5["question"]["options"]] == ["0", "1", "2", "3", "4"]
    anli = presets[-1]
    assert anli["default_split"] == "train_r3" and anli["question"]["id"] == "relation"
    offensive = presets[1]
    assert offensive["question"]["type"] == "noul"


def test_a_data_job_from_a_preset(studio):
    started = get(studio, preset="trec")
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    output = studio.data / "trec-train.jsonl"
    assert job["command"] == [
        "mlx-decision", "data", "--preset", "trec", "--split", "train", "-o", str(output),
    ]  # fmt: skip
    assert job["holds_gpu"] is False and job["title"] == "Get trec train"
    assert job["details"]["name"] == "trec-train" and job["output"] == str(output)
    # The same file twice is one too many; another name goes alongside.
    assert get(studio, preset="trec").status_code == 409
    sampled = get(studio, preset="sst5", split="validation", sample=200, seed=7, name="sst5-small")
    assert sampled.status_code == 202, sampled.text
    assert sampled.json()["job"]["command"][2:] == [
        "--preset", "sst5", "--split", "validation", "--sample", "200", "--seed", "7",
        "-o", str(studio.data / "sst5-small.jsonl"),
    ]  # fmt: skip
    assert sampled.json()["job"]["title"] == "Get sst5 validation (200 rows)"
    for each in (job, sampled.json()["job"]):
        stop(studio, each)


def test_refusals(studio):
    (studio.data / "taken.jsonl").write_text("{}\n")
    for body, status, param in [
        ({"preset": "imdb"}, 400, "preset"),
        ({"preset": "/etc/passwd"}, 400, "preset"),
        ({"preset": "trec", "split": "validation"}, 400, "split"),
        ({"preset": "trec", "sample": 0}, 400, "sample"),
        ({"preset": "trec", "sample": 501, "split": "test"}, 400, "sample"),
        ({"preset": "trec", "name": "../x"}, 400, "name"),
        ({"preset": "trec", "name": "taken"}, 409, "name"),
    ]:
        response = get(studio, **body)
        assert response.status_code == status, (body, response.text)
        assert response.json()["error"]["param"] == param
    assert studio.get("/studio/api/jobs").json()["jobs"] == []


def test_a_finished_data_job_tells_every_tab(studio):
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "2", "--every", "0.01"]
    job = get(studio, preset="trec", name="mine").json()["job"]
    deadline = time.monotonic() + 15
    while jobs.jobs[job["id"]].state != "ended":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert jobs.jobs[job["id"]].outcome == "done"
    recent = [message for _, (_, message) in studio.app_state.hub.recent]
    assert {"type": "workspace", "folder": "data", "name": "mine", "action": "saved"} in recent
