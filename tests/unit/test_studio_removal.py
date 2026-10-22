"""Deleting models the Studio made, and fit and evaluation entries."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.jobs import Job

TESTS = Path(__file__).resolve().parents[1]
HERE = "http://127.0.0.1:8000"
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def studio(tmp_path):
    """A Studio with a downloaded model (``named``), a converted one in the
    workspace (``conv``) with a fitted model on it (``conv-fit``)."""
    workspace = tmp_path / "ws"
    made = workspace / "models"
    write_model(tmp_path / "named")
    write_model(made / "conv")
    write_fitted(made / "conv-fit", made / "conv")
    pool = ModelPool(discover([tmp_path / "named", made]), budget=10**9, default="named")
    app = create_studio(pool, workspace=workspace, config=Config(tmp_path / "config.toml"))
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        client.made = made
        yield client


def delete(studio, name):
    return studio.delete(f"/studio/api/models/{name}", headers=JSON)


def listed(studio):
    return [model["name"] for model in studio.get("/studio/api/models").json()["models"]]


def test_a_converted_model_after_its_fitted_ones(studio):
    response = delete(studio, "conv")
    assert response.status_code == 409
    assert "fitted models are made from conv (conv-fit)" in response.json()["error"]["message"]
    # A loaded model is unloaded first.
    assert studio.post("/studio/api/models/load", json={"model": "conv-fit"}).status_code == 200
    response = delete(studio, "conv-fit")
    assert response.status_code == 200, response.text
    assert response.json() == {"model": "conv-fit", "path": str(studio.made / "conv-fit")}
    assert not (studio.made / "conv-fit").exists()
    assert [path.name for path in studio.made.iterdir()] == ["conv"]  # nothing left aside
    assert "conv-fit" not in studio.app_state.gpu.pool.specs
    assert "conv-fit" not in listed(studio)
    assert delete(studio, "conv").status_code == 200
    assert listed(studio) == ["named"]


def test_downloaded_unknown_and_default_models_stay(studio, tmp_path):
    response = delete(studio, "named")
    assert response.status_code == 403
    assert "not made by the Studio" in response.json()["error"]["message"]
    assert (tmp_path / "named").is_dir()
    assert delete(studio, "nope").status_code == 404
    studio.app_state.gpu.pool.default = "conv-fit"
    response = delete(studio, "conv-fit")
    assert response.status_code == 409 and "default model" in response.json()["error"]["message"]


def test_a_model_in_use_stays(studio):
    jobs = studio.app_state.jobs
    job = Job(
        id="j1",
        kind="eval",
        title="Evaluate conv-fit",
        argv=["training", "eval", "-m", "conv-fit"],
        holds_gpu=True,
        folder=studio.made.parent / "jobs" / "j1",
    )
    job.state = "running"
    jobs.jobs[job.id] = job
    response = delete(studio, "conv-fit")
    assert response.status_code == 409
    assert response.json()["error"]["message"].startswith("Evaluate conv-fit uses conv-fit")
    del jobs.jobs[job.id]
    # A batch keeps it loaded: busy, naming the batch.
    pool = studio.app_state.gpu.pool
    studio.post("/studio/api/models/load", json={"model": "conv-fit"})
    pool.pin("conv-fit", "batch tweets")
    response = delete(studio, "conv-fit")
    assert response.status_code == 503 and "batch tweets" in response.json()["error"]["message"]
    assert (studio.made / "conv-fit").is_dir() and "conv-fit" in pool.specs
    pool.unpin("conv-fit")


def wait_for(studio, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while (job := studio.get(f"/studio/api/jobs/{job_id}").json()["job"])["state"] != "ended":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return job


def test_a_fit_entry_with_or_without_its_model(studio):
    from test_studio_jobs_api import write_data

    write_data(studio.made.parent / "data" / "tickets.jsonl")
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(TESTS / "fake_job.py"), "--steps", "2", "--fit-events"]
    started = []
    for name in ("one", "two"):
        body = {"kind": "fit", "model": "conv", "data": "tickets", "name": name}
        response = studio.post("/studio/api/jobs", json=body)
        assert response.status_code == 202, response.text
        started.append(wait_for(studio, response.json()["job"]["id"]))
    deadline = time.monotonic() + 10
    while not {"one", "two"} <= set(studio.app_state.gpu.pool.specs):
        assert time.monotonic() < deadline, "the fitted models were not listed"
        time.sleep(0.02)

    one, two = started
    # The entry alone: its model stays.
    response = studio.delete(f"/studio/api/jobs/{one['id']}", headers=JSON)
    assert response.json() == {"job": one["id"], "model": None}
    assert not (studio.made.parent / "jobs" / one["id"]).exists()
    assert "one" in listed(studio)
    # With its model.
    response = studio.delete(f"/studio/api/jobs/{two['id']}?model=true", headers=JSON)
    assert response.json() == {"job": two["id"], "model": "two"}
    assert "two" not in listed(studio) and not (studio.made / "two").exists()
    assert [job["id"] for job in studio.get("/studio/api/jobs").json()["jobs"]] == []
    assert studio.delete(f"/studio/api/jobs/{one['id']}", headers=JSON).status_code == 404


def test_a_job_that_has_not_ended_stays(studio):
    jobs = studio.app_state.jobs
    job = Job(
        id="j2",
        kind="fit",
        title="Fit x",
        argv=["training", "fit"],
        holds_gpu=True,
        folder=studio.made.parent / "jobs" / "j2",
    )
    job.state = "running"
    jobs.jobs[job.id] = job
    response = studio.delete("/studio/api/jobs/j2", headers=JSON)
    assert response.status_code == 409 and "has not ended" in response.json()["error"]["message"]
    del jobs.jobs[job.id]


def test_a_fit_whose_model_was_deleted_on_disk_says_so(studio):
    import shutil

    from test_studio_jobs_api import write_data

    write_data(studio.made.parent / "data" / "tickets.jsonl")
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(TESTS / "fake_job.py"), "--steps", "2", "--fit-events"]
    body = {"kind": "fit", "model": "conv", "data": "tickets", "name": "gone"}
    response = studio.post("/studio/api/jobs", json=body)
    job = wait_for(studio, response.json()["job"]["id"])
    assert studio.get(f"/studio/api/jobs/{job['id']}/report").status_code == 200
    shutil.rmtree(studio.made / "gone")  # in Finder, say
    report = studio.get(f"/studio/api/jobs/{job['id']}/report")
    assert report.status_code == 404
    error = report.json()["error"]
    assert error["code"] == "model_gone"
    assert error["message"] == "The fitted model gone was deleted; its report went with it."
