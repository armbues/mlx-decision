"""Rescans: models a job (or the CLI) writes join the pool; gone ones leave it,
never while loaded."""

import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.catalog import Catalog

FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"
HERE = "http://127.0.0.1:8000"


def body(model: str) -> dict:
    return {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}, "model": model}


@pytest.fixture
def studio(tmp_path):
    """A Studio with two named models (``-m``-like) and fake jobs that write a
    fake model at their output."""
    named = tmp_path / "named"
    write_model(named / "fake")
    write_model(named / "other")
    pool = ModelPool(discover([named]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    jobs = app.state.jobs
    jobs.command = [
        sys.executable,
        str(FAKE_JOB),
        "--steps",
        "3",
        "--every",
        "0.02",
        "--fake-model",
    ]
    jobs.stop_grace = jobs.kill_grace = 0.6
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.workspace = tmp_path / "ws"
        client.named = named
        yield client


def names(studio) -> list[str]:
    return [model["name"] for model in studio.get("/studio/api/models").json()["models"]]


def wait_until(condition, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def rescan(studio) -> dict:
    response = studio.post("/studio/api/models/rescan", json={})
    assert response.status_code == 200, response.text
    return response.json()


def test_a_model_a_job_writes_joins_the_pool(studio):
    job = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "fake"}).json()["job"]
    assert job["details"] == {"model": "fake", "bits": 8, "name": "fake-q8"}
    wait_until(lambda: "fake-q8" in names(studio))
    assert studio.get(f"/studio/api/jobs/{job['id']}").json()["job"]["outcome"] == "done"
    assert names(studio) == ["fake", "other", "fake-q8"]
    listed = {model["name"]: model for model in studio.get("/studio/api/models").json()["models"]}
    assert listed["fake-q8"]["place"] == "studio"
    # It answers at once, by name, through the Jev API.
    assert studio.post("/v1/systemone", json=body("fake-q8")).status_code == 200
    assert rescan(studio) == {"added": [], "removed": []}  # nothing new since


def test_a_stopped_or_failed_job_adds_nothing(studio):
    jobs = studio.app_state.jobs
    jobs.command[2:] = ["--steps", "500", "--every", "0.02", "--fake-model"]
    job = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "fake"}).json()["job"]
    wait_until(lambda: jobs.jobs[job["id"]].events)
    studio.post(f"/studio/api/jobs/{job['id']}/stop", json={})
    wait_until(lambda: jobs.jobs[job["id"]].state == "ended")
    jobs.command[2:] = ["--steps", "2", "--end", "error", "--fake-model"]
    failed = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "other"}).json()
    wait_until(lambda: jobs.jobs[failed["job"]["id"]].state == "ended")
    time.sleep(0.2)  # a rescan would have been started by now
    assert names(studio) == ["fake", "other"]
    assert list((studio.workspace / "models").iterdir()) == []


def test_a_rescan_finds_what_came_and_went_but_keeps_loaded_models(studio):
    models = studio.workspace / "models"
    write_model(models / "made-by-cli")  # e.g. mlx-decision convert -o there
    write_fitted(models / "fake-fit", studio.named / "fake")
    assert rescan(studio) == {"added": ["fake-fit", "made-by-cli"], "removed": []}
    assert studio.post("/v1/systemone", json=body("fake-fit")).status_code == 200

    # Loaded (the fit loads its base too): its folder goes, it stays until unloaded.
    shutil.rmtree(models / "fake-fit")
    shutil.rmtree(models / "made-by-cli")
    assert rescan(studio) == {"added": [], "removed": ["made-by-cli"]}
    assert "fake-fit" in names(studio)
    studio.post("/studio/api/models/unload", json={"model": "fake-fit"})
    assert rescan(studio) == {"added": [], "removed": ["fake-fit"]}
    # A named model the scan does not list stays, whatever the scan says; one
    # whose folder is gone leaves too.
    assert names(studio) == ["fake", "other"]
    shutil.rmtree(studio.named / "other")
    assert rescan(studio) == {"added": [], "removed": ["other"]}
    assert names(studio) == ["fake"]


def test_a_rescan_runs_while_a_job_holds_the_gpu(studio):
    jobs = studio.app_state.jobs
    jobs.command[2:] = ["--steps", "500", "--every", "0.02"]
    job = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "fake"}).json()["job"]
    wait_until(lambda: jobs.jobs[job["id"]].events)
    write_model(studio.workspace / "models" / "new")
    assert rescan(studio) == {"added": ["new"], "removed": []}
    studio.post(f"/studio/api/jobs/{job['id']}/stop", json={})
    wait_until(lambda: jobs.jobs[job["id"]].state == "ended")


def test_a_name_taken_by_another_folder_is_skipped(studio):
    write_model(studio.workspace / "models" / "fake")  # the named fake keeps its name
    assert rescan(studio) == {"added": [], "removed": []}
    assert names(studio) == ["fake", "other"]


def test_the_catalog_survives_a_named_model_that_cannot_be_read(tmp_path):
    write_model(tmp_path / "ws-models" / "mine")
    catalog = Catalog([str(tmp_path / "missing")], tmp_path / "ws-models", include_cache=False)
    skipped = []
    found = catalog.scan(lambda path, reason: skipped.append(reason))
    assert [spec.name for spec in found] == ["mine"]
    assert skipped and skipped[0].startswith(str(tmp_path / "missing"))
    # One that cannot be read does not hide the others named with it.
    write_model(tmp_path / "kept" / "named")
    catalog.references.append(str(tmp_path / "kept"))
    assert sorted(spec.name for spec in catalog.scan()) == ["mine", "named"]


def test_pool_remove(tmp_path):
    for name in ("a", "b", "c"):
        write_model(tmp_path / name)
    pool = ModelPool(discover([tmp_path]), default="a", budget=10**9)
    changes = []
    pool.on_change = lambda: changes.append(pool.names)
    with pytest.raises(ValueError, match="default"):
        pool.remove("a")
    with pytest.raises(KeyError):
        pool.remove("zzz")
    pool.get("b")
    with pytest.raises(ValueError, match="loaded"):
        pool.remove("b")
    pool.remove("c")
    assert pool.names == ["a", "b"] and changes[-1] == ["a", "b"]
    assert "c" not in pool.specs and pool.loaded() == ["b"]
