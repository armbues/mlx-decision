"""Convert: what the sheet offers per model, and a real conversion as a job
(``python -m mlx_decision convert`` on the tiny Clef), stopped or to the end."""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from tiny_models import write_clef

TESTS = Path(__file__).resolve().parents[1]
HERE = "http://127.0.0.1:8000"
GB = 10**9

# Every tensor of a conversion slowed down (FAKE_SLOW_TENSORS seconds), in the
# job's own process, as tests/unit/test_download_convert_events.py does it.
SITECUSTOMIZE = """
import os, sys, time
if "FAKE_SLOW_TENSORS" in os.environ:
    import mlx_decision.backbones.qwen3_5.load as load
    real = load.mx.eval
    def slow(*arrays):
        time.sleep(float(os.environ["FAKE_SLOW_TENSORS"]))
        return real(*arrays)
    load.mx = type(sys)("mx_slow")
    load.mx.__dict__.update({**vars(__import__("mlx.core").core), "eval": slow})
"""


@pytest.fixture(scope="module")
def clef(tmp_path_factory):
    return write_clef(tmp_path_factory.mktemp("models") / "tiny-clef")


@pytest.fixture
def studio(tmp_path, clef, monkeypatch):
    """A Studio over the tiny Clef and fake models, running the real CLI for jobs."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    monkeypatch.setenv("PYTHONPATH", str(site))
    write_model(tmp_path / "other" / "fake")
    write_fitted(tmp_path / "other" / "fake-fit", tmp_path / "other" / "fake")
    pool = ModelPool(discover([clef, tmp_path / "other"]), budget=10 * GB)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    assert app.state.jobs.command == [sys.executable, "-m", "mlx_decision"]
    app.state.jobs.stop_grace = app.state.jobs.kill_grace = 3.0
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.workspace = tmp_path / "ws"
        yield client


def models(studio) -> dict[str, dict]:
    return {model["name"]: model for model in studio.get("/studio/api/models").json()["models"]}


def plan(studio, name):
    return studio.get("/studio/api/models/convert", params={"model": name})


def wait_for(studio, job_id, timeout=60.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        job = studio.get(f"/studio/api/jobs/{job_id}").json()["job"]
        if job["state"] == "ended":
            return job
        assert time.monotonic() < deadline, job
        time.sleep(0.05)


def test_what_convert_offers(studio):
    listed = models(studio)
    assert listed["tiny-clef"]["convertible"] is True
    assert listed["tiny-clef"]["convert_reason"] is None
    assert listed["fake"]["convertible"] is False
    assert listed["fake"]["convert_reason"] == "fake models are not stored quantized"
    assert "fitted model" in listed["fake-fit"]["convert_reason"]

    offered = plan(studio, "tiny-clef").json()
    assert offered["convertible"] is True and offered["reason"] is None
    eight, four = offered["options"]
    assert (eight["bits"], four["bits"]) == (8, 4)
    assert offered["source_bytes"] > eight["bytes"] > four["bytes"] > 0
    assert eight["memory_bytes"] == round(eight["bytes"] * 1.1)
    assert eight["fit"] == four["fit"] == "ok"
    refused = plan(studio, "fake").json()
    assert refused == {
        "model": "fake",
        "convertible": False,
        "reason": "fake models are not stored quantized",
        "options": [],
    }
    assert plan(studio, "nope").status_code == 404


def test_a_conversion_runs_as_a_job_and_the_copy_is_listed(studio):
    started = studio.post(
        "/studio/api/jobs", json={"kind": "convert", "model": "tiny-clef", "bits": 4}
    )
    assert started.status_code == 202, started.text
    job = wait_for(studio, started.json()["job"]["id"])
    assert job["outcome"] == "done", job
    assert job["title"] == "Convert tiny-clef-q4"
    copy = studio.workspace / "models" / "tiny-clef-q4"
    assert (copy / "config.json").exists()
    deadline = time.monotonic() + 10
    while "tiny-clef-q4" not in models(studio):
        assert time.monotonic() < deadline, "the copy was not listed"
        time.sleep(0.05)
    listed = models(studio)["tiny-clef-q4"]
    assert listed["precision"] == "4-bit" and listed["place"] == "studio"
    assert listed["convertible"] is False and "converted copy" in listed["convert_reason"]
    # The copy answers by name.
    body = {"state": "hi", "questions": {"u": {"type": "noul"}}, "model": "tiny-clef-q4"}
    assert studio.post("/v1/systemone", json=body).status_code == 200


def test_stop_leaves_no_folder(studio, monkeypatch):
    monkeypatch.setenv("FAKE_SLOW_TENSORS", "0.05")
    started = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "tiny-clef"})
    job_id = started.json()["job"]["id"]
    jobs = studio.app_state.jobs
    deadline = time.monotonic() + 30
    while not any(event.get("stage") == "convert" for event in jobs.jobs[job_id].events):
        assert time.monotonic() < deadline, jobs.jobs[job_id].events
        time.sleep(0.05)
    assert studio.post(f"/studio/api/jobs/{job_id}/stop", json={}).status_code == 200
    job = wait_for(studio, job_id)
    assert job["outcome"] == "cancelled", job
    assert job["last"] == {"stage": "cancelled", "message": "nothing was written"}
    assert list((studio.workspace / "models").iterdir()) == []  # no copy, no .partial
    assert "tiny-clef-q8" not in models(studio)
