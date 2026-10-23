"""Settings' workspace folders: listed with their sizes, each cleared whole,
refused (nothing cleared) while something in it is in use."""

import time

import pytest
from fastapi.testclient import TestClient
from test_studio_batches import write_tweets

from fake_backend import write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.jobs import Job

HERE = "http://127.0.0.1:8000"
API = "/studio/api/workspace/folders"
BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}


@pytest.fixture
def studio(tmp_path):
    """A Studio with a downloaded model (``fake``, the default), a converted
    one in the workspace (``conv``) with a fit on it, a set with history, a
    dataset and a batch on it."""
    workspace = tmp_path / "ws"
    made = workspace / "models"
    write_model(tmp_path / "fake")
    write_model(made / "conv")
    write_fitted(made / "conv-fit", made / "conv")
    pool = ModelPool(discover([tmp_path / "fake", made]), budget=10**9, default="fake")
    app = create_studio(pool, workspace=workspace, config=Config(tmp_path / "config.toml"))
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        client.root = workspace
        files = app.state.workspace
        files.save_set("tickets", BODY)
        assert (
            client.post(
                "/studio/api/run", json={"request": BODY, "model": "fake", "set": "tickets"}
            ).status_code
            == 200
        )
        write_tweets(workspace / "data")
        response = client.post("/studio/api/batches", json={"dataset": "tweets", "model": "fake"})
        assert response.status_code == 201, response.text
        deadline = time.monotonic() + 20
        while app.state.batches.running is not None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        yield client


def folders(studio):
    return {entry["folder"]: entry for entry in studio.get(API).json()["folders"]}


def clear(studio, folder):
    return studio.post(f"{API}/{folder}/clear", json={})


def test_the_folders_with_what_they_hold(studio):
    listed = studio.get(API).json()
    assert listed["editable"] is True
    assert [entry["folder"] for entry in listed["folders"]] == [
        "sets", "history", "data", "batches", "models", "jobs", "speed"
    ]  # fmt: skip
    found = folders(studio)
    assert {name: found[name]["count"] for name in found} == {
        "sets": 1, "history": 1, "data": 1, "batches": 1, "models": 2, "jobs": 0, "speed": 1
    }  # fmt: skip
    assert found["models"]["bytes"] > 0
    assert found["sets"]["path"] == str(studio.root / "sets")


def test_each_folder_cleared_as_deleting_each_entry(studio):
    assert clear(studio, "sets").json() == {"folder": "sets", "cleared": 1, "kept": 0}
    assert not studio.app_state.workspace.history_path("tickets").exists()  # with its history
    assert clear(studio, "data").json()["cleared"] == 1
    assert clear(studio, "batches").json()["cleared"] == 1
    assert studio.get("/studio/api/batches").json()["batches"] == []
    assert clear(studio, "speed").json()["cleared"] == 1
    # The models: unloaded first, taken out of the list, the folder emptied.
    studio.post("/studio/api/models/load", json={"model": "conv-fit"})
    (studio.root / "models" / ".leftover.partial").mkdir()
    assert clear(studio, "models").json()["cleared"] == 2
    assert list((studio.root / "models").iterdir()) == []
    assert list(studio.app_state.gpu.pool.specs) == ["fake"]
    found = folders(studio)
    assert all(found[name]["count"] == 0 for name in found)


def test_history_goes_and_the_sets_stay(studio):
    run = studio.post("/studio/api/run", json={"request": BODY, "model": "fake", "set": "tickets"})
    assert run.status_code == 200
    assert clear(studio, "history").json()["cleared"] == 1
    assert studio.get("/studio/api/history/tickets").json() == {"runs": []}
    assert folders(studio)["sets"]["count"] == 1


def stand_in(studio, job_id, state, **details):
    job = Job(
        id=job_id,
        kind="fit",
        title=f"Fit {job_id}",
        argv=["training", "fit", "-m", "conv"],
        holds_gpu=True,
        folder=studio.root / "jobs" / job_id,
        details=details,
    )
    job.state = state
    studio.app_state.jobs.jobs[job_id] = job
    return job


def test_what_is_in_use_is_refused_whole(studio):
    stand_in(studio, "j1", "running", data="tweets")
    response = clear(studio, "data")
    assert response.status_code == 409 and "Fit j1 reads a dataset" in response.text
    response = clear(studio, "models")
    assert response.status_code == 409 and "Fit j1 uses conv" in response.text
    assert folders(studio)["models"]["count"] == 2 and folders(studio)["data"]["count"] == 1
    del studio.app_state.jobs.jobs["j1"]
    studio.app_state.gpu.pool.default = "conv"
    response = clear(studio, "models")
    assert response.status_code == 409 and "default model" in response.text
    assert clear(studio, "nope").status_code == 404


def test_jobs_that_have_not_ended_stay(studio):
    for job_id in ("a", "b"):
        stand_in(studio, job_id, "ended").folder.mkdir(parents=True)
    stand_in(studio, "c", "running")
    assert clear(studio, "jobs").json() == {"folder": "jobs", "cleared": 2, "kept": 1}
    assert list(studio.app_state.jobs.jobs) == ["c"]
    del studio.app_state.jobs.jobs["c"]


def test_only_from_this_mac(tmp_path, studio):
    other = TestClient(studio.app, base_url=HERE, client=("192.168.1.20", 50000))
    other.cookies = studio.cookies
    response = other.post(f"{API}/sets/clear", json={})
    assert response.status_code in (401, 403)
    assert folders(studio)["sets"]["count"] == 1


def test_a_link_out_of_the_workspace_is_not_listed_nor_stops_clear(studio, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"state": "x", "questions": {"q": {"type": "noul"}}}')
    sets = studio.app_state.workspace.root / "sets"
    (sets / "link.json").symlink_to(outside)
    assert "link" not in [s["name"] for s in studio.get("/studio/api/sets").json()["sets"]]
    assert clear(studio, "sets").status_code == 200
    assert outside.is_file()
