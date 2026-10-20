"""Evaluate in the Studio: the plan (the data's labelled questions, the
warning for a fitted model fitted on this very file), the job's argv and
warnings, and refusals."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_studio_jobs_api import TEAM

from fake_backend import write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.data.format import Example, write_examples
from mlx_decision.fitted import MARKER_FILE
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.checks import file_sha256

HERE = "http://127.0.0.1:8000"
FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"


def write_data(path: Path, offset: int = 0) -> Path:
    examples = [
        Example(f"e{i}", {"state": f"ticket {offset + i}", "questions": {"team": TEAM}},
                {"team": ("tech", "billing")[i % 2]})
        for i in range(20)
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    write_examples(path, examples)
    return path


def fitted_on(folder: Path, base: Path, data: Path | None) -> Path:
    """A fitted model over ``base`` that records ``data``'s hash as its data's."""
    write_fitted(folder, base)
    marker = json.loads((folder / MARKER_FILE).read_text())
    marker["fit"]["data"] = file_sha256(data) if data else None
    (folder / MARKER_FILE).write_text(json.dumps(marker))
    return folder


@pytest.fixture
def studio(tmp_path):
    models = tmp_path / "models"
    write_model(models / "fake")
    data = tmp_path / "ws" / "data"
    train = write_data(data / "train.jsonl")
    write_data(data / "test.jsonl", offset=100)
    fitted_on(models / "fake-fit", models / "fake", train)
    fitted_on(models / "fake-unrecorded", models / "fake", None)
    pool = ModelPool(discover([models]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    app.state.jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "500"]
    app.state.jobs.stop_grace = app.state.jobs.kill_grace = 0.5
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.data = data
        client.models = models
        yield client


def plan(studio, **body):
    return studio.post("/studio/api/eval/plan", json=body)


def test_the_plan_and_the_training_data_warning(studio):
    base = plan(studio, model="fake", data="train").json()
    assert base == {"model": "fake", "compare": None, "data": "train", "examples": 20,
                    "questions": {"team": "choice"}, "warnings": []}  # fmt: skip
    # The fitted model on the file it was fitted on: said; on another file: not.
    same = plan(studio, model="fake-fit", data="train", compare="fake").json()
    assert same["warnings"] == [
        "train includes fake-fit's training data: fake-fit was fitted on this very file "
        "(the same SHA-256), so its scores here count examples it learned from"
    ]
    assert plan(studio, model="fake-fit", data="test", compare="fake").json()["warnings"] == []
    # The comparison is checked too; a fit that records no data hash is not.
    assert len(plan(studio, model="fake", compare="fake-fit", data="train").json()["warnings"]) == 1
    assert plan(studio, model="fake-unrecorded", data="train").json()["warnings"] == []


def test_plan_refusals(studio):
    for body, status, param in [
        ({"model": "nope", "data": "train"}, 404, "model"),
        ({"model": "fake", "data": "train", "compare": "nope"}, 404, "compare"),
        ({"model": "fake", "data": "train", "compare": "fake"}, 400, "compare"),
        ({"model": "fake", "data": "missing"}, 404, "data"),
        ({"model": "fake", "data": "../x"}, 400, "data"),
    ]:
        response = plan(studio, **body)
        assert response.status_code == status, (body, response.text)
        assert response.json()["error"]["param"] == param
    (studio.data / "broken.jsonl").write_text("{broken\n")
    broken = plan(studio, model="fake", data="broken")
    assert broken.status_code == 422 and "line 1" in broken.text


def test_an_eval_job_keeps_its_warnings(studio):
    body = {"kind": "eval", "model": "fake-fit", "compare": "fake", "data": "train"}
    started = studio.post("/studio/api/jobs", json=body)
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    assert job["command"][1:4] == ["training", "eval", "-m"]
    assert job["command"][-4:] == ["--data", str((studio.data / "train.jsonl").resolve()),
                                   "--compare", str(studio.models / "fake")]  # fmt: skip
    assert job["title"] == "Evaluate fake-fit against fake"
    assert job["details"]["warnings"] and "fake-fit was fitted on" in job["details"]["warnings"][0]
    assert job["holds_gpu"] is True
    studio.post(f"/studio/api/jobs/{job['id']}/stop", json={})
    same = studio.post("/studio/api/jobs", json={**body, "compare": "fake-fit"})
    assert same.status_code == 400 and same.json()["error"]["param"] == "compare"
