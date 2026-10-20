"""The fit form's plan: the same methods and estimates the fit wizard's menu
shows for the same data, the splits and labels, refusals, and the argv a fit
job gets per method."""

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.training import plan as kinds_plan
from mlx_decision.training import wizard
from mlx_decision.training.fitting import prepare
from tiny_models import write_clef, write_d3, write_marker

HERE = "http://127.0.0.1:8000"
FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"
TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    write_clef(folder / "tiny-clef")
    write_d3(folder / "tiny-d3")
    write_marker(folder / "tiny-laya", "laya")
    return folder


@pytest.fixture
def studio(tmp_path, models):
    pool = ModelPool(discover([models]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    labels = ["tech"] * 30 + ["billing"] * 12 + ["sales"] * 18
    lines = [
        json.dumps(
            {
                "state": f"ticket {i} " + "word " * (i % 7),
                "questions": {"team": TEAM},
                "labels": {"team": label},
            }
        )  # fmt: skip
        for i, label in enumerate(labels)
    ]
    data = tmp_path / "ws" / "data"
    data.mkdir(parents=True, exist_ok=True)  # the Studio made it
    (data / "tickets.jsonl").write_text("\n".join(lines) + "\n")
    app.state.jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "500"]
    app.state.jobs.stop_grace = app.state.jobs.kill_grace = 0.5
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.data = data / "tickets.jsonl"
        yield client


def plan(studio, model, **extra):
    return studio.post("/studio/api/fit/plan", json={"model": model, "data": "tickets", **extra})


@pytest.mark.parametrize("model", ["tiny-clef", "tiny-d3", "tiny-laya"])
def test_the_estimates_are_the_wizards(studio, models, model):
    planned = plan(studio, model)
    assert planned.status_code == 200, planned.text
    body = planned.json()
    prepared = prepare(models / model, studio.data, holdout=0.2, seed=0)
    menu = wizard._choices(prepared)  # what `training fit` asks in a terminal
    assert [(m["method"], m["head_only"]) for m in body["methods"]] == [key for key, _ in menu]
    for method, (_, text) in zip(body["methods"], menu, strict=True):
        peak = float(re.search(r"peak ([\d.]+) GB", text).group(1))
        assert round(method["peak_bytes"] / 1e9, 1) == peak, text
        kinds = kinds_plan.resolve(prepared.family, method["method"])
        assert method["kinds"] == kinds
        estimate = prepared.estimate(kinds, method["head_only"])
        assert method["peak_bytes"] == estimate.peak_bytes
        assert method["disk_bytes"] == estimate.disk_bytes
        assert method["fits"] == (estimate.peak_bytes <= body["working_set"])
        assert ("more than this Mac's working set" in text) == (not method["fits"])
    main = {"tiny-clef": "head", "tiny-d3": "readout", "tiny-laya": "finetune"}[model]
    assert body["methods"][0]["method"] == main and body["methods"][0]["auto"] is True
    assert body["methods"][-1]["method"] == "calibrate"
    if model == "tiny-clef":
        assert body["methods"][0]["disk_bytes"] > 0  # the stored states
    if model == "tiny-laya":
        assert body["methods"][1] | {} and body["methods"][1]["head_only"] is True
        assert body["methods"][1]["peak_bytes"] <= body["methods"][0]["peak_bytes"]
    assert body["name"] == f"{model}-tickets"


def test_splits_and_labels(studio):
    body = plan(studio, "tiny-laya", holdout=0.25, seed=3).json()
    splits = body["splits"]
    assert splits["examples"] == 60
    assert splits["fit"] + splits["dev"] + splits["holdout"] == 60
    assert splits["holdout"] == 15
    assert sum(body["labels"]["team"].values()) == splits["fit"] + splits["dev"]
    assert set(body["labels"]["team"]) == {"billing", "sales", "tech"}
    other = plan(studio, "tiny-laya", holdout=0.25, seed=4).json()
    assert other["splits"] == splits  # the seed moves examples, not the counts
    for share in (0, 0.01, 0.95):  # as the fit wizard: 0.05 to 0.9
        assert plan(studio, "tiny-laya", holdout=share).status_code == 400


def test_refusals(studio):
    assert plan(studio, "nope").status_code == 404
    assert (
        studio.post(
            "/studio/api/fit/plan", json={"model": "tiny-laya", "data": "missing"}
        ).status_code
        == 404
    )
    assert (
        studio.post("/studio/api/fit/plan", json={"model": "tiny-laya", "data": "../x"}).status_code
        == 400
    )
    assert plan(studio, "tiny-laya", holdout=0.95).status_code == 400
    studio.data.write_text("{broken\n")
    broken = plan(studio, "tiny-laya")
    assert broken.status_code == 422 and "line 1" in broken.text


@pytest.mark.parametrize(
    "model, body, flags",
    [
        ("tiny-clef", {"method": "head", "epochs": 2}, ["--method", "head", "--epochs", "2"]),
        ("tiny-d3", {"method": "readout"}, ["--method", "readout"]),
        ("tiny-laya", {"method": "finetune", "head_only": True},
         ["--method", "finetune", "--head-only"]),
        ("tiny-laya", {"method": "calibrate", "holdout": 0.3, "seed": 5, "early_stop": False},
         ["--method", "calibrate", "--holdout", "0.3", "--seed", "5", "--no-early-stop"]),
    ],
)  # fmt: skip
def test_the_argv_per_method(studio, models, model, body, flags):
    started = studio.post(
        "/studio/api/jobs", json={"kind": "fit", "model": model, "data": "tickets", **body}
    )
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    output = (studio.data.parent.parent / "models" / f"{model}-tickets").resolve()
    assert job["command"][1:] == [
        "training", "fit", "-m", str(models / model), "--data", str(studio.data.resolve()),
        "-o", str(output), *flags,
    ]  # fmt: skip
    studio.post(f"/studio/api/jobs/{job['id']}/stop", json={})


def test_head_only_goes_with_finetune(studio):
    body = {"kind": "fit", "model": "tiny-clef", "data": "tickets", "method": "head"}
    refused = studio.post("/studio/api/jobs", json={**body, "head_only": True})
    assert refused.status_code == 400 and refused.json()["error"]["param"] == "head_only"
