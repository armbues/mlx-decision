"""Public datasets: the presets listed with their question and splits, and a
``data`` job built from a preset's name, never a path."""

import json
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


def splits(*entries):
    """``("train", "train")`` or ``("test", "report", 100)``: split, role, sample."""
    return [{"split": e[0], "role": e[1], **({"sample": e[2]} if len(e) > 2 else {})}
            for e in entries]  # fmt: skip


def test_a_data_job_from_a_preset(studio):
    started = get(studio, preset="trec", splits=splits(("train", "train"), ("test", "evaluate")))
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    train, test = studio.data / "trec-train.jsonl", studio.data / "trec-test.jsonl"
    assert job["command"] == [
        "mlx-decision", "data", "--preset", "trec", "--split", "train", "--split", "test",
        "-o", str(studio.data / "trec-{split}.jsonl"), "--record", "--roles", "train,evaluate",
    ]  # fmt: skip
    assert job["holds_gpu"] is False and job["title"] == "Get trec train, test"
    assert job["details"]["name"] == "trec" and job["details"]["files"] == [
        "trec-train",
        "trec-test",
    ]
    assert job["output"] == str(train)
    assert job["outputs"] == [str(train), str(test), str(studio.data / "trec-train.source.json"),
                              str(studio.data / "trec-test.source.json")]  # fmt: skip
    # A file a running Get writes is taken; another name goes alongside.
    again = get(studio, preset="trec", splits=splits(("test", "train")))
    assert again.status_code == 409 and "Get trec train, test is writing trec-test" in again.text
    sampled = get(studio, preset="sst5", name="small", seed=7,
                  splits=splits(("train", "train", 200), ("validation", "report")))  # fmt: skip
    assert sampled.status_code == 202, sampled.text
    assert sampled.json()["job"]["command"][2:] == [
        "--preset", "sst5", "--split", "train", "--split", "validation", "--sample", "200",
        "--sample", "0", "--seed", "7", "-o", str(studio.data / "small-{split}.jsonl"),
        "--record", "--roles", "train,report",
    ]  # fmt: skip
    assert sampled.json()["job"]["title"] == "Get sst5 train (200 rows), validation"
    assert sampled.json()["job"]["details"]["seed"] == 7
    for each in (job, sampled.json()["job"]):
        stop(studio, each)


def test_refusals(studio):
    (studio.data / "taken-train.jsonl").write_text("{}\n")
    train = splits(("train", "train"))
    for body, status, param in [
        ({"preset": "imdb", "splits": train}, 400, "preset"),
        ({"preset": "/etc/passwd", "splits": train}, 400, "preset"),
        ({"preset": "trec"}, 400, "splits"),
        ({"preset": "trec", "splits": []}, 400, "splits"),
        ({"preset": "trec", "splits": splits(("validation", "train"))}, 400, "splits"),
        ({"preset": "trec", "splits": splits(("../x", "train"))}, 400, "splits"),
        (
            {"preset": "trec", "splits": splits(("train", "train"), ("train", "report"))},
            400,
            "splits",
        ),  # fmt: skip
        (
            {"preset": "trec", "splits": splits(("train", "train"), ("test", "train"))},
            400,
            "splits",
        ),  # fmt: skip
        ({"preset": "trec", "splits": splits(("train", "fit"))}, 400, "splits"),
        ({"preset": "trec", "splits": splits(("train", "train", 0))}, 400, "splits"),
        ({"preset": "trec", "splits": splits(("test", "train", 501))}, 400, "splits"),
        ({"preset": "trec", "splits": ["train"]}, 400, "splits"),
        ({"preset": "trec", "splits": train, "name": "../x"}, 400, "name"),
        ({"preset": "trec", "splits": train, "name": "taken"}, 409, "name"),
    ]:
        response = get(studio, **body)
        assert response.status_code == status, (body, response.text)
        assert response.json()["error"]["param"] == param, (body, response.text)
    assert studio.get("/studio/api/jobs").json()["jobs"] == []


def test_a_finished_data_job_tells_every_tab(studio):
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(FAKE_JOB), "--steps", "2", "--every", "0.01"]
    body = {"preset": "trec", "name": "mine",
            "splits": splits(("train", "train"), ("test", "report"))}  # fmt: skip
    job = get(studio, **body).json()["job"]
    deadline = time.monotonic() + 15
    while jobs.jobs[job["id"]].state != "ended":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert jobs.jobs[job["id"]].outcome == "done"
    recent = [message for _, (_, message) in studio.app_state.hub.recent]
    for name in ("mine-train", "mine-test"):
        assert {"type": "workspace", "folder": "data", "name": name, "action": "saved"} in recent
    # The list has each file's record; deleting a file deletes its record.
    listed = {item["name"]: item for item in studio.get("/studio/api/data").json()["files"]}
    assert listed["mine-test"]["record"]["role"] == "report"
    assert listed["mine-test"]["record"]["get"]["name"] == "mine"
    assert studio.get("/studio/api/data/mine-train").json()["record"]["role"] == "train"
    assert studio.request("DELETE", "/studio/api/data/mine-train", json={}).status_code == 204
    assert sorted(p.name for p in studio.data.iterdir()) == ["mine-test.jsonl",
                                                            "mine-test.source.json"]  # fmt: skip


def test_a_stopped_get_leaves_no_file(studio):
    body = {"preset": "trec", "splits": splits(("train", "train"), ("test", "report"))}
    job = get(studio, **body).json()["job"]
    jobs = studio.app_state.jobs
    deadline = time.monotonic() + 15
    while not list(studio.data.glob(".*.partial")):  # it has begun writing
        assert time.monotonic() < deadline
        time.sleep(0.02)
    stop(studio, job)
    while jobs.jobs[job["id"]].state != "ended":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert list(studio.data.iterdir()) == []


# Any dataset, with a question set's question


@pytest.fixture
def hub(monkeypatch):
    """A Hub stand-in: org/reviews has the configs en and de, each with the
    splits train and test; a split is 12 rows."""
    import datasets

    from mlx_decision.data import sources

    features = datasets.Features({
        "review": datasets.Value("string"),
        "sentiment": datasets.ClassLabel(names=["negative", "positive"]),
        "stars": datasets.Value("int64"),
        "note": datasets.Value("string"),
    })  # fmt: skip
    rows = {
        "review": [f"review {i}" for i in range(12)],
        "sentiment": [i % 2 for i in range(12)],
        "stars": [1 + i % 5 for i in range(12)],
        "note": [f"free text {i}" for i in range(12)],
    }

    def configs(repo, revision=None):
        if repo != "org/reviews":
            raise sources.SourceError(f"hf:{repo} cannot be read: Dataset '{repo}' doesn't exist")
        return ["en", "de"]

    def splits(repo, config=None, revision=None):
        return ["train", "test"]

    def load(repo, config=None, split=None, revision=None):
        dataset = datasets.Dataset.from_dict(rows, features=features)
        return sources._from_dataset(dataset, "reviews", split, {"repo": repo})

    monkeypatch.setattr(sources, "hub_configs", configs)
    monkeypatch.setattr(sources, "hub_splits", splits)
    monkeypatch.setattr(sources, "load_hub", load)


def preview(studio, **body):
    return studio.post("/studio/api/datasets/preview", json={"source": "hf:org/reviews", **body})


def test_the_preview_asks_for_config_and_split_then_reads(studio, hub):
    first = preview(studio).json()
    assert first == {"source": "hf:org/reviews", "configs": ["en", "de"], "config": None,
                     "splits": None, "split": None}  # fmt: skip
    second = preview(studio, config="en").json()
    assert second["splits"] == ["train", "test"] and second["split"] is None
    body = preview(studio, config="en", split="train").json()
    assert body["rows"] == 12
    assert [c["name"] for c in body["columns"]] == ["review", "sentiment", "stars", "note"]
    assert body["columns"][1] == {
        "name": "sentiment",
        "kind": "label",
        "names": ["negative", "positive"],
    }
    assert body["text_columns"] == ["review", "note"]
    labels = {column["name"]: column["classes"] for column in body["label_columns"]}
    assert labels["sentiment"] == [["negative", 6], ["positive", 6]]
    assert labels["stars"] == [["1", 3], ["2", 3], ["3", 2], ["4", 2], ["5", 2]]
    assert "review" not in labels  # 12 different texts are not classes
    assert body["text"] == "review" and body["label"] == "sentiment"
    assert len(body["sample"]) == 5 and body["sample"][0]["review"] == "review 0"
    assert [row["sentiment"] for row in body["sample"][:2]] == ["negative", "positive"]


def test_the_preview_refuses(studio, hub):
    for source in ("org/reviews", "hf:org", "hf:../x", "/etc/passwd", "hf:org/x y"):
        response = studio.post("/studio/api/datasets/preview", json={"source": source})
        assert response.status_code == 400, source
    missing = studio.post("/studio/api/datasets/preview", json={"source": "hf:org/none"})
    assert missing.status_code == 404 and "doesn't exist" in missing.text
    assert preview(studio, config="fr").status_code == 422
    assert preview(studio, config="en", split="dev").status_code == 422


QUESTIONS = {
    "mood": {"type": "choice", "instructions": "How does the reviewer feel?",
             "criteria": {"unhappy": "Not happy", "happy": "Happy"}},
    "angry": {"type": "noul", "instructions": "Is the reviewer angry?"},
    "level": {"type": "score", "criteria": ["low", "high"]},
}  # fmt: skip


def any_dataset(studio, **body):
    files = studio.app_state.workspace
    if not files.set_path("reviews").exists():
        files.save_set("reviews", {"state": "", "questions": QUESTIONS})
    fields = {"source": "hf:org/reviews", "config": "en", "text": "review",
              "splits": [{"split": "train", "role": "train"}],
              "label": "sentiment", "set": "reviews", "question": "mood",
              "label_map": {"negative": "unhappy", "positive": "happy"}}  # fmt: skip
    return get(studio, **(fields | body))


def test_a_job_from_any_dataset(studio, hub):
    two = [{"split": "train", "role": "train", "sample": 10, "rows": 12},
           {"split": "test", "role": "report"}]  # fmt: skip
    started = any_dataset(studio, splits=two, seed=2)
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    folder = studio.app_state.jobs.jobs[job["id"]].folder
    assert job["command"][1:] == [
        "data", "hf:org/reviews", "--config", "en", "--shape", "text",
        "--text", "review", "--label", "sentiment", "-q", f"{folder}/question.json",
        "--label-map", "negative=unhappy,positive=happy", "--split", "train", "--split", "test",
        "--sample", "10", "--sample", "0", "--seed", "2",
        "-o", str(studio.data / "reviews-{split}.jsonl"), "--record", "--roles", "train,report",
    ]  # fmt: skip
    assert json.loads((folder / "question.json").read_text()) == {"mood": QUESTIONS["mood"]}
    assert job["title"] == "Get org/reviews train (10 rows), test"
    assert job["details"]["label_map"] == {"negative": "unhappy", "positive": "happy"}
    assert job["details"]["files"] == ["reviews-train", "reviews-test"]
    stop(studio, job)
    # A noul: true and false; one class left out is fine.
    stars = {"1": "true", "5": "false"}
    noul = any_dataset(studio, question="angry", label="stars", label_map=stars, name="angry")
    assert noul.status_code == 202, noul.text
    assert "1=true,5=false" in noul.json()["job"]["command"]
    stop(studio, noul.json()["job"])


def test_a_job_from_any_dataset_refuses(studio, hub):
    for body, status, param in [
        ({"question": "level"}, 400, "question"),  # a score
        ({"question": "nope"}, 404, "question"),
        ({"set": "missing"}, 404, "set"),
        ({"set": "../x"}, 400, "set"),
        ({"label_map": {}}, 400, "label_map"),
        ({"label_map": {"negative": "sad"}}, 400, "label_map"),
        ({"label_map": {"a=b": "happy"}}, 400, "label_map"),
        ({"label": "review"}, 400, "label"),  # the same column as the state
        ({"splits": [{"split": "../x", "role": "train"}]}, 400, "splits"),
        ({"config": "../x"}, 400, "config"),
        ({"source": "hf:../x"}, 400, "source"),
        (
            {"splits": [{"split": "train", "role": "train", "sample": 13, "rows": 12}]},
            400,
            "splits",
        ),  # fmt: skip
    ]:
        response = any_dataset(studio, **body)
        assert response.status_code == status, (body, response.text)
        assert response.json()["error"]["param"] == param, (body, response.text)
