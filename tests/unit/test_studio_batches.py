"""Batches in the Studio: a dataset's rows answered by one model on the shared
worker, written to ``batches/NAME/`` as they go; stopped, run on, run again."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.dataset import Dataset, SplitOut, write_dataset
from mlx_decision.data.format import Example
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.batches import BatchError, read_lines

HERE = "http://127.0.0.1:8000"
TOPIC = {
    "type": "choice",
    "instructions": "What is the tweet about?",
    "criteria": {"analyst": "Analyst update", "fed": "Central banks", "other": "Other"},
}
ANSWERS = {"topic": {"column": "label", "map": {"0": "analyst", "1": "fed", "2": "other"}}}
VALUES = {"label": {"values": {v: {"rows": 4} for v in "012"}}}


def write_tweets(data, name="tweets", questions=True, count=12):
    """A dataset folder: train (``count`` rows) and test (6), the label's values
    0, 1, 2; with ``questions``, the question ``topic`` answered by them."""

    def rows(split, n):
        return [
            Example(f"{split}/{i}", {"state": f"tweet {split} {i}"}, {}, {"label": str(i % 3)})
            for i in range(n)
        ]

    dataset = Dataset(
        data / name,
        questions={"topic": TOPIC} if questions else None,
        answers=ANSWERS if questions else {},
        columns=VALUES,
    )
    outs = [
        SplitOut("train", rows("train", count), {"role": "train", "examples": count}),
        SplitOut("test", rows("test", 6), {"role": "evaluate", "examples": 6}),
    ]
    return write_dataset(data / name, dataset, outs)


@pytest.fixture
def models(tmp_path):
    write_model(tmp_path / "models" / "fake")
    write_model(tmp_path / "models" / "slow", delay=0.05)
    return tmp_path / "models"


@pytest.fixture
def studio(tmp_path, models):
    pool = ModelPool(discover([models]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.batches = app.state.batches
        client.data = (tmp_path / "ws" / "data").resolve()
        write_tweets(client.data)
        yield client


def start(studio, name):
    return studio.portal.call(studio.batches.start, name)


def stop(studio, name):
    return studio.portal.call(studio.batches.stop, name)


def wait(studio, name, states=("done", "stopped", "failed"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = studio.batches.summary(name)
        if info["state"] in states and studio.batches.running is None:
            return info
        time.sleep(0.02)
    raise AssertionError(f"{name} is still {studio.batches.summary(name)['state']}")


def answered_rows(studio, name):
    return [item["row"] for item in studio.batches.results(name)]


# A whole split, its files


def test_a_batch_answers_every_row_of_a_split(studio):
    folder = studio.batches.prepare("tweets", "train", "fake")
    name = folder.name
    assert name == "tweets-fake"
    assert studio.batches.summary(name)["state"] == "ready"
    start(studio, name)
    info = wait(studio, name)
    assert info["state"] == "done"
    assert (info["answered"], info["total"], info["errors"]) == (12, 12, 0)
    assert info["labelled"] == ["topic"]
    assert len(info["passes"]) == 1 and info["passes"][0]["ended"]
    results = studio.batches.results(name)
    assert [item["row"] for item in results] == list(range(12))
    first = results[0]
    assert set(first["response"]["answers"]) == {"topic"}
    assert first["input_tokens"] > 0 and first["latency_ms"] >= 0
    rows = studio.batches.rows(name)
    assert rows[1] == {
        "row": 1,
        "id": "train/1",
        "state": "tweet train 1",
        "labels": {"topic": "fed"},
    }
    assert studio.batches.questions(name) == {"topic": TOPIC}
    # The run's speed is recorded for the model, as the Playground's.
    fingerprint = studio.app_state.batches.models.fingerprint("fake")
    deadline = time.time() + 5
    while time.time() < deadline and not studio.app_state.workspace.read_speed(fingerprint):
        time.sleep(0.02)
    assert {r["source"] for r in studio.app_state.workspace.read_speed(fingerprint)} == {"batch"}


def test_the_terminal_answers_the_same_rows(studio):
    """``run -q questions.json --states rows.jsonl`` reads the batch's files as
    requests: the fields a request does not know are ignored."""
    from mlx_decision.types import parse_request

    folder = studio.batches.prepare("tweets", "test", "fake")
    questions = json.loads((folder / "questions.json").read_text())
    for row in read_lines(folder / "rows.jsonl"):
        parse_request({**questions, **row})


def test_a_sample_is_drawn_by_its_seed_in_the_files_order(studio):
    first = studio.batches.prepare("tweets", "train", "fake", sample=5, seed=3)
    again = studio.batches.prepare("tweets", "train", "fake", sample=5, seed=3)
    other = studio.batches.prepare("tweets", "train", "fake", sample=5, seed=4)
    rows = [r["row"] for r in read_lines(first / "rows.jsonl")]
    assert rows == sorted(rows) and len(rows) == 5
    assert rows == [r["row"] for r in read_lines(again / "rows.jsonl")]
    assert rows != [r["row"] for r in read_lines(other / "rows.jsonl")]
    assert (again.name, other.name) == ("tweets-fake-2", "tweets-fake-3")
    info = studio.batches.summary(first.name)
    assert (info["total"], info["sample"], info["seed"], info["rows_in_split"]) == (5, 5, 3, 12)
    # A sample as large as the split is the split.
    whole = studio.batches.summary(
        studio.batches.prepare("tweets", "train", "fake", sample=50).name
    )
    assert (whole["total"], whole["sample"], whole["seed"]) == (12, None, None)


# Questions


def test_a_question_sets_questions_on_a_dataset_without_any(studio):
    write_tweets(studio.data, "raw", questions=False)
    with pytest.raises(BatchError, match="no questions yet"):
        studio.batches.prepare("raw", "train", "fake")
    files = studio.app_state.workspace
    files.save_set("topics", {"state": "", "questions": {"topic": TOPIC}})
    name = studio.batches.prepare("raw", "train", "fake", questions_from="topics").name
    info = studio.batches.summary(name)
    assert (info["questions_from"], info["labelled"]) == ("topics", [])
    assert all(row["labels"] == {} for row in studio.batches.rows(name))
    start(studio, name)
    assert wait(studio, name)["answered"] == 12


def test_a_sets_question_keeps_the_labels_only_when_asked_the_same_way(studio):
    files = studio.app_state.workspace
    reworded = {**TOPIC, "instructions": "Which topic?"}
    other = {**TOPIC, "criteria": {"analyst": "A", "fed": "F"}}
    files.save_set("same", {"state": "", "questions": {"topic": reworded}})
    files.save_set("fewer", {"state": "", "questions": {"topic": other}})
    files.save_set("renamed", {"state": "", "questions": {"subject": TOPIC}})
    same = studio.batches.prepare("tweets", "test", "fake", questions_from="same")
    assert studio.batches.summary(same.name)["labelled"] == ["topic"]
    assert studio.batches.rows(same.name)[0]["labels"] == {"topic": "analyst"}
    for name in ("fewer", "renamed"):
        folder = studio.batches.prepare("tweets", "test", "fake", questions_from=name)
        assert studio.batches.summary(folder.name)["labelled"] == []


def test_what_cannot_run_is_refused_before_a_folder_is_made(studio):
    with pytest.raises(BatchError, match="no split 'dev'"):
        studio.batches.prepare("tweets", "dev", "fake")
    with pytest.raises(BatchError) as error:
        studio.batches.prepare("tweets", "train", "nothing")
    assert error.value.param == "model"
    with pytest.raises(BatchError, match="at least one row"):
        studio.batches.prepare("tweets", "train", "fake", sample=0)
    assert studio.batches.list() == []


# Stop, the rest, again


def test_stop_keeps_what_was_answered_and_the_rest_runs_on(studio):
    write_tweets(studio.data, "many", count=60)
    name = studio.batches.prepare("many", "train", "slow").name
    start(studio, name)
    deadline = time.time() + 10
    while studio.batches.summary(name)["answered"] < 3 and time.time() < deadline:
        time.sleep(0.01)
    stop(studio, name)
    info = wait(studio, name)
    assert info["state"] == "stopped"
    assert 3 <= info["answered"] < 60
    assert len(answered_rows(studio, name)) == info["answered"]
    start(studio, name)  # Run the Rest
    info = wait(studio, name, timeout=30)
    assert (info["state"], info["answered"], len(info["passes"])) == ("done", 60, 2)
    assert answered_rows(studio, name) == list(range(60))
    with pytest.raises(BatchError, match="answered every row"):
        start(studio, name)


def test_a_torn_last_line_is_answered_again(studio):
    name = studio.batches.prepare("tweets", "test", "fake").name
    results = studio.app_state.workspace.root / "batches" / name / "results.jsonl"
    results.write_text('{"row":0,"response":{}}\n{"row":1,"respo')
    assert answered_rows(studio, name) == [0]
    start(studio, name)
    wait(studio, name)
    assert answered_rows(studio, name) == list(range(6))


def test_run_again_makes_a_new_batch_of_the_same_rows(studio):
    first = studio.batches.prepare("tweets", "train", "fake", sample=4, seed=1).name
    start(studio, first)
    wait(studio, first)
    again = studio.batches.again(first).name
    assert again == "tweets-fake-2"
    assert studio.batches.rows(again) == studio.batches.rows(first)
    info = studio.batches.summary(again)
    assert (info["state"], info["answered"], info["total"], info["sample"]) == ("ready", 0, 4, 4)
    assert studio.batches.summary(first)["answered"] == 4  # the earlier one stays


def test_a_row_the_model_refuses_is_kept_as_an_error_and_the_rest_go_on(studio):
    name = studio.batches.prepare("tweets", "test", "fake").name
    rows = studio.app_state.workspace.root / "batches" / name / "rows.jsonl"
    lines = rows.read_text().splitlines()
    second = json.loads(lines[1])
    second["questions"] = {"topic": {"type": "choice", "criteria": {}}}  # no options
    lines[1] = json.dumps(second)
    rows.write_text("\n".join(lines) + "\n")
    start(studio, name)
    info = wait(studio, name)
    assert (info["state"], info["answered"], info["errors"]) == ("done", 6, 1)
    failed = studio.batches.results(name)[1]
    assert failed["row"] == 1 and failed["error"]["param"].startswith("questions")


# The worker is shared


def test_the_playground_answers_between_two_rows(studio):
    write_tweets(studio.data, "many", count=60)
    name = studio.batches.prepare("many", "train", "slow").name
    start(studio, name)
    request = {"state": "s", "questions": {"topic": TOPIC}}
    response = studio.post("/studio/api/run", json={"request": request, "model": "fake"})
    assert response.status_code == 200
    assert studio.batches.summary(name)["state"] == "running"
    stop(studio, name)
    wait(studio, name)


def test_one_batch_at_a_time_and_none_while_the_gpu_is_held(studio):
    write_tweets(studio.data, "many", count=60)
    first = studio.batches.prepare("many", "train", "slow").name
    second = studio.batches.prepare("tweets", "test", "fake").name
    start(studio, first)
    with pytest.raises(BatchError) as error:
        start(studio, second)
    assert (error.value.status, error.value.code) == (409, "batch_running")
    assert first in str(error.value)
    with pytest.raises(BatchError, match="is running"):
        studio.batches.delete(first)
    stop(studio, first)
    wait(studio, first)
    studio.app_state.gpu.holder = "Fit d3-edge"
    try:
        with pytest.raises(BatchError) as error:
            start(studio, second)
        assert error.value.code == "gpu_job_running" and "Fit d3-edge" in str(error.value)
    finally:
        studio.app_state.gpu.holder = None


# The Studio stops


def test_the_studio_stops_a_running_batch_and_a_restart_says_so(tmp_path, models):
    pool = ModelPool(discover([models]), budget=10**9)
    config = Config(tmp_path / "config.toml")
    app = create_studio(pool, workspace=tmp_path / "ws", config=config)
    write_tweets((tmp_path / "ws" / "data").resolve(), "many", count=200)
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        name = app.state.batches.prepare("many", "train", "slow").name
        client.portal.call(app.state.batches.start, name)
        time.sleep(0.2)
    info = app.state.batches.summary(name)
    assert info["state"] == "stopped" and 0 < info["answered"] < 200
    assert len(app.state.batches.results(name)) == info["answered"]

    # Killed instead: batch.json still says running; the next Studio says stopped.
    folder = tmp_path / "ws" / "batches" / name
    killed = {**info, "state": "running"}
    (folder / "batch.json").write_text(json.dumps(killed))
    restarted = create_studio(
        ModelPool(discover([models]), budget=10**9), workspace=tmp_path / "ws", config=config
    )
    info = restarted.state.batches.summary(name)
    assert info["state"] == "stopped" and "Studio stopped" in info["message"]


def test_delete_removes_the_folder(studio):
    name = studio.batches.prepare("tweets", "test", "fake").name
    assert [item["name"] for item in studio.batches.list()] == [name]
    studio.batches.delete(name)
    assert studio.batches.list() == []
    assert not (studio.app_state.workspace.root / "batches" / name).exists()


# The pool and the jobs while a batch runs


@pytest.fixture
def tight(tmp_path):
    """A Studio whose budget holds one of two models of 1,000 bytes."""
    from fake_backend import MARKER_FILE, write_sized_model

    models = tmp_path / "tight"
    write_sized_model(models / "slow", 1000)
    marker = models / "slow" / MARKER_FILE
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "delay": 0.05}))
    write_sized_model(models / "other", 1000)
    pool = ModelPool(discover([models]), budget=1100)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.batches = app.state.batches
        write_tweets((tmp_path / "ws" / "data").resolve(), "many", count=200)
        yield client


def test_a_model_that_needs_the_batchs_memory_waits_for_it(tight):
    name = tight.batches.prepare("many", "train", "slow").name
    start(tight, name)
    deadline = time.time() + 10
    while tight.batches.summary(name)["answered"] < 1 and time.time() < deadline:
        time.sleep(0.01)
    request = {"state": "s", "questions": {"topic": TOPIC}}
    response = tight.post("/studio/api/run", json={"request": request, "model": "other"})
    assert response.status_code == 503
    assert f"batch {name}" in response.json()["error"]["message"]
    assert response.headers["Retry-After"]
    unload = tight.post("/studio/api/models/unload", json={"model": "slow"})
    assert unload.status_code == 503 and f"batch {name}" in unload.json()["error"]["message"]
    assert tight.batches.summary(name)["state"] == "running"
    stop(tight, name)
    wait(tight, name)
    # Afterwards the pin is gone: the other model loads in its place.
    response = tight.post("/studio/api/run", json={"request": request, "model": "other"})
    assert response.status_code == 200
    assert tight.app_state.gpu.pool.loaded() == ["other"]


def test_a_gpu_job_is_refused_while_a_batch_runs_and_the_other_way(studio):
    from types import SimpleNamespace

    from mlx_decision.studio.jobs import JobBlocked

    write_tweets(studio.data, "many", count=200)
    name = studio.batches.prepare("many", "train", "slow").name
    start(studio, name)
    jobs = studio.app_state.jobs
    with pytest.raises(JobBlocked, match=f"batch {name} is running on slow"):
        jobs.submit("convert", ["convert"], holds_gpu=True, title="Convert slow")
    response = studio.post("/studio/api/jobs", json={"kind": "convert", "model": "fake"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "batch_running"
    assert jobs.list() == []  # nothing was made
    stop(studio, name)
    wait(studio, name)

    jobs.gpu_job = SimpleNamespace(title="Fit d3-edge")
    try:
        with pytest.raises(BatchError) as error:
            start(studio, name)
        assert (error.value.status, error.value.code) == (409, "gpu_job_running")
        assert "Fit d3-edge" in str(error.value)
    finally:
        jobs.gpu_job = None
