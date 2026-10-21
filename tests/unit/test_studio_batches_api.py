"""The batches' API: start, follow over the stream, results with metrics, the
table's pages and filters, export, stop, the rest, again, delete, and the
answers saved as a labelled dataset."""

import asyncio
import csv
import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from test_studio_batches import TOPIC, write_tweets

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.dataset import open_dataset, read_for
from mlx_decision.metrics import metrics, record
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.routes.batches import BatchMessages

HERE = "http://127.0.0.1:8000"
API = "/studio/api/batches"
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def studio(tmp_path):
    write_model(tmp_path / "models" / "fake")
    write_model(tmp_path / "models" / "slow", delay=0.05)
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.data = (tmp_path / "ws" / "data").resolve()
        write_tweets(client.data)
        yield client


def wait(studio, name, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = studio.get(f"{API}/{name}").json()["batch"]
        if info["state"] != "running" and studio.app_state.batches.running is None:
            return info
        time.sleep(0.02)
    raise AssertionError(f"{name} is still running")


def started(studio, **body):
    response = studio.post(API, json={"dataset": "tweets", "model": "fake", **body})
    assert response.status_code == 201, response.text
    return response.json()["batch"]["name"]


def test_a_batch_started_from_the_api_reports_its_metrics(studio):
    name = started(studio, split="train")
    info = wait(studio, name)
    assert (info["state"], info["answered"], info["total"]) == ("done", 12, 12)
    assert studio.get(API).json()["batches"][0]["name"] == name
    body = studio.get(f"{API}/{name}").json()
    topic = body["results"]["questions"]["topic"]
    assert topic["type"] == "choice" and sum(topic["answers"].values()) == 12
    assert list(topic["answers"]) == ["analyst", "fed", "other"]
    # The metrics are those of the answers against the labels, as training eval counts them.
    rows = {r["row"]: r for r in studio.app_state.batches.rows(name)}
    records = [
        record(r["response"]["answers"]["topic"], rows[r["row"]]["labels"]["topic"])
        for r in studio.app_state.batches.results(name)
    ]
    expected = metrics(records)
    assert topic["metrics"]["accuracy"] == round(expected["accuracy"], 4)
    assert topic["metrics"]["macro_f1"] == round(expected["macro_f1"], 4)
    assert topic["metrics"]["ece"] == round(expected["ece"], 4)
    assert topic["wrong"] == sum(r["predicted"] != r["gold"] for r in records)
    assert topic["labels"] == ["analyst", "fed", "other"]
    assert sum(sum(row.values()) for row in topic["confusion"].values()) == 12
    bins = topic["reliability"]
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 12
    # Stream messages: its states and its progress.
    kinds = [item[1]["type"] for _, item in studio.app_state.hub.recent]
    assert "batch_progress" in kinds
    states = [
        item[1]["batch"]["state"]
        for _, item in studio.app_state.hub.recent
        if item[1]["type"] == "batch"
    ]
    assert states[0] == "running" and states[-1] == "done"


def test_the_table_is_paged_and_filtered_on_the_server(studio):
    name = started(studio, split="train")
    wait(studio, name)
    page = studio.get(f"{API}/{name}/rows", params={"offset": 5, "limit": 4}).json()
    assert page["matching"] == 12 and [r["row"] for r in page["rows"]] == [5, 6, 7, 8]
    first = page["rows"][0]
    assert first["id"] == "train/5" and first["labels"] == {"topic": "other"}
    assert set(first["answers"]["topic"]) == {"answer", "confidence"}
    found = studio.get(f"{API}/{name}/rows", params={"search": "TRAIN 1"}).json()
    assert [r["row"] for r in found["rows"]] == [1, 10, 11]
    analysed = studio.get(f"{API}/{name}").json()["results"]["questions"]["topic"]
    for answer, count in analysed["answers"].items():
        some = studio.get(f"{API}/{name}/rows", params={"answer": answer}).json()
        assert some["matching"] == count
    wrong = studio.get(f"{API}/{name}/rows", params={"wrong": True}).json()
    assert wrong["matching"] == analysed["wrong"]
    assert all(r["answers"]["topic"]["answer"] != r["labels"]["topic"] for r in wrong["rows"])
    expected = studio.get(f"{API}/{name}/rows", params={"gold": "fed"}).json()
    assert expected["matching"] == 4 and {r["labels"]["topic"] for r in expected["rows"]} == {"fed"}
    everything = studio.get(f"{API}/{name}/rows", params={"unsure": 1.01}).json()
    assert everything["matching"] == 12
    none = studio.get(f"{API}/{name}/rows", params={"unsure": 0}).json()
    assert none["matching"] == 0
    whole = studio.get(f"{API}/{name}/rows", params={"row": 3}).json()["row"]
    assert whole["state"] == "tweet train 3" and whole["response"]["answers"]["topic"]
    assert whole["request"]["questions"] == {"topic": TOPIC}
    assert studio.get(f"{API}/{name}/rows", params={"question": "nope"}).status_code == 400
    assert studio.get(f"{API}/{name}/rows", params={"row": 99}).status_code == 404


def test_the_export_as_csv_and_json_lines(studio):
    name = started(studio, split="test")
    wait(studio, name)
    response = studio.get(f"{API}/{name}/export", params={"format": "csv"})
    assert response.status_code == 200
    assert f'filename="{name}.csv"' in response.headers["content-disposition"]
    table = list(csv.reader(io.StringIO(response.text)))
    assert table[0] == ["id", "state", "topic", "topic_confidence", "topic_expected", "error"]
    assert len(table) == 7 and table[1][:2] == ["test/0", "tweet test 0"]
    assert table[1][4] == "analyst"
    lines = studio.get(f"{API}/{name}/export", params={"format": "jsonl"}).text.splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert first["id"] == "test/0" and first["model"] == "fake" and "topic" in first["answers"]
    assert studio.get(f"{API}/{name}/export", params={"format": "xml"}).status_code == 400


def test_stop_the_rest_again_and_delete(studio):
    write_tweets(studio.data, "many", count=80)
    name = started(studio, dataset="many", model="slow", split="train")
    deadline = time.time() + 10
    while studio.get(f"{API}/{name}").json()["batch"]["answered"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    other = studio.post(API, json={"dataset": "tweets", "model": "fake"})
    assert other.status_code == 409 and other.json()["error"]["code"] == "batch_running"
    assert [b["name"] for b in studio.get(API).json()["batches"]] == [name]  # nothing left
    assert studio.delete(f"{API}/{name}", headers=JSON).status_code == 409
    assert studio.post(f"{API}/{name}/stop", headers=JSON).status_code == 200
    info = wait(studio, name)
    assert info["state"] == "stopped"
    assert studio.post(f"{API}/{name}/run", headers=JSON).json()["batch"]["state"] == "running"
    assert wait(studio, name, timeout=30)["answered"] == 80
    again = studio.post(f"{API}/{name}/again", headers=JSON)
    assert again.status_code == 201
    second = again.json()["batch"]["name"]
    assert second != name
    studio.post(f"{API}/{second}/stop", headers=JSON)
    wait(studio, second)
    assert studio.delete(f"{API}/{second}", headers=JSON).json() == {"deleted": second}
    assert studio.get(f"{API}/{second}").status_code == 404


def test_what_cannot_start_says_why(studio):
    write_tweets(studio.data, "raw", questions=False)
    missing = studio.post(API, json={"dataset": "nope", "model": "fake"})
    assert missing.status_code == 404
    unasked = studio.post(API, json={"dataset": "raw", "model": "fake"})
    assert unasked.status_code == 400 and unasked.json()["error"]["param"] == "questions"
    model = studio.post(API, json={"dataset": "tweets", "model": "nobody"})
    assert model.status_code == 400 and model.json()["error"]["param"] == "model"
    bad = studio.post(API, json={"dataset": "tweets", "model": "fake", "name": "../x"})
    assert bad.status_code == 400
    assert studio.get(API).json()["batches"] == []


def test_the_answers_saved_as_a_dataset_train_a_fit(studio):
    name = started(studio, split="train")
    wait(studio, name)
    saved = studio.post(f"{API}/{name}/dataset", json={"name": "tweets-by-fake"})
    assert saved.status_code == 201
    assert saved.json()["dataset"] == {
        "name": "tweets-by-fake",
        "examples": 12,
        "left_out": 0,
        "split": "train",
    }
    folder = studio.data / "tweets-by-fake"
    dataset, _ = open_dataset(folder)
    assert dataset.missing() == []
    assert dataset.questions == {"topic": TOPIC}
    # One split, train, from the batch's train split.
    assert dataset.splits == {"train": {"role": "train", "examples": 12, "split": "train"}}
    assert dataset.source["batch"] == name and dataset.source["model"] == "fake"
    answers = {
        r["row"]: r["response"]["answers"]["topic"]["choice"]
        for r in studio.app_state.batches.results(name)
    }
    examples = read_for(folder, "train").examples
    assert [e.labels["topic"] for e in examples] == [answers[i] for i in range(12)]
    assert [e.id for e in examples] == [f"train/{i}" for i in range(12)]
    listed = [item["name"] for item in studio.get("/studio/api/data").json()["files"]]
    assert "tweets-by-fake" in listed
    # Taken, a threshold nothing reaches, a bad threshold.
    again = studio.post(f"{API}/{name}/dataset", json={"name": "tweets-by-fake"})
    assert again.status_code == 409
    strict = studio.post(f"{API}/{name}/dataset", json={"name": "strict", "threshold": 1})
    assert strict.status_code == 400 and strict.json()["error"]["param"] == "threshold"
    assert not (studio.data / "strict").exists()
    assert (
        studio.post(f"{API}/{name}/dataset", json={"name": "x", "threshold": 2}).status_code == 400
    )


def test_a_threshold_keeps_only_the_sure_answers(studio):
    name = started(studio, split="train")
    wait(studio, name)
    confidences = sorted(
        r["response"]["answers"]["topic"]["confidence"]
        for r in studio.app_state.batches.results(name)
    )
    cut = confidences[len(confidences) // 2]
    kept = sum(c >= cut for c in confidences)
    saved = studio.post(f"{API}/{name}/dataset", json={"name": "sure", "threshold": cut}).json()
    assert (saved["dataset"]["examples"], saved["dataset"]["left_out"]) == (kept, 12 - kept)


def test_progress_messages_are_thinned():
    sent = []

    class Fake:
        name = "b"
        info = {"answered": 0, "total": 100, "errors": 0}

        def seconds_left(self):
            return 1.0

    async def go():
        messages = BatchMessages(lambda kind, data: sent.append((kind, dict(data))), interval=0.05)
        batch = Fake()
        for i in range(1, 21):
            batch.info["answered"] = i
            messages(batch, True, {})
        await asyncio.sleep(0.1)
        messages(batch, False, {"name": "b", "state": "done"})

    asyncio.run(go())
    progress = [data["answered"] for kind, data in sent if kind == "batch_progress"]
    assert progress == [1, 20]  # the first at once, the last when its turn came
    assert sent[-1] == ("batch", {"batch": {"name": "b", "state": "done"}})
