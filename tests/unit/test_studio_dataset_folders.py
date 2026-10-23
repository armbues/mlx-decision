"""Dataset folders in the Studio: listed and described by split, their rows
browsed before they have questions, questions and answers saved into
``dataset.json``, rows of columns uploaded as a folder, and fits and
evaluations naming a dataset and its split."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.dataset import DATASET_FILE, Dataset, SplitOut, write_dataset
from mlx_decision.data.format import Example
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio

HERE = "http://127.0.0.1:8000"
FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"
TOPIC = {
    "type": "choice",
    "instructions": "What is the tweet about?",
    "criteria": {"analyst": "Analyst update", "fed": "Central banks", "other": "Other"},
}
NAMES = {"0": "Analyst Update", "1": "Fed | Central Banks", "2": "Dividend"}


def write_tweets(data: Path) -> Path:
    """A dataset folder without questions: train (12 rows), validation (6)
    and test (6), the label's values 0, 1 and 2 (named from a card)."""

    def rows(split, count):
        return [
            Example(f"{split}/{i}", {"state": f"tweet {split} {i}"}, {}, {"label": str(i % 3)})
            for i in range(count)
        ]

    columns = {"label": {"values": {v: {"rows": 8, "name": n} for v, n in NAMES.items()}}}
    dataset = Dataset(data / "tweets", columns=columns, source={"source": "hf:org/tweets"})
    outs = [
        SplitOut("train", rows("train", 12), {"role": "train", "examples": 12}),
        SplitOut("validation", rows("validation", 6), {"role": "report", "examples": 6}),
        SplitOut("test", rows("test", 6), {"role": "evaluate", "examples": 6}),
    ]
    return write_dataset(data / "tweets", dataset, outs)


ANSWERS = {"topic": {"column": "label", "map": {"0": "analyst", "1": "fed", "2": "other"}}}


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
        client.folder = write_tweets(client.data)
        yield client


def questions(studio, body, name="tweets"):
    return studio.put(f"/studio/api/data/{name}/questions", json=body)


def test_a_folder_is_listed_with_its_splits_and_what_it_misses(studio):
    [item] = studio.get("/studio/api/data").json()["files"]
    assert item["name"] == "tweets" and item["kind"] == "folder"
    assert [(s["name"], s["role"], s["examples"]) for s in item["splits"]] == [
        ("train", "train", 12), ("validation", "report", 6), ("test", "evaluate", 6)]  # fmt: skip
    assert item["missing"] == ["it has no questions yet"] and item["questions"] == []
    assert item["source"] == {"source": "hf:org/tweets"}


def test_a_split_is_described_and_browsed_before_questions(studio):
    body = studio.get("/studio/api/data/tweets").json()
    assert body["split"] == "train" and body["examples"] == 12  # the train split by default
    assert body["path"] == str(studio.folder) and body["file"] == str(studio.folder / "train.jsonl")
    assert body["questions"] == [] and body["missing"] == ["it has no questions yet"]
    assert body["values"] == {"label": [["0", 4], ["1", 4], ["2", 4]]}
    assert body["columns"]["label"]["values"]["1"]["name"] == "Fed | Central Banks"
    assert body["sample"][0] == {"id": "train/0", "state": "tweet train 0", "labels": {},
                                 "values": {"label": "0"}}  # fmt: skip
    test = studio.get("/studio/api/data/tweets", params={"split": "test"}).json()
    assert test["split"] == "test" and test["examples"] == 6
    assert studio.get("/studio/api/data/tweets", params={"split": "dev"}).status_code == 404
    page = studio.get(
        "/studio/api/data/tweets/examples", params={"column": "label", "value": "2", "limit": 10}
    ).json()
    assert (page["split"], page["total"], page["matched"]) == ("train", 12, 4)
    assert [e["values"] for e in page["examples"]] == [{"label": "2"}] * 4
    assert (
        studio.get("/studio/api/data/tweets/examples", params={"column": "label"}).status_code
        == 400
    )
    whole = studio.get("/studio/api/data/tweets/examples/1", params={"split": "test"}).json()
    assert whole == {"index": 1, "id": "test/1", "state": "tweet test 1", "questions": {},
                     "labels": {}, "values": {"label": "1"}}  # fmt: skip
    # No questions: no question set to make from it yet.
    assert studio.post("/studio/api/data/tweets/set", json={}).status_code == 422


def test_questions_and_answers_are_saved_into_the_folder(studio):
    saved = questions(studio, {"questions": {"topic": TOPIC}, "answers": ANSWERS})
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["missing"] == []
    [topic] = body["questions"]
    assert topic["answered_by"] == "label" and topic["labelled"] == 12
    assert topic["labels"] == [["analyst", 4], ["fed", 4], ["other", 4]]
    written = json.loads((studio.folder / DATASET_FILE).read_text())
    assert list(written)[:3] == ["format", "questions", "answers"]  # first, the rest kept
    assert written["answers"]["topic"]["left_out"] == []
    assert written["columns"]["label"]["values"]["0"]["name"] == "Analyst Update"
    recent = [message for _, (_, message) in studio.app_state.hub.recent]
    assert {"type": "workspace", "folder": "data", "name": "tweets", "action": "saved"} in recent
    # Merged and left out: fed and dividend tweets as other, analyst left out.
    merged = {"topic": {"column": "label", "map": {"1": "other", "2": "other"}, "left_out": ["0"]}}
    body = questions(studio, {"questions": {"topic": TOPIC}, "answers": merged}).json()
    assert body["questions"][0]["labels"] == [["other", 8]] and body["missing"] == []
    # Unmapped values are what a fit misses.
    partial = {"topic": {"column": "label", "map": {"1": "fed"}}}
    body = questions(studio, {"questions": {"topic": TOPIC}, "answers": partial}).json()
    assert body["missing"] == ["topic: label values not mapped to an answer or left out: 0, 2"]
    # A set from it, now that it has questions.
    made = studio.post(
        "/studio/api/data/tweets/set", params={"split": "validation"}, json={}
    ).json()
    assert made == {"name": "tweets", "examples": 6}


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"questions": {"topic": {"type": "choice"}}}, "questions."),
        ({"questions": {"topic": TOPIC}, "answers": {"nope": ANSWERS["topic"]}}, "answers.nope"),
        ({"questions": {"topic": TOPIC},
          "answers": {"topic": {"column": "stars", "map": {}}}}, "answers.topic"),
        ({"questions": {"topic": TOPIC},
          "answers": {"topic": {"column": "label", "map": {"7": "fed"}}}}, "answers.topic.map"),
        ({"questions": {"topic": TOPIC},
          "answers": {"topic": {"column": "label", "map": {"0": "bank"}}}}, "answers.topic.map.0"),
        ({"questions": {"topic": TOPIC}, "answers": {"topic": {
            "column": "label", "map": {"0": "fed"}, "left_out": ["0"]}}}, "answers.topic.left_out"),
        ({"questions": [], "answers": {}}, "questions"),
    ],
)  # fmt: skip
def test_questions_that_do_not_check_are_refused(studio, body, param):
    before = (studio.folder / DATASET_FILE).read_text()
    response = questions(studio, body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"].startswith(param), response.text
    assert (studio.folder / DATASET_FILE).read_text() == before


def test_questions_wait_for_a_job_and_go_into_folders_only(studio):
    jobs = studio.app_state.jobs.jobs
    jobs["fit"] = SimpleNamespace(state="running", details={"data": "tweets"}, title="Fit x")
    try:
        busy = questions(studio, {"questions": {"topic": TOPIC}, "answers": ANSWERS})
        assert busy.status_code == 409 and "Fit x reads tweets" in busy.text
    finally:
        del jobs["fit"]
    plain = {"state": "s", "questions": {"topic": TOPIC}, "labels": {"topic": "fed"}}
    (studio.data / "plain.jsonl").write_text(json.dumps(plain) + "\n")
    assert questions(studio, {"questions": {}}, name="plain").status_code == 400
    assert questions(studio, {"questions": {}}, name="none").status_code == 404


def test_fits_and_evaluations_take_a_dataset_and_its_split(studio):
    def job(**body):
        return studio.post("/studio/api/jobs", json=body)

    # Not ready before its questions: the reason said.
    early = job(kind="fit", model="fake", data="tweets")
    assert early.status_code == 422 and "has no questions yet" in early.text
    plan = studio.post("/studio/api/fit/plan", json={"model": "fake", "data": "tweets"})
    assert plan.status_code == 422
    questions(studio, {"questions": {"topic": TOPIC}, "answers": ANSWERS})

    planned = studio.post("/studio/api/fit/plan", json={
        "model": "fake", "data": "tweets", "report": "tweets"}).json()  # fmt: skip
    assert (planned["split"], planned["report"], planned["report_split"]) == (
        "train", "tweets", "validation")  # fmt: skip
    assert planned["splits"]["holdout"] == 6 and planned["name"] == "fake-tweets"
    body = {"model": "fake", "data": "tweets", "report": "tweets", "report_split": "train"}
    same = studio.post("/studio/api/fit/plan", json=body)
    assert same.status_code == 400

    fit = job(kind="fit", model="fake", data="tweets", report="tweets")
    assert fit.status_code == 202, fit.text
    command = fit.json()["job"]["command"]
    assert command[command.index("--data") + 1] == str(studio.folder / "train.jsonl")
    assert command[command.index("--validation") + 1] == str(studio.folder / "validation.jsonl")
    details = fit.json()["job"]["details"]
    assert (details["data"], details["split"], details["report_split"]) == (
        "tweets", "train", "validation")  # fmt: skip
    studio.post(f"/studio/api/jobs/{fit.json()['job']['id']}/stop", json={})

    evaluated = studio.post("/studio/api/eval/plan", json={"model": "fake", "data": "tweets"})
    assert evaluated.json()["split"] == "test" and evaluated.json()["examples"] == 6
    on_train = studio.post("/studio/api/eval/plan", json={
        "model": "fake", "data": "tweets", "split": "train"}).json()  # fmt: skip
    assert on_train["examples"] == 12
    assert studio.post("/studio/api/eval/plan", json={
        "model": "fake", "data": "tweets", "split": "dev"}).status_code == 404  # fmt: skip


def test_rows_of_columns_are_uploaded_as_a_folder(studio):
    csv = "text,topic\nrates rise,fed\nbuy apple,analyst\nfed holds,fed\nno label,\n"
    described = studio.post("/studio/api/data/columns", json={"text": csv, "format": "csv"})
    assert described.status_code == 200, described.text
    body = described.json()
    assert body["rows"] == 4 and body["text"] == "text"
    assert [c["name"] for c in body["label_columns"]] == ["topic"]
    added = studio.post("/studio/api/data", json={
        "name": "mine", "text": csv, "format": "csv", "text_column": "text",
        "label_column": "topic"})  # fmt: skip
    assert added.status_code == 200, added.text
    body = added.json()
    assert body["kind"] == "folder" and body["examples"] == 3  # the row without a label: left out
    assert body["values"] == {"topic": [["analyst", 1], ["fed", 2]]}
    assert body["missing"] == ["it has no questions yet"]
    assert sorted(p.name for p in (studio.data / "mine").iterdir()) == [DATASET_FILE, "train.jsonl"]
    assert json.loads((studio.data / "mine" / DATASET_FILE).read_text())["format"] == 1
    assert not [p for p in studio.data.iterdir() if p.name.startswith(".")]
    again = studio.post("/studio/api/data", json={
        "name": "mine", "text": csv, "format": "csv", "text_column": "text",
        "label_column": "topic"})  # fmt: skip
    assert again.status_code == 409
    wrong = studio.post(
        "/studio/api/data",
        json={
            "name": "x",
            "text": csv,
            "format": "xml",
            "text_column": "text",
            "label_column": "topic",
        },
    )
    assert wrong.status_code == 400


def test_a_folder_is_deleted_whole(studio):
    assert studio.request("DELETE", "/studio/api/data/tweets", json={}).status_code == 204
    assert list(studio.data.iterdir()) == []
    assert studio.get("/studio/api/data/tweets").status_code == 404


def test_the_document_samples_rows_per_value(studio):
    doc = studio.get("/studio/api/data/tweets/document").json()
    assert (doc["split"], doc["column"], doc["total"]) == ("train", "label", 12)
    assert doc["questions"] == {} and doc["missing"] == ["it has no questions yet"]
    values = [row["values"]["label"] for row in doc["rows"]]
    assert values == ["0", "0", "1", "1", "2", "2"]  # two per value, in the column's order
    again = studio.get("/studio/api/data/tweets/document").json()
    assert [r["id"] for r in again["rows"]] == [r["id"] for r in doc["rows"]]  # the seed repeats
    other = studio.get("/studio/api/data/tweets/document", params={"seed": 5, "per": 4}).json()
    assert len(other["rows"]) == 12
    found = studio.get("/studio/api/data/tweets/document", params={"search": "train 1"}).json()
    assert found["matched"] == 3 and [r["id"] for r in found["rows"]] == [
        "train/1", "train/10", "train/11"]  # fmt: skip
    test = studio.get("/studio/api/data/tweets/document", params={"split": "test"}).json()
    assert test["split"] == "test" and len(test["rows"]) == 6
    assert studio.get("/studio/api/data/tweets/document", params={"per": 0}).status_code == 400
    plain = {"state": "s", "questions": {"topic": TOPIC}, "labels": {"topic": "fed"}}
    (studio.data / "plain.jsonl").write_text(json.dumps(plain) + "\n")
    assert studio.get("/studio/api/data/plain/document").status_code == 400


def test_a_save_names_the_version_it_was_loaded_from(studio):
    loaded = studio.get("/studio/api/data/tweets/document").json()["sha256"]
    body = {"questions": {"topic": TOPIC}, "answers": ANSWERS, "loaded": loaded}
    saved = questions(studio, body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["sha256"] == studio.get("/studio/api/data/tweets/document").json()["sha256"]
    stale = questions(studio, body)  # loaded from the version before
    assert stale.status_code == 409


def test_runs_on_a_dataset_keep_their_own_history(studio):
    request = {"state": "tweet", "questions": {"topic": TOPIC}}
    body = {"request": request, "model": "fake", "set": "data:tweets", "example": "train/0"}
    ran = studio.post("/studio/api/run", json=body)
    assert ran.status_code == 200, ran.text
    history = studio.app_state.workspace.root / "history" / "datasets" / "tweets.jsonl"
    assert len(history.read_text().splitlines()) == 1
    runs = studio.get("/studio/api/history/data:tweets").json()["runs"]
    assert runs[0]["example"] == "train/0"
    assert studio.post("/studio/api/run", json={"request": request, "model": "fake",
                                                "set": "data:../x"}).status_code == 400  # fmt: skip
    # Deleting the dataset deletes its history.
    assert studio.request("DELETE", "/studio/api/data/tweets", json={}).status_code == 204
    assert not history.exists()


def test_rows_fill_in_for_values_with_few_and_go_by_short_names(studio):
    folder = studio.data / "uneven"
    # The value "a" has 9 rows, "b" one, "c" none: two of a, one of b, and
    # three more of a fill in for b and c.
    labels = ["a"] * 9 + ["b"]
    rows = [
        Example(f"uneven:train/{i}", {"state": f"row {i}"}, {}, {"label": label})
        for i, label in enumerate(labels)
    ]
    columns = {"label": {"values": {v: {"rows": 1} for v in ("a", "b", "c")}}}
    write_dataset(folder, Dataset(folder, columns=columns),
                  [SplitOut("train", rows, {"role": "train"})])  # fmt: skip
    doc = studio.get("/studio/api/data/uneven/document").json()
    assert len(doc["rows"]) == 6  # two a value, three values
    assert [row["values"]["label"] for row in doc["rows"]] == ["a"] * 5 + ["b"]
    assert doc["rows"][0]["id"].startswith("uneven:train/")
    assert all(row["name"] == row["id"].removeprefix("uneven:") for row in doc["rows"])


def test_a_broken_dataset_does_not_empty_the_list(studio):
    broken = studio.data / "broken"
    broken.mkdir()
    (broken / "train.jsonl").write_text('{"state": "s"}\n')
    (broken / DATASET_FILE).write_text(json.dumps({"splits": {"train": "train"}}))
    listed = studio.get("/studio/api/data")
    assert listed.status_code == 200
    items = {item["name"]: item for item in listed.json()["files"]}
    assert "must be an object" in items["broken"]["error"] and "error" not in items["tweets"]


def test_a_plain_file_behind_a_folder_is_said(studio):
    plain = {"state": "s", "questions": {"topic": TOPIC}, "labels": {"topic": "fed"}}
    (studio.data / "tweets.jsonl").write_text(json.dumps(plain) + "\n")
    [item] = studio.get("/studio/api/data").json()["files"]
    assert item["kind"] == "folder" and item["hides"] == "tweets.jsonl"


def test_a_save_keeps_keys_it_does_not_know_and_writes_the_format(studio):
    body = json.loads((studio.folder / DATASET_FILE).read_text())
    del body["format"]  # as a dataset from before the field
    body["mixes"] = {"weights": [1, 2]}
    (studio.folder / DATASET_FILE).write_text(json.dumps(body))
    saved = questions(studio, {"questions": {"topic": TOPIC}, "answers": ANSWERS})
    assert saved.status_code == 200, saved.text
    written = json.loads((studio.folder / DATASET_FILE).read_text())
    assert written["format"] == 1 and written["mixes"] == {"weights": [1, 2]}


def test_a_dataset_from_a_newer_version_is_listed_with_its_error_and_not_saved(studio):
    body = json.loads((studio.folder / DATASET_FILE).read_text())
    (studio.folder / DATASET_FILE).write_text(json.dumps({**body, "format": 2}))
    before = (studio.folder / DATASET_FILE).read_text()
    [item] = studio.get("/studio/api/data").json()["files"]
    assert "its format is 2" in item["error"] and "a newer mlx-decision" in item["error"]
    opened = studio.get("/studio/api/data/tweets")
    assert opened.status_code >= 400 and "its format is 2" in opened.text
    saved = questions(studio, {"questions": {"topic": TOPIC}, "answers": ANSWERS})
    assert saved.status_code >= 400 and "its format is 2" in saved.text
    assert (studio.folder / DATASET_FILE).read_text() == before


def test_in_use_checks_ignore_the_names_case(studio):
    """macOS's disks find tweets as TWEETS too: the checks do the same."""
    jobs = studio.app_state.jobs.jobs
    jobs["fit"] = SimpleNamespace(state="running", details={"data": "tweets"}, title="Fit x")
    try:
        deleted = studio.delete(
            "/studio/api/data/TWEETS", headers={"Content-Type": "application/json"}
        )
        assert deleted.status_code == 409 and "Fit x reads TWEETS" in deleted.text
    finally:
        del jobs["fit"]
    assert (studio.folder / DATASET_FILE).is_file()


def test_a_name_a_job_writes_is_not_free(studio):
    folder = studio.data / "incoming"
    jobs = studio.app_state.jobs.jobs
    jobs["get"] = SimpleNamespace(state="running", outputs=[folder], title="Download incoming")
    try:
        csv = "text,topic\nrates rise,fed\n"
        upload = {"name": "INCOMING", "text": csv, "format": "csv", "text_column": "text",
                  "label_column": "topic"}  # fmt: skip
        response = studio.post("/studio/api/data", json=upload)
        assert response.status_code == 409 and "Download incoming is writing" in response.text
        line = {"state": "s", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
        plain = json.dumps(line)
        lines = studio.post("/studio/api/data", json={"name": "incoming", "text": plain})
        assert lines.status_code == 409
    finally:
        del jobs["get"]
    assert not folder.exists() and not (studio.data / "incoming.jsonl").exists()
