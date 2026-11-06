"""Mixes in the Studio: listed with their sources, described per split with
each source's rows, their rows and images browsed, refused where questions
are written, and read as one dataset by batches and the fit's plan."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from test_image_data import image_folder

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.dataset import DATASET_FILE
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio

HERE = "http://127.0.0.1:8000"
MIX = {
    "format": 2,
    "mix": {"sources": [{"path": "../pets"}, {"path": "../more", "held_out": True}], "seed": 0},
}


@pytest.fixture
def studio(tmp_path):
    """data/pets (6 pictures) and data/more (5), both naming their files
    images/N.png, and data/both, a mix of them with more held out."""
    write_model(tmp_path / "models" / "eyes", supports_images=True, open_images=True)
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.data = (tmp_path / "ws" / "data").resolve()
        client.data.mkdir(parents=True, exist_ok=True)
        image_folder(client.data / "pets")
        image_folder(client.data / "more", count=5)
        (client.data / "both").mkdir()
        (client.data / "both" / DATASET_FILE).write_text(json.dumps(MIX))
        yield client


def test_the_list_shows_a_mix_with_its_sources(studio):
    files = studio.get("/studio/api/data").json()["files"]
    [both] = [item for item in files if item["name"] == "both"]
    assert both["kind"] == "folder" and both["missing"] == []
    assert both["mix"] == {"sources": ["pets", "more"], "held_out": ["more"]}
    assert [split["name"] for split in both["splits"]] == ["train", "report", "evaluate"]


def test_a_split_is_described_with_each_sources_rows(studio):
    train = studio.get("/studio/api/data/both?split=train").json()
    # pets has no report split: a fifth of its rows reports; more is held out.
    assert train["examples"] == 5 and train["file"].endswith(f"both/{DATASET_FILE}")
    assert train["mix"]["sources"] == [
        {"name": "pets", "path": "../pets", "cap": None, "held_out": False, "rows": 5,
         "dataset": "pets"},
        {"name": "more", "path": "../more", "cap": None, "held_out": True, "rows": 0,
         "dataset": "more"},
    ]  # fmt: skip
    report = studio.get("/studio/api/data/both?split=report").json()
    assert {s["name"]: s["rows"] for s in report["mix"]["sources"]} == {"pets": 1, "more": 1}
    assert [q["id"] for q in report["questions"]] == ["pet"]


def test_rows_and_images_come_from_their_sources(studio):
    page = studio.get("/studio/api/data/both/examples?split=report").json()
    ids = [row["id"] for row in page["examples"]]
    assert len(ids) == 2 and ids[0].startswith("pets/") and ids[1].startswith("more/")
    for index in range(2):
        image = studio.get(f"/studio/api/data/both/examples/{index}/images/0?split=report")
        assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"


def test_a_changed_source_is_read_again(studio):
    assert studio.get("/studio/api/data/both?split=train").json()["examples"] == 5
    image_folder_rows = (studio.data / "pets" / "train.jsonl").read_text().splitlines()
    (studio.data / "pets" / "train.jsonl").write_text("\n".join(image_folder_rows[:5]) + "\n")
    assert studio.get("/studio/api/data/both?split=train").json()["examples"] == 4


def test_questions_are_written_on_the_sources_not_the_mix(studio):
    saved = studio.put("/studio/api/data/both/questions", json={"questions": {}, "answers": {}})
    assert saved.status_code == 400 and "both is a mix" in saved.text
    document = studio.get("/studio/api/data/both/document")
    assert document.status_code == 400 and "write or edit them on a source" in document.text


def test_a_missing_source_is_listed_with_what_is_missing(studio):
    body = {"format": 2, "mix": {"sources": [{"path": "../gone"}]}}
    (studio.data / "broken").mkdir()
    (studio.data / "broken" / DATASET_FILE).write_text(json.dumps(body))
    [broken] = [i for i in studio.get("/studio/api/data").json()["files"] if i["name"] == "broken"]
    assert broken["missing"] == ["the mix's source ../gone is missing"]
    assert broken["mix"] == {"sources": [], "held_out": []}


def test_a_batch_on_a_mix_keeps_each_sources_images_apart(studio):
    response = studio.post(
        "/studio/api/batches", json={"dataset": "both", "split": "report", "model": "eyes"}
    )
    assert response.status_code == 201, response.text
    name = response.json()["batch"]["name"]
    folder = studio.app_state.batches.folder(name)
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    images = [row["images"][0] for row in rows]
    assert images[0].startswith("pets/images/") and images[1].startswith("more/images/")
    assert all((folder / image).is_file() for image in images)
    deadline = time.time() + 20
    while time.time() < deadline:
        info = studio.get(f"/studio/api/batches/{name}").json()["batch"]
        if info["state"] != "running" and studio.app_state.batches.running is None:
            break
        time.sleep(0.02)
    assert (info["state"], info["answered"], info["errors"]) == ("done", 2, 0)


def test_the_fit_plan_reads_a_mix_as_one_dataset(studio):
    planned = studio.post("/studio/api/fit/plan", json={"model": "eyes", "data": "both"})
    assert planned.status_code == 200, planned.text
    assert planned.json()["splits"]["examples"] == 5


# A new mix


def new_mix(studio, **body):
    return studio.post("/studio/api/data/mix", json=body)


def test_a_new_mix_lists_the_workspaces_datasets(studio):
    sources = [{"name": "pets"}, {"name": "more", "held_out": True, "cap": 3}]
    made = new_mix(studio, name="pair", cap=4, seed=2, sources=sources)
    assert made.status_code == 201, made.text
    assert made.json()["name"] == "pair"
    assert made.json()["mix"] == {"sources": ["pets", "more"], "held_out": ["more"]}
    body = json.loads((studio.data / "pair" / DATASET_FILE).read_text())
    assert body["format"] == 2 and "created" in body
    assert body["mix"] == {"sources": [{"path": "../pets"},
                                       {"path": "../more", "cap": 3, "held_out": True}],
                           "cap": 4, "seed": 2}  # fmt: skip
    assert not list((studio.data / "pair").glob(".*"))  # no staged file left
    train = studio.get("/studio/api/data/pair?split=train").json()
    assert {s["name"]: s["rows"] for s in train["mix"]["sources"]} == {"pets": 3, "more": 0}
    planned = studio.post("/studio/api/fit/plan", json={"model": "eyes", "data": "pair"})
    assert planned.status_code == 200, planned.text


def test_a_plain_file_is_a_source_by_its_file(studio):
    line = {"state": "s", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
    (studio.data / "notes.jsonl").write_text(json.dumps(line) + "\n")
    made = new_mix(studio, name="pair", sources=[{"name": "pets"}, {"name": "notes"}])
    assert made.status_code == 201, made.text
    body = json.loads((studio.data / "pair" / DATASET_FILE).read_text())
    assert [s["path"] for s in body["mix"]["sources"]] == ["../pets", "../notes.jsonl"]


def two(first="pets", second="more", held=False, **body):
    sources = [{"name": first, "held_out": held}, {"name": second, "held_out": held}]
    return {"name": "pair", "sources": sources, **body}


@pytest.mark.parametrize(
    ("body", "status", "message"),
    [
        ({"name": "pair", "sources": [{"name": "pets"}]}, 400, "two datasets or more"),
        (two(name=".x"), 400, "a mix's name"),
        (two(second="gone"), 404, "no dataset is named gone"),
        (two(cap=0), 400, "cap must be"),
        (two(held=True), 400, "every source of the mix is held out"),
        (two(name="pets"), 409, "exists already"),
        (two(second="both"), 422, "is a mix itself"),
    ],
)
def test_a_mix_that_cannot_be_says_why(studio, body, status, message):
    made = new_mix(studio, **body)
    assert made.status_code == status and message in made.text, made.text
    assert not (studio.data / "pair").exists()
