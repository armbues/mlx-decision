"""An image dataset through the Studio: Run as Batch (the batch keeps its own
links to the images and answers from them, also after the dataset's images
are gone), its answers kept as a dataset with the images, the Playground's
rows with their images, and the fit's plan."""

import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from test_image_data import image_folder

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.dataset import read_for
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio

HERE = "http://127.0.0.1:8000"
API = "/studio/api/batches"


@pytest.fixture
def studio(tmp_path):
    # "eyes" reads images (and opens each, as the real backends do); "text" does not.
    write_model(tmp_path / "models" / "eyes", supports_images=True, open_images=True)
    write_model(tmp_path / "models" / "text")
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"))
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.app_state = app.state
        client.data = (tmp_path / "ws" / "data").resolve()
        client.data.mkdir(parents=True, exist_ok=True)
        client.pets = image_folder(client.data / "pets")
        yield client


def wait(studio, name, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = studio.get(f"{API}/{name}").json()["batch"]
        if info["state"] != "running" and studio.app_state.batches.running is None:
            return info
        time.sleep(0.02)
    raise AssertionError(f"{name} is still running")


def batch(studio, model="eyes", **body):
    response = studio.post(API, json={"dataset": "pets", "model": model, **body})
    assert response.status_code == 201, response.text
    return response.json()["batch"]["name"]


def test_a_batch_answers_from_its_own_links_to_the_images(studio):
    name = batch(studio)
    folder = studio.app_state.batches.folder(name)
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    assert rows[0]["images"] == ["images/0.png"]  # by its path in the batch's folder
    linked, source = folder / "images" / "0.png", studio.pets / "images" / "0.png"
    assert linked.stat().st_ino == source.stat().st_ino  # a hard link, not a copy
    info = wait(studio, name)
    assert (info["state"], info["answered"], info["errors"]) == ("done", 6, 0)
    # The batch stands alone: the dataset's images gone, Run Again still answers.
    for file in (studio.pets / "images").iterdir():
        file.unlink()
    again = studio.post(f"{API}/{name}/again", json={})
    assert again.status_code == 201, again.text
    other = again.json()["batch"]["name"]
    assert (studio.app_state.batches.folder(other) / "images" / "5.png").is_file()
    studio.post(f"{API}/{other}/start", json={})
    info = wait(studio, other)
    assert (info["state"], info["answered"], info["errors"]) == ("done", 6, 0)


def test_a_model_without_images_answers_each_row_with_its_refusal(studio):
    info = wait(studio, batch(studio, model="text"))
    assert info["errors"] == 6


def test_a_path_leaving_the_batch_folder_is_refused_for_its_row(studio):
    name = batch(studio)
    folder = studio.app_state.batches.folder(name)
    rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text().splitlines()]
    rows[0]["images"] = ["../../data/pets/images/0.png"]
    (folder / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    info = wait(studio, name)
    assert (info["answered"], info["errors"]) == (6, 1)
    first = studio.app_state.batches.results(name)[0]
    assert "leaves the batch's folder" in first["error"]["message"]


def test_the_answers_kept_as_a_dataset_keep_the_images(studio):
    name = batch(studio)
    wait(studio, name)
    saved = studio.post(f"{API}/{name}/dataset", json={"name": "pets-by-eyes"})
    assert saved.status_code == 201, saved.text
    folder = studio.data / "pets-by-eyes"
    assert sorted(p.name for p in (folder / "images").iterdir()) == [f"{i}.png" for i in range(6)]
    read = read_for(folder)  # as a fit reads it: every image opens
    assert read.examples[0].request["images"] == [str(folder / "images" / "0.png")]


def test_the_playgrounds_rows_carry_their_images(studio):
    body = studio.get("/studio/api/data/pets/document", params={"per": 3}).json()
    first = body["rows"][0]
    data = (studio.pets / "images" / f"{first['index']}.png").read_bytes()
    assert first["images"] == ["data:image/png;base64," + base64.b64encode(data).decode()]
    (studio.pets / "images" / "0.png").unlink()
    rows = studio.get("/studio/api/data/pets/document", params={"per": 3}).json()["rows"]
    gone = next(row for row in rows if row["index"] == 0)
    assert gone["images"] == [] and gone["images_missing"] == 1
    # The Playground sends a row back as data URLs: the run takes it.
    run = studio.post("/studio/api/run", json={"model": "eyes", "request": {
        "state": first["state"], "images": first["images"],
        "questions": body["questions"]}})  # fmt: skip
    assert run.status_code == 200, run.text


def test_the_fit_plan_reads_the_images_and_refuses_a_model_without(studio):
    planned = studio.post("/studio/api/fit/plan", json={"model": "eyes", "data": "pets"})
    assert planned.status_code == 200, planned.text
    assert planned.json()["splits"]["examples"] == 6
    refused = studio.post("/studio/api/fit/plan", json={"model": "text", "data": "pets"})
    assert refused.status_code == 422 and "does not read images" in refused.text


def test_other_pictures_with_the_same_state_are_not_shared_examples(studio):
    from mlx_decision.data.format import Example
    from mlx_decision.training.fitting import shared_states

    def example(image):
        return Example("x", {"state": "", "images": [image]}, {})

    train = [example("images/0.png"), example("images/1.png")]
    assert shared_states(train, [example("images/2.png")]) == 0
    assert shared_states(train, [example("images/1.png")]) == 1
    text = [Example("t", {"state": "same"}, {})]
    assert shared_states(text, [Example("r", {"state": "same"}, {})]) == 1
