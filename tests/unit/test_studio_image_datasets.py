"""Image datasets on the Studio's Datasets page: a thumbnail per row and the
image whole (served only from inside the dataset's folder), and Add of a
file naming image files beside it (the steps, not an upload)."""

import base64
import io
import json
import shutil

import pytest
import test_studio_data
from test_image_data import COLOURS, PET, image_folder, png
from test_studio_data import upload

from mlx_decision.data.dataset import read_for

PIL = pytest.importorskip("PIL.Image")
studio = test_studio_data.studio  # the fixture


def image(studio, name, index, number=0, **params):
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return studio.get(f"/studio/api/data/{name}/examples/{index}/images/{number}?{query}")


@pytest.fixture
def pets(studio):
    """data/pets: six pictures, red and blue in turn, 16 x 16 pixels."""
    return image_folder(studio.workspace / "data" / "pets")


def test_rows_say_how_many_images_they_have(studio, pets):
    page = studio.get("/studio/api/data/pets/examples").json()
    assert [row["images"] for row in page["examples"]] == [1] * 6
    whole = studio.get("/studio/api/data/pets/examples/1").json()
    assert whole["images"] == 1 and whole["state"] == "a picture"
    described = studio.get("/studio/api/data/pets").json()
    assert described["sample"][0]["images"] == 1 and described["sample"][0]["index"] == 0


def test_an_image_as_a_thumbnail_and_whole(studio, pets):
    large = PIL.new("RGB", (640, 400), COLOURS["blue"])
    buffer = io.BytesIO()
    large.save(buffer, "PNG")
    (pets / "images" / "1.png").write_bytes(buffer.getvalue())
    thumb = image(studio, "pets", 1)
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert PIL.open(io.BytesIO(thumb.content)).size == (160, 100)
    full = image(studio, "pets", 1, size="full")
    assert full.headers["content-type"] == "image/png"
    assert full.content == (pets / "images" / "1.png").read_bytes()
    assert image(studio, "pets", 1, size="huge").status_code == 400
    assert image(studio, "pets", 1, number=1).status_code == 404
    assert image(studio, "pets", 99).status_code == 404


def test_only_files_inside_the_dataset_and_data_urls_are_served(studio, pets, tmp_path):
    outside = tmp_path / "secret.png"
    outside.write_bytes(png())
    rows = (pets / "train.jsonl").read_text().splitlines()
    first = json.loads(rows[0])
    first["images"] = [str(outside)]  # read where it is: not served
    second = json.loads(rows[1])
    second["images"] = ["../../../secret.png"]  # out of the folder
    third = json.loads(rows[2])
    third["images"] = ["data:image/png;base64," + base64.b64encode(png()).decode()]
    rows[:3] = [json.dumps(first), json.dumps(second), json.dumps(third)]
    (pets / "train.jsonl").write_text("\n".join(rows) + "\n")
    assert image(studio, "pets", 0, size="full").status_code == 404
    assert image(studio, "pets", 1, size="full").status_code == 404
    served = image(studio, "pets", 2, size="full")
    assert served.status_code == 200 and served.content == png()


def test_adding_a_file_with_images_beside_it_says_how(studio, tmp_path):
    source = image_folder(tmp_path / "mine")
    # A lone file of training lines: each with its question.
    rows = [json.loads(row) | {"questions": {"pet": PET}}
            for row in (source / "train.jsonl").read_text().splitlines()]  # fmt: skip
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (source / "train.jsonl").write_text(text)
    response = upload(studio, "mine", text)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "images_beside"
    assert error["images"] == {"files": 6, "entries": ["images"], "absolute": False}
    assert not (studio.workspace / "data" / "mine.jsonl").exists()
    # The steps the sheet gives: the file as NAME/train.jsonl, its images
    # beside it, a dataset.json of {"format": 1}.
    target = studio.workspace / "data" / "mine"
    target.mkdir()
    shutil.copy(source / "train.jsonl", target / "train.jsonl")
    shutil.copytree(source / "images", target / "images")
    (target / "dataset.json").write_text('{"format": 1}\n')
    described = studio.get("/studio/api/data/mine").json()
    assert described["examples"] == 6 and described["missing"] == []
    assert described["questions"][0]["id"] == "pet"
    assert len(read_for(target).examples) == 6  # as a fit reads it
    assert image(studio, "mine", 0).status_code == 200
