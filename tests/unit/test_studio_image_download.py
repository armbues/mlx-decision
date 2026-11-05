"""Download of an image dataset in the Studio: the preview's image columns as
thumbnails and the image shape guessed; the ``data`` job of the image shape."""

import io

import pytest
import test_studio_datasets
from test_studio_datasets import get, stop

datasets = pytest.importorskip("datasets")
studio = test_studio_datasets.studio  # the fixture
PIL = pytest.importorskip("PIL.Image")


def picture(colour, size=(320, 200)) -> bytes:
    buffer = io.BytesIO()
    PIL.new("RGB", size, colour).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def pets_hub(monkeypatch):
    """A Hub stand-in: org/pets, one config, splits train and test, a picture
    column, a class label, an id and a caption."""
    from mlx_decision.data import sources

    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=["cat", "dog"]),
        "image_id": datasets.Value("string"),
        "caption": datasets.Value("string"),
    })  # fmt: skip
    rows = {
        "image": [{"bytes": picture((200, 30, 30) if i % 2 else (30, 30, 200)), "path": None}
                  for i in range(8)],
        "label": [i % 2 for i in range(8)],
        "image_id": [f"p{i}" for i in range(8)],
        "caption": [f"photo {i}" for i in range(8)],
    }  # fmt: skip

    def load(repo, config=None, split=None, revision=None):
        dataset = datasets.Dataset.from_dict(rows, features=features)
        return sources._from_dataset(dataset, "pets", split, {"repo": repo})

    monkeypatch.setattr(sources, "hub_configs", lambda repo, revision=None: [])
    monkeypatch.setattr(sources, "hub_splits", lambda repo, config=None, revision=None: ["train"])
    small = {"train": sources.SplitSize(8, 20_000)}
    monkeypatch.setattr(sources, "hub_split_sizes", lambda repo, config=None: small)
    monkeypatch.setattr(sources, "load_hub", load)
    monkeypatch.setattr(sources, "hub_card", lambda repo, revision=None: None)


def test_the_preview_shows_thumbnails_and_guesses_the_image_shape(studio, pets_hub):
    body = studio.post("/studio/api/datasets/preview", json={"source": "hf:org/pets"}).json()
    assert body["split"] == "train" and body["rows"] == 8
    assert body["image_columns"] == ["image"]
    assert {"name": "image", "kind": "image", "names": None} in body["columns"]
    first = body["sample"][0]["image"]
    assert first.startswith("data:image/jpeg;base64,")
    import base64

    thumb = PIL.open(io.BytesIO(base64.b64decode(first.split(",", 1)[1])))
    assert max(thumb.size) == 160 and thumb.size == (160, 100)  # the aspect kept
    assert body["guess"] == {
        "shape": "image",
        "image": "image",
        "text": "caption",
        "label": "label",
    }


def test_a_job_of_the_image_shape(studio, pets_hub):
    fields = {"source": "hf:org/pets", "shape": "image", "image": "image", "label": "label",
              "splits": [{"split": "train", "role": "train"}]}  # fmt: skip
    started = get(studio, **fields, text="caption")
    assert started.status_code == 202, started.text
    job = started.json()["job"]
    argv = job["command"]
    at = argv.index("--shape")
    assert argv[at : at + 8] == ["--shape", "image", "--image", "image", "--label", "label",
                                 "--text", "caption"]  # fmt: skip
    assert job["details"]["image"] == "image" and job["details"]["text"] == "caption"
    stop(studio, job)
    without = get(studio, **fields, name="pets-plain")
    assert without.status_code == 202, without.text
    assert "--text" not in without.json()["job"]["command"]
    stop(studio, without.json()["job"])
    same = get(studio, **{**fields, "label": "image"}, name="pets-same")
    assert same.status_code == 400 and same.json()["error"]["param"] == "label"
    assert get(studio, **{**fields, "image": None}, name="x").status_code == 400
