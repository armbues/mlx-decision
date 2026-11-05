"""Training data with images: paths relative to the data, every image opened by
the check, image bytes in the data's hash, text-only bases refusing image data
before they load, and a dataset folder of images fitted end to end."""

import base64
import io
import json
from pathlib import Path

import pytest

from mlx_decision.data.dataset import open_dataset, read_for
from mlx_decision.data.format import DataError, parse_examples, read_examples
from mlx_decision.errors import DecisionError
from mlx_decision.training import evaluate, fit
from tiny_models import write_d3, write_marker

PIL = pytest.importorskip("PIL.Image")

PET = {"type": "choice", "criteria": {"red": "A red picture", "blue": "A blue picture"}}
COLOURS = {"red": (220, 20, 20), "blue": (20, 20, 220)}


def png(colour=(200, 30, 30), size=(16, 16)) -> bytes:
    buffer = io.BytesIO()
    PIL.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


def data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def line(labels=None, state="a picture", **extra) -> str:
    body = {**extra, "state": state, "questions": {"pet": PET}, "labels": labels or {"pet": "red"}}
    return json.dumps(body)


def error_of(*lines) -> DataError:
    with pytest.raises(DataError) as caught:
        parse_examples(lines)
    return caught.value


def image_folder(folder: Path, count: int = 6) -> Path:
    """A dataset folder of ``count`` pictures, red and blue in turn, their
    lines naming them as ``images/N.png``."""
    (folder / "images").mkdir(parents=True)
    rows = []
    for index in range(count):
        colour = "red" if index % 2 == 0 else "blue"
        (folder / "images" / f"{index}.png").write_bytes(png(COLOURS[colour]))
        rows.append({"id": f"p{index}", "state": "a picture", "images": [f"images/{index}.png"],
                     "labels": {"pet": colour}})  # fmt: skip
    (folder / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    described = {"format": 1, "questions": {"pet": PET}, "splits": {"train": {"role": "train"}}}
    (folder / "dataset.json").write_text(json.dumps(described))
    return folder


def test_data_urls_and_images_in_the_state_are_read():
    url = data_url(png())
    examples = parse_examples([
        line(images=[url]),
        line(state={"text": "hi", "photo": url}),
    ])  # fmt: skip
    assert examples[0].request["images"] == [url]
    assert examples[1].request["state"]["photo"] == url  # lifted by the model, not here


def test_paths_are_relative_to_the_file_and_read_as_absolute(tmp_path, monkeypatch):
    (tmp_path / "pics").mkdir()
    (tmp_path / "pics" / "a.png").write_bytes(png())
    path = tmp_path / "data.jsonl"
    path.write_text(line(images=["pics/a.png"]) + "\n")
    monkeypatch.chdir(tmp_path.parent)  # elsewhere: the file's folder counts
    [example] = read_examples(path)
    assert example.request["images"] == [str(tmp_path / "pics" / "a.png")]
    [same] = open_dataset(path)[0].read()
    assert same.request == example.request


def test_a_missing_unreadable_or_oversized_image_is_an_error_of_its_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("text.png").write_text("not a picture")
    Path("big.png").write_bytes(png(size=(64, 64)))
    good = line(images=[data_url(png())])
    error = error_of(good, line(images=["gone.png"]))
    assert (error.line, error.param, error.message) == (
        2,
        "images.0",
        "gone.png: no such image file",
    )
    error = error_of(line(images=["text.png"]))
    assert error.param == "images.0" and "text.png: not a readable" in error.message
    error = error_of(line(images=[data_url(png()), "data:image/png;base64,AAAA"]))
    assert error.param == "images.1" and "not a readable" in error.message
    error = error_of(line(images=[42]))
    assert error.message == "an image is a file path or a data URL"
    monkeypatch.setattr(PIL, "MAX_IMAGE_PIXELS", 1000)
    error = error_of(line(images=["big.png"]))
    assert "big.png: the image is too large" in error.message
    broken = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(80)).decode()
    error = error_of(line(state={"text": "hi", "photo": broken}))
    assert error.param == "state.photo" and "not a readable" in error.message


def test_videos_are_still_refused():
    error = error_of(line(videos=["clip.mp4"]))
    assert (error.param, error.message) == ("videos", "training data cannot hold videos")


def test_a_folder_reads_its_images_and_browsing_does_not_open_them(tmp_path):
    folder = image_folder(tmp_path / "pets")
    read = read_for(folder)
    assert [e.request["images"] for e in read.examples[:2]] == [
        [str(folder / "images" / "0.png")],
        [str(folder / "images" / "1.png")],
    ]
    (folder / "images" / "3.png").unlink()
    dataset, _ = open_dataset(folder)
    assert len(dataset.read("train", lenient=True)) == 6  # a browse opens nothing
    with pytest.raises(DataError) as error:
        read_for(folder)
    assert str(error.value) == "train.jsonl line 4: images.0: images/3.png: no such image file"


def test_the_hash_covers_the_image_files(tmp_path):
    folder = image_folder(tmp_path / "pets")
    before = read_for(folder).sha256
    (folder / "images" / "2.png").write_bytes(png((20, 220, 20)))
    after = read_for(folder).sha256
    assert after != before
    (folder / "images" / "2.png").write_bytes(png(COLOURS["red"]))
    assert read_for(folder).sha256 == before


def test_text_data_hashes_as_before(tmp_path):
    import hashlib

    path = tmp_path / "text.jsonl"
    path.write_text(line() + "\n")
    assert read_for(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("family", ["laya", "julia"])
def test_a_text_only_base_refuses_image_data_before_loading(tmp_path, monkeypatch, family):
    base = write_marker(tmp_path / family, family)
    folder = image_folder(tmp_path / "pets")

    def no_load(*args, **kwargs):
        raise AssertionError("the model was loaded")

    monkeypatch.setattr("mlx_decision.training.fitting.load", no_load)
    monkeypatch.setattr("mlx_decision.model.load", no_load)
    with pytest.raises(
        DecisionError, match=rf"example p0 first\) and {family} does not read images"
    ):
        fit(base, folder, output=tmp_path / "fitted")
    with pytest.raises(DecisionError, match="does not read images"):
        evaluate(base, folder)
    assert not (tmp_path / "fitted").exists()


def test_a_folder_of_images_fits_a_readout_on_tiny_d3(tmp_path, monkeypatch):
    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 100)
    base = write_d3(tmp_path / "tiny-d3")
    folder = image_folder(tmp_path / "pets")
    monkeypatch.chdir(tmp_path.parent)
    result = fit(base, folder, output=tmp_path / "fitted")
    assert result.kinds == ["readout", "calibrate"]
    assert result.report["data"]["sha256"] == read_for(folder).sha256
    import mlx_decision

    model = mlx_decision.load(result.path)
    answer = model.decide("a picture", {"pet": PET}, images=[str(folder / "images" / "0.png")])
    assert set(answer.answers["pet"].probabilities) == {"red", "blue"}
