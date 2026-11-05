"""``data``'s image shape: Hugging Face ``Image`` columns and image paths in a
CSV written into a dataset folder's ``images/``, guessed from one image
column and a class label; ``--show`` listing the files; the menus."""

import io
import json
from pathlib import Path

import pytest
from test_wizards import drive, typed
from typer.testing import CliRunner

from mlx_decision.cli import app
from mlx_decision.data.dataset import read_for
from mlx_decision.data.make import DataPlan
from mlx_decision.data.shapes import guess
from mlx_decision.data.sources import IMAGE, SourceError, open_source
from mlx_decision.data.wizard import ask_data

datasets = pytest.importorskip("datasets")
PIL = pytest.importorskip("PIL.Image")

COLOURS = {"red": (220, 20, 20), "blue": (20, 20, 220)}


def picture(colour: str, kind: str = "PNG", size=(24, 16)) -> bytes:
    buffer = io.BytesIO()
    PIL.new("RGB", size, COLOURS[colour]).save(buffer, kind)
    return buffer.getvalue()


@pytest.fixture
def pictures(tmp_path) -> Path:
    """A saved dataset of 8 pictures, red and blue in turn, with a class label,
    an id column (text) and one picture stored as BMP."""
    names = ["red" if i % 2 == 0 else "blue" for i in range(8)]
    images = [{"bytes": picture(n, "BMP" if i == 3 else "PNG"), "path": None}
              for i, n in enumerate(names)]  # fmt: skip
    features = datasets.Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=["red", "blue"]),
        "image_id": datasets.Value("string"),
    })  # fmt: skip
    data = datasets.Dataset.from_dict(
        {"image": images, "label": [0 if n == "red" else 1 for n in names],
         "image_id": [f"p{i}" for i in range(8)]},
        features=features,
    )  # fmt: skip
    folder = tmp_path / "pictures"
    datasets.DatasetDict({"train": data}).save_to_disk(folder)
    return folder


def run(*args: str):
    return CliRunner().invoke(app, ["data", *args])


def test_an_image_column_and_a_label_guess_the_image_shape(pictures):
    table = open_source(str(pictures), split="train")
    assert table.columns["image"].kind == IMAGE
    assert "image (image, e.g. null)" in table.listing()  # the path, not the bytes
    mapping = guess(table)
    # image_id is text, but not named like a state: left out.
    assert mapping.shape == "image"
    assert mapping.columns == {"image": "image", "label": "label", "text": None}


def test_show_lists_the_files_and_their_sizes(pictures):
    result = run(str(pictures), "--split", "train", "--question", "Which colour?", "--show", "2")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    first = json.loads(lines[0])
    assert first["images"] == ["images/train-00000.png"] and first["state"] == ""
    assert first["labels"] == {"pictures": "red"}
    assert lines[1] == "  images/train-00000.png: 24x16 PNG, 1 KB"
    assert "8 examples (image shape: image=image, label=label)" in result.stdout
    assert "images: 8 files in images/" in result.stdout


def test_a_folder_gets_the_images_and_fits_on_tiny_d3(pictures, tmp_path, monkeypatch):
    out = tmp_path / "colours"
    result = run(str(pictures), "--split", "train", "--question", "Which colour?", "-o", str(out))
    assert result.exit_code == 0, result.output
    files = sorted(p.name for p in (out / "images").iterdir())
    # The BMP picture is written as PNG.
    assert files == [f"train-{i:05d}.png" for i in range(8)]
    assert PIL.open(out / "images" / "train-00003.png").format == "PNG"
    described = json.loads((out / "dataset.json").read_text())
    assert described["source"]["shape"] == "image"
    assert described["answers"]["pictures"]["map"] == {"red": "red", "blue": "blue"}
    read = read_for(out)
    assert read.examples[0].request["images"] == [str(out / "images" / "train-00000.png")]
    # What data wrote trains as it is.
    from mlx_decision.training import fit
    from tiny_models import write_d3

    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 50)
    fitted = fit(write_d3(tmp_path / "tiny-d3"), out, output=tmp_path / "fitted")
    assert fitted.report["data"]["examples"] == 8


def test_without_a_question_the_folder_keeps_the_values(pictures, tmp_path):
    out = tmp_path / "raw"
    result = run(str(pictures), "--split", "train", "-o", str(out))
    assert result.exit_code == 0, result.output
    line = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert line == {"id": "pictures:train/0", "state": "", "images": ["images/train-00000.png"],
                    "values": {"label": "red"}}  # fmt: skip
    assert len(list((out / "images").iterdir())) == 8


def test_a_lone_file_is_refused(pictures, tmp_path):
    result = run(str(pictures), "--split", "train", "--question", "Which colour?",
                 "-o", str(tmp_path / "colours.jsonl"))  # fmt: skip
    assert result.exit_code != 0
    assert "writes into a dataset folder" in result.output
    assert not (tmp_path / "colours.jsonl").exists()


def test_paths_in_a_csv_unreadable_images_and_a_caption(tmp_path):
    (tmp_path / "pics").mkdir()
    (tmp_path / "pics" / "a.png").write_bytes(picture("red"))
    (tmp_path / "pics" / "b.jpg").write_bytes(picture("blue", "JPEG"))
    (tmp_path / "pics" / "c.png").write_text("not a picture")
    csv = tmp_path / "pics.csv"
    csv.write_text("file,text,label\npics/a.png,a red one,red\npics/b.jpg,a blue one,blue\n"
                   "pics/c.png,broken,red\npics/gone.png,missing,blue\n")  # fmt: skip
    table = open_source(str(csv))
    assert table.columns["file"].kind == IMAGE
    assert guess(table).columns == {"image": "file", "label": "label", "text": "text"}
    out = tmp_path / "out"
    result = run(str(csv), "--question", "Which colour?", "-o", str(out))
    assert result.exit_code == 0, result.output
    assert "skipped 1 rows: unreadable image" in result.stderr
    assert "skipped 1 rows: no image" in result.stderr
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert [(r["state"], r["images"]) for r in rows] == [
        ("a red one", ["images/rows-00000.png"]),
        ("a blue one", ["images/rows-00001.jpg"]),
    ]
    assert (out / "images" / "rows-00001.jpg").read_bytes() == (
        tmp_path / "pics" / "b.jpg"
    ).read_bytes()


def test_two_image_columns_need_the_flag(tmp_path):
    (tmp_path / "a.png").write_bytes(picture("red"))
    csv = tmp_path / "two.csv"
    csv.write_text("front,back,label\na.png,a.png,red\n")
    with pytest.raises(SourceError, match="image column of the image shape is not clear"):
        guess(open_source(str(csv)))
    result = run(str(csv), "--image", "back", "--question", "Which?", "--show", "1")
    assert result.exit_code == 0, result.output
    assert "image=back" in result.stdout


def test_the_menus_offer_the_shape_and_a_folder(pictures, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    keys = typed(
        "",  # shape: image, guessed (the only split is not asked)
        "",  # image column
        "",  # label column
        "",  # text column: (none), the guess
        "",  # type: choice
        "",  # id
        "Which colour?",
        "",  # the options as they are
        "",  # write to: pictures (a folder)
        "y!",
    )
    plan, made = drive(ask_data, keys, DataPlan(source=str(pictures)))
    assert plan.shape == "image" and plan.columns["text"] is None
    assert plan.output == "pictures" and plan.folder
    assert len(made.images) == 8
