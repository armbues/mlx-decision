"""``training label``: a teacher's answers to a dataset's rows written as a new
dataset (soft labels by default, top answers with --hard), measured against
the data's own labels when it has them, with the note on the teacher's
calibration; samples, thresholds, images, mixes, overwrite and cancel."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_backend import TEAM, write_fitted, write_model
from mlx_decision.cli import app
from mlx_decision.data.dataset import DATASET_FILE, open_dataset, read_for
from mlx_decision.errors import DecisionError
from mlx_decision.training import Cancelled, fit, label

# The fake answers the first option most: 3/6, 2/6, 1/6 for three options.
FAKE = {"billing": 0.5, "tech": 1 / 3, "sales": 1 / 6}


def tickets(folder: Path, count: int = 30, labelled: bool = True) -> Path:
    """A dataset folder: train (``count`` rows, a third each team) and test (6)."""
    folder.mkdir(parents=True)
    teams = ["billing", "tech", "sales"]
    for split, n in (("train", count), ("test", 6)):
        rows = [{"id": f"{split}{i}", "state": f"ticket {i}"} for i in range(n)]
        if labelled:
            for i, row in enumerate(rows):
                row["labels"] = {"team": teams[i % 3]}
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = {"questions": {"team": TEAM},
            "splits": {"train": {"role": "train"}, "test": {"role": "evaluate"}}}  # fmt: skip
    (folder / DATASET_FILE).write_text(json.dumps(body))
    return folder


@pytest.fixture
def base(tmp_path) -> Path:
    return write_model(tmp_path / "fake")


def test_soft_labels_are_the_teachers_distributions(base, tmp_path):
    data = tickets(tmp_path / "tickets")
    result = label(base, data, output=tmp_path / "taught")
    assert (result.examples, result.left_out, result.refused, result.labels) == (30, 0, 0, "soft")
    examples = read_for(tmp_path / "taught").examples
    assert len(examples) == 30 and examples[0].id == "train0"
    assert examples[0].targets["team"] == pytest.approx(FAKE)
    assert all(example.labels == {"team": "billing"} for example in examples)
    dataset, _ = open_dataset(tmp_path / "taught")
    assert dataset.questions == {"team": TEAM}
    assert dataset.splits == {"train": {"role": "train", "examples": 30, "split": "train"}}
    source = dataset.source
    assert source["source"] == "teacher:fake" and source["teacher"] == {"name": "fake"}
    assert source["labels"] == "soft" and source["split"] == "train"
    # The data's own labels: the teacher scored on them (a third right).
    assert result.measured == {"team": {"accuracy": pytest.approx(1 / 3, abs=1e-4),
                                        "ece": pytest.approx(1 / 6, abs=1e-4)}}  # fmt: skip
    assert source["measured"] == result.measured
    assert result.note.startswith("fake's probabilities are off on the labelled rows")
    # A student fits on them.
    fitted = fit(base, tmp_path / "taught", output=tmp_path / "student")
    assert fitted.metrics["team"]["after"]["n"] == 6


def test_hard_labels_and_another_split(base, tmp_path):
    data = tickets(tmp_path / "tickets")
    result = label(base, data, output=tmp_path / "hard", labels="hard", split="test")
    assert result.examples == 6
    examples = read_for(tmp_path / "hard").examples
    assert all(e.targets is None and e.labels == {"team": "billing"} for e in examples)
    assert open_dataset(tmp_path / "hard")[0].source["split"] == "test"


def test_an_unlabelled_pool_is_labelled_and_the_release_noted(base, tmp_path):
    pool = tickets(tmp_path / "pool", labelled=False)
    result = label(base, pool, output=tmp_path / "taught")
    assert result.examples == 30 and result.measured == {}
    assert result.note.startswith("fake has no closing calibration, so how far")
    assert "measured" not in open_dataset(tmp_path / "taught")[0].source


def test_a_fitted_teacher_with_its_calibration(base, tmp_path):
    teacher = write_fitted(tmp_path / "taught-fake", base)
    pool = tickets(tmp_path / "pool", labelled=False)
    result = label(teacher, pool, output=tmp_path / "out")
    assert result.teacher.base == "fake" and result.teacher.calibrated == {"team"}
    assert result.note is None
    # It answers with its fit (temperature 2 flattens the fake's answers).
    target = read_for(tmp_path / "out").examples[0].targets["team"]
    assert target["billing"] < FAKE["billing"] and target["sales"] > FAKE["sales"]


def test_samples_and_thresholds(base, tmp_path):
    data = tickets(tmp_path / "tickets")
    sampled = label(base, data, output=tmp_path / "five", sample=5, seed=1)
    assert sampled.examples == 5
    assert open_dataset(tmp_path / "five")[0].source["sample"] == 5
    # The fake's confidence is 0.25 (above an even split): none is that sure.
    with pytest.raises(DecisionError, match="no row has an answer to keep"):
        label(base, data, output=tmp_path / "sure", threshold=0.5)
    assert not (tmp_path / "sure").exists()
    kept = label(base, data, output=tmp_path / "kept", threshold=0.2)
    assert kept.examples == 30


def test_rows_without_questions_are_refused(base, tmp_path):
    path = tmp_path / "bare.jsonl"
    path.write_text(json.dumps({"id": "a", "state": "s"}) + "\n")
    with pytest.raises(DecisionError, match="row a has no questions"):
        label(base, path, output=tmp_path / "out")


def test_an_existing_output_is_kept_unless_overwritten(base, tmp_path):
    data = tickets(tmp_path / "tickets")
    label(base, data, output=tmp_path / "out", sample=3)
    with pytest.raises(FileExistsError):
        label(base, data, output=tmp_path / "out")
    label(base, data, output=tmp_path / "out", overwrite=True)
    assert len(read_for(tmp_path / "out").examples) == 30
    assert not list(tmp_path.glob(".out*"))


def test_cancel_writes_nothing(base, tmp_path):
    data = tickets(tmp_path / "tickets")
    with pytest.raises(Cancelled):
        label(base, data, output=tmp_path / "out", cancel=lambda: True)
    assert not (tmp_path / "out").exists()


def test_a_mixs_images_are_linked_in_apart(tmp_path):
    from test_image_data import image_folder

    teacher = write_model(tmp_path / "eyes", supports_images=True, open_images=True)
    image_folder(tmp_path / "pets")
    image_folder(tmp_path / "more", count=2)
    (tmp_path / "both").mkdir()
    body = {"format": 2, "mix": {"sources": [{"path": "../pets"}, {"path": "../more"}]}}
    (tmp_path / "both" / DATASET_FILE).write_text(json.dumps(body))
    result = label(teacher, tmp_path / "both", output=tmp_path / "out")
    assert result.examples == 7  # a fifth of each reports: pets 5 rows train, more both
    lines = [
        json.loads(line) for line in (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    ]
    images = [line["images"][0] for line in lines]
    assert images[0].startswith("pets/images/") and images[-1].startswith("more/images/")
    assert all((tmp_path / "out" / image).is_file() for image in images)
    assert [e.id for e in read_for(tmp_path / "out").examples][0].startswith("pets/")


def test_the_command(base, tmp_path, monkeypatch):
    data = tickets(tmp_path / "tickets")
    monkeypatch.chdir(tmp_path)
    run = CliRunner().invoke
    result = run(app, ["training", "label", "-m", str(base), "--data", str(data)])
    assert result.exit_code == 0, result.output
    assert "30 rows labelled by fake, as soft labels (its probabilities)" in result.stdout
    assert "teacher on the data's labels, team: accuracy 33.3%, ECE 16.7%" in result.stdout
    assert "note: fake's probabilities are off" in result.stdout
    assert (tmp_path / "tickets-fake" / "train.jsonl").is_file()  # the default output
    again = run(app, ["training", "label", "-m", str(base), "--data", str(data)])
    assert again.exit_code != 0 and "pass --overwrite" in again.output
    events = run(app, ["training", "label", "-m", str(base), "--data", str(data), "--hard",
                       "-o", str(tmp_path / "hard"), "--events"])  # fmt: skip
    assert events.exit_code == 0, events.output
    last = json.loads(events.stdout.strip().splitlines()[-1])
    assert last["stage"] == "done" and last["metrics"]["labels"] == "hard"
    stages = {json.loads(line)["stage"] for line in events.stdout.strip().splitlines()}
    assert stages >= {"check", "load", "score", "write", "done"}
    missing = run(app, ["training", "label", "--data", str(data)])
    assert missing.exit_code != 0 and "pass -m TEACHER" in missing.output


def test_a_tiny_laya_teaches_and_a_student_fits(tmp_path):
    from tiny_models import write_marker

    teacher = write_marker(tmp_path / "laya", "laya")
    data = tickets(tmp_path / "tickets", count=60)
    result = label(teacher, data, output=tmp_path / "taught")
    assert result.examples == 60 and set(result.measured) == {"team"}
    examples = read_for(tmp_path / "taught").examples
    for example in examples:
        assert sum(example.targets["team"].values()) == pytest.approx(1)
        assert example.labels["team"] == max(
            example.targets["team"], key=example.targets["team"].get
        )
    student = fit(write_model(tmp_path / "fake"), tmp_path / "taught", output=tmp_path / "s")
    assert student.metrics["team"]["after"]["n"] == 12
