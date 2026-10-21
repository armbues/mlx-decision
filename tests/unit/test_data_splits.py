"""``data`` with several splits: one file per split in a dataset folder, a
sample once or per split, the roles in ``dataset.json``, the folder written
whole, and a stop that leaves nothing."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlx_decision.cli import app
from mlx_decision.watchdog import WATCH_VARIABLE

datasets = pytest.importorskip("datasets")

# The command line as its own process; reading the split FAKE_SLOW_SPLIT
# takes 30 s, so a stop lands after the other split was read.
LAUNCHER = """
import os, time
import mlx_decision.data.make as make
real = make.DataPlan.make
def slow(self):
    if self.split == os.environ.get("FAKE_SLOW_SPLIT"):
        time.sleep(30)
    return real(self)
make.DataPlan.make = slow
from mlx_decision.cli import app
app(prog_name="mlx-decision")
"""


@pytest.fixture
def saved(tmp_path) -> Path:
    """A saved dataset with a train (30 rows) and a test (12 rows) split."""

    def split(count, offset):
        texts = [f"ticket number {offset + i}" for i in range(count)]
        labels = [i % 3 == 0 for i in range(count)]
        names = ["tech" if not label else "billing" for label in labels]
        return datasets.Dataset.from_dict({"text": texts, "label": names})

    folder = tmp_path / "tickets"
    datasets.DatasetDict({"train": split(30, 0), "test": split(12, 100)}).save_to_disk(folder)
    return folder


def command(source: Path, output: str, *more: str) -> list[str]:
    return [
        "data", str(source), "--text", "text", "--label", "label", "--question", "Which team?",
        "--id", "team", "-o", output, *more,
    ]  # fmt: skip


def events(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_two_splits_with_events(saved, tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    folder = out / "tickets"
    more = ["--split", "train", "--split", "test", "--sample", "10", "--sample", "0"]
    more += ["--seed", "3", "--roles", "train,evaluate", "--events"]
    result = CliRunner().invoke(app, command(saved, str(folder), *more))
    assert result.exit_code == 0, result.output
    lines = events(result.stdout)
    stages = [(line["stage"], line.get("step"), line.get("total")) for line in lines]
    assert stages == [
        ("check", 1, 2), ("read", 1, 2), ("check", 2, 2), ("read", 2, 2),
        ("write", None, None), ("done", None, None),
    ]  # fmt: skip
    assert lines[1]["metrics"]["split"] == "train" and lines[1]["metrics"]["examples"] == 10
    assert lines[3]["metrics"]["examples"] == 12  # 0: the whole split
    done = lines[-1]["metrics"]
    assert done["path"] == str(folder)
    assert [(f["split"], f["role"], f["examples"]) for f in done["files"]] == [
        ("train", "train", 10), ("test", "evaluate", 12)]  # fmt: skip
    assert len((folder / "train.jsonl").read_text().splitlines()) == 10
    assert len((folder / "test.jsonl").read_text().splitlines()) == 12
    assert sorted(p.name for p in out.iterdir()) == ["tickets"]  # no .partial left
    body = json.loads((folder / "dataset.json").read_text())
    assert body["splits"]["train"] == {"role": "train", "split": "train", "rows": 30,
                                       "sample": 10, "seed": 3, "examples": 10}  # fmt: skip
    assert body["splits"]["test"]["sample"] is None and body["splits"]["test"]["seed"] is None
    assert body["source"]["source"] == str(saved)
    assert body["questions"]["team"]["instructions"] == "Which team?"


def test_without_events_and_one_sample_for_every_split(saved, tmp_path):
    folder = tmp_path / "t"
    more = ["--split", "train", "--split", "test", "--sample", "5"]
    result = CliRunner().invoke(app, command(saved, str(folder), *more))
    assert result.exit_code == 0, result.output
    for split in ("train", "test"):
        assert len((folder / f"{split}.jsonl").read_text().splitlines()) == 5
    assert f"wrote {folder}: train (train, 5 examples), test (report, 5 examples)" in result.output


def test_one_split_into_a_file_with_the_question_given(saved, tmp_path):
    question = tmp_path / "question.json"
    criteria = {"t": "Bugs", "b": "Money"}
    team = {"type": "choice", "instructions": "Which team?", "criteria": criteria}
    question.write_text(json.dumps({"team": team}))
    output = tmp_path / "one.jsonl"
    args = ["data", str(saved), "--split", "train", "--text", "text", "--label", "label",
            "-q", str(question), "--label-map", "tech=t,billing=b", "-o", str(output),
            "--events"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert events(result.stdout)[-1]["metrics"]["path"] == str(output)
    first = json.loads(output.read_text().splitlines()[0])
    assert first["questions"]["team"]["criteria"] == criteria and first["labels"] == {"team": "b"}


@pytest.mark.parametrize(
    ("more", "message"),
    [
        (["--split", "train", "--split", "test"], "-o names a dataset folder"),
        (["--split", "train", "--split", "train"], "each split once"),
        (
            [
                "--split",
                "train",
                "--split",
                "test",
                "--sample",
                "1",
                "--sample",
                "2",
                "--sample",
                "3",
            ],
            "once for every split, or once per --split",
        ),  # fmt: skip
        (["--split", "train", "--roles", "train,report"], "one entry per --split"),
        (["--split", "train", "--roles", "fit"], "none of train, report, evaluate"),
    ],
)
def test_what_does_not_go_together(saved, tmp_path, more, message):
    name = "x.jsonl" if message == "-o names a dataset folder" else "x"
    output = str(tmp_path / name)
    for events_flag in ([], ["--events"]):
        result = CliRunner().invoke(app, command(saved, output, *more, *events_flag))
        assert result.exit_code != 0
        assert message in result.output, result.output
        if events_flag:
            assert events(result.stdout)[-1]["stage"] == "error"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["tickets"]
    shown = CliRunner().invoke(
        app,
        command(
            saved,
            str(tmp_path / "s"),
            "--split",
            "train",
            "--split",
            "test",
            "--show",
            "2",
        ),  # fmt: skip
    )
    assert shown.exit_code != 0 and "--show takes one --split" in shown.output


def test_a_stop_after_one_split_leaves_no_folder(saved, tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    args = command(saved, str(out / "t"), "--split", "train", "--split", "test",
                   "--roles", "train,report", "--events")  # fmt: skip
    env = {**os.environ, "FAKE_SLOW_SPLIT": "test", WATCH_VARIABLE: "1"}
    process = subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip
    for line in process.stdout:
        event = json.loads(line)
        if event["stage"] == "check" and event.get("step") == 2:
            break
    process.stdin.close()  # the Studio gone: the watchdog stops the run
    process.wait(timeout=10)
    rest = process.stdout.read()
    assert process.returncode == 130
    assert events(rest)[-1] == {"stage": "cancelled", "message": "nothing was written"}
    assert list(out.iterdir()) == []  # neither the folder nor its partial
