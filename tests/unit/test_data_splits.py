"""``data`` with several splits: one file per split (``{split}`` in ``-o``),
a sample once or per split, a record beside each file (``--record``,
``--roles``), every file written whole, and a stop that leaves none."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlx_decision.cli import app
from mlx_decision.data.make import read_record, record_path
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


def test_two_splits_with_events_and_records(saved, tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    pattern = str(out / "tickets-{split}.jsonl")
    more = ["--split", "train", "--split", "test", "--sample", "10", "--sample", "0"]
    more += ["--seed", "3", "--record", "--roles", "train,evaluate", "--events"]
    result = CliRunner().invoke(app, command(saved, pattern, *more))
    assert result.exit_code == 0, result.output
    lines = events(result.stdout)
    stages = [(line["stage"], line.get("step"), line.get("total")) for line in lines]
    assert stages == [
        ("check", 1, 2), ("read", 1, 2), ("check", 2, 2), ("read", 2, 2),
        ("write", None, None), ("done", None, None),
    ]  # fmt: skip
    assert lines[1]["metrics"]["split"] == "train" and lines[1]["metrics"]["examples"] == 10
    assert lines[3]["metrics"]["examples"] == 12  # 0: the whole split
    train, test = out / "tickets-train.jsonl", out / "tickets-test.jsonl"
    done = lines[-1]["metrics"]
    assert done["paths"] == [str(train), str(test)]
    assert [(f["split"], f["examples"]) for f in done["files"]] == [("train", 10), ("test", 12)]
    assert len(train.read_text().splitlines()) == 10
    assert len(test.read_text().splitlines()) == 12
    assert sorted(p.name for p in out.iterdir()) == [
        "tickets-test.jsonl",
        "tickets-test.source.json",
        "tickets-train.jsonl",
        "tickets-train.source.json",
    ]  # fmt: skip (no .partial left)

    record = read_record(train)
    assert record_path(train) == out / "tickets-train.source.json"
    assert (record["source"], record["split"], record["rows"]) == (str(saved), "train", 30)
    assert (record["sample"], record["seed"], record["examples"]) == (10, 3, 10)
    assert record["role"] == "train" and record["preset"] is None
    assert sum(record["labels"]["team"].values()) == 10
    assert record["get"] == {
        "name": "tickets",
        "files": [
            {"name": "tickets-train", "split": "train", "role": "train"},
            {"name": "tickets-test", "split": "test", "role": "evaluate"},
        ],
    }
    assert read_record(test)["role"] == "evaluate" and read_record(test)["sample"] is None
    assert read_record(test)["seed"] is None


def test_without_events_and_one_sample_for_every_split(saved, tmp_path):
    pattern = str(tmp_path / "t-{split}.jsonl")
    more = ["--split", "train", "--split", "test", "--sample", "5"]
    result = CliRunner().invoke(app, command(saved, pattern, *more))
    assert result.exit_code == 0, result.output
    for split in ("train", "test"):
        assert len((tmp_path / f"t-{split}.jsonl").read_text().splitlines()) == 5
        assert f"wrote {tmp_path / f't-{split}.jsonl'}" in result.output
    assert not record_path(tmp_path / "t-train.jsonl").exists()  # no --record: no record


def test_one_split_with_a_record_and_the_question_given(saved, tmp_path):
    question = tmp_path / "question.json"
    criteria = {"t": "Bugs", "b": "Money"}
    team = {"type": "choice", "instructions": "Which team?", "criteria": criteria}
    question.write_text(json.dumps({"team": team}))
    output = tmp_path / "one.jsonl"
    args = ["data", str(saved), "--split", "train", "--text", "text", "--label", "label",
            "-q", str(question), "--label-map", "tech=t,billing=b", "-o", str(output),
            "--record", "--events"]  # fmt: skip
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert events(result.stdout)[-1]["metrics"]["path"] == str(output)  # one split: as before
    record = read_record(output)
    assert record["question"]["team"]["criteria"] == {"t": "Bugs", "b": "Money"}
    assert record["label_map"] == {"tech": "t", "billing": "b"}
    assert record["role"] is None
    assert record["get"] == {"name": "one", "files": [{"name": "one", "split": "train",
                                                        "role": None}]}  # fmt: skip


@pytest.mark.parametrize(
    ("more", "message"),
    [
        (["--split", "train", "--split", "test"], "put {split} in -o"),
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
        (["--split", "train", "--roles", "train"], "--roles goes with --record"),
        (["--split", "train", "--record", "--roles", "train,report"], "one role per --split"),
        (["--split", "train", "--record", "--roles", "fit"], "each is one of train, report"),
    ],
)
def test_what_does_not_go_together(saved, tmp_path, more, message):
    name = "x.jsonl" if message == "put {split} in -o" else "x-{split}.jsonl"
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
            str(tmp_path / "s-{split}.jsonl"),
            "--split",
            "train",
            "--split",
            "test",
            "--show",
            "2",
        ),  # fmt: skip
    )
    assert shown.exit_code != 0 and "--show takes one --split" in shown.output


def test_a_stop_after_one_split_leaves_no_file(saved, tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    args = command(saved, str(out / "t-{split}.jsonl"), "--split", "train", "--split", "test",
                   "--record", "--roles", "train,report", "--events")  # fmt: skip
    env = {**os.environ, "FAKE_SLOW_SPLIT": "test", WATCH_VARIABLE: "1"}
    process = subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip
    seen = []
    for line in process.stdout:
        event = json.loads(line)
        seen.append((event["stage"], event.get("step")))
        if event["stage"] == "check" and event.get("step") == 2:
            break
    assert seen == [("check", 1), ("read", 1), ("check", 2)]
    process.send_signal(signal.SIGINT)
    rest, _ = process.communicate(timeout=10)
    assert process.returncode == 130
    assert events(rest)[-1] == {"stage": "cancelled", "message": "nothing was written"}
    assert list(out.iterdir()) == []  # neither the train file nor any partial
