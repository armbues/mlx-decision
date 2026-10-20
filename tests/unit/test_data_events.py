"""``data --events``: the progress as event lines, the file written whole, and
a stop that leaves nothing."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from typer.testing import CliRunner

from mlx_decision.cli import app
from mlx_decision.watchdog import WATCH_VARIABLE

TESTS = Path(__file__).resolve().parents[1]

# The command line as its own process, with the dataset's reading slowed
# down (FAKE_SLOW_DATA seconds) so a stop lands while it reads.
LAUNCHER = """
import os, sys, time
if "FAKE_SLOW_DATA" in os.environ:
    import mlx_decision.data.make as make
    real = make.DataPlan.make
    def slow(self):
        time.sleep(float(os.environ["FAKE_SLOW_DATA"]))
        return real(self)
    make.DataPlan.make = slow
from mlx_decision.cli import app
app(prog_name="mlx-decision")
"""


def write_csv(path: Path) -> Path:
    rows = ["text,label"] + [
        f"ticket number {i},{'tech' if i % 3 else 'billing'}" for i in range(30)
    ]
    path.write_text("\n".join(rows) + "\n")
    return path


def args(source: Path, output: Path) -> list[str]:
    return [
        "data", str(source), "--text", "text", "--label", "label", "--question", "Which team?",
        "-o", str(output), "--events",
    ]  # fmt: skip


def events(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


def test_the_events_and_the_file(tmp_path):
    output = tmp_path / "out" / "tickets.jsonl"
    output.parent.mkdir()
    result = CliRunner().invoke(app, args(write_csv(tmp_path / "t.csv"), output))
    assert result.exit_code == 0, result.output
    lines = events(result.stdout)
    assert [line["stage"] for line in lines] == ["check", "read", "write", "done"]
    assert lines[0]["message"].startswith("opening ")
    assert lines[1]["metrics"]["examples"] == 30
    labels = lines[1]["metrics"]["labels"]
    [counts] = labels.values()
    assert counts == {"tech": 20, "billing": 10}
    assert lines[-1]["metrics"] == {"path": str(output), "examples": 30, "labels": labels}
    assert len(output.read_text().splitlines()) == 30
    assert sorted(p.name for p in output.parent.iterdir()) == ["tickets.jsonl"]  # no .partial


def test_an_error_is_an_event(tmp_path):
    source = write_csv(tmp_path / "t.csv")
    output = tmp_path / "out.jsonl"
    result = CliRunner().invoke(app, [*args(source, output)[:3], "nope", *args(source, output)[4:]])
    assert result.exit_code == 1
    last = events(result.stdout)[-1]
    assert last["stage"] == "error" and "nope" in last["message"]
    assert not output.exists()
    shown = CliRunner().invoke(app, [*args(source, output), "--show", "3"])
    assert events(shown.stdout)[-1]["message"] == "--show and --events do not go together"


def run(source: Path, output: Path) -> subprocess.Popen:
    env = {**os.environ, "FAKE_SLOW_DATA": "30", WATCH_VARIABLE: "1"}
    return subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, *args(source, output)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip


def read_until(process: subprocess.Popen, stage: str) -> None:
    for line in process.stdout:
        if json.loads(line)["stage"] == stage:
            return
    raise AssertionError(f"no {stage} event")


def test_ctrl_c_stops_it_and_nothing_is_written(tmp_path):
    output = tmp_path / "out.jsonl"
    process = run(write_csv(tmp_path / "t.csv"), output)
    read_until(process, "check")
    time.sleep(0.3)
    process.send_signal(signal.SIGINT)
    rest, _ = process.communicate(timeout=10)
    assert process.returncode == 130
    assert events(rest)[-1] == {"stage": "cancelled", "message": "nothing was written"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.csv"]


def test_a_closed_stdin_stops_it(tmp_path):
    output = tmp_path / "out.jsonl"
    process = run(write_csv(tmp_path / "t.csv"), output)
    read_until(process, "check")
    started = time.monotonic()
    process.stdin.close()
    process.stdin = None
    process.communicate(timeout=10)
    assert process.returncode == 130 and time.monotonic() - started < 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.csv"]
