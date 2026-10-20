"""The stdin watchdog: a run whose parent closed its stdin stops and leaves nothing."""

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fake_backend import TEAM, write_model
from mlx_decision.data.format import Example, write_examples
from mlx_decision.watchdog import GRACE_S, watch_stdin

TESTS = Path(__file__).resolve().parents[1]

# The command line with the fake model family registered first, as its own process.
LAUNCHER = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); import fake_backend; "
    "from mlx_decision.cli import app; app(prog_name='mlx-decision')"
)


def cli(*args: str, stdin=subprocess.PIPE) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, str(TESTS), *args],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture
def data(tmp_path):
    labels = ("tech",) * 8 + ("billing", "sales")
    path = tmp_path / "tickets.jsonl"
    write_examples(
        path,
        [
            Example(
                f"e{i}",
                {"state": f"ticket {i}", "questions": {"team": TEAM}},
                {"team": labels[i % len(labels)]},
            )
            for i in range(100)
        ],
    )
    return path


def read_until(process: subprocess.Popen, stage: str) -> list[dict]:
    """Event lines up to the first of ``stage``."""
    events = []
    for line in process.stdout:
        events.append(json.loads(line))
        if events[-1]["stage"] == stage:
            return events
    raise AssertionError(f"no {stage!r} event: {events}")


def ended_within(process: subprocess.Popen, seconds: float) -> tuple[int, list[dict], str]:
    """Wait for the process; its exit code, the event lines left and stderr."""
    process.stdin = None  # closed by the test (communicate would flush it)
    rest, errors = process.communicate(timeout=seconds + 5)
    return process.returncode, [json.loads(line) for line in rest.splitlines()], errors


def test_only_a_pipe_is_watched():
    flag = threading.Event()
    with open(os.devnull) as null, watch_stdin(flag, stdin=null) as watched:
        assert not watched  # EOF at once, but /dev/null is not a parent
    with watch_stdin(flag, stdin=io.StringIO("")) as watched:
        assert not watched  # no file behind it
    assert not flag.is_set()


def test_a_closed_pipe_sets_the_cancel_flag_and_an_ended_block_ignores_it():
    flag = threading.Event()
    read, write = os.pipe()
    with os.fdopen(read) as stdin, watch_stdin(flag, stdin=stdin, grace=60) as watched:
        assert watched
        os.write(write, b"dropped\n")
        assert not flag.wait(0.2)
        os.close(write)
        assert flag.wait(2)
    # The block ended within the grace: the watchdog stands down, no exit.

    flag = threading.Event()
    read, write = os.pipe()
    with os.fdopen(read) as stdin:
        with watch_stdin(flag, stdin=stdin, grace=0.1):
            pass
        os.close(write)
        assert not flag.wait(0.5)


def test_a_stuck_run_removes_its_leftovers_and_exits_130(tmp_path):
    """The run ignores its cancel flag: after the grace it is ended from outside."""
    folder, file = tmp_path / ".out.partial", tmp_path / ".out.part"
    folder.mkdir()
    (folder / "weights").write_text("half")
    file.write_text("half")
    script = (
        "import sys, threading, time; from mlx_decision.watchdog import watch_stdin\n"
        "with watch_stdin(threading.Event(), sys.argv[1:]):\n"
        "    print('ready', flush=True); time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(folder), str(file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout.readline() == "ready\n"
    process.stdin.close()
    started = time.monotonic()
    process.wait(timeout=GRACE_S + 5)
    assert GRACE_S - 0.5 < time.monotonic() - started < GRACE_S + 1
    assert process.returncode == 130
    assert "the run did not stop: quitting" in process.stderr.read()
    assert list(tmp_path.iterdir()) == []


def test_fit_events_stop_when_stdin_closes_and_write_nothing(tmp_path, data):
    base = write_model(tmp_path / "base", delay=0.05)
    process = cli("training", "fit", "-m", str(base), "--data", str(data),
                  "-o", str(tmp_path / "fitted"), "--events")  # fmt: skip
    read_until(process, "before")
    process.stdin.close()
    started = time.monotonic()
    code, rest, errors = ended_within(process, 5)
    assert time.monotonic() - started < 5
    assert code == 130
    assert rest[-1] == {"stage": "cancelled", "message": "nothing was written"}
    assert "stdin closed: stopping after the current step" in errors
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base", "tickets.jsonl"]


def test_a_fit_stuck_in_a_step_ends_within_5_s(tmp_path, data):
    base = write_model(tmp_path / "base", delay=60)  # one answer takes a minute
    process = cli("training", "fit", "-m", str(base), "--data", str(data),
                  "-o", str(tmp_path / "fitted"), "--events")  # fmt: skip
    for line in process.stdout:
        if "metrics" in json.loads(line):  # the plan: the first answer comes next
            break
    time.sleep(0.5)  # into that answer
    process.stdin.close()
    started = time.monotonic()
    code, rest, errors = ended_within(process, 5)
    assert time.monotonic() - started < 5
    assert code == 130 and rest == []  # quit from outside: no last event
    assert "the run did not stop: quitting" in errors
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base", "tickets.jsonl"]


def test_a_fit_that_ends_first_is_unaffected(tmp_path, data):
    base = write_model(tmp_path / "base")
    process = cli("training", "fit", "-m", str(base), "--data", str(data),
                  "-o", str(tmp_path / "fitted"), "--events")  # fmt: skip
    read_until(process, "done")
    process.stdin.close()
    code, rest, _ = ended_within(process, 5)
    assert code == 0 and rest == []
    assert (tmp_path / "fitted" / "mlx_decision.json").exists()


def test_without_a_pipe_on_stdin_nothing_changes(tmp_path, data):
    """/dev/null on stdin (a script, cron) reads EOF at once: no cancel."""
    base = write_model(tmp_path / "base")
    process = cli("training", "fit", "-m", str(base), "--data", str(data),
                  "-o", str(tmp_path / "fitted"), "--events", stdin=subprocess.DEVNULL)  # fmt: skip
    code, events, errors = ended_within(process, 30)
    assert code == 0, errors
    assert events[-1]["stage"] == "done"


def test_eval_events_stop_when_stdin_closes(tmp_path, data):
    base = write_model(tmp_path / "base", delay=0.05)
    process = cli("training", "eval", "-m", str(base), "--data", str(data), "--events")
    read_until(process, "score")
    process.stdin.close()
    code, rest, _ = ended_within(process, 5)
    assert code == 130
    assert rest[-1] == {"stage": "cancelled", "message": "stopped before the end"}
