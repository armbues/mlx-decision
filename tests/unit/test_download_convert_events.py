"""``download --events`` and ``convert --events``: event lines, Ctrl-C and a closed stdin."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

import fake_backend  # noqa: F401 - registers the fake family
import fake_hub
import mlx_decision.download as download
from mlx_decision.cli import app
from mlx_decision.convert import convert
from mlx_decision.events import Cancelled
from mlx_decision.watchdog import WATCH_VARIABLE
from tiny_models import write_clef

TESTS = Path(__file__).resolve().parents[1]
SIZES = {"model.safetensors": 3_000_000, "tokenizer.json": 500_000}
TOTAL = sum(SIZES.values()) + len('{"family": "fake"}')

# The command line with the fake family and the fake Hub (from FAKE_HUB_*),
# and with every tensor of a conversion slowed down (FAKE_SLOW_TENSORS), as
# its own process.
LAUNCHER = """
import os, sys, time
sys.path.insert(0, sys.argv.pop(1))
import fake_backend, fake_hub
if "FAKE_HUB_SIZES" in os.environ:
    fake_hub.from_env()
if "FAKE_SLOW_TENSORS" in os.environ:
    import mlx_decision.backbones.qwen3_5.load as load
    real = load.mx.eval
    def slow(*arrays):
        time.sleep(float(os.environ["FAKE_SLOW_TENSORS"]))
        return real(*arrays)
    load.mx = type(sys)("mx_slow")
    load.mx.__dict__.update({**vars(__import__("mlx.core").core), "eval": slow})
from mlx_decision.cli import app
app(prog_name="mlx-decision")
"""


def lines(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines()]


def invoke(*args):
    return CliRunner().invoke(app, list(args))


@pytest.fixture
def hub(monkeypatch, tmp_path):
    monkeypatch.setattr(download.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    fake_hub.install(SIZES, patch=monkeypatch.setattr)


def process(*args, env=None) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", LAUNCHER, str(TESTS), *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, WATCH_VARIABLE: "1", **(env or {})},  # as the Studio runs it
    )


def read_until(job: subprocess.Popen, stage: str) -> list[dict]:
    events = []
    for line in job.stdout:
        events.append(json.loads(line))
        if events[-1]["stage"] == stage:
            return events
    raise AssertionError(f"no {stage!r} event: {events}")


def ended(job: subprocess.Popen, seconds: float = 5) -> tuple[int, list[dict], str]:
    """Wait for ``job``: its exit code, the event lines left and stderr."""
    if job.stdin is not None and job.stdin.closed:
        job.stdin = None  # closed by the test (communicate would flush it)
    started = time.monotonic()
    rest, errors = job.communicate(timeout=seconds + 10)
    assert time.monotonic() - started < seconds, errors
    return job.returncode, lines(rest), errors


def stop(job: subprocess.Popen, how: str) -> None:
    if how == "sigint":
        job.send_signal(signal.SIGINT)
    else:
        job.stdin.close()


# download


def test_download_events_into_a_folder(hub, tmp_path):
    result = invoke("download", fake_hub.REPO, "--local-dir", str(tmp_path / "m"), "--events")
    assert result.exit_code == 0, result.output
    events = lines(result.stdout)
    assert events[0] == {"stage": "check", "message": f"checking {fake_hub.REPO}"}
    assert events[1]["stage"] == "check"
    assert events[1]["metrics"] == {"family": "fake", "bytes": TOTAL}
    steps = [event for event in events if event["stage"] == "download"]
    assert steps[-1]["step"] == steps[-1]["total"] == TOTAL
    assert [event["step"] for event in steps] == sorted(event["step"] for event in steps)
    assert events[-1] == {
        "stage": "done",
        "metrics": {"path": str(tmp_path / "m"), "family": "fake"},
        "message": f"downloaded {fake_hub.REPO} to {tmp_path / 'm'}",
    }
    assert (tmp_path / "m" / "model.safetensors").stat().st_size == SIZES["model.safetensors"]
    assert not (tmp_path / ".m.partial").exists()


def test_download_events_into_the_cache(hub, tmp_path):
    result = invoke("download", fake_hub.REPO, "--events")
    assert result.exit_code == 0, result.output
    done = lines(result.stdout)[-1]
    snapshot = tmp_path / "hub" / "models--org--fake" / "snapshots" / fake_hub.SHA
    assert done["metrics"]["path"] == str(snapshot)


def test_download_events_ask_nothing_and_end_errors_with_an_event(hub, tmp_path, monkeypatch):
    import mlx_decision.cli as cli

    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: True)  # a terminal, yet no menus
    result = invoke("download", "--events")
    assert result.exit_code == 1
    assert lines(result.stdout)[-1] == {
        "stage": "error",
        "message": "give a repo id, e.g. mlx-decision download Cloudflare/clef-flash",
    }
    result = invoke("download", "org/missing", "--events")
    assert result.exit_code == 1
    assert lines(result.stdout)[-1]["message"].startswith("org/missing: not found on the Hub")


def test_download_without_events_prints_lines_as_before(hub, tmp_path):
    result = invoke("download", fake_hub.REPO, "--local-dir", str(tmp_path / "m"))
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    assert f"downloaded {fake_hub.REPO} to {tmp_path / 'm'}" in result.stderr


@pytest.mark.parametrize("how", ["sigint", "stdin"])
@pytest.mark.parametrize("into", ["folder", "cache"])
def test_a_stopped_download_leaves_nothing(tmp_path, how, into):
    env = {
        "FAKE_HUB_SIZES": json.dumps({"model.safetensors": 20_000_000}),
        "FAKE_HUB_DELAY": "0.1",  # 2 s for the whole download
        "HF_HUB_CACHE": str(tmp_path / "hub"),
    }
    where = ["--local-dir", str(tmp_path / "models" / "fake")] if into == "folder" else []
    job = process("download", fake_hub.REPO, *where, "--events", env=env)
    read_until(job, "download")
    stop(job, how)
    code, rest, errors = ended(job)
    assert code == 130, errors
    assert rest[-1] == {"stage": "cancelled", "message": "nothing was kept"}
    if into == "folder":
        assert list((tmp_path / "models").iterdir()) == []
    else:
        assert list((tmp_path / "hub").iterdir()) == []


def test_a_download_that_ends_first_is_unaffected(tmp_path):
    env = {"FAKE_HUB_SIZES": json.dumps(SIZES), "HF_HUB_CACHE": str(tmp_path / "hub")}
    folder = str(tmp_path / "m")
    job = process("download", fake_hub.REPO, "--local-dir", folder, "--events", env=env)
    read_until(job, "done")
    job.stdin.close()
    code, rest, _ = ended(job)
    assert code == 0 and rest == []
    assert (tmp_path / "m" / "model.safetensors").exists()


# convert


@pytest.fixture
def release(tmp_path):
    return write_clef(tmp_path / "tiny-clef")


def test_convert_events(release, tmp_path):
    out = tmp_path / "tiny-q8"
    result = invoke("convert", "-m", str(release), "-o", str(out), "-q", "--events")
    assert result.exit_code == 0, result.output
    events = lines(result.stdout)
    assert events[0] == {"stage": "load", "message": f"reading {release}"}
    stages = [event["stage"] for event in events]
    assert stages[1:] == ["convert"] * (len(stages) - 3) + ["write", "done"]
    assert events[-3]["step"] == events[-3]["total"] > 10  # the last tensor, always shown
    assert events[-1] == {"stage": "done", "metrics": {"path": str(out)}, "message": f"wrote {out}"}
    assert (out / "mlx_decision.json").exists()
    assert result.stderr == ""


def test_convert_without_events_prints_lines_as_before(release, tmp_path):
    result = invoke("convert", "-m", str(release), "-o", str(tmp_path / "copy"))
    assert result.exit_code == 0, result.output
    assert result.stdout == f"wrote {tmp_path / 'copy'}\n"
    assert result.stderr == f"converting {release} to {tmp_path / 'copy'} ...\n"


def test_a_cancelled_conversion_writes_nothing(release, tmp_path):
    flag = threading.Event()
    seen = []

    def on_event(event):
        seen.append(event.stage)
        if event.stage == "convert":
            flag.set()

    with pytest.raises(Cancelled):
        convert(release, tmp_path / "out", on_event=on_event, cancel=flag, bits=8)
    assert seen[-1] == "convert" and "write" not in seen
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tiny-clef"]


@pytest.mark.parametrize("how", ["sigint", "stdin"])
def test_a_stopped_conversion_leaves_nothing(release, tmp_path, how):
    out = tmp_path / "tiny-q8"
    job = process("convert", "-m", str(release), "-o", str(out), "-q", "--events",
                  env={"FAKE_SLOW_TENSORS": "0.2"})  # fmt: skip
    read_until(job, "convert")
    stop(job, how)
    code, rest, errors = ended(job)
    assert code == 130, errors
    assert rest[-1] == {"stage": "cancelled", "message": "nothing was written"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["tiny-clef"]


def test_a_killed_studio_leaves_no_partial_download(tmp_path):
    """The Studio killed: its pipes close, stdout and stderr too. The cancel
    line cannot be written, and the partial folder still goes."""
    env = {
        "FAKE_HUB_SIZES": json.dumps({"model.safetensors": 20_000_000}),
        "FAKE_HUB_DELAY": "0.1",
        "HF_HUB_CACHE": str(tmp_path / "hub"),
    }
    job = process("download", fake_hub.REPO, "--local-dir", str(tmp_path / "models" / "fake"),
                  "--events", env=env)  # fmt: skip
    read_until(job, "download")
    for pipe in (job.stdin, job.stdout, job.stderr):
        pipe.close()
    started = time.monotonic()
    code = job.wait(timeout=15)
    assert time.monotonic() - started < 5
    assert code == 130
    assert list((tmp_path / "models").iterdir()) == []
