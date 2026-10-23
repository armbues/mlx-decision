"""How a command ends: exit codes, ``error: ...`` instead of a traceback, and the
last ``--events`` line whatever stops the run."""

import json
import sys

import pytest
from typer.testing import CliRunner

import mlx_decision.cli as cli
import mlx_decision.download as download
from fake_backend import write_model
from mlx_decision.cli import app


def invoke(*args, env=None):
    return CliRunner().invoke(app, list(args), env=env)


def last_event(output: str) -> dict | None:
    lines = [line for line in output.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else None


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_DECISION_HOME", str(tmp_path / "home"))


@pytest.mark.parametrize(
    ("raised", "code", "stage"),
    [(RuntimeError("the Hub is down"), 1, "error"), (KeyboardInterrupt(), 130, "cancelled")],
)
def test_an_events_run_ends_with_its_last_line_whatever_stops_it(monkeypatch, raised, code, stage):
    def check(repo_id):
        raise raised

    monkeypatch.setattr(download, "check_repo", check)
    result = invoke("download", "--events", "org/model")
    assert result.exit_code == code
    assert last_event(result.stdout)["stage"] == stage
    if stage == "error":
        assert last_event(result.stdout)["message"] == "the Hub is down"


def test_a_failure_before_the_first_event_still_ends_with_one(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": true}}\n')
    result = invoke("training", "fit", "--events", "--data", str(data))  # no -m
    assert result.exit_code == 1
    assert last_event(result.stdout) == {"stage": "error", "message": "pass -m MODEL (the base)"}


def test_a_run_its_arguments_refuse_prints_no_events():
    result = invoke("download", "--events", "--bits", "5", "org/model")
    assert result.exit_code == 2 and last_event(result.stdout) is None


def test_a_port_out_of_range_is_wrong_usage(tmp_path):
    model = write_model(tmp_path / "fake")
    result = invoke("server", "-m", str(model), "--port", "99999")
    assert result.exit_code == 2 and "99999" in result.output


def test_a_server_whose_default_model_does_not_load_ends_with_its_error(tmp_path, monkeypatch):
    import uvicorn

    from mlx_decision.gpu import Gpu

    async def broken(self):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(Gpu, "start", broken)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("served anyway"))
    result = invoke("server", "-m", str(write_model(tmp_path / "fake")))
    assert result.exit_code == 1
    assert "error: fake could not be loaded: weights missing" in result.stderr


def test_the_entry_point_shows_an_unexpected_error_without_a_traceback(monkeypatch, capsys):
    def boom(**kwargs):
        raise RuntimeError("something broke")

    monkeypatch.setattr(cli, "app", boom)
    monkeypatch.setattr(sys, "argv", ["mlx-decision", "run"])
    monkeypatch.delenv(cli.DEBUG_VARIABLE, raising=False)
    with pytest.raises(SystemExit) as stopped:
        cli.entry()
    assert stopped.value.code == 1
    assert capsys.readouterr().err.startswith("error: something broke")
    monkeypatch.setenv(cli.DEBUG_VARIABLE, "1")
    with pytest.raises(RuntimeError, match="something broke"):
        cli.entry()
