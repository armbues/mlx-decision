"""``mlx-decision training fit`` and ``training eval``, ``evaluate`` and the console pieces."""

import io
import json
import os
import signal
import weakref

import pytest
from typer.testing import CliRunner

import mlx_decision.model
from fake_backend import TEAM, write_model
from mlx_decision.cli import app, default_output
from mlx_decision.data.format import Example, training_splits, write_examples
from mlx_decision.metrics import confusion
from mlx_decision.training import Cancelled, Event, evaluate, fit
from mlx_decision.training.console import Progress, duration, interrupt_cancels

URGENT = {"type": "noul", "instructions": "Is it urgent?"}


def examples(count: int = 100) -> list[Example]:
    """The fake model favours billing; the labels are mostly tech."""
    labels = ("tech",) * 8 + ("billing", "sales")
    return [
        Example(
            f"e{i}",
            {"state": f"ticket {i}", "questions": {"team": TEAM, "urgent": URGENT}},
            {"team": labels[i % len(labels)]},
        )
        for i in range(count)
    ]


@pytest.fixture
def base(tmp_path):
    return write_model(tmp_path / "base")


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "tickets.jsonl"
    write_examples(path, examples())
    return path


@pytest.fixture
def cli():
    runner = CliRunner()
    return lambda *args: runner.invoke(app, ["training", *args])


def test_fit_shows_the_plan_and_reports_before_and_after(tmp_path, base, data, cli, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cli("fit", "-m", str(base), "--data", str(data))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "base-tickets" / "mlx_decision.json").exists()  # the default output
    assert "data: 100 examples: 20 holdout, 40 fit, 40 dev" in result.stderr
    assert "labels: team: billing 8, sales 8, tech 64" in result.stderr
    assert "method: calibrate" in result.stderr
    assert "warning: question 'team'" in result.stderr
    assert "team (20 holdout examples)" in result.stdout
    assert "accuracy        10.0%       80.0%" in result.stdout
    assert "wrote base-tickets" in result.stdout


def test_fit_events_are_json_lines_ending_with_done(tmp_path, base, data, cli):
    result = cli("fit", "-m", str(base), "--data", str(data), "-o", str(tmp_path / "f"),
                 "--events")  # fmt: skip
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0] == {"stage": "check", "message": "reading the data"}
    assert events[-1]["stage"] == "done" and "team" in events[-1]["metrics"]
    assert not result.stderr.strip() or "warning" not in result.stdout


def test_fit_errors_name_the_flag_and_end_the_events(tmp_path, base, data, cli):
    (tmp_path / "taken").mkdir()
    result = cli("fit", "-m", str(base), "--data", str(data), "-o", str(tmp_path / "taken"))
    assert result.exit_code == 1
    assert f"error: {tmp_path / 'taken'} exists; pass --overwrite to replace it" in result.stderr
    result = cli("fit", "-m", str(base), "--data", str(data), "-o", str(tmp_path / "taken"),
                 "--overwrite")  # fmt: skip
    assert result.exit_code == 0, result.output
    result = cli("fit", "-m", str(base), "--data", str(data), "-o", str(tmp_path / "x"),
                 "--method", "readout", "--events")  # fmt: skip
    assert result.exit_code == 1
    last = json.loads(result.stdout.splitlines()[-1])
    assert last["stage"] == "error" and "cannot be fitted with readout" in last["message"]


def test_a_cancelled_fit_exits_130(tmp_path, base, data, cli, monkeypatch):
    def cancelled(*args, **kwargs):
        raise Cancelled()

    monkeypatch.setattr("mlx_decision.training.fit", cancelled)
    result = cli("fit", "-m", str(base), "--data", str(data), "--events")
    assert result.exit_code == 130
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "stage": "cancelled",
        "message": "nothing was written",
    }
    assert "cancelled; nothing was written" in result.stderr


def test_the_first_ctrl_c_cancels_and_the_second_interrupts():
    stream = io.StringIO()
    before = signal.getsignal(signal.SIGINT)
    with interrupt_cancels(stream) as stop:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop.is_set()
        assert "stopping after the current step" in stream.getvalue()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) is before


def test_eval_scores_a_fitted_model_beside_its_base(tmp_path, base, data, cli):
    fitted = fit(base, data, output=tmp_path / "support").path
    result = cli("eval", "-m", str(fitted), "--data", str(data), "--compare", str(base))
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == "team (choice, 100 examples)"
    assert lines[1].split() == ["support", "fake"]  # the fake family's name
    assert lines[2].split() == ["accuracy", "80.0%", "10.0%"]
    assert "  confusion, fake (rows: gold, columns: predicted)" in lines
    as_json = cli("eval", "-m", str(fitted), "--data", str(data), "--json")
    assert as_json.exit_code == 0, as_json.output
    body = json.loads(as_json.stdout)
    [model] = body["models"]
    assert model["name"] == "support"
    assert model["questions"]["team"]["confusion"]["tech"] == {"billing": 0, "tech": 80, "sales": 0}
    assert "urgent" not in model["questions"]  # asked, not labelled
    assert body["data"]["examples"] == 100


def test_eval_without_labels_says_so(tmp_path, base, cli):
    path = tmp_path / "plain.jsonl"
    write_examples(path, [Example("a", {"state": "x", "questions": {"team": TEAM}}, {})])
    result = cli("eval", "-m", str(base), "--data", str(path))
    assert result.exit_code == 1
    assert "error: no example has a label" in result.stderr


def test_evaluate_matches_the_fit_report_and_loads_one_model_at_a_time(
    tmp_path, base, data, monkeypatch
):
    result = fit(base, data, output=tmp_path / "support")
    holdout = training_splits(examples(), seed=0).holdout
    loaded = []
    real_load = mlx_decision.model.load

    def load(*args, **kwargs):
        assert all(ref() is None for ref in loaded), "the previous model is still loaded"
        model = real_load(*args, **kwargs)
        loaded.append(weakref.ref(model))
        return model

    monkeypatch.setattr(mlx_decision.model, "load", load)
    events: list[Event] = []
    evaluated = evaluate(result.path, holdout, compare=base, on_event=events.append)
    assert [m.name for m in evaluated.models] == ["support", "fake"]
    after, before = (m.questions["team"] for m in evaluated.models)
    for key in ("accuracy", "macro_f1", "ece"):
        assert after[key] == result.metrics["team"]["after"][key]
        assert before[key] == result.metrics["team"]["before"][key]
    assert [e.stage for e in events if e.step is None] == ["load", "load", "done"]


def test_confusion_counts_gold_against_predicted():
    rows = [
        {"gold": "a", "predicted": "a"},
        {"gold": "a", "predicted": "b"},
        {"gold": "b", "predicted": None, "refused": "too long"},
        {"gold": "c", "predicted": "a"},
    ]
    assert confusion(rows, ["a", "b"]) == {
        "a": {"a": 1, "b": 1, "c": 0, "refused": 0},
        "b": {"a": 0, "b": 0, "c": 0, "refused": 1},
        "c": {"a": 1, "b": 0, "c": 0, "refused": 0},
    }


def test_progress_redraws_in_a_terminal_and_steps_by_tenths_elsewhere():
    terminal = io.StringIO()
    progress = Progress(terminal, terminal=True)
    for step in range(1, 5):
        progress(Event("before", step, 4, eta_s=3.0))
    progress(Event("calibrate", message="fitting the calibration"))
    text = terminal.getvalue()
    assert text.count("\r") == 4 and "4/4\033[K\n" in text
    assert text.endswith("calibrate: fitting the calibration\n")
    plain = io.StringIO()
    progress = Progress(plain, terminal=False)
    for step in range(1, 101):
        progress(Event("features", step, 100, eta_s=1.0))
    assert len(plain.getvalue().splitlines()) == 11  # 1 (0%), then each tenth


def test_durations_and_default_outputs(tmp_path):
    assert [duration(s) for s in (42, 185, 4320)] == ["42 s", "3 min 5 s", "1 h 12 min"]
    data = tmp_path / "tickets.jsonl"
    assert str(default_output("Cloudflare/clef-flash", data)) == "clef-flash-tickets"
    assert str(default_output("_models/laya/", data)) == "laya-tickets"
