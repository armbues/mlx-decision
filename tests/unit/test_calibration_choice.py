"""The closing calibration's choice: ``auto``, ``full``, ``temperature`` or
``none``, through ``fit``, the fitted folder, the report and the CLI."""

import json

import pytest
from test_readout_fit import STATES, TEAM, URGENT, labelled
from typer.testing import CliRunner

import mlx_decision
from mlx_decision.cli import app
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import fit, fitting
from mlx_decision.training.console import format_fit
from mlx_decision.training.wizard import FitPlan
from tiny_models import write_d3


@pytest.fixture
def short_search(monkeypatch):
    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 50)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return write_d3(tmp_path_factory.mktemp("tiny") / "tiny-d3")


def test_none_drops_the_closing_calibration_and_needs_another_kind():
    assert fitting.check_calibration("none", ["readout", "calibrate"]) == ["readout"]
    assert fitting.check_calibration("auto", ["readout", "calibrate"]) == ["readout", "calibrate"]
    with pytest.raises(ValueError, match="would change nothing"):
        fitting.check_calibration("none", ["calibrate"])
    with pytest.raises(ValueError, match="one of auto, full, temperature, none"):
        fitting.check_calibration("bias", ["calibrate"])


def test_auto_wants_enough_examples_of_every_option_and_no_training(monkeypatch):
    monkeypatch.setattr(fitting, "BIAS_MIN_PER_OPTION", 3)
    three = ["a", "a", "a", "b", "b", "b"]
    assert fitting._uses_bias("auto", three, ["a", "b"], trained=False)
    assert not fitting._uses_bias("auto", three, ["a", "b"], trained=True)
    assert not fitting._uses_bias("auto", ["a", "a", "a", "b", "b"], ["a", "b"], trained=False)
    assert not fitting._uses_bias("auto", ["a"] * 9, ["a", "b"], trained=False)  # b never seen
    assert fitting._uses_bias("full", ["a"], ["a", "b"], trained=True)
    assert not fitting._uses_bias("temperature", three, ["a", "b"], trained=False)


def calibrations(result) -> dict:
    return {entry["question"]: entry["bias"] for entry in result.report["calibrations"]}


@pytest.mark.parametrize("choice", ["full", "temperature"])
def test_full_and_temperature(tmp_path, base, short_search, choice):
    result = fit(base, labelled(), output=tmp_path / "fitted", calibration=choice)
    assert result.kinds == ["readout", "calibrate"]
    biased = {q: bias is not None for q, bias in calibrations(result).items()}
    assert biased == {"team": choice == "full", "urgent": choice == "full"}
    expected = {"choice": choice, "questions": {"team": choice, "urgent": choice}}
    assert result.report["calibration"] == expected
    assert result.report["options"]["calibration"] == choice
    marker = json.loads((result.path / MARKER_FILE).read_text())
    assert marker["fit"]["calibration"] == expected
    assert f"calibration ({choice}): team {choice}, urgent {choice}" in format_fit(
        result.report, str(result.path)
    )


def test_auto_after_a_readout_and_alone(tmp_path, base, short_search, monkeypatch):
    result = fit(base, labelled(), output=tmp_path / "readout")
    assert result.report["calibration"] == {
        "choice": "auto", "questions": {"team": "temperature", "urgent": "temperature"}
    }  # fmt: skip
    # Alone: the calibration reads the whole training part, 96 examples, 48 of each.
    alone = fit(base, labelled(), output=tmp_path / "alone", method="calibrate")
    assert alone.report["calibration"]["questions"] == {"team": "full", "urgent": "full"}
    monkeypatch.setattr(fitting, "BIAS_MIN_PER_OPTION", 60)
    few = fit(base, labelled(), output=tmp_path / "few", method="calibrate")
    assert few.report["calibration"]["questions"] == {
        "team": "temperature",
        "urgent": "temperature",
    }


def test_none_answers_as_the_readout_alone(tmp_path, base, short_search):
    calibrated = fit(base, labelled(), output=tmp_path / "full", calibration="full")
    bare = fit(base, labelled(), output=tmp_path / "none", calibration="none")
    assert bare.kinds == ["readout"] and bare.report["calibrations"] == []
    assert bare.report["calibration"] == {"choice": "none", "questions": {}}
    assert "calibration: none" in format_fit(bare.report, str(bare.path))
    marker = json.loads((bare.path / MARKER_FILE).read_text())
    assert marker["fit"]["kinds"] == ["readout"]
    # The same readout (the same seed); the full fit without its calibrations
    # answers as the one fitted with none.
    uncalibrated = mlx_decision.load(calibrated.path)
    uncalibrated.backend.fit.calibrations.clear()
    plain = mlx_decision.load(bare.path)
    for state in STATES:
        questions = {"team": TEAM, "urgent": URGENT}
        got = plain.decide(state, questions).answers
        want = uncalibrated.decide(state, questions).answers
        assert got["urgent"].noul == pytest.approx(want["urgent"].noul, abs=1e-6)
        for option, p in got["team"].probabilities.items():
            assert p == pytest.approx(want["team"].probabilities[option], abs=1e-6)


def test_the_cli_takes_the_choice(tmp_path, base, short_search):
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(e.to_json()) + "\n" for e in labelled()))
    runner = CliRunner()
    refused = runner.invoke(app, ["training", "fit", "-m", str(base), "--data", str(data),
                                  "-o", str(tmp_path / "x"), "--method", "calibrate",
                                  "--calibration", "none", "--yes"])  # fmt: skip
    assert refused.exit_code != 0 and "would change nothing" in refused.output
    done = runner.invoke(app, ["training", "fit", "-m", str(base), "--data", str(data),
                               "-o", str(tmp_path / "t"), "--calibration", "temperature",
                               "--yes"])  # fmt: skip
    assert done.exit_code == 0, done.output
    assert "calibration (temperature): team temperature" in done.output
    plan = FitPlan(model="m", data="d", output="o", calibration="none")
    assert plan.arguments()[-2:] == ["--calibration", "none"]
    assert "--calibration" not in FitPlan(model="m", data="d", output="o").arguments()
