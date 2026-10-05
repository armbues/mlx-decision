"""The Python API, run through the fake backend."""

import json

import pytest

import mlx_decision
from fake_backend import FakeBackend, write_model
from mlx_decision import Choice, DecisionModel, Noul, Result, Score
from mlx_decision.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

TRIAGE = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {"technical": "Bugs", "billing": "Payments", "sales": "Pricing"},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm", "Frustrated", "Angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "The message is urgent"},
}


def test_load_finds_the_family_through_the_marker(fake_model):
    assert isinstance(fake_model, DecisionModel)
    assert isinstance(fake_model.backend, FakeBackend)
    assert fake_model.name == "fake"


def test_load_passes_options_to_the_backend(fake_model_path):
    model = mlx_decision.load(fake_model_path, max_input_tokens=3)
    assert model.backend.capabilities.max_input_tokens == 3


def test_load_rejects_a_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        mlx_decision.load(tmp_path / "nowhere")


def test_load_rejects_an_unknown_folder(tmp_path):
    with pytest.raises(ValueError, match="not a supported decision model"):
        mlx_decision.load(tmp_path)


def test_decide_answers_every_question(fake_model):
    result = fake_model.decide("My Stripe integration fails", TRIAGE)
    assert isinstance(result, Result)
    assert list(result.answers) == list(TRIAGE)
    department, frustration, urgent = result.answers.values()
    assert isinstance(department, ChoiceAnswer)
    assert department.choice == "technical"
    assert isinstance(frustration, ScoreAnswer)
    assert isinstance(urgent, NoulAnswer)
    assert urgent.noul == 0.75
    assert result.model == "fake"
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 0
    assert result.truncated is False


def test_decide_takes_typed_helpers(fake_model):
    helpers = {
        "department": Choice(instructions="Which team", criteria={"a": None, "b": "B"}),
        "frustration": Score(criteria=["low", "high"]),
        "is_urgent": Noul(instructions="Urgent?"),
    }
    result = fake_model.decide("text", helpers)
    assert result.answers["department"].choice == "a"
    assert result.answers["frustration"].legend == {"0": "low", "1": "high"}


def test_decide_request_matches_decide(fake_model):
    body = {"state": {"ticket": "Stripe fails"}, "questions": TRIAGE, "model": "jev-latest"}
    assert fake_model.decide_request(body) == fake_model.decide(body["state"], TRIAGE)


def test_to_wire_rounds_to_four_decimals_and_hides_truncated(fake_model_path):
    write_model(
        fake_model_path,
        fixed={"department": {"technical": 0.123456, "billing": 0.654321, "sales": 0.222223}},
    )
    result = mlx_decision.load(fake_model_path).decide("a b c", TRIAGE)
    wire = result.to_wire()
    assert wire["answers"]["department"]["probabilities"] == {
        "technical": 0.1235,
        "billing": 0.6543,
        "sales": 0.2222,
    }
    assert result.answers["department"].probabilities["technical"] == 0.123456
    assert set(wire) == {"model", "answers", "usage"}
    assert wire["usage"] == {"input_tokens": 3, "output_tokens": 0}
    assert json.loads(json.dumps(wire)) == wire


def test_wire_answers_carry_their_type(fake_model):
    wire = fake_model.decide("text", TRIAGE).to_wire()
    assert {answer["type"] for answer in wire["answers"].values()} == {"choice", "score", "noul"}
    assert set(wire["answers"]["is_urgent"]) == {"type", "noul"}
    assert set(wire["answers"]["department"]) == {"type", "choice", "confidence", "probabilities"}
    assert set(wire["answers"]["frustration"]) == {
        "type",
        "score",
        "confidence",
        "legend",
        "probabilities",
    }


def test_truncation_is_reported(fake_model_path):
    model = mlx_decision.load(fake_model_path, max_input_tokens=2)
    result = model.decide("one two three", TRIAGE)
    assert result.truncated is True
    assert result.usage.input_tokens == 2


def test_without_a_metal_gpu_loading_says_why(fake_model_path, monkeypatch):
    import mlx.core as mx
    from typer.testing import CliRunner

    from mlx_decision.cli import app
    from mlx_decision.errors import PlatformError

    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(PlatformError, match="Apple Silicon"):
        mlx_decision.load(fake_model_path)
    result = CliRunner().invoke(app, ["run", "-m", str(fake_model_path), "-s", "x", "--noul", "q?"])
    assert result.exit_code == 1
    assert "error: mlx-decision runs models on Apple Silicon Macs" in result.stderr
