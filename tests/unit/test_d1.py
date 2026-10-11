"""The d1 family on a tiny random release (word-level tokenizer)."""

import json

import mlx.core as mx
import numpy as np
import pytest
from typer.testing import CliRunner

import mlx_decision
from mlx_decision.cli import app
from mlx_decision.errors import DecisionError
from mlx_decision.models.d1.prompt import prefix, suffix
from mlx_decision.models.d1.readout import probabilities
from mlx_decision.registry import detect_family
from mlx_decision.types import parse_request
from tiny_models import write_d1

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Charges", "technical": "Faults", "sales": None},
    },
    "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "angry"]},
}
STATE = "I was charged twice , please refund ."


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_d1(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def test_detected_as_d1_and_not_other_lfm2_vl_folders(folder, tmp_path):
    assert detect_family(folder).name == "d1"
    config = json.loads((folder / "config.json").read_text())
    del config["auto_map"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not a supported decision model"):
        detect_family(tmp_path)


def test_info(model):
    info = model.info()
    assert info["family"] == "d1"
    assert info["precision"] == "float32"
    assert info["max_input_tokens"] == 32768
    assert info["max_choice_options"] == 255
    assert info["max_score_levels"] == 10
    assert info["supports_images"] is False
    assert info["requires_instructions"] is False


def full_vocabulary_answer(model, state, question_id, question):
    """The readout over the whole vocabulary's log-probabilities, as the release reads it."""
    backend = model.backend
    tokenizer = backend.tokenizer
    data = {**question, "instructions": question.get("instructions") or question_id}
    ids = tokenizer.encode(prefix(state) + suffix(data, tokenizer), add_special_tokens=False).ids
    hidden = backend.model(mx.array([ids]))[0, -1]
    logits = (backend.model.model.embed_tokens.as_linear(hidden)).astype(mx.float32)
    logprobs = np.array(logits - mx.logsumexp(logits))
    return probabilities(data, tokenizer, logprobs)


@pytest.mark.parametrize("question_id", list(QUESTIONS))
def test_one_question_reads_the_option_tokens_as_over_the_whole_vocabulary(model, question_id):
    question = QUESTIONS[question_id]
    result = model.decide(STATE, {question_id: question})
    answer = result.answers[question_id]
    got = list(answer.probabilities.values()) if question["type"] != "noul" else [answer.noul]
    want = full_vocabulary_answer(model, STATE, question_id, question)
    assert np.allclose(got, want[: len(got)], atol=1e-4)


def test_several_questions_share_the_tokenized_prefix(model):
    backend = model.backend
    encoded = backend.encode(parse_request({"state": STATE, "questions": QUESTIONS}))
    tokenizer = backend.tokenizer
    shared = tokenizer.encode(prefix(STATE), add_special_tokens=False).ids
    for question in encoded:
        tail = tokenizer.encode(suffix(question.question, tokenizer), add_special_tokens=False).ids
        assert question.input_ids == shared + tail
    result = model.decide(STATE, QUESTIONS)
    assert result.usage.input_tokens == sum(len(q.input_ids) for q in encoded)
    assert list(result.answers["team"].probabilities) == ["billing", "technical", "sales"]
    assert sum(result.answers["anger"].probabilities.values()) == pytest.approx(1)


def test_missing_instructions_become_the_question_id(model):
    backend = model.backend
    question = {"type": "noul"}
    (encoded,) = backend.encode(parse_request({"state": STATE, "questions": {"urgent": question}}))
    assert encoded.question["instructions"] == "urgent"
    assert "QUESTION:\nurgent\n" in prefix(STATE) + suffix(encoded.question, backend.tokenizer)
    assert 0 < model.decide(STATE, {"urgent": question}).answers["urgent"].noul < 1


def test_limits(model):
    many = {f"o{i}": f"option {i}" for i in range(256)}
    with pytest.raises(DecisionError, match="at most 255 options"):
        model.decide(STATE, {"q": {"type": "choice", "instructions": "Pick", "criteria": many}})
    levels = [f"level {i}" for i in range(11)]
    with pytest.raises(DecisionError):
        model.decide(STATE, {"q": {"type": "score", "instructions": "Rate", "criteria": levels}})
    with pytest.raises(DecisionError, match="does not support images") as error:
        model.decide_request({"state": STATE, "questions": QUESTIONS, "images": ["data:x"]})
    assert error.value.param == "images"


def test_prompts_over_the_limit_are_refused(folder):
    model = mlx_decision.load(folder, max_input_tokens=40)
    with pytest.raises(DecisionError, match="the limit is 40") as error:
        model.decide(" ".join(["refund"] * 60), {"refund": QUESTIONS["refund"]})
    assert error.value.param == "questions.refund"


def test_load_options(folder):
    for bad in (0, 32769):
        with pytest.raises(ValueError, match="max_input_tokens"):
            mlx_decision.load(folder, max_input_tokens=bad)
    with pytest.raises(ValueError, match="d1 models do not take noul_floor"):
        mlx_decision.load(folder, noul_floor=0.9)


def test_run(folder):
    result = CliRunner().invoke(
        app, ["run", "-m", str(folder), "-s", STATE, "--noul", "refund=Wants money back", "--json"]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["model"] == folder.name
    assert 0 < body["answers"]["refund"]["noul"] < 1
