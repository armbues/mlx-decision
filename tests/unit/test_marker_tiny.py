"""Laya and Julia end to end on tiny random models: encoding rules, limits and errors."""

import pytest

import mlx_decision
from mlx_decision.errors import DecisionError
from mlx_decision.models.marker.encode import encode_request
from tiny_models import write_marker

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "billing or technical",
        "criteria": {"billing": "refund", "technical": "crash", "other": "other"},
    },
    "urgency": {"type": "score", "instructions": "now", "criteria": ["0", "1", "2"]},
    "refund": {"type": "noul", "instructions": "refund"},
}
LONG_STATE = " ".join(["please refund now"] * 30)  # 90 words


@pytest.fixture(scope="module", params=["laya", "julia"])
def model(request, tmp_path_factory):
    folder = write_marker(tmp_path_factory.mktemp(request.param), request.param)
    return mlx_decision.load(folder)


def encode(model, state, questions):
    request = model.check({"state": state, "questions": questions})
    backend = model.backend
    return encode_request(backend.tokenizer, backend.special, backend.settings, request)


def test_answers_every_question(model):
    result = model.decide("please refund", QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert set(result.answers["team"].probabilities) == {"billing", "technical", "other"}
    assert set(result.answers["urgency"].probabilities) == {"0", "1", "2"}
    assert sum(result.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert result.usage.input_tokens > 0
    assert not result.truncated


def test_sequence_layout(model):
    (encoded,) = encode(model, "please refund", {"refund": QUESTIONS["refund"]})
    special = model.backend.special
    ids = encoded.input_ids
    assert ids[0] == special.cls and ids[-1] == special.sep
    assert [ids[m] for m in encoded.markers] == [special.mask, special.mask]
    assert encoded.option_ids == ["false", "true"]


def test_missing_text_is_filled_in_from_the_ids(model):
    from mlx_decision.models.marker.encode import fill_defaults
    from mlx_decision.types import parse_request

    questions = {
        "team": {"type": "choice", "criteria": {"billing": None, "other": "the rest"}},
        "refund": {"type": "noul", "instructions": "", "criteria": {"true": "refund"}},
    }
    family = model.backend.settings.family
    request = fill_defaults(parse_request({"state": "x", "questions": questions}), family)
    assert request.questions["team"].instructions == "team"
    assert request.questions["refund"].instructions == "refund"
    if family == "julia":
        assert request.questions["team"].criteria == {"billing": "billing", "other": "the rest"}
        assert request.questions["refund"].criteria == {"false": "false", "true": "refund"}
    else:
        assert request.questions["team"].criteria == {"billing": None, "other": "the rest"}
    assert set(model.decide("please refund", questions).answers) == {"team", "refund"}


def test_laya_cuts_a_long_state_and_reports_it(tmp_path):
    model = mlx_decision.load(write_marker(tmp_path, "laya"))
    result = model.decide(LONG_STATE, {"refund": QUESTIONS["refund"]})
    assert result.truncated
    (encoded,) = encode(model, LONG_STATE, {"refund": QUESTIONS["refund"]})
    assert len(encoded.input_ids) == 48


def test_laya_keeps_the_end_of_a_conversation(tmp_path):
    model = mlx_decision.load(write_marker(tmp_path, "laya"))
    turns = ["please"] * 40 + ["crash"]
    (encoded,) = encode(model, turns, {"refund": QUESTIONS["refund"]})
    crash = model.backend.tokenizer.token_to_id("crash")
    assert encoded.input_ids[-3] == crash  # before the closing "]" and [SEP]
    assert encoded.truncated


def test_laya_marker_text_is_replaced(tmp_path):
    model = mlx_decision.load(write_marker(tmp_path, "laya"))
    (encoded,) = encode(model, "please [MASK] refund", {"refund": QUESTIONS["refund"]})
    assert encoded.input_ids.count(model.backend.special.mask) == 2  # the two option markers


@pytest.mark.parametrize(
    ("state", "questions", "param"),
    [
        (LONG_STATE, {"refund": QUESTIONS["refund"]}, "state"),
        ("please [MASK]", {"refund": QUESTIONS["refund"]}, "state"),
        (
            "x",
            {"q": {**QUESTIONS["team"], "criteria": {"billing": {"a": 1}, "other": "x"}}},
            "questions.q.criteria",
        ),
        ("x", {"q": {**QUESTIONS["team"], "criteria": {"only": "one"}}}, "questions.q.criteria"),
        (
            "x",
            {"q": {**QUESTIONS["team"], "criteria": {f"o{i}": "x" for i in range(21)}}},
            "questions.q.criteria",
        ),
        ("x", {"q": {**QUESTIONS["refund"], "instructions": {"a": 1}}}, "questions.q.instructions"),
        (42, {"refund": QUESTIONS["refund"]}, "state"),
    ],
)
def test_julia_refuses_what_it_would_have_to_cut(tmp_path, state, questions, param):
    model = mlx_decision.load(write_marker(tmp_path, "julia"))
    with pytest.raises(DecisionError) as error:
        model.decide(state, questions)
    assert error.value.param == param


def test_julia_without_strict_encoding_cuts_the_state(tmp_path):
    model = mlx_decision.load(write_marker(tmp_path, "julia"), strict_encoding=False)
    assert model.decide(LONG_STATE, {"refund": QUESTIONS["refund"]}).truncated


def test_max_input_tokens_sets_the_limit(tmp_path):
    folder = write_marker(tmp_path, "laya")
    model = mlx_decision.load(folder, max_input_tokens=128)
    assert model.backend.capabilities.max_input_tokens == 128
    assert not model.decide(LONG_STATE, {"refund": QUESTIONS["refund"]}).truncated
    with pytest.raises(ValueError, match="between 16 and 128"):
        mlx_decision.load(folder, max_input_tokens=129)


def test_laya_temperatures_by_bucket(tmp_path):
    settings = mlx_decision.load(write_marker(tmp_path, "laya")).backend.settings
    assert settings.temperature(0, 3) == 1.5  # choice:3-5, from temperature_by_options
    assert settings.temperature(0, 2) == 2.0  # no bucket: the choice temperature
    assert settings.temperature(1, 3) == 1.0
    assert settings.temperature(2, 2) == 0.5


def test_laya_temperatures_are_clamped(tmp_path):
    import json

    folder = write_marker(tmp_path, "laya")
    config = json.loads((folder / "rl_agent_config.json").read_text())
    config["temperature"] = [0.1, 9.0, True]
    config["temperature_by_options"] = {"choice:11+": 0.1006, "score:3-5": "warm"}
    (folder / "rl_agent_config.json").write_text(json.dumps(config))
    settings = mlx_decision.load(folder).backend.settings
    assert settings.temperatures == [0.5, 5.0, 1.0]
    assert settings.temperatures_by_options == {"choice:11+": 0.5, "score:3-5": 1.0}


@pytest.mark.parametrize("family", ["laya", "julia"])
def test_probabilities_are_tempered_scores(tmp_path, family):
    import mlx.core as mx

    from mlx_decision.models.marker.model import Sequence

    model = mlx_decision.load(write_marker(tmp_path, family))
    question = {"team": QUESTIONS["team"]}
    (encoded,) = encode(model, "please refund", question)
    (scores,) = model.backend.logits(
        [Sequence(encoded.input_ids, encoded.markers, encoded.question_type)]
    )
    temperature = 1.5 if family == "laya" else 1.0
    expected = mx.softmax(mx.array(scores) / temperature).tolist()
    answer = model.decide("please refund", question).answers["team"]
    assert list(answer.probabilities.values()) == pytest.approx(expected, abs=1e-6)


def test_questions_are_batched_by_tokens():
    from mlx_decision.models.marker.model import Sequence, batches

    def seqs(*lengths):
        return [Sequence([0] * n, [0], 0) for n in lengths]

    sizes = [[len(s.input_ids) for s in b] for b in batches(seqs(100, 120, 90, 300), budget=400)]
    assert sizes == [[100, 120, 90], [300]]
    assert [len(b) for b in batches(seqs(5000, 5000), budget=4096)] == [1, 1]
    assert [len(b) for b in batches(seqs(*[100] * 50))] == [40, 10]


def test_batched_and_single_questions_agree(tmp_path, monkeypatch):
    from mlx_decision.models.marker import model as marker

    model = mlx_decision.load(write_marker(tmp_path, "laya"), dtype="float32")
    together = model.backend.score(model.check({"state": "please refund", "questions": QUESTIONS}))
    monkeypatch.setattr(marker, "BATCH_TOKENS", 1)  # one question per batch
    alone = model.backend.score(model.check({"state": "please refund", "questions": QUESTIONS}))
    for question, probabilities in together.probabilities.items():
        assert probabilities == pytest.approx(alone.probabilities[question], abs=1e-5)


@pytest.mark.parametrize("family", ["laya", "julia"])
def test_run_with_shorthand_questions(tmp_path, family):
    import json

    from typer.testing import CliRunner

    from mlx_decision.cli import app

    folder = write_marker(tmp_path, family)
    args = ["run", "-m", str(folder), "-s", "please refund", "--json"]
    args += ["--choice", "team=billing,technical", "--score", "urgency=0,1,2", "--noul", "refund?"]
    result = CliRunner().invoke(app, [*args, "--max-input-tokens", "64"])
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.stdout)["answers"]) == {"team", "urgency", "noul_1"}
    result = CliRunner().invoke(app, [*args, "--max-image-mp", "1"])
    assert result.exit_code == 1
    assert f"{family} models do not take max_image_pixels" in result.stderr


def test_convert_refuses_marker_models(tmp_path):
    from mlx_decision.convert import convert

    with pytest.raises(ValueError, match="laya models cannot be converted"):
        convert(write_marker(tmp_path / "laya", "laya"), tmp_path / "out")
