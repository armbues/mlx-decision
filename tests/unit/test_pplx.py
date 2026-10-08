"""pplx on a tiny random model: prompt, limits, errors and the decision config."""

import json

import pytest

import mlx_decision
from mlx_decision.errors import DecisionError
from mlx_decision.models.pplx.prompt import render
from mlx_decision.types import parse_request
from tiny_models import PPLX_CODES, write_pplx

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "billing or technical",
        "criteria": {"billing": "refund", "technical": None},
    },
    "urgency": {"type": "score", "instructions": "now", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "refund"},
}


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_pplx(tmp_path_factory.mktemp("pplx") / "tiny-pplx")


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def prompt_of(state, question) -> str:
    request = parse_request({"state": state, "questions": {"q": question}})
    text = render(request.state, request.questions["q"], PPLX_CODES)
    return text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


def test_info(model):
    info = model.info()
    assert info["family"] == "pplx"
    assert info["max_input_tokens"] == 8192
    assert not info["truncates_input"]
    assert info["max_choice_options"] == info["max_score_levels"] == 255
    assert info["precision"] == "float32"


def test_answers_every_question(model):
    result = model.decide("please refund", QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert set(result.answers["team"].probabilities) == {"billing", "technical"}
    assert sum(result.answers["urgency"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert 0 <= result.answers["refund"].noul <= 1
    assert result.usage.input_tokens > 0
    assert not result.truncated


def test_prompt_layout():
    text = prompt_of({"ticket": "Grüße", "n": 3}, QUESTIONS["team"])
    assert text == (
        'State:\n{"ticket": "Grüße", "n": 3}\n\n'
        "Question:\nbilling or technical\n\n"
        "Options:\nA: billing: refund\nB: technical\n\n"
        "Return only the letter code of the best option."
    )


def test_prompt_defaults():
    noul = prompt_of(3, {"type": "noul", "criteria": {"true": ""}})
    assert noul.startswith("State:\n3\n\nQuestion:\nChoose the best matching option.\n\n")
    assert "Options:\nA: No / false\nB: Yes / true\n\n" in noul
    score = prompt_of("s", {"type": "score", "instructions": {"q": 1}, "criteria": ["a", {"b": 2}]})
    assert 'Question:\n{"q": 1}\n\nOptions:\nA: a\nB: {"b": 2}\n\n' in score
    choice = prompt_of("s", {"type": "choice", "criteria": {"x": ["y", "z"]}})
    assert 'Options:\nA: x: ["y", "z"]\n\n' in choice


def test_255_options_answered_256_refused(model):
    criteria = {f"o{i}": None for i in range(255)}
    result = model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert len(result.answers["many"].probabilities) == 255
    criteria["o255"] = None
    with pytest.raises(DecisionError, match="at most 255") as error:
        model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert error.value.param == "questions.many.criteria"
    levels = [str(i) for i in range(256)]
    with pytest.raises(DecisionError, match="at most 255"):
        model.decide("please", {"s": {"type": "score", "criteria": levels}})


def test_input_over_the_limit_is_refused_before_any_pass(model, folder, monkeypatch):
    short = {"short": QUESTIONS["refund"]}
    (encoded,) = model.backend.encode(parse_request({"state": "please", "questions": short}))
    limit = len(encoded.input_ids) + 5
    model = mlx_decision.load(folder, max_input_tokens=limit)
    monkeypatch.setattr(model.backend, "probabilities", lambda *_: pytest.fail("ran a pass"))
    questions = {**short, "long": {**QUESTIONS["refund"], "instructions": "refund " * 60}}
    with pytest.raises(DecisionError, match=f"the limit is {limit} and input is not cut") as error:
        model.decide("please", questions)
    assert error.value.param == "questions.long"
    monkeypatch.undo()
    assert model.decide("please", short).answers  # the short one alone fits


@pytest.mark.parametrize("limit", [0, 8193])
def test_the_limit_can_only_be_lowered(folder, limit):
    with pytest.raises(ValueError, match="between 1 and 8192"):
        mlx_decision.load(folder, max_input_tokens=limit)


def test_prefix_cache_is_not_an_option(folder):
    with pytest.raises(ValueError, match="pplx models do not take prefix_cache_gb"):
        mlx_decision.load(folder, prefix_cache_gb=1.0)


def edit_config(folder, tmp_path, **changes):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    config = json.loads((copy / "decision_config.json").read_text())
    (copy / "decision_config.json").write_text(json.dumps({**config, **changes}))
    return copy


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pooling": "mean"}, "last-token pooling"),
        ({"attention_mode": "sliding"}, "unknown attention mode"),
        ({"temperature": 0}, "temperature"),
        ({"format_version": 2}, "format"),
    ],
)
def test_decision_config_is_checked(folder, tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        mlx_decision.load(edit_config(folder, tmp_path, **changes))


def test_causal_attention_mode(folder, tmp_path):
    model = mlx_decision.load(edit_config(folder, tmp_path, attention_mode="causal"))
    assert model.backend.backbone.language_model.model.causal
