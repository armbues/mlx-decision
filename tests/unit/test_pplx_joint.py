"""pplx v1.2 (joint requests) on tiny random models: detection, limits, chunks and errors."""

import json
import shutil

import pytest

import mlx_decision
from mlx_decision.errors import DecisionError
from mlx_decision.models.pplx.backend import JointBackend
from mlx_decision.models.pplx.joint import escape, escaped_tokens, render
from mlx_decision.registry import detect_family
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


@pytest.fixture(scope="module", params=["27b", "flash"])
def folder(request, tmp_path_factory):
    return write_pplx(
        tmp_path_factory.mktemp("pplx") / f"tiny-{request.param}", joint=request.param
    )


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def copy_of(folder, tmp_path, decision=None, joint=None, remove=()):
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    for name, changes in (("decision_config.json", decision), ("joint_config.json", joint)):
        if changes:
            config = json.loads((copy / name).read_text())
            (copy / name).write_text(json.dumps({**config, **changes}))
    for name in remove:
        (copy / name).unlink()
    return copy


def test_detected_and_loaded_as_joint(folder, model):
    assert detect_family(folder).name == "pplx"
    assert isinstance(model.backend, JointBackend)
    info = model.info()
    assert info["family"] == "pplx"
    assert info["max_input_tokens"] == 32768
    assert model.backend.capabilities.max_image_pixels == 1638400
    assert not info["truncates_input"]
    assert info["max_choice_options"] == info["max_score_levels"] == 255


def test_one_pass_answers_every_question(model, monkeypatch):
    passes = []
    original = model.backend.hidden_states
    monkeypatch.setattr(
        model.backend, "hidden_states", lambda *a, **k: passes.append(1) or original(*a, **k)
    )
    result = model.decide("please refund", QUESTIONS)
    assert len(passes) == 1
    assert set(result.answers) == set(QUESTIONS)
    assert set(result.answers["team"].probabilities) == {"billing", "technical"}
    assert sum(result.answers["urgency"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert 0 <= result.answers["refund"].noul <= 1
    (chunk,) = model.backend.encode(
        parse_request({"state": "please refund", "questions": QUESTIONS})
    )
    assert result.usage.input_tokens == len(chunk.input_ids)
    assert not result.truncated


def test_questions_see_each_other(model):
    """Unlike v1.1, another question in the request changes an answer."""
    alone = model.decide("please refund", {"refund": QUESTIONS["refund"]})
    together = model.decide("please refund", QUESTIONS)
    assert alone.answers["refund"].noul != pytest.approx(together.answers["refund"].noul, abs=1e-6)


def test_prompt_layout():
    request = parse_request({"state": {"n": 3}, "questions": QUESTIONS})
    text = render(request.state, list(request.questions.values()), PPLX_CODES, ["<|im_end|>"])
    user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>\n<|im_start|>assistant", 1)[0]
    assert user == (
        'State:\n{"n": 3}\n\n'
        "Question 0:\nbilling or technical\n"
        "Options:\nA: billing: refund\nB: technical\n<|fim_suffix|>\n\n"
        "Question 1:\nnow\nOptions:\nA: low\nB: mid\nC: high\n<|fim_suffix|>\n\n"
        "Question 2:\nrefund\nOptions:\nA: No / false\nB: Yes / true\n<|fim_suffix|>"
    )
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_escaping(folder):
    tokens = escaped_tokens(folder)
    assert tokens[0] == "<|vision_start|>"  # the longest first
    assert set(tokens) == {
        "<|im_end|>",
        "<|endoftext|>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|fim_suffix|>",
    }
    assert escape("a <|fim_suffix|> b <|im_start|>", tokens) == "a ＜|fim_suffix|＞ b <|im_start|>"
    assert escape({"x": "<|im_end|>"}, tokens) == '{"x": "＜|im_end|＞"}'


def test_image_markers_in_text_are_escaped_not_refused(model):
    question = {"refund": {"type": "noul", "instructions": "what is <|vision_start|>?"}}
    result = model.decide("look at <|image_pad|> here", question, images=[png(64, 64)])
    assert result.answers["refund"].noul >= 0


def test_255_options_answered_256_refused(model):
    criteria = {f"o{i}": None for i in range(255)}
    result = model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert len(result.answers["many"].probabilities) == 255
    criteria["o255"] = None
    with pytest.raises(DecisionError, match="at most 255") as error:
        model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert error.value.param == "questions.many.criteria"


def test_a_request_over_the_limit_is_answered_in_chunks(folder, model):
    request = parse_request({"state": "please refund", "questions": QUESTIONS})
    (whole,) = model.backend.encode(request)
    limited = mlx_decision.load(folder, max_input_tokens=len(whole.input_ids) - 1)
    chunks = limited.backend.encode(request)
    assert [c.question_ids for c in chunks] == [["team"], ["urgency", "refund"]]
    # Each chunk numbers its questions from 0 and keeps the whole state.
    assert all(len(c.markers) == len(c.question_ids) for c in chunks)
    result = limited.decide("please refund", QUESTIONS)
    assert list(result.answers) == list(QUESTIONS)
    assert result.usage.input_tokens == sum(len(c.input_ids) for c in chunks)
    alone = limited.decide("please refund", {"team": QUESTIONS["team"]})
    assert result.answers["team"].probabilities == pytest.approx(
        alone.answers["team"].probabilities, abs=1e-6
    )


def test_a_question_that_does_not_fit_alone_is_refused_before_any_pass(folder, model, monkeypatch):
    short = {"short": QUESTIONS["refund"]}
    (chunk,) = model.backend.encode(parse_request({"state": "please", "questions": short}))
    limit = len(chunk.input_ids) + 5
    limited = mlx_decision.load(folder, max_input_tokens=limit)
    monkeypatch.setattr(limited.backend, "hidden_states", lambda *_: pytest.fail("ran a pass"))
    questions = {**short, "long": {**QUESTIONS["refund"], "instructions": "refund " * 60}}
    with pytest.raises(DecisionError, match=f"the limit is {limit} and input is not cut") as error:
        limited.decide("please", questions)
    assert error.value.param == "questions.long"


@pytest.mark.parametrize("limit", [0, 262145])
def test_the_limit_is_1_to_262144(folder, limit):
    with pytest.raises(ValueError, match="between 1 and 262144"):
        mlx_decision.load(folder, max_input_tokens=limit)


def test_the_limit_can_be_raised(folder):
    assert mlx_decision.load(folder, max_input_tokens=100_000).info()["max_input_tokens"] == 100_000


def test_prefix_cache_is_not_an_option(folder):
    with pytest.raises(ValueError, match="pplx models do not take prefix_cache_gb"):
        mlx_decision.load(folder, prefix_cache_gb=1.0)


@pytest.mark.parametrize(
    ("decision", "remove", "message"),
    [
        ({"format_version": 1}, (), "unsupported decision config format"),
        ({"token_ids": [0] * 255}, (), "answer codes do not match the tokenizer"),
        ({"token_ids": None}, (), "answer codes do not match the tokenizer"),
        ({"max_pixels": 100}, (), "max_pixels"),
        (None, ("tokenizer_config.json",), "tokenizer_config.json is missing"),
    ],
)
def test_the_folder_is_checked(folder, tmp_path, decision, remove, message):
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        mlx_decision.load(copy_of(folder, tmp_path, decision=decision, remove=remove))


def test_format_2_without_joint_config_is_refused(folder, tmp_path):
    copy = copy_of(folder, tmp_path, remove=("joint_config.json",))
    with pytest.raises(ValueError, match="unsupported decision config format"):
        mlx_decision.load(copy)


def png(width: int, height: int) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def test_images_up_to_the_release_maximum(model):
    question = {"refund": QUESTIONS["refund"]}
    plain = model.decide("please", question).usage.input_tokens
    small = model.decide("please", question, images=[png(10, 10)]).usage.input_tokens
    large = model.decide("please", question, images=[png(2400, 1600)]).usage.input_tokens
    assert small - plain == 64 + 2
    # 1.6 MP at 32 x 32 pixels a token.
    assert 1500 < large - plain - 2 <= 1600


def test_the_image_maximum_can_be_lowered(folder):
    model = mlx_decision.load(folder, max_image_pixels=262144)
    question = {"refund": QUESTIONS["refund"]}
    plain = model.decide("please", question).usage.input_tokens
    large = model.decide("please", question, images=[png(2400, 1600)]).usage.input_tokens
    assert 200 < large - plain - 2 <= 256
