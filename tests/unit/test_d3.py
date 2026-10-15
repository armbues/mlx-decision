"""d3 on a tiny random model: detection, prompt, limits, images and convert."""

import io
import json
import shutil

import pytest

import mlx_decision
from mlx_decision.convert import convert
from mlx_decision.errors import DecisionError
from mlx_decision.models.d3.prompt import render
from mlx_decision.registry import MARKER_FILE, detect_family
from mlx_decision.types import parse_request
from tiny_models import PPLX_CODES, write_d3

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
    return write_d3(tmp_path_factory.mktemp("d3") / "tiny-d3")


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def user_text(state, question) -> str:
    request = parse_request({"state": state, "questions": {"q": question}})
    text = render(request.state, request.questions["q"], PPLX_CODES)
    return text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


def png(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def test_detected_as_d3_not_pplx(folder):
    assert detect_family(folder).name == "d3"


def test_info(model):
    info = model.info()
    assert info["family"] == "d3"
    assert info["max_input_tokens"] == 262144
    assert not info["truncates_input"]
    assert info["max_choice_options"] == 255
    assert info["max_score_levels"] == 10
    assert info["supports_images"]


def test_answers_every_question(model):
    result = model.decide("please refund", QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert set(result.answers["team"].probabilities) == {"billing", "technical"}
    assert sum(result.answers["urgency"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert 0 <= result.answers["refund"].noul <= 1
    assert not result.truncated


def test_prompt_layout():
    text = user_text({"ticket": "Grüße", "n": 3}, QUESTIONS["team"])
    assert text == (
        'State:\n{"ticket": "Grüße", "n": 3}\n\n'
        "Question:\nbilling or technical\n\n"
        "Options:\nA: billing: refund\nB: technical\n\n"
        "Reply with only the code of the best option."
    )


@pytest.mark.parametrize("state", [None, ""])
def test_an_empty_state_is_shown_as_empty(state):
    text = user_text(state, {"type": "noul", "criteria": {"true": ""}})
    assert text.startswith("State:\n(empty)\n\nQuestion:\nChoose the best matching option.\n\n")
    assert "Options:\nA: No / false\nB: Yes / true\n\n" in text


def test_system_prompt_and_answer_slot():
    request = parse_request({"state": "s", "questions": {"q": {"type": "noul"}}})
    text = render(request.state, request.questions["q"], PPLX_CODES, images=2)
    assert text.startswith("<|im_start|>system\nYou are a decision engine. ")
    assert (
        "<|im_start|>user\n" + "<|vision_start|><|image_pad|><|vision_end|>" * 2 + "State:" in text
    )
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_score_levels(model):
    levels = [str(i) for i in range(10)]
    result = model.decide("please", {"s": {"type": "score", "criteria": levels}})
    assert len(result.answers["s"].probabilities) == 10
    with pytest.raises(DecisionError, match="at most 10 levels"):
        model.decide("please", {"s": {"type": "score", "criteria": [*levels, "10"]}})
    for criteria in (["ok", "ok"], ["", "ok"]):
        with pytest.raises(DecisionError, match="distinct and nonempty") as error:
            model.decide("please", {"s": {"type": "score", "criteria": criteria}})
        assert error.value.param == "questions.s.criteria"


def test_255_options_answered_256_refused(model):
    criteria = {f"o{i}": None for i in range(255)}
    result = model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert len(result.answers["many"].probabilities) == 255
    criteria["o255"] = None
    with pytest.raises(DecisionError, match="at most 255"):
        model.decide("please", {"many": {"type": "choice", "criteria": criteria}})


@pytest.mark.parametrize("limit", [0, 262145])
def test_the_limit_can_only_be_lowered(folder, limit):
    with pytest.raises(ValueError, match="between 1 and 262144"):
        mlx_decision.load(folder, max_input_tokens=limit)


def test_prefix_cache_is_not_an_option(folder):
    with pytest.raises(ValueError, match="d3 models do not take prefix_cache_gb"):
        mlx_decision.load(folder, prefix_cache_gb=1.0)


def test_images_up_to_the_releases_bounds(model, folder):
    question = {"refund": QUESTIONS["refund"]}
    plain = model.decide("please", question).usage.input_tokens
    small = model.decide("please", question, images=[png(10, 10)]).usage.input_tokens
    large = model.decide("please", question, images=[png(2000, 1000)]).usage.input_tokens
    # Each image adds its pad tokens plus the start and end markers; one token
    # per 32 x 32 pixels, at least 65,536 and at most 1,638,400 pixels.
    assert small - plain - 2 == 64
    assert 1400 < large - plain - 2 <= 1600
    lowered = mlx_decision.load(folder, max_image_pixels=262144)
    capped = lowered.decide("please", question, images=[png(2000, 1000)]).usage.input_tokens
    assert 200 < capped - plain - 2 <= 256


def test_videos_are_refused(model):
    with pytest.raises(DecisionError, match="videos are not supported"):
        model.decide_request(
            {
                "state": "s",
                "questions": {"q": {"type": "noul"}},
                "videos": ["data:video/mp4;base64,AA=="],
            }
        )


def test_converted_copy_is_d3(folder, tmp_path):
    (folder / "NOTICE").write_text("test notice\n")
    out = convert(folder, tmp_path / "q8", bits=8)
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["family"] == "d3"
    assert (out / "NOTICE").exists() and (out / "decision_config.json").exists()
    copy = mlx_decision.load(out)
    assert copy.info()["family"] == "d3" and copy.info()["precision"] == "8-bit"
    original = mlx_decision.load(folder).decide("please refund", QUESTIONS)
    answers = copy.decide("please refund", QUESTIONS)
    assert answers.answers["team"].choice == original.answers["team"].choice


def test_a_d3_folder_no_longer_loads_as_pplx(folder, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    config = json.loads((copy / "decision_config.json").read_text())
    del config["prompt"]
    (copy / "decision_config.json").write_text(json.dumps(config))
    assert detect_family(copy).name == "pplx"  # what tells them apart
