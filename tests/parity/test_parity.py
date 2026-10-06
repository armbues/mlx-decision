"""Clef on MLX against Cloudflare's PyTorch reference, on the parity set.

``reference.json`` is written by ``scripts/make_parity_reference.py``. Requests
with images open them through ``parity_images.py``, as the reference did.
"""

import json
from pathlib import Path

import pytest
from parity_images import open_image

from mlx_decision.models.clef.encode import encode_request
from mlx_decision.types import parse_request

HERE = Path(__file__).parent
REQUESTS = {case["id"]: case for case in json.loads((HERE / "requests.json").read_text())}
REFERENCE = json.loads((HERE / "reference.json").read_text())["results"]
CASE_IDS = list(REQUESTS)

# Largest allowed difference of any probability; largest mean difference over
# the whole set; reference top-two margin above which the top option must agree.
# Measured: max 0.035 on a near tie (the reference moves 0.014 there between
# CPU and MPS), mean 0.001.
TOLERANCE = 0.04
MEAN_TOLERANCE = 0.0025
MARGIN = 0.04

pytestmark = pytest.mark.weights


def request(case_id: str):
    case = REQUESTS[case_id]
    body = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [open_image(spec) for spec in case["images"]]
    return parse_request(body)


def image_tokens(case_id: str, clef_path: Path) -> list[int]:
    from mlx_decision.models.clef.model import prepare_images

    if not REQUESTS[case_id].get("images"):
        return []
    images = prepare_images(request(case_id).images, clef_path, None)
    return [image.tokens for image in images]


def test_every_request_has_a_reference():
    assert set(REQUESTS) == set(REFERENCE)


@pytest.fixture(scope="session")
def tokenizer(clef_path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(clef_path / "tokenizer.json"))


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_tokens_and_spans_match(tokenizer, clef_path, case_id):
    encoded = encode_request(
        tokenizer, request(case_id), image_tokens=image_tokens(case_id, clef_path)
    )
    reference = REFERENCE[case_id]
    assert list(encoded.input_ids) == reference["input_ids"]
    assert encoded.truncated == reference["truncated"]
    ours = [
        {
            "id": q.question_id,
            "type": q.question_type,
            "span": list(q.question_span),
            "option_spans": [list(span) for span in q.option_spans],
            "option_ids": list(q.option_ids),
        }
        for q in encoded.questions
    ]
    theirs = [{k: v for k, v in q.items() if k != "probabilities"} for q in reference["questions"]]
    assert ours == theirs


def test_the_set_covers_truncation():
    assert any(reference["truncated"] for reference in REFERENCE.values())
    assert not all(reference["truncated"] for reference in REFERENCE.values())
    assert any(REFERENCE[c]["truncated"] for c in CASE_IDS if REQUESTS[c].get("images"))


def test_the_set_covers_images():
    with_images = [case for case in REQUESTS.values() if case.get("images")]
    assert any(len(case["images"]) > 1 for case in with_images)
    assert any(isinstance(spec, dict) for case in with_images for spec in case["images"])


@pytest.fixture(scope="session")
def outputs(clef):
    """Backend output for every request, computed once."""
    return {case_id: clef.backend.score(request(case_id)) for case_id in CASE_IDS}


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_probabilities_match(outputs, case_id):
    output = outputs[case_id]
    reference = REFERENCE[case_id]
    assert output.input_tokens == len(reference["input_ids"])
    assert output.truncated == reference["truncated"]
    for question in reference["questions"]:
        expected = question["probabilities"]
        ours = [output.probabilities[question["id"]][o] for o in question["option_ids"]]
        worst = max(abs(a - b) for a, b in zip(ours, expected, strict=True))
        assert worst <= TOLERANCE, f"{question['id']}: max |dp| {worst:.4f}"
        first, second = sorted(expected, reverse=True)[:2]
        if first - second > MARGIN:
            assert ours.index(max(ours)) == expected.index(first), question["id"]


def test_mean_difference_over_the_set(outputs):
    differences = [
        abs(outputs[case_id].probabilities[question["id"]][option] - expected)
        for case_id, reference in REFERENCE.items()
        for question in reference["questions"]
        for option, expected in zip(question["option_ids"], question["probabilities"], strict=True)
    ]
    mean = sum(differences) / len(differences)
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f} over {len(differences)} probabilities"


def test_truncation_reaches_the_result(clef):
    case_id = next(case_id for case_id, ref in REFERENCE.items() if ref["truncated"])
    result = clef.decide_request(request(case_id))
    assert result.truncated is True
    assert result.usage.input_tokens == clef.backend.capabilities.max_input_tokens


def test_a_schema_that_does_not_fit_is_an_error(tokenizer):
    from mlx_decision import DecisionError

    with pytest.raises(DecisionError) as caught:
        encode_request(tokenizer, request("twenty_mixed"), max_length=1000)
    assert caught.value.param == "questions"
    assert "the limit is 1000" in caught.value.message


def test_the_state_is_cut_to_the_limit(tokenizer):
    full = encode_request(tokenizer, request("ticket"))
    limit = len(full.input_ids) - 10
    encoded = encode_request(tokenizer, request("ticket"), max_length=limit)
    assert encoded.truncated and not full.truncated
    assert len(encoded.input_ids) == limit
    # The schema and the closing prompt are kept whole; only the state shrinks.
    assert encoded.input_ids[-300:] == full.input_ids[-300:]


def test_image_markers_in_user_text_are_refused_with_images(tokenizer):
    from mlx_decision import DecisionError

    for state, instructions, param in [
        ("a <|image_pad|> b", "q", "state"),
        ("ok", "see <|vision_start|>", "questions"),
    ]:
        body = {"state": state, "questions": {"q": {"type": "noul", "instructions": instructions}}}
        with pytest.raises(DecisionError) as caught:
            encode_request(tokenizer, parse_request(body), image_tokens=[64])
        assert caught.value.param == param
        # Without images they are ordinary input, as in the reference.
        encode_request(tokenizer, parse_request(body))
