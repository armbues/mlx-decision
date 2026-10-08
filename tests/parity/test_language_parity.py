"""Clef on the second parity set (many languages, long options), against its reference.

The requests are ``marker/requests.json`` (made for Laya and Julia), which has
text whose combining marks the tokenizer must keep with their letters (Hindi,
Bengali, Thai, vowelled Arabic). ``reference-languages.json`` is written by
``scripts/make_parity_reference.py --requests marker/requests.json --out
reference-languages.json``.
"""

import json
from pathlib import Path

import pytest
from test_parity import MARGIN, MEAN_TOLERANCE, TOLERANCE

from mlx_decision.models.clef.encode import encode_request
from mlx_decision.types import parse_request

HERE = Path(__file__).parent
REQUESTS = {
    case["id"]: case for case in json.loads((HERE / "marker" / "requests.json").read_text())
}
REFERENCE = json.loads((HERE / "reference-languages.json").read_text())["results"]
CASE_IDS = list(REQUESTS)

pytestmark = pytest.mark.weights


def request(case_id: str):
    case = REQUESTS[case_id]
    return parse_request({"state": case["state"], "questions": case["questions"]})


@pytest.fixture(scope="module")
def outputs(clef):
    return {case_id: clef.backend.score(request(case_id)) for case_id in CASE_IDS}


def test_every_request_has_a_reference():
    assert set(REQUESTS) == set(REFERENCE)


def test_the_set_has_combining_marks():
    import unicodedata

    for case_id in ("lang_ar", "lang_hi", "lang_bn", "lang_th"):
        assert any(unicodedata.category(c).startswith("M") for c in REQUESTS[case_id]["state"])


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_tokens_and_spans_match(tokenizer, case_id):
    encoded = encode_request(tokenizer, request(case_id))
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


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_probabilities_match(outputs, case_id):
    output = outputs[case_id]
    reference = REFERENCE[case_id]
    assert output.input_tokens == len(reference["input_ids"])
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
