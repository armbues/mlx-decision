"""Confidence formulas and answer building, checked against the Jev documentation."""

import pytest

from mlx_decision.answers import build_answer, choice_confidence, score_confidence
from mlx_decision.types import Choice, Noul, Score


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        # Worked examples from Jev's confidence page.
        ((0.6, 0.3, 0.1), 0.4),
        ((0.6, 0.2, 0.2), 0.4),
        # Edges: certainty, an even split, two options (same as |2p - 1|).
        ((1.0, 0.0, 0.0), 1.0),
        ((0.25, 0.25, 0.25, 0.25), 0.0),
        ((0.8, 0.2), 0.6),
    ],
)
def test_choice_confidence(probabilities, expected):
    assert choice_confidence(probabilities) == pytest.approx(expected)


def test_choice_confidence_of_a_single_option_is_one():
    assert choice_confidence([1.0]) == 1.0


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        # Worked examples from Jev's confidence page.
        ((0.0, 0.5, 0.5), 0.25),
        ((0.5, 0.0, 0.5), 0.0),
        ((0.0, 0.57, 0.43), 1 - 0.43 / (2 / 3)),
        # Five levels: the even spread measured from the middle is 1.2.
        ((0.2, 0.2, 0.2, 0.2, 0.2), 0.0),
        ((0.0, 0.0, 0.7, 0.3, 0.0), 1 - 0.3 / 1.2),
        ((0.0, 1.0, 0.0, 0.0, 0.0), 1.0),
    ],
)
def test_score_confidence(probabilities, expected):
    assert score_confidence(probabilities) == pytest.approx(expected)


def test_score_confidence_rounds_like_the_docs():
    assert round(score_confidence((0.0, 0.57, 0.43)), 2) == 0.35


def test_score_confidence_takes_the_first_of_tied_levels():
    # As in the docs' code (`index(max(...))`): peak at level 0, so the spread is
    # 0.45 * 1 + 0.1 * 2; peaking at level 1 would give 0.45 + 0.1.
    assert score_confidence((0.45, 0.45, 0.1)) == pytest.approx(1 - 0.65 / (2 / 3))


def test_noul_answer_is_the_probability_of_true():
    answer = build_answer(Noul(instructions="Urgent?"), {"false": 0.3, "true": 0.7})
    assert answer.noul == 0.7
    assert answer.model_dump() == {"type": "noul", "noul": 0.7}


def test_choice_answer_keeps_the_callers_option_order():
    question = Choice(criteria={"zeta": None, "alpha": "A", "mid": "M"})
    # Backends may return options in any order (Clef reads them sorted).
    answer = build_answer(question, {"alpha": 0.2, "mid": 0.1, "zeta": 0.7})
    assert list(answer.probabilities) == ["zeta", "alpha", "mid"]
    assert answer.choice == "zeta"
    assert answer.confidence == pytest.approx((0.7 - 1 / 3) / (2 / 3))


def test_choice_answer_ties_go_to_the_first_option():
    question = Choice(criteria={"b": None, "a": None})
    assert build_answer(question, {"a": 0.5, "b": 0.5}).choice == "b"


def test_score_answer():
    question = Score(criteria=["calm", {"level": "upset"}, "angry"])
    answer = build_answer(question, {"0": 0.0, "1": 0.57, "2": 0.43})
    assert answer.score == pytest.approx(0.57 + 2 * 0.43)
    assert answer.legend == {"0": "calm", "1": {"level": "upset"}, "2": "angry"}
    assert answer.probabilities == {"0": 0.0, "1": 0.57, "2": 0.43}
    assert answer.confidence == pytest.approx(1 - 0.43 / (2 / 3))
