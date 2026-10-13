"""Metrics: records from answers, accuracy, macro-F1, ECE and MAE."""

import pytest

from mlx_decision.metrics import metrics, record


def test_record_choice():
    answer = {
        "type": "choice",
        "choice": "b",
        "confidence": 0.5,
        "probabilities": {"a": 0.3, "b": 0.7},
    }
    r = record(answer, "b")
    assert (r["predicted"], r["p_max"], r["confidence"]) == ("b", 0.7, 0.5)
    assert "score" not in r


def test_record_noul_predicts_yes_from_one_half():
    r = record({"type": "noul", "noul": 0.5}, "true")
    assert (r["predicted"], r["p_max"], r["noul"]) == ("true", 0.5, 0.5)
    r = record({"type": "noul", "noul": 0.2}, "true")
    assert (r["predicted"], r["p_max"]) == ("false", 0.8)


def test_record_score_keeps_the_most_probable_level_and_the_expected_one():
    answer = {
        "type": "score",
        "score": 1.4,
        "confidence": 0.6,
        "legend": {},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    }
    r = record(answer, "1")
    assert (r["predicted"], r["p_max"], r["score"]) == ("2", 0.5, 1.4)


def test_metrics():
    records = [
        {"gold": "a", "predicted": "a", "p_max": 0.9},
        {"gold": "a", "predicted": "b", "p_max": 0.9, "truncated": True},
        {"gold": "b", "predicted": "b", "p_max": 0.7},
        {"gold": "b", "predicted": None, "refused": "too long"},
    ]
    m = metrics(records)
    assert (m["n"], m["refused"], m["truncated"]) == (4, 1, 1)
    assert m["accuracy"] == 0.5
    # a: tp 1, fp 0, fn 1 -> 2/3; b: tp 1, fp 1, fn 1 -> 1/2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    # bin 0.9: 2 answered, accuracy 0.5 -> 0.4; bin 0.7: accuracy 1 -> 0.3
    assert m["ece"] == pytest.approx(2 / 3 * 0.4 + 1 / 3 * 0.3)
    assert "mae" not in m


def test_metrics_score_mae_leaves_out_refusals():
    records = [
        {"gold": "1", "predicted": "1", "p_max": 0.6, "score": 1.5},
        {"gold": "3", "predicted": "2", "p_max": 0.5, "score": 2.0},
        {"gold": "0", "predicted": None, "refused": "x"},
    ]
    m = metrics(records)
    assert m["mae"] == pytest.approx(0.75)
    assert (m["n"], m["refused"]) == (3, 1)
