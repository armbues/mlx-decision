"""Soft labels: the format, the loss and calibration against distributions, the
metrics' log loss, every trained kind end to end, and the typed-decisions preset."""

import json
import math

import mlx.core as mx
import pytest

from fake_backend import TEAM, write_model
from mlx_decision.data.format import DataError, parse_example, parse_examples, write_examples
from mlx_decision.fitted import FIT_FILE
from mlx_decision.metrics import log_loss, metrics, record
from mlx_decision.training import fit
from mlx_decision.training.calibrate import fit_calibration
from mlx_decision.training.loss import decision_loss, gold_arrays, nll_rows

URGENT = {"type": "noul", "instructions": "Is it urgent?"}
LEVELS = {"type": "score", "criteria": ["Low", "Medium", "High"]}


def line(labels: dict, **questions) -> dict:
    questions = questions or {"team": TEAM, "urgent": URGENT, "level": LEVELS}
    return {"state": "a ticket", "questions": questions, "labels": labels}


# The format


def test_soft_labels_are_kept_with_their_most_likely_answer():
    example = parse_example(
        line({"team": {"tech": 0.6, "sales": 0.4}, "urgent": 0.3, "level": {"2": 0.7, "0": 0.3}}),
        1,
    )
    assert example.labels == {"team": "tech", "urgent": "false", "level": "2"}
    assert example.targets == {
        "team": {"billing": 0.0, "tech": 0.6, "sales": 0.4},  # every option, in order
        "urgent": {"true": 0.3, "false": 0.7},
        "level": {"0": 0.3, "1": 0.0, "2": 0.7},
    }
    # A family's own order of the options.
    assert example.gold("team", ["sales", "tech", "billing"]) == (0.4, 0.6, 0.0)
    assert example.gold("urgent", ["true", "false"]) == (0.3, 0.7)
    # Ties: the question's first option, a noul's true.
    tied = parse_example(line({"team": {"sales": 0.5, "tech": 0.5}, "urgent": 0.5}), 1)
    assert tied.labels == {"team": "tech", "urgent": "true"}


def test_a_soft_label_on_one_answer_is_the_hard_label():
    hard = parse_example(line({"team": "sales", "urgent": True, "level": 1}), 1)
    one_hot = parse_example(
        line({"team": {"sales": 1.0, "tech": 0.0}, "urgent": 1, "level": {"1": 1}}), 1
    )
    assert one_hot == hard and one_hot.targets is None
    assert parse_example(line({"urgent": 0.0}), 1).labels == {"urgent": "false"}


def test_a_rounded_distribution_is_kept_and_a_rough_one_renormalised():
    kept = parse_example(line({"team": {"tech": 0.333333, "sales": 0.666666}}), 1)
    assert kept.targets["team"]["tech"] == 0.333333
    rough = parse_example(line({"team": {"tech": 0.2004, "sales": 0.8}}), 1)
    assert sum(rough.targets["team"].values()) == pytest.approx(1, abs=1e-12)


def test_soft_labels_round_trip_through_a_file(tmp_path):
    from mlx_decision.data.format import read_examples

    example = parse_example(line({"team": {"tech": 0.6, "sales": 0.4}, "urgent": 0.25}), 1)
    assert example.to_json()["labels"] == {
        "team": {"billing": 0.0, "tech": 0.6, "sales": 0.4},
        "urgent": 0.25,
    }
    write_examples(tmp_path / "data.jsonl", [example])
    assert read_examples(tmp_path / "data.jsonl") == [example]


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ({"team": {"nobody": 1.0}}, 'names "nobody", none of the options (billing, tech, sales)'),
        ({"team": {"tech": -0.1, "sales": 1.1}}, "is a number from 0 to 1, not -0.1"),
        ({"team": {"tech": True}}, "is a number from 0 to 1, not true"),
        ({"team": {"tech": 0.5, "sales": 0.4}}, "sum to 1, not 0.9"),
        ({"urgent": 1.5}, "a soft noul label is the probability of true, from 0 to 1"),
        ({"level": {"3": 1.0}}, 'names "3", none of the options (0, 1, 2)'),
    ],
)
def test_a_bad_soft_label_names_its_line(label, message):
    good = json.dumps(line({"team": "tech"}))
    with pytest.raises(DataError) as error:
        parse_examples([good, json.dumps(line(label))])
    question = next(iter(label))
    assert str(error.value).startswith(f"line 2: labels.{question}: ")
    assert message in str(error.value)


# The loss and the calibration


def test_hard_rows_keep_the_hard_path():
    labels, soft = gold_arrays([2, 0, 1], 3)
    assert labels.tolist() == [2, 0, 1] and soft is None
    labels, soft = gold_arrays([2, (0.2, 0.5, 0.3), (0.6, 0.4)], 3)
    assert labels.tolist() == [2, 1, 0]
    targets, rows = soft
    for row, want in zip(
        targets.tolist(), [[0, 0, 1], [0.2, 0.5, 0.3], [0.6, 0.4, 0]], strict=True
    ):
        assert row == pytest.approx(want)
    assert rows.tolist() == [False, True, True]


def test_the_soft_loss_by_hand():
    logits = mx.array([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0], [1.0, -1.0, 0.0]])
    valid = mx.array([[True, True, True], [True, True, False], [True, True, True]])
    golds = [0, (0.3, 0.7), (0.2, 0.5, 0.3)]
    labels, soft = gold_arrays(golds, 3)
    mixed = decision_loss(logits, valid, labels, soft=soft)
    # The hard row as the hard path computes it, the soft rows by hand.
    hard = decision_loss(logits[:1], valid[:1], mx.array([0])).item()

    def soft_row(z, t):
        p = [math.exp(v) / sum(math.exp(w) for w in z) for v in z]
        entropy = -sum(ti * math.log(pi) for ti, pi in zip(t, p, strict=True))
        return entropy + sum((pi - ti) ** 2 for pi, ti in zip(p, t, strict=True))

    want = (hard + soft_row([0.5, 1.5], golds[1]) + soft_row([1.0, -1.0, 0.0], golds[2])) / 3
    assert mixed.item() == pytest.approx(want, rel=1e-5)
    # Without smoothing, a one-hot soft row is its hard row.
    one_hot = gold_arrays([(1.0, 0.0, 0.0), 1], 3)
    as_soft = decision_loss(logits[:2], valid[:2], one_hot[0], smoothing=0.0, soft=one_hot[1])
    as_hard = decision_loss(logits[:2], valid[:2], mx.array([0, 1]), smoothing=0.0)
    assert as_soft.item() == pytest.approx(as_hard.item(), rel=1e-6)
    # Each row's negative log-likelihood, soft rows against their targets.
    rows = nll_rows(mx.array([[0.0, math.log(3.0)]]), *gold_arrays([(0.5, 0.5)], 2))
    assert rows.item() == pytest.approx(-0.5 * math.log(0.25) - 0.5 * math.log(0.75), rel=1e-6)


def test_answers_that_are_their_soft_labels_keep_temperature_one():
    import random

    rng = random.Random(0)
    rows = []
    for _ in range(500):
        z = [rng.gauss(0, 1) for _ in range(3)]
        rows.append([math.exp(v) / sum(math.exp(w) for w in z) for v in z])
    calibration = fit_calibration(rows, [tuple(row) for row in rows], bias=False)
    assert calibration.temperature == pytest.approx(1.0, abs=0.02)
    # Against their argmax as hard labels the answers look underconfident.
    hard = fit_calibration(rows, [row.index(max(row)) for row in rows], bias=False)
    assert hard.temperature < 0.5


# The metrics


def test_the_log_loss_is_against_a_soft_label_when_there_is_one():
    answer = {"type": "choice", "choice": "a", "confidence": 0.4,
              "probabilities": {"a": 0.6, "b": 0.4}}  # fmt: skip
    assert record(answer, "a")["log_loss"] == pytest.approx(-math.log(0.6))
    soft = record(answer, "a", {"a": 0.5, "b": 0.5})
    assert soft["log_loss"] == pytest.approx(-0.5 * math.log(0.6) - 0.5 * math.log(0.4))
    assert soft["gold"] == "a"  # accuracy against the most likely answer
    noul = record({"type": "noul", "noul": 0.8}, "true", {"true": 0.7, "false": 0.3})
    assert noul["log_loss"] == pytest.approx(-0.7 * math.log(0.8) - 0.3 * math.log(0.2))
    assert log_loss({"a": 0.0}, "a") == pytest.approx(-math.log(1e-12))
    m = metrics([record(answer, "a"), {"gold": "b", "predicted": None, "refused": "x"}])
    assert m["log_loss"] == pytest.approx(-math.log(0.6)) and m["accuracy"] == 0.5
    assert "log_loss" not in metrics([{"gold": "a", "predicted": "a", "p_max": 1.0}])


# Every trained kind, end to end


def softened(examples, weight: float = 0.75):
    """The examples with each label softened: ``weight`` on it, the rest spread
    over the other options (a noul: the probability of true)."""
    result = []
    for example in examples:
        data = example.to_json()
        labels = {}
        for question_id, label in example.labels.items():
            question = example.request["questions"][question_id]
            if question["type"] == "noul":
                labels[question_id] = weight if label == "true" else 1 - weight
                continue
            options = list(question["criteria"])
            rest = (1 - weight) / (len(options) - 1)
            labels[question_id] = {o: weight if o == label else rest for o in options}
        result.append(parse_example({**data, "labels": labels}, 1))
    return result


def one_hot(examples):
    return softened(examples, 1.0)


def test_a_one_hot_soft_label_fits_exactly_as_the_hard_one(tmp_path):
    from test_training import examples

    base = write_model(tmp_path / "base")
    hard = examples()
    assert one_hot(hard) == hard
    fit(base, hard, output=tmp_path / "hard")
    fit(base, one_hot(hard), output=tmp_path / "one-hot")
    fit(base, softened(hard), output=tmp_path / "soft")
    stored = (tmp_path / "hard" / FIT_FILE).read_bytes()
    assert (tmp_path / "one-hot" / FIT_FILE).read_bytes() == stored
    assert (tmp_path / "soft" / FIT_FILE).read_bytes() != stored


def fits_both_ways(tmp_path, base, data, **options):
    """A fit on ``data`` and on its softened copy; the soft one's report."""
    hard = fit(base, data, output=tmp_path / "hard", **options)
    soft = fit(base, softened(data), output=tmp_path / "soft", **options)
    assert soft.kinds == hard.kinds
    assert (tmp_path / "soft" / FIT_FILE).read_bytes() != (
        tmp_path / "hard" / FIT_FILE
    ).read_bytes()
    for entry in soft.metrics.values():
        assert entry["after"]["log_loss"] > 0
        assert set(entry["after"]) >= {"accuracy", "log_loss"}
    return soft


def test_a_readout_fits_soft_labels(tmp_path, monkeypatch):
    from test_readout_fit import labelled

    from tiny_models import write_d1

    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 100)
    result = fits_both_ways(tmp_path, write_d1(tmp_path / "tiny-d1"), labelled(80))
    assert result.kinds == ["readout", "calibrate"]


def test_a_head_fits_soft_labels(tmp_path, monkeypatch):
    from test_head_fit import labelled

    from mlx_decision.training import head as head_fit
    from tiny_models import write_clef

    monkeypatch.setattr(head_fit, "LEARNING_RATE", 3e-3)
    monkeypatch.setattr(head_fit, "BATCH_REQUESTS", 4)
    result = fits_both_ways(tmp_path, write_clef(tmp_path / "tiny-clef"), labelled(40), epochs=1)
    assert result.kinds == ["head", "calibrate"]


def test_a_fine_tune_fits_soft_labels(tmp_path):
    from test_finetune import labelled

    from tiny_models import write_marker

    base = write_marker(tmp_path / "tiny-laya", "laya")
    result = fits_both_ways(tmp_path, base, labelled(30), epochs=1)
    assert result.kinds == ["finetune", "calibrate"]


def test_training_eval_scores_against_soft_labels(tmp_path):
    from test_training import examples

    from mlx_decision.training.evaluation import evaluate

    base = write_model(tmp_path / "base")
    hard, soft = examples(20), softened(examples(20), 0.6)
    on_hard = evaluate(base, hard).models[0].questions["team"]
    on_soft = evaluate(base, soft).models[0].questions["team"]
    assert on_soft["accuracy"] == on_hard["accuracy"]  # the argmax is the label
    assert on_soft["log_loss"] != on_hard["log_loss"]


# The typed-decisions preset


def typed_row() -> dict:
    questions = {"risk": LEVELS, "urgent": URGENT, "team": TEAM}
    gold = {
        "risk": {"type": "score", "label": "1", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
        "urgent": {"type": "noul", "label": "false", "probabilities": {"false": 0.6, "true": 0.4}},
        "team": {"type": "choice", "label": "tech",
                 "probabilities": {"billing": 0.1, "tech": 0.9, "sales": 0.0}},
    }  # fmt: skip
    return {
        "id": "cs_000001",
        "state": json.dumps({"ticket": "it broke"}),
        "questions": json.dumps(questions),
        "gold": json.dumps(gold),
    }


def test_the_typed_decisions_preset_makes_soft_labels():
    from mlx_decision.data.presets import PRESETS, examples

    preset = PRESETS["typed-decisions"]
    assert not preset.coverage
    [item] = examples(preset, [typed_row()], "train")
    assert item["id"] == "typed_decisions:train/cs_000001"
    assert item["state"] == {"ticket": "it broke"}
    example = parse_example(item, 1)
    assert example.labels == {"risk": "1", "urgent": "false", "team": "tech"}
    assert example.targets["urgent"] == {"true": 0.4, "false": 0.6}
    assert example.targets["team"] == {"billing": 0.1, "tech": 0.9, "sales": 0.0}


def test_a_dataset_folder_keeps_soft_labels(tmp_path):
    from mlx_decision.data.dataset import line_json

    example = parse_example(line({"team": {"tech": 0.6, "sales": 0.4}, "urgent": True}), 1)
    obj = line_json(example, shared_questions=True, mapped=set())
    assert obj["labels"] == {"team": {"billing": 0.0, "tech": 0.6, "sales": 0.4}, "urgent": True}
    assert "questions" not in obj
