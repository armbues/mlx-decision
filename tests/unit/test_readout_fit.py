"""The readout fit: the search on synthetic features, and pplx, d3, H2O and d1 end to end
on their tiny models."""

import json
import threading

import mlx.core as mx
import pytest

import mlx_decision
from mlx_decision.data.format import Example, training_splits
from mlx_decision.fitted import FIT_FILE
from mlx_decision.models.readout import Features, Readout, group_logits, option_logits
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import Cancelled, fit, plan
from mlx_decision.training.fitting import _readouts
from mlx_decision.training.readout import fit_readout, search_steps
from mlx_decision.training.scoring import Scored, reread, score
from mlx_decision.types import parse_request
from tiny_models import write_d1, write_d3, write_h2o, write_pplx

TEAM = {"type": "choice", "criteria": {"billing": "refund", "technical": "crash"}}
URGENT = {"type": "noul", "instructions": "Is it urgent?"}
STATES = ["please refund", "crash now", "refund now please", "crash"]
WRITERS = {"h2o": write_h2o, "d1": write_d1, "pplx": write_pplx, "d3": write_d3}


def labelled(count: int = 120) -> list[Example]:
    """Examples whose team and urgency follow words in the state."""
    return [
        Example(
            f"e{i}",
            {"state": STATES[i % 4], "questions": {"team": TEAM, "urgent": URGENT}},
            {
                "team": "billing" if "refund" in STATES[i % 4] else "technical",
                "urgent": "true" if "now" in STATES[i % 4] else "false",
            },
        )  # fmt: skip
        for i in range(count)
    ]


@pytest.fixture
def short_search(monkeypatch):
    """A search of 100 steps per setting, enough for the tiny models."""
    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 100)


def synthetic(count: int, seed: int):
    """Hidden states labelled by true rows the release has in the wrong order
    (rows close enough for the search's steps to reach)."""
    key = mx.random.key(seed)
    hidden = mx.random.normal((count, 8), key=key) * 10
    true_rows = mx.random.normal((3, 8), key=mx.random.key(99)) * 0.1
    labels = (hidden @ true_rows.T).argmax(axis=1).tolist()
    return hidden, labels, true_rows[mx.array([1, 2, 0])]


def accuracy(rows, bias, hidden, labels) -> float:
    logits = group_logits(rows, hidden, [[0], [1], [2]], 1.0, bias)
    return (logits.argmax(axis=1) == mx.array(labels)).mean().item()


def test_batched_logits_are_the_single_ones():
    hidden = mx.random.normal((5, 4), key=mx.random.key(0))
    rows = mx.random.normal((4, 4), key=mx.random.key(1))
    groups = [[0, 2], [1], [3, 0, 1]]
    batch = group_logits(rows, hidden, groups, 2.0, mx.array([0.1, 0.0, -0.1]))
    for i in range(5):
        features = Features(["x", "y", "z"], hidden[i], rows, groups, 2.0)
        readout = Readout(["x", "y", "z"], rows, mx.array([0.1, 0.0, -0.1]))
        # A batch is a matrix product, which MLX may run at reduced precision on
        # the GPU (MLX_ENABLE_TF32); one hidden state is a matrix-vector product.
        assert mx.allclose(batch[i], option_logits(features, readout), rtol=1e-2, atol=1e-2)


def test_the_search_learns_rows_the_release_has_wrong():
    hidden, labels, release = synthetic(400, 0)
    dev_hidden, dev_labels, _ = synthetic(100, 1)
    zero = mx.zeros(3)
    assert accuracy(release, zero, dev_hidden, dev_labels) < 0.2
    steps = []
    readout, picked = fit_readout(
        ["a", "b", "c"], release, [[0], [1], [2]], 1.0,
        (hidden, labels), (dev_hidden, dev_labels), on_step=steps.append,
    )  # fmt: skip
    assert readout is not None and readout.option_ids == ["a", "b", "c"]
    assert picked.steps > 0 and picked.dev_nll < picked.dev_nll_release
    assert accuracy(readout.rows, readout.bias, dev_hidden, dev_labels) > 0.9
    assert steps[-1] == search_steps()
    again, _ = fit_readout(
        ["a", "b", "c"], release, [[0], [1], [2]], 1.0, (hidden, labels), (dev_hidden, dev_labels)
    )
    assert mx.array_equal(again.rows, readout.rows) and mx.array_equal(again.bias, readout.bias)


def test_the_search_stops_when_cancelled():
    hidden, labels, release = synthetic(50, 0)
    with pytest.raises(Cancelled):
        fit_readout(
            ["a", "b", "c"], release, [[0], [1], [2]], 1.0,
            (hidden, labels), (hidden, labels), cancelled=lambda: True,
        )  # fmt: skip


def test_no_readout_for_a_question_whose_examples_score_other_tokens():
    def item(i: int, rows: mx.array, part: str) -> Scored:
        example = Example(
            f"{part}{i}", {"state": "x", "questions": {"team": TEAM}}, {"team": "billing"}
        )
        hidden = mx.random.normal((4,), key=mx.random.key(i))
        features = Features(["billing", "technical"], hidden, rows, [[0], [1]], 1.0)
        request = parse_request(example.request)
        return Scored(example, request, {}, features={"team": features})

    rows = mx.ones((2, 4))
    events = []
    fit_part = [item(i, rows, "f") for i in range(5)] + [item(5, rows * 2, "f")]
    fitted, report = _readouts(fit_part, [item(9, rows, "d")], events.append, None, 0.0)
    assert fitted.readouts == {}
    assert report[0]["skipped"].startswith("its examples score other tokens")
    assert any("no readout for 'team'" in (e.message or "") for e in events)
    fitted, report = _readouts(fit_part[:5], [], events.append, None, 0.0)
    assert report[0]["skipped"] == "it needs examples in both the fit and the dev part"


def test_readout_is_the_main_kind_of_the_reading_families(tmp_path):
    for family in WRITERS:
        assert plan.resolve(family, "auto") == ["readout", "calibrate"]
    splits = training_splits(labelled(200))
    base = write_h2o(tmp_path / "tiny-h2o")
    readout = plan.estimate(["readout", "calibrate"], base, {}, splits)
    calibrate = plan.estimate(["calibrate"], base, {}, splits)
    assert readout.peak_bytes == calibrate.peak_bytes + 400 * plan.READOUT_FEATURE_BYTES


@pytest.mark.parametrize("family", list(WRITERS))
def test_every_reading_family_fits_a_readout_end_to_end(tmp_path, family, short_search):
    base = WRITERS[family](tmp_path / f"tiny-{family}")
    data = labelled()
    events = []
    result = fit(base, data, output=tmp_path / "fitted", on_event=events.append)
    assert result.kinds == ["readout", "calibrate"]
    stages = list(dict.fromkeys(event.stage for event in events))
    assert stages == ["check", "before", "features", "train", "calibrate", "after", "write",
                      "done"]  # fmt: skip
    for question in result.report["training"]["questions"]:
        assert question["steps"] > 0 and question["dev_nll"] < question["dev_nll_release"]
    for question_id in ("team", "urgent"):
        metrics = result.metrics[question_id]
        assert metrics["after"]["accuracy"] > metrics["before"]["accuracy"]
    tensors = mx.load(str(result.path / FIT_FILE))
    assert {
        "readout.0.rows",
        "readout.0.bias",
        "readout.1.rows",
        "calibration.1.temperature",
    } <= set(tensors)
    marker = json.loads((result.path / MARKER_FILE).read_text())
    assert marker["fit"]["kinds"] == ["readout", "calibrate"]

    # The fitted model serves what the fit scored: its answers to the holdout are
    # the features read out again with the fitted rows, then calibrated.
    model = mlx_decision.load(result.path)
    unfitted = mlx_decision.load(base)
    holdout = training_splits(data, seed=0).holdout
    scored = reread(score(unfitted, holdout, features=True), model.backend.readouts)
    for item in scored:
        served = model.backend.score(item.request).probabilities
        adjusted = model.backend.adjust(item.request.questions, item.raw)
        for question_id, values in served.items():
            for option, p in values.items():
                assert p == pytest.approx(adjusted[question_id][option], abs=1e-5)


def test_the_same_seed_gives_the_same_readout(tmp_path, short_search):
    base = write_d1(tmp_path / "tiny-d1")
    data = labelled(80)
    fit(base, data, output=tmp_path / "a", seed=1)
    fit(base, data, output=tmp_path / "b", seed=1)
    assert (tmp_path / "a" / FIT_FILE).read_bytes() == (tmp_path / "b" / FIT_FILE).read_bytes()


def test_a_fit_cancelled_while_training_writes_nothing(tmp_path):
    base = write_d1(tmp_path / "tiny-d1")
    stop = threading.Event()

    def on_event(event):
        if event.stage == "train" and event.step:
            stop.set()

    with pytest.raises(Cancelled):
        fit(base, labelled(80), output=tmp_path / "out" / "fit", on_event=on_event, cancel=stop)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []
