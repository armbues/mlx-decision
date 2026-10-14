"""Fine-tuning Laya and Julia on their tiny models: hooks, training, the fitted folder."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlx_decision
from mlx_decision.data.format import Example, training_splits
from mlx_decision.fitted import FIT_FILE
from mlx_decision.models.marker.model import load_fitted_weights
from mlx_decision.pool import ModelPool, discover
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import Cancelled, fit, plan
from mlx_decision.training.checkpoint import checkpointed
from mlx_decision.training.finetune import Item, batches, train
from mlx_decision.types import parse_request
from tiny_models import write_marker

TEAM = {"type": "choice", "criteria": {"billing": "refund", "technical": "crash"}}
STATES = ["please refund", "crash now", "refund now please", "crash", "now refund", "crash please"]


def labelled(count: int = 60) -> list[Example]:
    """Examples whose team follows a word in the state (refund: billing, else technical)."""
    return [
        Example(
            f"e{i}",
            {"state": STATES[i % len(STATES)], "questions": {"team": TEAM}},
            {"team": "billing" if "refund" in STATES[i % len(STATES)] else "technical"},
        )
        for i in range(count)
    ]


@pytest.fixture(params=["laya", "julia"])
def base(request, tmp_path):
    return write_marker(tmp_path / f"tiny-{request.param}", request.param)


def items(model, examples):
    result = []
    for example in examples:
        by_question = model.backend.items(model.check(example.request))
        item = by_question["team"]
        result.append(Item(item, item.option_ids.index(example.labels["team"])))
    return result


def parameters(module) -> dict:
    return {k: v for k, v in tree_flatten(module.parameters())}


def test_scores_are_the_served_logits(base):
    model = mlx_decision.load(base)
    request = parse_request({"state": "please refund", "questions": {"team": TEAM}})
    logits, valid = model.backend.scores(list(model.backend.items(request).values()))
    served = model.backend.score(request).probabilities["team"]
    assert valid.tolist() == [[True, True]]
    assert mx.softmax(logits, axis=-1)[0].tolist() == pytest.approx(list(served.values()), abs=1e-4)


def test_training_lowers_the_loss_and_keeps_the_serving_dtype(base, monkeypatch):
    # Tiny models need a larger step than real ones to move in a few batches.
    monkeypatch.setattr("mlx_decision.training.finetune.LEARNING_RATE", 1e-3)
    model = mlx_decision.load(base)
    data = items(model, labelled())
    before = parameters(model.backend.trainable())
    rows = []
    trained = train(model.backend, data, data[:12], epochs=3, on_epoch=rows.append)
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert rows[-1]["loss"] < rows[0]["loss"]
    assert trained.best == min(rows, key=lambda row: row["dev_loss"])["epoch"]
    after = parameters(model.backend.trainable())
    assert {v.dtype for v in after.values()} == {v.dtype for v in before.values()} == {mx.float16}
    assert {v.dtype for v in trained.tensors.values()} == {mx.bfloat16}
    assert set(trained.tensors) == set(before)
    assert any(not mx.array_equal(after[k], before[k]) for k in before)


def test_head_only_leaves_the_encoder_alone(base):
    model = mlx_decision.load(base)
    before = parameters(model.backend.trainable())
    trained = train(model.backend, items(model, labelled()), [], epochs=1, head_only=True)
    after = parameters(model.backend.trainable())
    assert trained.tensors and all(k.startswith("head.") for k in trained.tensors)
    assert all(mx.array_equal(after[k], before[k]) for k in before if k.startswith("encoder."))
    assert model.backend.encoder.trainable_parameters()  # unfrozen again


def test_a_cancelled_training_stops_between_steps_in_the_serving_dtype(base):
    model = mlx_decision.load(base)
    steps = []
    with pytest.raises(Cancelled):
        train(
            model.backend,
            items(model, labelled()),
            [],
            on_step=lambda step, total, loss: steps.append(step),
            cancelled=lambda: len(steps) >= 2,
        )
    assert steps == [1, 2]
    after = parameters(model.backend.trainable())
    assert {value.dtype for value in after.values()} == {mx.float16}


def test_the_optimizer_matches_mlx_adamw_with_clipping():
    import mlx.optimizers as optim

    from mlx_decision.training.finetune import MixedAdamW

    weights = {"a": mx.array([0.5, -1.0, 2.0]), "b": mx.array([[0.1, 0.2], [0.3, 0.4]])}
    grads = [{"a": mx.array([3.0, -4.0, 0.5]), "b": mx.array([[1.0, 0.0], [0.0, -2.0]])}] * 3
    ours = MixedAdamW(weights, lambda step: 0.01 * (step + 1), 0.01)
    reference = optim.AdamW(learning_rate=lambda step: 0.01 * (step + 1), weight_decay=0.01)
    params = {k: v.astype(mx.float32) for k, v in weights.items()}
    for g in grads:
        clipped, _ = optim.clip_grad_norm(g, 1.0)
        params = reference.apply_gradients(clipped, params)
        ours.update(g)
    for key in weights:
        assert mx.allclose(ours.master[key], params[key], atol=1e-6).item(), key


def test_checkpointing_gives_the_same_gradients_and_is_undone(base):
    model = mlx_decision.load(base, dtype="float32")
    parts = model.backend.trainable()
    batch = items(model, labelled(4))
    layer_class = type(parts.encoder.layers[0])
    original = layer_class.__call__

    def loss(batch):
        logits, _ = model.backend.scores([entry.item for entry in batch])
        return logits.sum()

    grad = nn.value_and_grad(parts, loss)
    plain = dict(tree_flatten(grad(batch)[1]))
    with checkpointed(layer_class):
        assert layer_class.__call__ is not original
        again = dict(tree_flatten(grad(batch)[1]))
    assert layer_class.__call__ is original
    for key, value in plain.items():
        assert mx.allclose(value, again[key], atol=1e-5).item(), key


def test_batches_hold_tokens_and_rows_within_bounds(base):
    model = mlx_decision.load(base)
    data = items(model, labelled(100))
    groups = batches(data)
    assert sum(map(len, groups)) == 100
    assert all(len(g) <= 32 and len(g) * max(e.item.length for e in g) <= 4096 for g in groups)


def test_a_fine_tune_is_written_reloaded_and_served(tmp_path, base):
    data = labelled()
    events = []
    result = fit(base, data, output=tmp_path / "tuned", epochs=2, on_event=events.append)
    assert result.kinds == ["finetune", "calibrate"]
    stages = list(dict.fromkeys(event.stage for event in events))
    assert stages == ["check", "before", "train", "calibrate", "after", "write", "done"]
    training = result.report["training"]
    assert [row["epoch"] for row in training["epochs"]] == [1, 2]
    stored = mx.load(str(tmp_path / "tuned" / FIT_FILE))
    assert any(k.startswith("encoder.") for k in stored)
    assert stored["head.scorer_out.weight"].dtype == mx.bfloat16
    marker = json.loads((tmp_path / "tuned" / MARKER_FILE).read_text())
    assert marker["fit"]["kinds"] == ["finetune", "calibrate"]
    # Loaded again, it answers the holdout as the report says.
    model = mlx_decision.load(tmp_path / "tuned")
    holdout = training_splits(data, seed=0).holdout
    predicted = [model.decide_request(e.request).answers["team"].choice for e in holdout]
    right = sum(p == e.labels["team"] for p, e in zip(predicted, holdout, strict=True))
    assert result.metrics["team"]["after"]["accuracy"] == pytest.approx(
        right / len(holdout), abs=1e-4
    )
    plain = mlx_decision.load(base).decide_request(holdout[0].request).answers["team"]
    assert model.decide_request(holdout[0].request).answers["team"] != plain
    # In a pool it loads a base of its own.
    pool = ModelPool(discover([tmp_path / "tuned", base]), budget=10**12)
    assert pool.get("tuned").backend.base is not pool.get(base.name).backend


def test_head_only_fits_store_the_head(tmp_path, base):
    result = fit(base, labelled(), output=tmp_path / "head", epochs=1, head_only=True)
    stored = mx.load(str(tmp_path / "head" / FIT_FILE))
    assert stored and all(k.startswith(("head.", "calibration.")) for k in stored)
    assert result.report["options"]["head_only"] is True
    mlx_decision.load(tmp_path / "head")
    with pytest.raises(ValueError, match="head_only applies to finetune"):
        fit(base, labelled(), output=tmp_path / "x", method="calibrate", head_only=True)


def test_fine_tuned_weights_must_fit_the_model(tmp_path, base):
    model = mlx_decision.load(base)
    file = tmp_path / "wrong.safetensors"
    mx.save_safetensors(str(file), {"head.scorer_out.weight": mx.zeros((3, 3))})
    with pytest.raises(ValueError, match="does not fit this model"):
        load_fitted_weights(model.backend.trainable(), file)


def test_the_fine_tune_estimate_counts_the_trained_parameters(base):
    full = plan.kind_bytes("finetune", base, {})
    head = plan.kind_bytes("finetune", base, {}, head_only=True)
    assert 0 < head < full
    assert plan.kind_bytes("calibrate", base, {}) == 0
    splits = training_splits(labelled(200))
    tuned = plan.estimate(["finetune", "calibrate"], base, {}, splits)
    calibrated = plan.estimate(["calibrate"], base, {}, splits)
    assert tuned.peak_bytes > calibrated.peak_bytes


def test_an_epoch_worse_on_dev_stops_the_run(base, monkeypatch):
    model = mlx_decision.load(base)
    losses = iter([0.5, 0.4, 0.45, 0.1])  # epoch 3 is worse than epoch 2: no epoch 4
    monkeypatch.setattr("mlx_decision.training.finetune.dev_loss", lambda *a: next(losses))
    data = items(model, labelled(16))
    trained = train(model.backend, data[:12], data[12:], epochs=4, head_only=True)
    assert [row["epoch"] for row in trained.epochs] == [1, 2, 3]
    assert (trained.best, trained.stopped) == (2, True)
    assert trained.epochs[-1]["stopped"]
    every = iter([0.5, 0.4, 0.45, 0.1])
    monkeypatch.setattr("mlx_decision.training.finetune.dev_loss", lambda *a: next(every))
    trained = train(model.backend, data[:12], data[12:], epochs=4, head_only=True,
                    early_stop=False)  # fmt: skip
    assert (trained.best, trained.stopped, len(trained.epochs)) == (4, False, 4)
    again = iter([0.5, 0.4, 0.3])
    monkeypatch.setattr("mlx_decision.training.finetune.dev_loss", lambda *a: next(again))
    trained = train(model.backend, data[:12], data[12:], epochs=3, head_only=True)
    assert (trained.best, trained.stopped, len(trained.epochs)) == (3, False, 3)
