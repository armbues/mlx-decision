"""Clef's head fit on the tiny model: stored states, training, the fitted folder."""

import json
import threading
from collections import namedtuple

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

import mlx_decision
from mlx_decision.data.format import Example, training_splits
from mlx_decision.fitted import FIT_FILE, open_folder
from mlx_decision.models.clef.model import load, load_fitted_head
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import Cancelled, fit, plan
from mlx_decision.training import head as head_fit
from mlx_decision.training.scoring import rehead, score
from mlx_decision.types import parse_request
from tiny_models import write_clef

# The tiny tokenizer knows only a few words, so states and options use them.
TEAM = {"type": "choice", "criteria": {"billing": "billing", "technical": "technical"}}
URGENT = {"type": "noul", "instructions": "yes"}
STATES = ["billing yes", "technical no", "sales billing yes", "no technical"]


def labelled(count: int = 120) -> list[Example]:
    """Examples whose team and urgency follow words in the state."""
    return [
        Example(
            f"e{i}",
            {"state": STATES[i % 4], "questions": {"team": TEAM, "urgent": URGENT}},
            {
                "team": "billing" if "billing" in STATES[i % 4] else "technical",
                "urgent": "true" if "yes" in STATES[i % 4] else "false",
            },
        )  # fmt: skip
        for i in range(count)
    ]


@pytest.fixture
def base(tmp_path):
    return write_clef(tmp_path / "tiny-clef")


@pytest.fixture
def quick(monkeypatch):
    """Larger steps for the tiny head, whose random start learns slowly."""
    monkeypatch.setattr(head_fit, "LEARNING_RATE", 3e-3)
    monkeypatch.setattr(head_fit, "BATCH_REQUESTS", 4)


def request(state: str = "billing yes"):
    return parse_request({"state": state, "questions": {"team": TEAM, "urgent": URGENT}})


def test_the_head_reads_what_score_answers(base):
    backend = load(base)
    inputs = backend.head_inputs(request())
    assert inputs.hidden.shape[0] == inputs.input_ids.size
    assert backend.read_head(inputs) == backend.score(request()).probabilities
    assert backend.head_size(request()) == inputs.hidden.nbytes


def test_the_store_gives_back_what_it_kept(tmp_path, base, monkeypatch):
    monkeypatch.setattr(head_fit, "SHARD_REQUESTS", 2)
    backend = load(base)
    store = head_fit.StateStore(tmp_path / "states")
    kept = [backend.head_inputs(request(state)) for state in STATES + STATES[:1]]
    indexes = [store.add(inputs) for inputs in kept]
    assert indexes == list(range(5))
    assert len(list((tmp_path / "states").glob("*.safetensors"))) == 2  # the fifth pending
    for index, inputs in zip(indexes, kept, strict=True):
        again = store.get(index)
        assert mx.array_equal(again.hidden, inputs.hidden)
        assert again.questions == inputs.questions
        assert backend.read_head(again) == backend.read_head(inputs)
    assert store.bytes > 0
    store.remove()
    assert not (tmp_path / "states").exists()


def test_states_read_back_are_not_kept_in_memory(tmp_path):
    from mlx_decision.models.clef.model import HeadInputs

    store = head_fit.StateStore(tmp_path / "states")
    for i in range(24):
        hidden = mx.full((1024, 1024), float(i), dtype=mx.float32)  # 4 MB each
        store.add(HeadInputs(hidden, mx.array([1]), ()))
    store.flush()
    mx.clear_cache()
    start = mx.get_active_memory()
    for i in range(24):
        hidden = store.get(i).hidden
        assert hidden[0, 0].item() == i
        del hidden
    assert mx.get_active_memory() - start < 16 * 2**20
    store.remove()


def test_a_trained_head_loads_over_the_release(tmp_path, base):
    backend = load(base)
    tensors = {f"head.{k}": v + 1 for k, v in tree_flatten(backend.head.parameters())}
    mx.save_safetensors(str(tmp_path / "fit.safetensors"), tensors)
    fitted = load(base, fit_weights=tmp_path / "fit.safetensors")
    name, value = tree_flatten(fitted.head.parameters())[0]
    assert mx.array_equal(value, tensors[f"head.{name}"])
    del tensors[f"head.{name}"]
    mx.save_safetensors(str(tmp_path / "part.safetensors"), tensors)
    with pytest.raises(ValueError, match="holds no trained head"):
        load_fitted_head(backend.head, tmp_path / "part.safetensors")


def test_training_lowers_the_loss_and_keeps_the_best_epoch(tmp_path, base, quick):
    backend = load(base)
    store = head_fit.StateStore(tmp_path / "states")
    inputs = backend.head_inputs(request())
    true = inputs.questions[1].option_ids.index("true")
    items = [head_fit.Item(store.add(inputs), {0: 0, 1: true}) for _ in range(12)]
    dtype = tree_flatten(backend.head.parameters())[0][1].dtype
    start = head_fit.dev_loss(backend, store, items[8:])
    steps = []
    trained = head_fit.train(
        backend, store, items[:8], items[8:], epochs=4,
        on_step=lambda step, total, loss: steps.append((step, total)),
    )  # fmt: skip
    assert steps[-1] == (8, 8)
    losses = [row["dev_loss"] for row in trained.epochs]
    assert losses[-1] < start / 2
    assert trained.best == 1 + losses.index(min(losses))
    assert tree_flatten(backend.head.parameters())[0][1].dtype == dtype
    kept = {f"head.{k}": v for k, v in tree_flatten(backend.head.parameters())}
    assert set(kept) == set(trained.tensors)
    assert head_fit.dev_loss(backend, store, items[8:]) == pytest.approx(min(losses), rel=1e-3)
    store.remove()


def test_head_is_the_main_kind_of_clef(tmp_path, base):
    assert plan.resolve("clef", "auto") == ["head", "calibrate"]
    splits = training_splits(labelled(200))
    estimate = plan.estimate(["head", "calibrate"], base, {}, splits)
    calibrate = plan.estimate(["calibrate"], base, {}, splits)
    assert estimate.peak_bytes > calibrate.peak_bytes
    assert estimate.disk_bytes > 0 and "disk_gb" in estimate.as_json()
    assert "disk_gb" not in calibrate.as_json()
    short = plan.kind_bytes("head", base, {}, tokens=100)
    assert plan.kind_bytes("head", base, {}, tokens=1100) == short + 1000 * plan.HEAD_TOKEN_BYTES


def test_clef_fits_its_head_end_to_end(tmp_path, base, quick):
    data = labelled()
    events = []
    result = fit(base, data, output=tmp_path / "out" / "fitted", on_event=events.append)
    assert result.kinds == ["head", "calibrate"]
    stages = list(dict.fromkeys(event.stage for event in events))
    assert stages == ["check", "before", "features", "train", "calibrate", "after", "write",
                      "done"]  # fmt: skip
    stored = [e for e in events if e.message and e.message.startswith("storing the hidden")]
    assert len(stored) == 1 and "removed afterwards" in stored[0].message
    assert stored[0].metrics["states_gb"] >= 0
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["fitted"]  # states removed
    training = result.report["training"]
    assert len(training["epochs"]) == 3 and 1 <= training["best_epoch"] <= 3
    assert set(result.metrics) == {"team", "urgent"}
    tensors = mx.load(str(result.path / FIT_FILE))
    backend = load(base)
    names = {f"head.{k}" for k, _ in tree_flatten(backend.head.parameters())}
    assert names <= set(tensors)
    marker = json.loads((result.path / MARKER_FILE).read_text())
    assert marker["fit"]["kinds"] == ["head", "calibrate"]
    assert not open_folder(result.path).shares_base  # loads a base of its own

    # The fitted model serves what the fit scored: the stored states read by the
    # trained head, then calibrated.
    model = mlx_decision.load(result.path)
    holdout = training_splits(data, seed=0).holdout
    store = head_fit.StateStore(tmp_path / "check")
    unfitted = mlx_decision.load(base)
    scored = score(unfitted, holdout, store=store)
    scored = rehead(scored, store, model.backend.base)
    for item in scored:
        served = model.backend.score(item.request).probabilities
        adjusted = model.backend.adjust(item.request.questions, item.raw)
        for question_id, values in served.items():
            for option, p in values.items():
                assert p == pytest.approx(adjusted[question_id][option], abs=1e-5)
    store.remove()


def test_a_head_fit_cancelled_while_training_leaves_nothing(tmp_path, base, quick):
    stop = threading.Event()

    def on_event(event):
        if event.stage == "train" and event.step:
            stop.set()

    with pytest.raises(Cancelled):
        fit(base, labelled(60), output=tmp_path / "out" / "fit", on_event=on_event, cancel=stop)
    assert list((tmp_path / "out").iterdir()) == []


def test_a_head_fit_refuses_a_disk_without_room(tmp_path, base, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda path: usage(10**12, 10**12, 10**6))
    with pytest.raises(ValueError, match=r"the hidden states need about .* MB on disk in .*, "
                       r"where 1.0 MB are free"):  # fmt: skip
        fit(base, labelled(60), output=tmp_path / "out" / "fit")
    assert list((tmp_path / "out").iterdir()) == []


def test_an_epoch_worse_on_dev_stops_the_head_fit(tmp_path, base, quick, monkeypatch):
    backend = load(base)
    store = head_fit.StateStore(tmp_path / "states")
    inputs = backend.head_inputs(request())
    items = [head_fit.Item(store.add(inputs), {0: 0}) for _ in range(6)]
    losses = iter([0.5, 0.6, 0.1])
    monkeypatch.setattr(head_fit, "dev_loss", lambda *a, **k: next(losses))
    trained = head_fit.train(backend, store, items[:4], items[4:], epochs=3)
    assert (len(trained.epochs), trained.best, trained.stopped) == (2, 1, True)
    losses = iter([0.5, 0.6, 0.1])
    trained = head_fit.train(backend, store, items[:4], items[4:], epochs=3, early_stop=False)
    assert (len(trained.epochs), trained.best, trained.stopped) == (3, 3, False)
    store.remove()
