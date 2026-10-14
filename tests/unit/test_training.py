"""Training: the fit's stages, events and report, and calibration end to end."""

import hashlib
import json
import math
import threading

import mlx.core as mx
import pytest

import mlx_decision
from fake_backend import TEAM, write_fitted, write_model
from mlx_decision.data.format import DataError, Example, training_splits, write_examples
from mlx_decision.fitted import FIT_FILE, Calibration, calibrate
from mlx_decision.memory import ModelTooLargeError
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import Cancelled, Event, fit, plan
from mlx_decision.training.calibrate import fit_calibration
from mlx_decision.training.loss import decision_loss
from tiny_models import write_clef, write_d1, write_h2o, write_marker, write_pplx

URGENT = {"type": "noul", "instructions": "Is it urgent?"}


def examples(count: int = 100, labels=("tech",) * 8 + ("billing", "sales")) -> list[Example]:
    """``count`` examples asking ``TEAM`` (the fake model favours billing, then tech)
    and an unlabelled noul, labelled by cycling through ``labels``."""
    return [
        Example(
            f"e{i}",
            {"state": f"ticket {i}", "questions": {"team": TEAM, "urgent": URGENT}},
            {"team": labels[i % len(labels)]},
        )
        for i in range(count)
    ]


@pytest.fixture
def base(tmp_path):
    return write_model(tmp_path / "base")


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "data.jsonl"
    write_examples(path, examples())
    return path


def test_a_calibration_fit_moves_answers_towards_the_labels(tmp_path, base, data):
    result = fit(base, data, output=tmp_path / "support")
    assert result.path == tmp_path / "support"
    assert result.kinds == ["calibrate"]
    team = result.metrics["team"]
    assert team["before"]["accuracy"] == pytest.approx(0.1)  # always billing
    assert team["after"]["accuracy"] == pytest.approx(0.8)  # always tech
    assert team["before"]["n"] == team["after"]["n"] == 20
    model = mlx_decision.load(result.path)
    answer = model.decide("anything", {"team": TEAM}).answers["team"]
    assert (model.name, answer.choice) == ("support", "tech")
    marker = json.loads((result.path / MARKER_FILE).read_text())
    assert marker["family"] == "fake"
    assert marker["base"] == {"path": str(base.absolute())}
    assert marker["fit"]["kinds"] == ["calibrate"]
    assert marker["fit"]["questions"] == [
        {"id": "team", "type": "choice", "options": ["billing", "tech", "sales"]}
    ]
    assert marker["fit"]["seed"] == 0
    assert marker["fit"]["data"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert marker["fit"]["metrics"]["team"]["after"]["accuracy"] == pytest.approx(0.8)
    report = json.loads((result.path / "training.json").read_text())
    assert report == json.loads(json.dumps(result.report))
    assert report["data"] | {"labels": None} == {
        "file": str(data),
        "sha256": marker["fit"]["data"],
        "examples": 100,
        "holdout": 20,
        "fit": 40,
        "dev": 40,  # a tenth, at least 50, at most half
        "labels": None,
        "refused": 0,
        "truncated": 0,
    }
    assert report["data"]["labels"] == {"team": {"billing": 8, "sales": 8, "tech": 64}}
    [calibration] = report["calibrations"]
    assert calibration["question"] == "team" and len(calibration["bias"]) == 3
    assert "urgent" not in result.metrics  # asked, not labelled: not trained on
    assert report["warnings"] == [
        "question 'team': fewer than 10 training examples for billing (8), sales (8)"
    ]


def test_events_follow_the_stages_and_serialize(tmp_path, base, data):
    events: list[Event] = []
    fit(base, data, output=tmp_path / "fit", on_event=events.append)
    stages = list(dict.fromkeys(event.stage for event in events))
    assert stages == ["check", "before", "features", "calibrate", "after", "write", "done"]
    before = [e for e in events if e.stage == "before" and e.step is not None]
    assert [(e.step, e.total) for e in before] == [(i, 20) for i in range(1, 21)]
    assert all(e.eta_s is not None for e in before)
    features = [e for e in events if e.stage == "features"]
    assert features[-1].eta_s == 0
    check = [e for e in events if e.stage == "check" and e.metrics]
    assert check[0].metrics["kinds"] == ["calibrate"]
    assert set(check[0].metrics["estimate"]) == {"peak_gb"}
    assert any(e.message and e.message.startswith("warning: ") for e in events)
    for event in events:
        json.dumps(event.as_json())
    assert Event("train", 3, 9).as_json() == {"stage": "train", "step": 3, "total": 9}


@pytest.mark.parametrize("stage", ["before", "features"])
def test_a_cancelled_fit_writes_nothing(tmp_path, base, data, stage):
    stop = threading.Event()

    def on_event(event):
        if event.stage == stage and event.step == 2:
            stop.set()

    with pytest.raises(Cancelled):
        fit(base, data, output=tmp_path / "out" / "fit", on_event=on_event, cancel=stop)
    assert not (tmp_path / "out").exists() or list((tmp_path / "out").iterdir()) == []


def test_the_same_seed_gives_the_same_fit(tmp_path, base, data):
    first = fit(base, data, output=tmp_path / "a", seed=3)
    again = fit(base, data, output=tmp_path / "b", seed=3)
    other = fit(base, data, output=tmp_path / "c", seed=4)
    assert first.report["calibrations"] == again.report["calibrations"]
    assert (tmp_path / "a" / FIT_FILE).read_bytes() == (tmp_path / "b" / FIT_FILE).read_bytes()
    assert first.report["calibrations"] != other.report["calibrations"]


def test_a_validation_file_replaces_the_holdout(tmp_path, base, data):
    validation = tmp_path / "validation.jsonl"
    write_examples(validation, examples(30))
    result = fit(base, data, output=tmp_path / "fit", validation=validation)
    assert result.metrics["team"]["after"]["n"] == 30
    assert result.report["data"] | {"labels": None} == result.report["data"] | {
        "holdout": 30,
        "fit": 50,
        "dev": 50,
        "labels": None,
    }
    assert result.report["options"]["validation"] == str(validation)


def test_biases_only_for_a_question_asked_the_same_way_everywhere(tmp_path, base):
    varied = examples()
    for example in varied[::2]:
        example.request["questions"]["team"] = {**TEAM, "instructions": "Who handles it?"}
    result = fit(base, varied, output=tmp_path / "fit")
    [calibration] = result.report["calibrations"]
    assert calibration["bias"] is None
    assert result.report["data"]["file"] is None


def test_a_fitted_model_is_refitted_on_its_base(tmp_path, base, data):
    old = write_fitted(tmp_path / "old", base)
    result = fit(old, data, output=tmp_path / "new")
    marker = json.loads((result.path / MARKER_FILE).read_text())
    assert marker["base"] == {"path": str(base.absolute())}
    assert result.report["base"]["name"] == "base"


def test_problems_are_found_before_the_model_loads(tmp_path, base, data, monkeypatch):
    import mlx_decision.training.fitting as fitting

    def no_load(*args, **kwargs):
        raise AssertionError("loaded")

    monkeypatch.setattr(fitting, "load", no_load)
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError, match="taken exists"):
        fit(base, data, output=tmp_path / "taken")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": 3}}\n')
    with pytest.raises(DataError, match="line 1: labels.q: a noul label is true or false"):
        fit(base, bad, output=tmp_path / "fit")
    with pytest.raises(ValueError, match="no fitting method is called magic"):
        fit(base, data, output=tmp_path / "fit", method="magic")
    sized = tmp_path / "sized"
    write_model(sized)
    mx.save_safetensors(str(sized / "model.safetensors"), {"w": mx.zeros(1000, mx.uint8)})
    monkeypatch.setattr("mlx_decision.memory.working_set", lambda: 500)
    with pytest.raises(ModelTooLargeError, match="sized needs about"):
        fit(sized, data, output=tmp_path / "fit")


def test_kinds_per_family():
    assert plan.kinds("laya") == ["finetune", "calibrate"]
    assert plan.kinds("fake") == ["calibrate"]
    assert plan.resolve("h2o", "auto") == ["readout", "calibrate"]
    assert plan.resolve("julia", "calibrate") == ["calibrate"]
    assert plan.resolve("laya", "auto") == ["finetune", "calibrate"]
    with pytest.raises(ValueError, match="laya models cannot be fitted with readout; they take "
                       "auto, finetune, calibrate"):  # fmt: skip
        plan.resolve("laya", "readout")


def test_a_run_over_the_working_set_names_the_smaller_runs_that_fit(tmp_path, monkeypatch):
    from mlx_decision.training.fitting import prepare

    laya = write_marker(tmp_path / "laya", "laya")
    prepared = prepare(laya, examples(60))
    full = prepared.estimate(["finetune", "calibrate"], False).peak_bytes
    head = prepared.estimate(["finetune", "calibrate"], True).peak_bytes
    assert head < full
    monkeypatch.setattr("mlx_decision.memory.working_set", lambda: head)
    with pytest.raises(ModelTooLargeError) as error:
        fit(laya, examples(60), output=tmp_path / "fit")
    message = str(error.value)
    assert "GB for training)" in message and "--no-memory-check" not in message
    assert "What would fit: training the head alone (--head-only, about" in message
    assert "or a calibration alone (--method calibrate, about" in message
    assert not (tmp_path / "fit").exists()


def test_the_estimate_counts_memory_only(tmp_path, base):
    estimate = plan.estimate(["calibrate"], base, {}, training_splits(examples(10)))
    assert not hasattr(estimate, "seconds") and set(estimate.as_json()) == {"peak_gb"}


def test_the_calibration_fit_recovers_a_temperature():
    # Answers a model gives at temperature 1, with labels drawn as if the truth
    # were twice as sure (temperature 0.5).
    import random

    rng = random.Random(0)
    rows, labels = [], []
    for _ in range(4000):
        logits = [rng.gauss(0, 1) for _ in range(3)]
        p = [math.exp(z) / sum(math.exp(y) for y in logits) for z in logits]
        q = calibrate(dict(enumerate(p)), Calibration(0.5))
        rows.append(p)
        labels.append(rng.choices(range(3), weights=list(q.values()))[0])
    calibration = fit_calibration(rows, labels, bias=False)
    assert calibration.temperature == pytest.approx(0.5, rel=0.1)
    assert calibration.bias is None
    assert fit_calibration(rows, labels).temperature == pytest.approx(0.5, rel=0.1)


def test_the_decision_loss_by_hand():
    logits = mx.array([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]])
    valid = mx.array([[True, True, True], [True, True, False]])
    labels = mx.array([0, 1])
    plain = decision_loss(logits, valid, labels, smoothing=0.0, brier=0.0)
    p1 = [math.exp(z) for z in (2.0, 0.0, -1.0)]
    p2 = [math.exp(z) for z in (0.5, 1.5)]
    want = (-math.log(p1[0] / sum(p1)) - math.log(p2[1] / sum(p2))) / 2
    assert plain.item() == pytest.approx(want, rel=1e-5)
    full = decision_loss(logits, valid, labels)
    assert full.item() > plain.item() * 0.9  # smoothing and Brier add up


# Every family, calibrated end to end on its tiny model.

WRITERS = {
    "clef": write_clef,
    "pplx": write_pplx,
    "h2o": write_h2o,
    "d1": write_d1,
    "laya": lambda folder: write_marker(folder, "laya"),
    "julia": lambda folder: write_marker(folder, "julia"),
}
CHOICE = {"type": "choice", "criteria": {"billing": "refund", "technical": "crash"}}
STATES = ["please refund", "crash now", "refund now please", "crash"]


@pytest.mark.parametrize("family", list(WRITERS))
def test_every_family_calibrates_end_to_end(tmp_path, family):
    base = WRITERS[family](tmp_path / f"tiny-{family}")
    data = [
        Example(
            f"e{i}",
            {"state": STATES[i % 4], "questions": {"team": CHOICE, "urgent": URGENT}},
            {
                "team": "billing" if "refund" in STATES[i % 4] else "technical",
                "urgent": "true" if "now" in STATES[i % 4] else "false",
            },
        )  # fmt: skip
        for i in range(40)
    ]
    result = fit(base, data, output=tmp_path / "fitted", method="calibrate")
    model = mlx_decision.load(result.path)
    assert model.info()["family"] == family
    assert model.info()["fit"]["questions"] == ["team", "urgent"]
    # The report's "after" is what the fitted model answers when loaded.
    holdout = training_splits(data, seed=0).holdout
    for question_id in ("team", "urgent"):
        answers = [model.decide_request(e.request).answers[question_id] for e in holdout]
        predicted = [
            a.choice if hasattr(a, "choice") else ("true" if a.noul >= 0.5 else "false")
            for a in answers
        ]
        right = sum(p == e.labels[question_id] for p, e in zip(predicted, holdout, strict=True))
        accuracy = right / len(holdout)
        assert result.metrics[question_id]["after"]["accuracy"] == pytest.approx(accuracy)
