"""The readout families' hooks on tiny models: features, per-call readouts and finish."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_decision
from mlx_decision.models.readout import Features, Readout, option_logits, read_out
from mlx_decision.types import parse_request
from tiny_models import write_d1, write_h2o, write_pplx

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Plans"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
}
REQUEST = parse_request({"state": "My invoice is wrong, please refund me", "questions": QUESTIONS})
WRITERS = {"h2o": write_h2o, "d1": write_d1, "pplx": write_pplx}


@pytest.fixture(scope="module", params=list(WRITERS))
def backend(request, tmp_path_factory):
    family = request.param
    folder = WRITERS[family](tmp_path_factory.mktemp(family) / f"tiny-{family}")
    return mlx_decision.load(folder).backend


def by_hand(features: Features) -> list[float]:
    """The readout in NumPy (float64), from the features alone."""
    logits = np.array(features.rows) @ np.array(features.hidden)
    scores = np.array([logits[group].max() for group in features.groups])
    scores = scores / features.temperature
    weights = np.exp(scores - scores.max())
    return list(weights / weights.sum())


def test_features_reproduce_the_served_probabilities(backend):
    with mx.stream(mx.cpu):
        served = backend.score(REQUEST, finish=False).probabilities
        features = backend.features(REQUEST)
        again = {question_id: read_out(f) for question_id, f in features.items()}
    assert list(features) == list(QUESTIONS)
    for question_id, values in served.items():
        f = features[question_id]
        assert f.option_ids == list(values)
        assert f.hidden.dtype == f.rows.dtype == mx.float32
        assert f.rows.shape[1] == f.hidden.shape[0]
        assert by_hand(f) == pytest.approx(list(values.values()), abs=1e-5)
        assert again[question_id] == list(values.values())


def test_score_is_the_readout_then_finish(backend):
    raw = backend.score(REQUEST, finish=False).probabilities
    served = backend.score(REQUEST).probabilities
    for question_id, question in REQUEST.questions.items():
        assert served[question_id] == backend.finish(question, raw[question_id])


def test_h2o_finish_holds_the_yes_no_floor(tmp_path):
    model = mlx_decision.load(write_h2o(tmp_path / "tiny-h2o"))
    backend = model.backend
    raw = backend.score(REQUEST, finish=False).probabilities["refund"]
    assert max(raw.values()) < backend.noul_floor  # the tiny model is unsure
    served = backend.score(REQUEST).probabilities["refund"]
    side = raw["true"] >= 0.5
    assert served["true"] == pytest.approx(backend.noul_floor if side else 1 - backend.noul_floor)
    choice = {"a": 0.6, "b": 0.4}
    assert backend.finish(REQUEST.questions["team"], choice) is choice


def zero_readout(f: Features) -> Readout:
    return Readout(f.option_ids, f.rows, mx.zeros(len(f.groups)))


def test_readouts_replace_the_rows_of_their_questions_only(backend):
    with mx.stream(mx.cpu):
        features = backend.features(REQUEST)
        base = backend.score(REQUEST, finish=False).probabilities
        same = backend.score(
            REQUEST, {q: zero_readout(f) for q, f in features.items()}, finish=False
        ).probabilities
        team = features["team"]
        bias = mx.array([0.0, 0.0, 30.0])
        readout = Readout(team.option_ids, team.rows * 0, bias)
        moved = backend.score(REQUEST, {"team": readout}, finish=False).probabilities
    for question_id in QUESTIONS:
        assert list(same[question_id].values()) == pytest.approx(
            list(base[question_id].values()), abs=1e-6
        )
    assert moved["team"]["sales"] > 0.999
    assert moved["urgency"] == base["urgency"] and moved["refund"] == base["refund"]


def test_a_readout_for_other_options_is_not_used(backend):
    features = backend.features(REQUEST)
    team = features["team"]
    other = Readout(["billing", "technical", "legal"], team.rows * 0, mx.array([0, 0, 30.0]))
    base = backend.score(REQUEST).probabilities
    assert backend.score(REQUEST, {"team": other}).probabilities == base
    with pytest.raises(ValueError, match="for other options"):
        option_logits(team, other)
    wrong = Readout(team.option_ids, team.rows[:1], mx.zeros(3))
    with pytest.raises(ValueError, match="do not fit"):
        backend.score(REQUEST, {"team": wrong})


def test_grouped_rows_take_each_options_best_token():
    rows = mx.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
    hidden = mx.array([1.0, 2.0])  # token logits 1, 2, 2, 6
    features = Features(["x", "y", "z"], hidden, rows, [[0, 2], [1], [3, 0, 1]], 2.0)
    logits = option_logits(features)
    assert logits.tolist() == pytest.approx([1.0, 1.0, 3.0])
    assert read_out(features) == pytest.approx(by_hand(features), abs=1e-6)
    readout = Readout(["x", "y", "z"], rows, mx.array([1.0, 0.0, -1.0]))
    assert option_logits(features, readout).tolist() == pytest.approx([2.0, 1.0, 2.0])


def test_h2o_features_of_a_request_with_images_use_the_image_temperature(tmp_path):
    image = pytest.importorskip("PIL.Image").new("RGB", (64, 48), (200, 30, 30))
    backend = mlx_decision.load(write_h2o(tmp_path / "tiny-h2o")).backend
    request = parse_request({"state": "See the photo", "questions": QUESTIONS, "images": [image]})
    with mx.stream(mx.cpu):
        served = backend.score(request, finish=False).probabilities
        features = backend.features(request)
    for question_id, f in features.items():
        assert f.temperature == backend.config.image.temperature
        assert by_hand(f) == pytest.approx(list(served[question_id].values()), abs=1e-5)


def test_the_readout_families_are_readable(backend, tmp_path):
    from mlx_decision.backend import Readable
    from tiny_models import write_marker

    assert isinstance(backend, Readable)
    laya = mlx_decision.load(write_marker(tmp_path / "laya", "laya")).backend
    assert not isinstance(laya, Readable)
