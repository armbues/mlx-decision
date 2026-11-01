"""The LFM3 text model, vision tower and projector against transformers' code.

``tests/parity/d1lfm3/backbone.json`` holds what the release's vendored
transformers LFM3 code computes on the tiny random model of
``write_lfm3_vl`` (``scripts/make_lfm3_reference.py``, CPU, float32).
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_decision.backbones.lfm3 import language
from mlx_decision.backbones.lfm3.language import SetLayout
from mlx_decision.backbones.lfm3.load import load_text_model, load_vision
from tiny_models import write_lfm3_vl

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "parity" / "d1lfm3" / "backbone.json").read_text()
)
TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_lfm3_vl(tmp_path_factory.mktemp("lfm3") / "tiny-lfm3", seed=FIXTURE["seed"])


@pytest.fixture(scope="module")
def model(folder):
    return load_text_model(folder)


def layout(data: dict) -> SetLayout:
    return SetLayout(data["lengths"], data["state"], mx.array(data["question"], dtype=mx.int32))


def largest_difference(ours, theirs) -> float:
    return float(np.abs(np.array(ours, dtype=np.float64) - np.array(theirs)).max())


def test_a_causal_pass_is_transformers(model):
    with mx.stream(mx.cpu):
        hidden = model(mx.array(FIXTURE["causal"]["ids"]))
    assert largest_difference(hidden, FIXTURE["causal"]["hidden"]) < TOLERANCE


def test_a_pass_under_set_masks_is_transformers(model):
    data = FIXTURE["set"]
    with mx.stream(mx.cpu):
        hidden = model(mx.array(data["ids"]), layout=layout(data))
    for row, length in enumerate(data["lengths"]):
        assert largest_difference(hidden[row, :length], data["hidden"][row]) < TOLERANCE


def test_set_masks_differ_from_causal(model):
    """The fixture is not passing on a causal pass alone."""
    data = FIXTURE["set"]
    with mx.stream(mx.cpu):
        causal = model(mx.array(data["ids"]))
    length = data["lengths"][0]
    assert largest_difference(causal[0, :length], data["hidden"][0]) > 1e-3


def test_the_vision_tower_and_projector_are_transformers(folder):
    data = FIXTURE["vision"]
    vision = load_vision(folder)
    pixels = mx.array(data["pixels"])
    with mx.stream(mx.cpu):
        tower = vision.vision_tower(pixels, data["grids"])
        features = vision(pixels, data["grids"])
    assert largest_difference(tower, data["tower"]) < TOLERANCE
    assert largest_difference(features, data["features"]) < TOLERANCE


def test_blocks_of_queries_equal_one_dense_mask(model, monkeypatch):
    """Long rows go a block of queries at a time; the result is the same."""
    from mlx_decision.backbones.qwen3_5 import base

    rng = np.random.default_rng(1)
    state, questions = 200, [90, 130, 70]
    length = state + sum(questions) + 5
    question = np.full((2, length), -1, dtype=np.int32)
    start = state
    for index, size in enumerate(questions):
        question[0, start : start + size] = index
        start += size
    question[1, state : state + 110] = 0
    data = {
        "lengths": [start, state + 110],
        "state": [state, state],
        "question": question.tolist(),
    }
    ids = mx.array(rng.integers(1, 300, (2, length)))
    with mx.stream(mx.cpu):
        whole = model(ids, layout=layout(data))
        # Blocks of 64 queries everywhere: state queries, question queries, sliding.
        monkeypatch.setattr(language, "SCORE_BUDGET", 64 * 2 * 4 * length)
        monkeypatch.setattr(base, "SCORE_BUDGET", 64 * 2 * 4 * length)
        monkeypatch.setattr(language, "SLIDING_WHOLE", 16)
        monkeypatch.setattr(language, "SLIDING_BLOCK", 64)
        blocks = model(ids, layout=layout(data))
    for row, real in enumerate(data["lengths"]):
        assert largest_difference(blocks[row, :real], whole[row, :real]) < 1e-6


def test_sliding_blocks_equal_one_mask():
    """Sliding attention in blocks over just the keys each window reaches."""
    rng = np.random.default_rng(2)
    q, k, v = (mx.array(rng.normal(0, 1, (1, 2, 3000, 8)).astype(np.float32)) for _ in range(3))
    with mx.stream(mx.cpu):
        whole = language.sliding_attention(q, k, v, 0.35, 256)
        saved = language.SLIDING_WHOLE, language.SLIDING_BLOCK
        language.SLIDING_WHOLE, language.SLIDING_BLOCK = 100, 300
        try:
            blocks = language.sliding_attention(q, k, v, 0.35, 256)
        finally:
            language.SLIDING_WHOLE, language.SLIDING_BLOCK = saved
    assert largest_difference(blocks, whole) < 1e-6


def test_padding_leaves_real_tokens_alone(model):
    """A row padded further answers as before at its real tokens."""
    data = FIXTURE["set"]
    ids = [row + [7] * 9 for row in data["ids"]]
    question = [row + [-1] * 9 for row in data["question"]]
    wider = {**data, "question": question}
    with mx.stream(mx.cpu):
        hidden = model(mx.array(ids), layout=layout(wider))
    for row, length in enumerate(data["lengths"]):
        assert largest_difference(hidden[row, :length], data["hidden"][row]) < TOLERANCE
