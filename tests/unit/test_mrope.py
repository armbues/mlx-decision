"""Multimodal rotary positions: the text half, checked against MLX's own RoPE."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_decision.backbones.qwen3_5.mrope import MRoPE

# clef-flash: head_dim 256, partial_rotary_factor 0.25, interleaved sections.
HEAD_DIM = 256
DIMS = 64
BASE = 10_000_000
SECTION = [11, 11, 10]


@pytest.fixture
def mrope():
    return MRoPE(DIMS, BASE, SECTION)


def queries(batch=2, heads=4, length=40, dtype=mx.float32):
    x = np.random.default_rng(0).standard_normal((batch, heads, length, HEAD_DIM))
    return mx.array(x.astype(np.float32)).astype(dtype)


def text_positions(batch, length, start=0):
    return mx.broadcast_to(mx.arange(start, start + length)[None, None], (3, batch, length))


@pytest.mark.parametrize("start", [0, 1000, 16000])
def test_text_positions_match_rope(mrope, start):
    x = queries()
    ours = mrope(x, text_positions(2, 40, start))
    reference = nn.RoPE(DIMS, traditional=False, base=BASE)(x, offset=start)
    # Both round position * frequency in float32, differently; the error grows
    # with the position.
    tolerance = 1e-5 + 2e-7 * (start + 40)
    assert mx.abs(ours - reference).max().item() < tolerance


def test_channels_beyond_the_rotary_part_are_untouched(mrope):
    x = queries()
    ours = mrope(x, text_positions(2, 40, 7))
    assert mx.array_equal(ours[..., DIMS:], x[..., DIMS:])


def test_keeps_the_input_dtype(mrope):
    x = queries(dtype=mx.bfloat16)
    assert mrope(x, text_positions(2, 40)).dtype == mx.bfloat16


def test_axes_are_interleaved_within_their_sections(mrope):
    axes = mrope._axes.tolist()
    assert len(axes) == DIMS // 2
    assert [i for i, a in enumerate(axes) if a == 1] == list(range(1, 33, 3))[: SECTION[1]]
    assert [i for i, a in enumerate(axes) if a == 2] == list(range(2, 30, 3))[: SECTION[2]]
    assert axes.count(0) == DIMS // 2 - SECTION[1] - SECTION[2]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_moving_one_axis_rotates_only_its_frequencies(mrope, axis):
    x = queries(batch=1, length=8)
    base = np.array(text_positions(1, 8, 5))
    moved = base.copy()
    moved[axis] += 17
    changed = mx.abs(mrope(x, mx.array(moved)) - mrope(x, mx.array(base))).max(axis=(0, 1, 2))
    half = DIMS // 2
    own = [i for i, a in enumerate(mrope._axes.tolist()) if a == axis]
    expected = np.zeros(HEAD_DIM, dtype=bool)
    expected[own] = True
    expected[[i + half for i in own]] = True
    assert np.array_equal(np.array(changed) > 1e-6, expected)
