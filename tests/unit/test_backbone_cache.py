"""The backbone continuing from a cache equals one pass over the whole input."""

import mlx.core as mx
import pytest

from mlx_decision.backbones.qwen3_5.cache import fork_cache
from tiny_models import qwen3_5

LENGTH = 24


@pytest.fixture(scope="module")
def model():
    return qwen3_5(seed=3)


def tokens(seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.randint(0, 100, (1, LENGTH))


def image_positions() -> mx.array:
    """[3, 1, L]: text, then a 2 x 3 image block, then text continuing after it."""
    t, h, w = [], [], []
    for i in range(6):
        t.append(i), h.append(i), w.append(i)
    for row in range(2):
        for column in range(3):
            t.append(6), h.append(6 + row), w.append(6 + column)
    for i in range(LENGTH - 12):
        t.append(9 + i), h.append(9 + i), w.append(9 + i)
    return mx.array([t, h, w])[:, None, :]


def split_pass(model, ids, cuts, positions=None):
    cache = model.make_cache()
    parts = []
    bounds = [0, *cuts, ids.shape[1]]
    for start, end in zip(bounds, bounds[1:], strict=False):
        part_positions = None if positions is None else positions[:, :, start:end]
        parts.append(model(ids[:, start:end], position_ids=part_positions, cache=cache))
    return mx.concatenate(parts, axis=1), cache


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("cuts", [[10], [2], [LENGTH - 2], [5, 17]])
def test_continuing_from_a_cache_equals_one_pass(model, with_images, cuts):
    ids = tokens()
    positions = image_positions() if with_images else None
    whole = model(ids, position_ids=positions)
    split, _ = split_pass(model, ids, cuts, positions)
    assert split.shape == whole.shape
    assert mx.array_equal(split, whole).item()


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("cuts", [[1], [LENGTH - 1], [1, 2]])
def test_single_token_parts_on_the_cpu(model, with_images, cuts):
    # On the GPU, float32 matmuls over several rows use TF32 and a single row
    # does not, so a one-token part differs by about 1e-3 with or without a
    # cache; the CPU computes both exactly.
    ids = tokens()
    positions = image_positions() if with_images else None
    with mx.stream(mx.cpu):
        whole = model(ids, position_ids=positions)
        split, _ = split_pass(model, ids, cuts, positions)
        assert mx.allclose(split, whole, atol=1e-5, rtol=1e-5).item()


def test_a_forked_cache_serves_several_continuations(model):
    ids, other = tokens(0), tokens(1)
    prefix = 9
    cache = model.make_cache()
    model(ids[:, :prefix], cache=cache)
    kept = [list(c.state) for c in cache]
    for rest in (ids[:, prefix:], other[:, prefix:]):
        continued = model(rest, cache=fork_cache(cache))
        whole = model(mx.concatenate([ids[:, :prefix], rest], axis=1))
        assert mx.array_equal(continued, whole[:, prefix:]).item()
    # The kept cache still holds the prefix's arrays, unchanged.
    for c, before in zip(cache, kept, strict=True):
        for now, then in zip(c.state, before, strict=True):
            assert now is then


def test_cache_contents(model):
    cache = model.make_cache()
    model(tokens()[:, :7], cache=cache)
    attention = [c for c in cache if hasattr(c, "keys")]
    linear = [c for c in cache if not hasattr(c, "keys")]
    assert len(attention) == len(model.language_model.layers) // 4
    assert all(c.offset == 7 and c.keys.shape[-2] == 7 for c in attention)
    kernel = model.language_model.args.linear_conv_kernel_dim
    assert all(c[0].shape[:2] == (1, kernel - 1) for c in linear)
    # The recurrent state is kept in float32, whatever the weights' precision.
    assert all(c[1].dtype == mx.float32 for c in linear)
    assert sum(c.nbytes for c in cache) > 0
