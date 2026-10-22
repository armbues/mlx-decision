"""MLX's cache of freed buffers is capped when a model loads, so answering many
requests of varying length keeps memory flat."""

import random

import mlx.core as mx
import pytest

import mlx_decision
from mlx_decision import memory
from tiny_models import write_pplx


@pytest.fixture
def cache_limit():
    """Restores the cache limit a test changes."""
    previous = mx.set_cache_limit(memory.MIN_CACHE_BYTES)
    mx.set_cache_limit(previous)
    yield
    mx.set_cache_limit(previous)


def current_limit() -> int:
    limit = mx.set_cache_limit(0)
    mx.set_cache_limit(limit)
    return limit


def test_the_cap_is_a_tenth_of_the_working_set_and_at_least_1_gb(cache_limit, monkeypatch):
    mx.set_cache_limit(2**40)
    monkeypatch.setattr(memory, "working_set", lambda: 64 * 2**30)
    assert memory.cap_buffer_cache() == int(6.4 * 2**30) == current_limit()
    mx.set_cache_limit(2**40)
    monkeypatch.setattr(memory, "working_set", lambda: 4 * 2**30)
    assert memory.cap_buffer_cache() == 2**30 == current_limit()


def test_a_lower_limit_set_before_is_kept(cache_limit):
    mx.set_cache_limit(2**20)
    assert memory.cap_buffer_cache() == 2**20 == current_limit()


def test_loading_a_model_caps_the_cache(cache_limit, tmp_path):
    mx.set_cache_limit(2**40)
    mlx_decision.load(write_pplx(tmp_path / "tiny", joint="flash"))
    assert current_limit() == max(memory.MIN_CACHE_BYTES, int(memory.working_set() * 0.1))


def test_memory_stays_flat_over_200_requests_of_varying_length(tmp_path):
    model = mlx_decision.load(write_pplx(tmp_path / "tiny", joint="flash"))
    questions = {
        "team": {"type": "choice", "criteria": {"billing": "refund", "technical": "crash"}},
        "urgent": {"type": "noul", "instructions": "now"},
    }
    words = ["billing", "refund", "crash", "please", "now"]
    rng = random.Random(0)
    active = []
    for _ in range(200):
        model.decide(" ".join(rng.choice(words) for _ in range(rng.randint(5, 400))), questions)
        active.append(mx.get_active_memory())
    # Nothing kept between requests (a few bytes come and go).
    assert max(active[20:]) - active[20] < 2**20
    assert mx.get_cache_memory() <= current_limit()
