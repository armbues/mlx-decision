"""Prefix reuse in the Clef backend, on tiny random models (no weights needed)."""

import io

import mlx.core as mx
import numpy as np
import pytest
from typer.testing import CliRunner

import mlx_decision
from mlx_decision.cli import app
from mlx_decision.models.clef.prefix import PrefixCache, prefix_key
from tiny_models import write_clef, write_marker

STATE = "billing yes technical no sales billing yes sales technical no billing"
FIRST = {
    "team": {"type": "choice", "criteria": {"billing": None, "sales": None}},
    "ok": {"type": "noul"},
}
OTHER = {"level": {"type": "score", "criteria": ["no", "yes", "billing"]}}


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_clef(tmp_path_factory.mktemp("models") / "tiny-clef", vision=True)


def probabilities(model, state=STATE, questions=FIRST, images=None) -> dict:
    request = {"state": state, "questions": questions}
    if images:
        request["images"] = images
    request = model.check(request)
    return model.backend.score(request).probabilities


def close(a: dict, b: dict, tolerance: float = 1e-5) -> bool:
    return a.keys() == b.keys() and all(
        abs(a[q][o] - b[q][o]) <= tolerance for q in a for o in a[q]
    )


def png(width=32, height=32, color=(200, 30, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_on_by_default_and_off_with_zero(folder):
    assert mlx_decision.load(folder).backend.prefix_cache.max_bytes == 2 * 10**9
    assert mlx_decision.load(folder, prefix_cache_gb=0).backend.prefix_cache is None
    with pytest.raises(ValueError, match="prefix_cache_gb"):
        mlx_decision.load(folder, prefix_cache_gb=-1)


def test_a_kept_state_answers_new_questions_as_a_fresh_model(folder):
    model = mlx_decision.load(folder)
    probabilities(model, questions=FIRST)
    warm = probabilities(model, questions=OTHER)
    cache = model.backend.prefix_cache
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    # The same request on a fresh model (a cold prefix) gives the same numbers.
    cold = probabilities(mlx_decision.load(folder), questions=OTHER)
    assert warm == cold
    # And one pass without reuse agrees within float tolerance.
    assert close(warm, probabilities(mlx_decision.load(folder, prefix_cache_gb=0), questions=OTHER))


def test_another_state_is_not_reused(folder):
    model = mlx_decision.load(folder)
    probabilities(model)
    probabilities(model, state=STATE + " sales")
    assert (model.backend.prefix_cache.hits, len(model.backend.prefix_cache)) == (0, 2)


def test_a_state_cut_differently_is_not_reused(folder):
    # A limit the state does not fit: longer questions leave less room for it.
    model = mlx_decision.load(folder, max_input_tokens=260)
    long_state = " ".join(["billing technical sales"] * 100)
    short = {"ok": {"type": "noul"}}
    longer = {"ok": {"type": "noul"}, "more": {"type": "noul"}}
    first = model.decide(long_state, short)
    second = model.decide(long_state, longer)
    assert first.truncated and second.truncated
    assert model.backend.prefix_cache.hits == 0
    # The same cut again is reused.
    model.decide(long_state, {"yes": {"type": "noul"}})
    assert model.backend.prefix_cache.hits == 1


def test_images_are_part_of_the_prefix(folder):
    model = mlx_decision.load(folder)
    red, blue = png(), png(color=(20, 30, 200))
    off = mlx_decision.load(folder, prefix_cache_gb=0)
    first = probabilities(model, images=[red])
    assert close(first, probabilities(off, images=[red]))
    # Another image is another prefix; the same image with other questions
    # is reused without running the vision tower again.
    probabilities(model, images=[blue])
    assert model.backend.prefix_cache.hits == 0
    calls = []
    vision = model.backend.vision
    model.backend.vision = lambda *args: calls.append(args) or vision(*args)
    model.backend.vision.args = vision.args
    warm = probabilities(model, questions=OTHER, images=[red])
    assert model.backend.prefix_cache.hits == 1
    assert calls == []
    assert close(warm, probabilities(off, questions=OTHER, images=[red]))


def test_answers_through_the_api_are_unchanged(folder):
    on, off = mlx_decision.load(folder), mlx_decision.load(folder, prefix_cache_gb=0)
    for questions in (FIRST, OTHER, FIRST):
        a, b = on.decide(STATE, questions), off.decide(STATE, questions)
        assert a.usage == b.usage and a.truncated == b.truncated
        for question_id, answer in a.answers.items():
            assert answer.type == b.answers[question_id].type


def entry(size: int):
    from mlx_decision.backbones.qwen3_5.cache import ArraysCache

    cache = ArraysCache(2)
    cache[0] = mx.zeros((size,), dtype=mx.uint8)
    return [cache], mx.zeros((0,), dtype=mx.uint8)


def test_least_recently_used_entries_go_first():
    cache = PrefixCache(max_bytes=300)
    for key in "abc":
        cache.put(key, *entry(100))
    assert cache.get("a") is not None  # now the most recently used
    cache.put("d", *entry(100))
    assert cache.get("b") is None
    assert all(cache.get(key) is not None for key in "acd")
    assert cache.nbytes == 300


def test_an_entry_larger_than_the_limit_is_used_once_but_not_kept():
    cache = PrefixCache(max_bytes=100)
    cache.put("small", *entry(50))
    continued = cache.put("big", *entry(200))
    assert continued.cache[0][0].shape == (200,)
    assert cache.get("big") is None
    assert cache.get("small") is not None


def test_entries_are_handed_out_as_forks():
    cache = PrefixCache(max_bytes=1000)
    cache.put("a", *entry(10))
    one, two = cache.get("a"), cache.get("a")
    one.cache[0][0] = mx.ones((3,))
    assert two.cache[0][0].shape == (10,)
    assert cache.get("a").cache[0][0].shape == (10,)


def test_the_key_covers_tokens_and_pixels():
    pixels = np.zeros((4, 6), dtype=np.float32)
    assert prefix_key([1, 2, 3]) == prefix_key([1, 2, 3])
    assert prefix_key([1, 2, 3]) != prefix_key([1, 2, 4])
    assert prefix_key([1, 2], [pixels]) != prefix_key([1, 2])
    assert prefix_key([1, 2], [pixels]) != prefix_key([1, 2], [pixels + 1])
    assert prefix_key([1, 2], [pixels]) != prefix_key([1, 2], [pixels.reshape(6, 4)])


def test_cli_flag(folder, tmp_path):
    runner = CliRunner()
    args = ["run", "-s", STATE, "--noul", "ok", "--json"]
    assert runner.invoke(app, [*args, "-m", str(folder), "--prefix-cache", "0"]).exit_code == 0
    assert runner.invoke(app, [*args, "-m", str(folder), "--prefix-cache", "0.5"]).exit_code == 0
    laya = write_marker(tmp_path / "laya", "laya")
    assert runner.invoke(app, [*args, "-m", str(laya)]).exit_code == 0
    refused = runner.invoke(app, [*args, "-m", str(laya), "--prefix-cache", "1"])
    assert refused.exit_code == 1
    assert "do not take prefix_cache_gb" in refused.stderr
