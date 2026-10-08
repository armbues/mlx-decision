"""The backbone in pplx's layout: no ``model.`` prefix, no lm_head, non-causal full attention."""

import mlx.core as mx
import pytest

from mlx_decision.backbones.qwen3_5.load import load_text_model
from mlx_decision.backbones.qwen3_5.vision import load_vision_model
from tiny_models import write_pplx


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_pplx(tmp_path_factory.mktemp("pplx"))


@pytest.fixture(scope="module")
def noncausal(folder):
    return load_text_model(folder, lm_head=False, causal=False)


def test_loads_without_lm_head(folder, noncausal):
    assert "lm_head" not in noncausal.language_model
    with pytest.raises(ValueError, match="lm_head"):
        load_text_model(folder)


def test_vision_tower_loads_from_the_release_names(folder):
    model = load_vision_model(folder)
    assert model.patch_embed.proj.weight.shape == (32, 3 * 2 * 16 * 16)


def test_full_attention_sees_later_tokens(folder, noncausal):
    causal = load_text_model(folder, lm_head=False)
    ids = mx.array([[11, 12, 13, 14, 15, 16]])
    changed = mx.array([[11, 12, 13, 14, 15, 99]])
    for model, sees_later in ((causal, False), (noncausal, True)):
        before = model(ids)[0, :-1]
        after = model(changed)[0, :-1]
        assert (mx.abs(before - after).max().item() > 1e-4) == sees_later


def test_noncausal_refuses_a_cache(noncausal):
    with pytest.raises(ValueError, match="cache"):
        noncausal(mx.array([[1, 2, 3]]), cache=noncausal.make_cache())
