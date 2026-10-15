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


def test_long_attention_runs_in_blocks_of_queries(folder, noncausal, monkeypatch):
    from mlx_decision.backbones.qwen3_5 import base

    # On the CPU: on the GPU, float32 matmuls use TF32, rounded differently
    # for blocks of other shapes.
    ids = mx.array([list(range(10, 210))])
    with mx.stream(mx.cpu):
        whole = noncausal(ids)
    calls = []
    fused = mx.fast.scaled_dot_product_attention

    def counted(queries, *args, **kwargs):
        calls.append(queries.shape[2])
        return fused(queries, *args, **kwargs)

    # Room for 64 queries' scores against the 200 keys, per head.
    heads = noncausal.language_model.args.num_attention_heads
    monkeypatch.setattr(base, "SCORE_BUDGET", 64 * heads * 200)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", counted)
    with mx.stream(mx.cpu):
        blocks = noncausal(ids)
    assert set(calls) == {64, 8}  # three blocks of 64 and the last 8, per layer
    assert mx.abs(blocks - whole).max().item() < 1e-5


def test_layer_types_set_the_full_attention_layers():
    from mlx_decision.backbones.qwen3_5.qwen3_5 import Qwen3_5TextModel, TextModelArgs
    from tiny_models import PPLX_TEXT

    # The every-fourth-layer pattern without layer_types; a pruned model's own with them.
    stock = Qwen3_5TextModel(TextModelArgs.from_dict({**PPLX_TEXT, "layer_types": None}))
    assert [layer.is_linear for layer in stock.layers] == [True, True, True, False] * 2
    pruned = ["linear_attention", "full_attention", "full_attention", "full_attention"]
    args = TextModelArgs.from_dict({**PPLX_TEXT, "num_hidden_layers": 4, "layer_types": pruned})
    model = Qwen3_5TextModel(args)
    assert [layer.is_linear for layer in model.layers] == [True, False, False, False]
    assert model.fa_idx == 1
    with pytest.raises(ValueError, match="layer_types"):
        TextModelArgs.from_dict({**PPLX_TEXT, "layer_types": pruned})
