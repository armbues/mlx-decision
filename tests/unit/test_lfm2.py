"""The LFM2 text model (adapted from mlx-lm) on a tiny random model, on the CPU."""

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import lfm2 as upstream

from mlx_decision.backbones.lfm2.lfm2 import Model, ModelArgs
from mlx_decision.backbones.lfm2.load import load_text_model
from mlx_decision.backbones.qwen3_5.cache import fork_cache

CONFIG = {
    "model_type": "lfm2",
    "vocab_size": 300,
    "hidden_size": 64,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 1024,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "block_dim": 64,
    "block_multiple_of": 16,
    "block_ffn_dim_multiplier": 1.0,
    "block_auto_adjust_ff_dim": False,
    "intermediate_size": 96,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    "layer_types": ["conv", "conv", "full_attention", "conv", "full_attention"],
    "tie_word_embeddings": True,
}


@pytest.fixture(autouse=True)
def cpu():
    with mx.stream(mx.cpu):
        yield


@pytest.fixture(scope="module")
def models():
    """Ours and mlx-lm's with the same random weights (float32)."""
    mx.random.seed(0)
    ours = Model(ModelArgs.from_dict(CONFIG))
    theirs = upstream.Model(upstream.ModelArgs.from_dict(CONFIG))
    theirs.load_weights(tree_flatten(ours.parameters()), strict=True)
    mx.eval(ours.parameters(), theirs.parameters())
    return ours, theirs


def ids(n: int, start: int = 1) -> mx.array:
    return mx.array([[(start + 7 * i) % 300 for i in range(n)]])


def test_hidden_states_and_logits_equal_mlx_lms(models):
    ours, theirs = models
    x = ids(23)
    hidden = ours(x)
    assert mx.array_equal(hidden, theirs.model(x))
    rows = mx.array([5, 17, 299])
    assert mx.allclose(hidden @ ours.output_embedding_rows(rows).T, theirs(x)[..., rows], atol=1e-5)


def test_a_pass_continued_from_a_cache_equals_one_pass(models):
    ours, _ = models
    x = ids(30)
    whole = ours(x)
    cache = ours.make_cache()
    head = ours(x[:, :19], cache=cache)
    tail = ours(x[:, 19:], cache=cache)
    assert mx.allclose(mx.concatenate([head, tail], axis=1), whole, atol=1e-5)


def test_forks_of_a_prefix_continue_independently(models):
    ours, _ = models
    prefix, a, b = ids(12), ids(5, start=40), ids(9, start=90)
    cache = ours.make_cache()
    ours(prefix, cache=cache)
    kept = [c.state for c in cache]
    out_a = ours(a, cache=fork_cache(cache))
    out_b = ours(b, cache=fork_cache(cache))
    assert mx.allclose(out_a, ours(mx.concatenate([prefix, a], axis=1))[:, 12:], atol=1e-5)
    assert mx.allclose(out_b, ours(mx.concatenate([prefix, b], axis=1))[:, 12:], atol=1e-5)
    # The prefix's own cache is as it was.
    for c, state in zip(cache, kept, strict=True):
        for now, before in zip(c.state, state, strict=True):
            assert now is before


def test_loads_the_lfm2_vl_release_layout(models, tmp_path):
    """Text weights under model.language_model.*, conv kernels as stored by PyTorch,
    a vision tower and projector that are not read."""
    ours, _ = models
    weights = {}
    for name, value in tree_flatten(ours.parameters()):
        if name.endswith("conv.conv.weight"):
            value = value.transpose(0, 2, 1)  # PyTorch's [channels, 1, taps]
        weights["model.language_model." + name[len("model.") :]] = value
    weights["model.vision_tower.vision_model.post_layernorm.weight"] = mx.ones((8,))
    weights["model.multi_modal_projector.linear_1.weight"] = mx.ones((4, 8))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config = {"model_type": "lfm2_vl", "text_config": CONFIG, "vision_config": {"hidden_size": 8}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_text_model(tmp_path)
    x = ids(17)
    assert mx.array_equal(loaded(x), ours(x))


def test_needs_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    with pytest.raises(FileNotFoundError):
        load_text_model(tmp_path)
