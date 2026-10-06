"""The ModernBERT encoder against transformers' implementation, on a tiny random model."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_decision.backbones.modernbert import Model, ModelArgs

# Old-style keys, as in the released ModernBERT and mmBERT config.json files. A small
# window (4 tokens each side) and 30-token inputs exercise several attention blocks.
TINY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "local_attention": 8,
    "max_position_embeddings": 64,
    "pad_token_id": 0,
    "hidden_activation": "gelu",
    "norm_eps": 1e-5,
}
LENGTHS = [30, 17, 3]


def token_ids(lengths, seed=0):
    rng = np.random.default_rng(seed)
    width = max(lengths)
    ids = np.zeros((len(lengths), width), dtype=np.int64)
    mask = np.zeros_like(ids)
    for row, n in enumerate(lengths):
        ids[row, :n] = rng.integers(1, TINY["vocab_size"], n)
        mask[row, :n] = 1
    return ids, mask


@pytest.fixture(scope="module")
def reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.modernbert")
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    config = ModernBertConfig(**TINY)
    model = ModernBertModel._from_config(
        config, attn_implementation="sdpa", dtype=torch.float32
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.5)
    return model


def mlx_model(reference, config=TINY) -> Model:
    model = Model(ModelArgs.from_config(config))
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    model.load_weights(list(weights.items()), strict=True)
    return model


def run(model, ids, mask=None):
    # MLX's float32 matmuls on the GPU are less exact; the CPU gives a fair comparison.
    with mx.stream(mx.cpu):
        out = model(mx.array(ids), None if mask is None else mx.array(mask))
        mx.eval(out)
    return np.array(out)


def test_config_layouts_agree(reference):
    old = ModelArgs.from_config(TINY)
    new = ModelArgs.from_config(reference.config.to_dict())
    assert old == new
    assert old.global_layers == (True, False, False, True)
    assert old.window == 4


def test_equal_to_the_reference_in_a_padded_batch(reference):
    import torch

    ids, mask = token_ids(LENGTHS)
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
    out = run(mlx_model(reference), ids, mask)
    real = mask.astype(bool)
    np.testing.assert_allclose(
        out[real], expected.last_hidden_state.numpy()[real], rtol=1e-4, atol=1e-4
    )


def test_padding_does_not_change_the_tokens(reference):
    model = mlx_model(reference)
    ids, mask = token_ids(LENGTHS)
    batched = run(model, ids, mask)
    for row, n in enumerate(LENGTHS):
        alone = run(model, ids[row : row + 1, :n])
        np.testing.assert_allclose(batched[row, :n], alone[0], rtol=1e-5, atol=1e-5)


def test_short_input_within_one_window(reference):
    import torch

    ids, _ = token_ids([5])
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor(ids)).last_hidden_state.numpy()
    np.testing.assert_allclose(run(mlx_model(reference), ids), expected, rtol=1e-4, atol=1e-4)


def test_new_style_rope_parameters():
    args = ModelArgs.from_config(
        {
            **{k: v for k, v in TINY.items() if not k.endswith("rope_theta")},
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            "rope_parameters": {
                "full_attention": {"rope_theta": 1000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 500.0, "rope_type": "default"},
            },
        }
    )
    assert args.global_layers == (True, False, True, False)
    assert (args.global_rope_theta, args.local_rope_theta) == (1000.0, 500.0)


@pytest.mark.parametrize("key", ["attention_bias", "mlp_bias", "norm_bias"])
def test_unsupported_configs_are_refused(key):
    with pytest.raises(ValueError, match=key):
        ModelArgs.from_config({**TINY, key: True})


def test_sanitize_keeps_only_the_prefixed_weights():
    weights = {"encoder.final_norm.weight": mx.ones(2), "scorer.0.weight": mx.ones(2)}
    assert list(Model.sanitize(weights, "encoder.")) == ["final_norm.weight"]


@pytest.mark.weights
def test_julia_encoder_matches_transformers(julia_path):
    """The real mmBERT-small encoder of Julia 1 on a few hundred tokens of text."""
    import json

    torch = pytest.importorskip("torch")
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer
    from transformers import AutoConfig, AutoModel

    config = json.loads((julia_path / "encoder" / "config.json").read_text())
    weights = Model.sanitize(load_file(str(julia_path / "model.safetensors")), "encoder.")
    model = Model(ModelArgs.from_config(config))
    model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
    reference = AutoModel.from_config(
        AutoConfig.from_pretrained(julia_path / "encoder"), attn_implementation="sdpa"
    ).eval()
    reference.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)

    tokenizer = Tokenizer.from_file(str(julia_path / "tokenizer" / "tokenizer.json"))
    text = " ".join(
        ["The invoice from March was charged twice, so the customer asks for a refund."] * 20
        + [
            "Die App stürzt ab, wenn ich die Einstellungen öffne.",
            "設定を開くたびにクラッシュします。",
        ]
    )
    ids = np.array([tokenizer.encode(text).ids])
    assert ids.shape[1] > 300
    with torch.inference_mode():
        expected = reference(input_ids=torch.tensor(ids)).last_hidden_state.numpy()
    out = run(model, ids)
    # Over 22 layers single values drift by up to about 1e-3, as between torch on
    # the CPU and on MPS; the mean and the direction of every token must hold.
    error = np.abs(out - expected)
    assert error.mean() < 1e-4
    assert error.max() < 1e-2
    cosine = (
        (out * expected).sum(-1) / np.linalg.norm(out, axis=-1) / np.linalg.norm(expected, axis=-1)
    )
    assert cosine.min() > 0.99999
