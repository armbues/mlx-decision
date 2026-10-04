"""Saving, quantizing and reloading the Qwen3.5 backbone, on a tiny random model."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_decision.backbones.qwen3_5.load import (
    load_text_model,
    quantize_text_model,
    save_text_model,
)
from tiny_models import QWEN3_5 as TINY
from tiny_models import qwen3_5

IDS = mx.array([[1, 5, 9, 42, 7, 200, 3, 3, 18, 255]])


@pytest.fixture
def model():
    return qwen3_5()


def quantized_paths(model):
    return {
        path
        for path, module in model.named_modules()
        if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }


def test_save_and_load_round_trip(model, tmp_path):
    save_text_model(model, tmp_path, TINY)
    assert sorted(p.name for p in tmp_path.glob("model*.safetensors")) == [
        "model-00001-of-00001.safetensors"
    ]
    loaded = load_text_model(tmp_path)
    assert mx.array_equal(loaded(IDS), model(IDS))


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_model_reloads_identically(model, tmp_path, bits):
    reference = model(IDS)
    quantization = quantize_text_model(model, bits=bits, group_size=32)
    assert quantization == {"group_size": 32, "bits": bits, "mode": "affine"}
    assert "language_model.lm_head" in quantized_paths(model)
    assert "language_model.model.embed_tokens" in quantized_paths(model)

    save_text_model(model, tmp_path, {**TINY, "quantization": quantization})
    loaded = load_text_model(tmp_path)
    assert quantized_paths(loaded) == quantized_paths(model)
    assert mx.array_equal(loaded(IDS), model(IDS))
    assert not mx.array_equal(loaded(IDS), reference)


def test_more_bits_stay_closer():
    errors = {}
    for bits in (4, 8):
        model = qwen3_5()
        reference = model(IDS)
        quantize_text_model(model, bits=bits, group_size=32)
        errors[bits] = mx.abs(model(IDS) - reference).mean().item()
    assert errors[8] < 0.02
    assert errors[8] < errors[4] / 4


def test_output_embeddings_can_keep_their_precision(model, tmp_path):
    rows = model.language_model.output_embedding_rows(IDS[0])
    quantize_text_model(model, bits=4, group_size=32, output_embeddings=False)
    assert "language_model.lm_head" not in quantized_paths(model)
    assert "language_model.model.embed_tokens" in quantized_paths(model)
    assert mx.array_equal(model.language_model.output_embedding_rows(IDS[0]), rows)

    save_text_model(model, tmp_path, {**TINY, "quantization": {"group_size": 32, "bits": 4}})
    assert "language_model.lm_head" not in quantized_paths(load_text_model(tmp_path))


def test_quantized_output_embedding_rows_are_dequantized(model):
    rows = model.language_model.output_embedding_rows(IDS[0])
    quantize_text_model(model, bits=8, group_size=32)
    quantized = model.language_model.output_embedding_rows(IDS[0])
    assert quantized.shape == rows.shape
    assert mx.abs(quantized - rows).max().item() < 0.01


def test_large_models_are_sharded(model, tmp_path, monkeypatch):
    from mlx_decision.backbones.qwen3_5 import load

    monkeypatch.setattr(load, "SHARD_BYTES", 50_000)
    save_text_model(model, tmp_path, TINY)
    assert len(list(tmp_path.glob("model-*.safetensors"))) > 1
    assert mx.array_equal(load_text_model(tmp_path)(IDS), model(IDS))


def test_per_layer_bits_round_trip(model, tmp_path):
    from mlx_decision.backbones.qwen3_5.load import quantizable_layers

    layers = quantizable_layers(model, group_size=32)
    down = "language_model.model.layers.0.mlp.down_proj"
    up = "language_model.model.layers.1.mlp.up_proj"
    assert {down, up, "language_model.lm_head"} <= set(layers)
    quantization = quantize_text_model(
        model, bits=4, group_size=32, layer_bits={down: 8, up: None, "language_model.lm_head": 4}
    )
    assert quantization == {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
        down: {"group_size": 32, "bits": 8, "mode": "affine"},
    }
    modules = dict(model.named_modules())
    assert modules[down].bits == 8
    assert isinstance(modules[up], nn.Linear) and not isinstance(modules[up], nn.QuantizedLinear)
    assert modules["language_model.lm_head"].bits == 4

    save_text_model(model, tmp_path, {**TINY, "quantization": quantization})
    loaded = load_text_model(tmp_path)
    loaded_modules = dict(loaded.named_modules())
    assert loaded_modules[down].bits == 8
    assert up not in quantized_paths(loaded)
    assert mx.array_equal(loaded(IDS), model(IDS))
