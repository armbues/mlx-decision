"""Sensitivity measurement and bit allocation for mixed-precision quantization."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_decision.backbones.qwen3_5.load import quantizable_layers
from mlx_decision.mixed import (
    Unit,
    allocate,
    average_bits,
    base_bits,
    group_units,
    kl_divergence,
    measure_sensitivity,
)
from tiny_models import qwen3_5

IDS = mx.array([[1, 5, 9, 42, 7, 200, 3, 3, 18, 255]])


def test_group_units():
    layers = {
        "lm.model.embed_tokens": 5_000_000,
        "lm.model.layers.0.mlp.up_proj": 2_000_000,
        "lm.model.layers.0.mlp.down_proj": 2_000_000,
        "lm.model.layers.0.attn.q_proj": 1_500_000,
        "lm.model.layers.0.attn.gate": 1_000,
        "lm.model.layers.1.mlp.up_proj": 2_000_000,
        "lm.lm_head": 5_000_000,
    }
    units, small = group_units(layers)
    assert {unit.name: (unit.layers, unit.params) for unit in units} == {
        "lm.model.embed_tokens": (("lm.model.embed_tokens",), 5_000_000),
        "lm.model.layers.0.mlp": (
            ("lm.model.layers.0.mlp.up_proj", "lm.model.layers.0.mlp.down_proj"),
            4_000_000,
        ),
        "lm.model.layers.0.attn": (("lm.model.layers.0.attn.q_proj",), 1_500_000),
        "lm.model.layers.1.mlp": (("lm.model.layers.1.mlp.up_proj",), 2_000_000),
        "lm.lm_head": (("lm.lm_head",), 5_000_000),
    }
    assert small == ["lm.model.layers.0.attn.gate"]


def test_kl_divergence():
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert kl_divergence([1.0, 0.0], [0.5, 0.5]) == pytest.approx(0.6931, abs=1e-3)
    assert kl_divergence([0.9, 0.1], [0.6, 0.4]) > kl_divergence([0.9, 0.1], [0.8, 0.2])


@pytest.mark.parametrize(
    ("target", "base"), [(2, 2), (4.0, 3), (4.5, 4), (5.0, 4), (5.9, 5), (7.5, 6), (8, 6)]
)
def test_base_bits(target, base):
    assert base_bits(target) == base


def test_base_bits_needs_two():
    with pytest.raises(ValueError):
        base_bits(1.5)


def test_allocation_favours_sensitive_units_within_budget():
    units = [Unit("a", ("a",), 100), Unit("b", ("b",), 100), Unit("c", ("c",), 200)]
    sensitivity = {"a": 1.0, "b": 0.01, "c": 0.1}
    bits = allocate(units, sensitivity, base=4, target=4.5)
    assert bits["a"] > 4
    assert bits["b"] == 4
    assert average_bits(units, bits) <= 4.5
    # No budget above the base: nothing moves.
    assert allocate(units, sensitivity, base=4, target=4.0) == {"a": 4, "b": 4, "c": 4}
    # Plenty of budget: everything reaches 8 bits.
    assert set(allocate(units, sensitivity, base=4, target=8).values()) == {8}


def test_allocation_steps_through_allowed_bits():
    units = [Unit("a", ("a",), 100)]
    assert allocate(units, {"a": 1.0}, base=6, target=7.5) == {"a": 6}
    assert allocate(units, {"a": 1.0}, base=6, target=8) == {"a": 8}


def test_sensitivity_sweep_restores_the_model():
    model = qwen3_5()
    before = model(IDS)

    def evaluate():
        hidden = model(IDS)[0]
        return [mx.softmax(row[:8].astype(mx.float32)).tolist() for row in hidden]

    units, _ = group_units(quantizable_layers(model, group_size=32))
    seen = []
    sensitivity = measure_sensitivity(
        model,
        units,
        bits=2,
        group_size=32,
        evaluate=evaluate,
        progress=lambda i, n, unit, value: seen.append((i, n, unit.name)),
    )
    assert set(sensitivity) == {unit.name for unit in units}
    # Every unit moves the hidden states, except lm_head, which they never pass.
    assert sensitivity.pop("language_model.lm_head") == 0
    assert all(value > 0 for value in sensitivity.values())
    assert len(seen) == len(units) and seen[-1][0] == len(units)
    assert mx.array_equal(model(IDS), before)
    assert not any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules())
