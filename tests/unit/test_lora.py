"""The LoRA fit's training path on the tiny d3 and d1 LFM3 models: an all-zero
adapter answers as the base does, and the gradients through the served
passes (DeltaNet's chunked rule, checkpointing) match finite differences."""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from test_image_data import COLOURS, PET, png

import mlx_decision
from mlx_decision.adapters import LoRALinear
from mlx_decision.backend import Adaptable
from mlx_decision.training.checkpoint import checkpointed
from mlx_decision.training.lora import Item, Trainee, adapters, add_adapters, logits, loss
from tiny_models import write_d1lfm3, write_d3

WRITERS = {"d3": write_d3, "d1lfm3": write_d1lfm3}
QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "billing or technical",
        "criteria": {"billing": "refund", "technical": None},
    },
    "urgency": {"type": "score", "instructions": "now", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "refund"},
}
STATES = ["please refund my order now", "the app crashes", "a refund"]


@pytest.fixture(params=list(WRITERS))
def model(request, tmp_path):
    return mlx_decision.load(WRITERS[request.param](tmp_path / request.param))


@contextmanager
def on_cpu():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


def trainee_for(backend, passes, gold=0):
    """A trainee with the release's rows and zero biases for every question of
    ``passes``, and the items asking all of them."""
    rows, biases, items = [], [], []
    for passed in passes:
        golds = {}
        for question_id, features in passed.questions.items():
            golds[question_id] = (len(rows), gold)
            rows.append(features.rows)
            biases.append(mx.zeros(len(features.groups)))
        items.append(Item(passed, golds))
    return Trainee(backend.network(), rows, biases), items


def answers(backend, trainee, batch):
    padded, valid, _ = logits(backend, trainee, batch)
    probabilities = mx.softmax(mx.where(valid, padded, -1e9), axis=-1)
    out, row = [], 0
    for item in batch:
        found = {}
        for question_id, features in item.passed.questions.items():
            count = len(features.option_ids)
            values = probabilities[row, :count].tolist()
            found[question_id] = dict(zip(features.option_ids, values, strict=True))
            row += 1
        out.append(found)
    return out


def close(found: dict, expected: dict, tolerance: float = 2e-5) -> None:
    assert found.keys() == expected.keys()
    for question_id, values in expected.items():
        assert found[question_id].keys() == values.keys()
        for option, value in values.items():
            assert found[question_id][option] == pytest.approx(value, abs=tolerance)


def test_the_families_are_adaptable(model):
    assert isinstance(model.backend, Adaptable)
    targets = model.backend.adapter_targets()
    layers = [m for _, m in model.backend.network().named_modules()
              if isinstance(m, model.backend.decoder_layer)]  # fmt: skip
    assert len(targets) == len(layers) > 0


def zero_adapter_answers(model):
    backend = model.backend
    requests = [model.check({"state": s, "questions": QUESTIONS}) for s in STATES]
    served = [backend.score(r).probabilities for r in requests]
    count = add_adapters(backend, rank=4, seed=1)
    assert count > 0
    assert all(isinstance(layer, LoRALinear) for _, m, names in backend.adapter_targets()
               for layer in (getattr(m, n) for n in names))  # fmt: skip
    passes = [backend.training_passes(r) for r in requests]
    trainee, items = trainee_for(backend, [p for found in passes for p in found])
    trainee.train()
    together = answers(backend, trainee, items)
    alone = [answers(backend, trainee, [item])[0] for item in items]
    by_request, start = [], 0
    for found in passes:
        merged, merged_alone = {}, {}
        for i in range(start, start + len(found)):
            merged |= together[i]
            merged_alone |= alone[i]
        start += len(found)
        by_request.append((merged, merged_alone))
    return served, by_request


def test_an_all_zero_adapter_answers_as_the_base_on_text(model):
    """Each pass alone answers exactly as served; all passes of all requests in
    one right-padded batch within float32 rounding."""
    served, by_request = zero_adapter_answers(model)
    for expected, (_, alone) in zip(served, by_request, strict=True):
        assert alone == expected


def test_a_padded_batch_answers_as_each_pass_alone(model):
    # On the CPU: the GPU's float32 matmuls differ by shape at about 1e-4.
    with on_cpu():
        served, by_request = zero_adapter_answers(model)
    for expected, (together, _) in zip(served, by_request, strict=True):
        close(together, expected, 1e-6)


def test_an_all_zero_adapter_answers_as_the_base_on_images(model, tmp_path):
    backend = model.backend
    images = []
    for name, colour in COLOURS.items():
        (tmp_path / f"{name}.png").write_bytes(png(colour, (40, 24)))
        images.append(str(tmp_path / f"{name}.png"))
    request = model.check(
        {"state": "a picture", "images": images, "questions": {"pet": PET, **QUESTIONS}}
    )
    served = backend.score(request).probabilities
    add_adapters(backend, rank=4)
    passes = backend.training_passes(request)
    assert all(p.images for p in passes)
    trainee, items = trainee_for(backend, passes)
    trainee.train()
    merged = {}
    for item in items:
        merged |= answers(backend, trainee, [item])[0]
    assert merged == served
    with pytest.raises(ValueError, match="alone"):
        backend.pass_hidden([p for p in passes] * 2)


def test_the_adapters_change_the_network_once_trained(model):
    backend = model.backend
    request = model.check({"state": STATES[0], "questions": QUESTIONS})
    passes = backend.training_passes(request)
    before = [h.tolist() for h in backend.pass_hidden(passes)]
    add_adapters(backend, rank=4)
    for _, module, names in backend.adapter_targets():
        for name in names:
            layer = getattr(module, name)
            layer.lora_b = mx.random.normal(layer.lora_b.shape) * 0.5
    after = [h.tolist() for h in backend.pass_hidden(passes)]
    assert not np.allclose(np.array(before[0]), np.array(after[0]), atol=1e-3)
    tensors = adapters(backend)
    assert len(tensors) == 2 * sum(len(names) for _, _, names in backend.adapter_targets())
    assert all(name.startswith("layers.") for name in tensors)


def perturbed_trainee(model, seed=0):
    backend = model.backend
    requests = [model.check({"state": s, "questions": QUESTIONS}) for s in STATES[:2]]
    add_adapters(backend, rank=2, seed=seed)
    rng = np.random.default_rng(seed)
    for _, module, names in backend.adapter_targets():
        for name in names:
            layer = getattr(module, name)
            layer.lora_b = mx.array(rng.normal(0, 0.1, layer.lora_b.shape).astype(np.float32))
    passes = [p for r in requests for p in backend.training_passes(r)]
    trainee, items = trainee_for(backend, passes, gold=1)
    trainee.train()
    return backend, trainee, items


def test_gradients_match_finite_differences(model):
    """Directional derivatives of every trainable tensor kind against central
    differences, on the CPU (float32 matmuls on the GPU are too coarse)."""
    with on_cpu():
        backend, trainee, items = perturbed_trainee(model)
        loss_and_grad = nn.value_and_grad(trainee, lambda: loss(backend, trainee, items))
        _, grads = loss_and_grad()
        grads = dict(tree_flatten(grads))
        params = dict(tree_flatten(trainee.trainable_parameters()))
        assert grads.keys() == params.keys()
        rng = np.random.default_rng(1)
        picked = [n for n in params if n.endswith("lora_a")][:2]
        picked += [n for n in params if n.endswith("lora_b")][-2:]
        picked += ["rows.0", "bias.2"]
        for name in picked:
            direction = rng.normal(size=params[name].shape).astype(np.float32)
            direction = mx.array(direction / np.linalg.norm(direction))
            analytic = (grads[name] * direction).sum().item()
            # Small: the update is scaled by 8 at rank 2, so the loss bends fast.
            step = 3e-3
            values = []
            for sign in (1, -1):
                moved = {**params, name: params[name] + sign * step * direction}
                trainee.update(tree_unflatten(list(moved.items())))
                values.append(loss(backend, trainee, items).item())
            trainee.update(tree_unflatten(list(params.items())))
            numeric = (values[0] - values[1]) / (2 * step)
            assert analytic == pytest.approx(numeric, rel=0.02, abs=1e-4), name


def test_checkpointing_leaves_the_gradients_alone(model):
    with on_cpu():
        backend, trainee, items = perturbed_trainee(model, seed=3)
        loss_and_grad = nn.value_and_grad(trainee, lambda: loss(backend, trainee, items))
        value, plain = loss_and_grad()
        with checkpointed(backend.decoder_layer):
            again, kept = loss_and_grad()
        assert again.item() == pytest.approx(value.item(), rel=1e-6)
        kept = dict(tree_flatten(kept))
        for name, grad in tree_flatten(plain):
            assert mx.allclose(kept[name], grad, rtol=1e-4, atol=1e-7).item(), name


# Adapters in a fitted folder


def adapted_pair(quantized: bool):
    mx.random.seed(0)
    module = nn.Module()
    module.proj = nn.Linear(64, 32, bias=False)
    if quantized:
        module.proj = nn.QuantizedLinear.from_linear(module.proj, group_size=32, bits=8)
    targets = [("layers.0.mlp", module, ("proj",))]
    return module, targets


def write_adapters(path, rank=4, scale=4.0, extra=None):
    tensors = {
        "lora.layers.0.mlp.proj.lora_a": mx.random.normal((64, rank)) * 0.1,
        "lora.layers.0.mlp.proj.lora_b": mx.random.normal((rank, 32)) * 0.1,
        "lora.scale": mx.array(scale),
        "readout.0.rows": mx.zeros((2, 8)),
        **(extra or {}),
    }
    mx.save_safetensors(str(path), tensors)
    return tensors


def test_fused_adapters_answer_as_unfused_ones(tmp_path):
    # On the CPU: the GPU's float32 matmuls are coarser than the difference.
    with on_cpu():
        from mlx_decision.adapters import load_adapters

        tensors = write_adapters(tmp_path / "fit.safetensors")
        x = mx.random.normal((3, 64))
        module, targets = adapted_pair(quantized=False)
        unfused = LoRALinear.from_base(module.proj, 4, 4.0)
        unfused.lora_a = tensors["lora.layers.0.mlp.proj.lora_a"]
        unfused.lora_b = tensors["lora.layers.0.mlp.proj.lora_b"]
        expected = unfused(x)
        assert load_adapters(targets, tmp_path / "fit.safetensors") == 1
        assert type(module.proj) is nn.Linear  # fused
        assert mx.allclose(module.proj(x), expected, atol=1e-5).item()


def test_a_quantized_base_keeps_its_weights_and_applies_the_adapters_unfused(tmp_path):
    # On the CPU: the GPU's float32 matmuls are coarser than the difference.
    with on_cpu():
        from mlx_decision.adapters import load_adapters

        write_adapters(tmp_path / "fit.safetensors")
        module, targets = adapted_pair(quantized=True)
        stored = module.proj.weight
        load_adapters(targets, tmp_path / "fit.safetensors")
        assert isinstance(module.proj, LoRALinear)
        assert isinstance(module.proj.linear, nn.QuantizedLinear)
        assert mx.array_equal(module.proj.linear.weight, stored).item()
        x = mx.random.normal((3, 64))
        fused = module.proj.fuse(dequantize=True)
        assert mx.allclose(module.proj(x), fused(x), atol=1e-4).item()


def test_adapters_that_do_not_fit_the_model_are_refused(tmp_path):
    from mlx_decision.adapters import load_adapters

    write_adapters(tmp_path / "extra.safetensors", extra={
        "lora.layers.9.mlp.proj.lora_a": mx.zeros((64, 4)),
        "lora.layers.9.mlp.proj.lora_b": mx.zeros((4, 32)),
    })  # fmt: skip
    _, targets = adapted_pair(quantized=False)
    with pytest.raises(ValueError, match="does not have"):
        load_adapters(targets, tmp_path / "extra.safetensors")
    mx.save_safetensors(str(tmp_path / "short.safetensors"), {"lora.scale": mx.array(2.0)})
    _, targets = adapted_pair(quantized=False)
    with pytest.raises(ValueError, match="no adapter for layers.0.mlp.proj"):
        load_adapters(targets, tmp_path / "short.safetensors")
    write_adapters(tmp_path / "wide.safetensors", extra={
        "lora.layers.0.mlp.proj.lora_a": mx.zeros((32, 4)),
    })  # fmt: skip
    _, targets = adapted_pair(quantized=False)
    with pytest.raises(ValueError, match="other shapes"):
        load_adapters(targets, tmp_path / "wide.safetensors")
