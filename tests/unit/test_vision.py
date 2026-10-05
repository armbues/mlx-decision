"""The vision tower against transformers' implementation, on a tiny random model."""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_decision.backbones.qwen3_5.load import has_vision_weights
from mlx_decision.backbones.qwen3_5.vision import VisionArgs, VisionModel, load_vision_model

TINY = {
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_heads": 4,
    "in_channels": 3,
    "patch_size": 4,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 24,
    "num_position_embeddings": 36,
    "hidden_act": "gelu_pytorch_tanh",
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
}
GRIDS = [(1, 4, 6), (1, 2, 2), (1, 8, 4)]


def patches(grids, seed=0):
    rows = sum(t * h * w for t, h, w in grids)
    return np.random.default_rng(seed).standard_normal((rows, 3 * 2 * 4 * 4)).astype(np.float32)


@pytest.fixture(scope="module")
def reference():
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    torch.manual_seed(0)
    model = modeling.Qwen3_5VisionModel._from_config(
        Qwen3_5VisionConfig(**TINY), dtype=torch.float32
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.5)
    return model


def weights_of(reference) -> dict[str, mx.array]:
    return {f"model.visual.{k}": mx.array(v.numpy()) for k, v in reference.state_dict().items()}


def test_equal_to_the_reference(reference):
    import torch

    model = VisionModel(VisionArgs.from_config({"vision_config": TINY}))
    model.load_weights(list(model.sanitize(weights_of(reference)).items()), strict=True)
    pixels = patches(GRIDS)
    with torch.inference_mode():
        expected = reference(torch.tensor(pixels), grid_thw=torch.tensor(GRIDS)).pooler_output
    # MLX's float32 matmuls on the GPU are less exact; the CPU gives a fair comparison.
    with mx.stream(mx.cpu):
        out = model(mx.array(pixels), GRIDS)
        mx.eval(out)
    assert out.shape == (sum(t * h * w for t, h, w in GRIDS) // 4, 24)
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-4, atol=1e-4)


def test_images_do_not_attend_to_each_other(reference):
    model = VisionModel(VisionArgs.from_config({"vision_config": TINY}))
    model.load_weights(list(model.sanitize(weights_of(reference)).items()), strict=True)
    pixels = patches(GRIDS)
    with mx.stream(mx.cpu):
        together = model(mx.array(pixels), GRIDS)
        alone = model(mx.array(pixels[:24]), GRIDS[:1])
        mx.eval(together, alone)
    np.testing.assert_allclose(np.array(together[:6]), np.array(alone), rtol=1e-5, atol=1e-5)


def test_loads_from_a_release_folder(reference, tmp_path):
    weights = weights_of(reference)
    mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), weights)
    index = {"weight_map": dict.fromkeys(weights, "model-00001-of-00001.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": TINY}))
    assert has_vision_weights(tmp_path)
    model = load_vision_model(tmp_path)
    assert model.patch_embed.proj.weight.shape == (32, 96)
