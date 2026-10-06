"""The marker head of Laya and Julia against PyTorch's modules, and the release layouts."""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_decision.models.marker.head import MarkerHead
from mlx_decision.registry import detect_family

DIMS, LAYERS = 128, 2  # two attention heads of 64, as d // 64 in the head


@pytest.fixture(scope="module")
def reference():
    """The head as Laya and Julia build it, with their parameter names."""
    torch = pytest.importorskip("torch")
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                DIMS, DIMS // 64, 4 * DIMS, 0.0, batch_first=True, norm_first=True
            )
            self.head = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.type_emb = nn.Embedding(3, DIMS)
            self.scorer = nn.Sequential(
                nn.LayerNorm(DIMS), nn.Linear(DIMS, DIMS), nn.GELU(), nn.Linear(DIMS, 1)
            )
            self.act_head = nn.Sequential(nn.Linear(DIMS + 4, 256), nn.GELU(), nn.Linear(256, 2))
            self.register_buffer("temperature", torch.ones(3))

        def forward(self, hidden, attention_mask, markers, question_types):
            h = hidden + self.type_emb(question_types)[:, None, :]
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=~attention_mask.bool())
            index = markers[:, :, None].expand(-1, -1, DIMS)
            return self.scorer(torch.gather(h, 1, index)).squeeze(-1)

    torch.manual_seed(0)
    model = Head().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.3)
    return model


def mlx_head(reference) -> MarkerHead:
    head = MarkerHead(DIMS, LAYERS)
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    head.load_weights(list(MarkerHead.sanitize(weights).items()), strict=True)
    return head


def test_equal_to_the_reference_in_a_padded_batch(reference):
    import torch

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((3, 12, DIMS)).astype(np.float32)
    lengths = [12, 9, 5]
    mask = np.array([[1] * n + [0] * (12 - n) for n in lengths])
    markers = np.array([[1, 4, 7], [2, 3, 3], [1, 2, 1]])  # short rows repeat a marker
    types = np.array([0, 1, 2])
    with torch.inference_mode():
        expected = reference(*(torch.tensor(x) for x in (hidden, mask, markers, types))).numpy()
    with mx.stream(mx.cpu):
        out = mlx_head(reference)(
            mx.array(hidden), mx.array(mask), mx.array(types), mx.array(markers)
        )
        mx.eval(out)
    assert out.dtype == mx.float32
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-4, atol=1e-4)


def test_sanitize_drops_what_the_head_does_not_use(reference):
    weights = {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
    weights["encoder.final_norm.weight"] = mx.ones(DIMS)
    kept = MarkerHead.sanitize(weights)
    assert not any(k.startswith(("act_head", "encoder", "temperature")) for k in kept)
    assert "layers.0.self_attn.in_proj.weight" in kept
    assert "scorer_out.bias" in kept


@pytest.mark.parametrize(
    ("marker", "family"), [("rl_agent_config.json", "laya"), ("julia_config.json", "julia")]
)
def test_release_layouts_are_recognised(tmp_path, marker, family):
    (tmp_path / marker).write_text(json.dumps({}))
    assert detect_family(tmp_path).name == family


@pytest.mark.weights
def test_julia_loads_through_the_api(julia_path):
    import mlx_decision

    model = mlx_decision.load(julia_path)
    settings = model.backend.settings
    assert (settings.family, settings.max_length, settings.head_length) == ("julia", 8192, 512)
    assert settings.strict
    assert model.backend.capabilities.max_choice_options == 20
    assert model.backend.special.mask_text == "<mask>"


@pytest.mark.weights
def test_laya_loads_through_the_api(laya_path):
    import mlx_decision

    model = mlx_decision.load(laya_path, dtype="float32")
    settings = model.backend.settings
    assert (settings.family, settings.max_length, settings.head_length) == ("laya", 512, 192)
    assert settings.temperatures_by_options["choice:11+"] == pytest.approx(0.1006, abs=1e-4)
    assert model.backend.encoder.embeddings.norm.weight.dtype == mx.float32
    assert model.backend.special.mask_text == "[MASK]"
