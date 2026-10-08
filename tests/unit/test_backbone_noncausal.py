"""The backbone in pplx's layout: no ``model.`` prefix, no lm_head, non-causal full attention."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_decision.backbones.qwen3_5.load import load_text_model
from mlx_decision.backbones.qwen3_5.vision import load_vision_model
from tiny_models import write_pplx

REFERENCE = Path(__file__).parents[1] / "parity" / "pplx" / "reference.json"


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


def test_logits_match_the_release_code(folder, noncausal):
    """Last hidden state times the readout, as the release's own code computes it.

    On the CPU: the GPU computes float32 matmuls in TF32, which moves these
    logits by up to 0.02; the CPU matches the reference to about 5e-6. The
    first text question of each request up to 1,024 tokens, for time (about
    40 ms each on the CPU).
    """
    readout = mx.load(str(folder / "readout.safetensors"))["weight"]
    results = json.loads(REFERENCE.read_text())["results"]
    worst = 0.0
    checked = 0
    for questions in results.values():
        expected = next(iter(questions.values()))
        if "image_grids" in expected or expected.get("tokens", 0) > 1024:
            continue
        with mx.stream(mx.cpu):
            hidden = noncausal(mx.array([expected["ids"]]))[0, -1]
            logits = np.array(readout[: expected["options"]] @ hidden)
        worst = max(worst, float(np.abs(logits - expected["logits"]).max()))
        checked += 1
    assert checked > 60
    assert worst < 5e-5
