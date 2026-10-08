"""Converting a pplx release into an MLX folder, on a tiny random pplx."""

import io
import json

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_decision
from mlx_decision import memory
from mlx_decision.convert import convert
from mlx_decision.memory import ModelTooLargeError, check_fits, weight_bytes
from mlx_decision.registry import MARKER_FILE
from tiny_models import write_pplx

QUESTIONS = {
    "team": {"type": "choice", "criteria": {"billing": "refund", "technical": None}},
    "refund": {"type": "noul", "instructions": "refund"},
}


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    folder = write_pplx(tmp_path_factory.mktemp("pplx") / "tiny-pplx")
    (folder / "NOTICE").write_text("test notice\n")
    return folder


@pytest.fixture(scope="module")
def q8(release, tmp_path_factory):
    return convert(release, tmp_path_factory.mktemp("q8") / "tiny-pplx-q8", bits=8)


def png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (9, 99, 199)).save(buffer, format="PNG")
    return buffer.getvalue()


def quantized(backbone) -> set[str]:
    return {
        path
        for path, module in backbone.named_modules()
        if isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding)
    }


def test_unquantized_copy_answers_like_the_release(release, tmp_path):
    out = convert(release, tmp_path / "copy")
    body = {"state": "please refund", "questions": QUESTIONS, "images": [png()]}
    original = mlx_decision.load(release).decide_request(body)
    copy = mlx_decision.load(out).decide_request(body)
    assert copy.answers == original.answers
    assert copy.usage.input_tokens == original.usage.input_tokens
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker == {
        "family": "pplx",
        "format": 1,
        "source": release.name,
        "quantization": None,
        "vision": True,
    }


def test_folder_layout(release, q8):
    names = {path.name for path in q8.iterdir()}
    assert names == {
        "config.json",
        "model-00001-of-00001.safetensors",
        "model-vision.safetensors",
        "model.safetensors.index.json",
        "readout.safetensors",
        "decision_config.json",
        "tokenizer.json",
        "LICENSE",
        "NOTICE",
        MARKER_FILE,
    }
    for name in ("readout.safetensors", "decision_config.json", "tokenizer.json", "NOTICE"):
        assert (q8 / name).read_bytes() == (release / name).read_bytes()
    config = json.loads((q8 / "config.json").read_text())
    assert config["quantization"] == {"group_size": 64, "bits": 8, "mode": "affine"}


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_copy_keeps_readout_and_vision(release, q8, tmp_path, bits):
    out = q8 if bits == 8 else convert(release, tmp_path / "q4", bits=4)
    model = mlx_decision.load(out, vision=True)
    backend = model.backend
    assert model.info()["precision"] == f"{bits}-bit"
    layers = quantized(backend.backbone)
    assert "language_model.model.embed_tokens" in layers
    assert not any("lm_head" in path for path in layers)
    assert quantized(backend.vision) == set()
    readout = mx.load(str(release / "readout.safetensors"))["weight"]
    assert mx.array_equal(backend.readout, readout).item()
    body = {"state": "please refund", "questions": QUESTIONS, "images": [png()]}
    result = model.decide_request(body)
    assert sum(result.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert json.loads((out / MARKER_FILE).read_text())["quantization"]["bits"] == bits


def test_8bit_stays_close_to_the_release(release, q8):
    original = mlx_decision.load(release).decide("please refund", QUESTIONS)
    copy = mlx_decision.load(q8).decide("please refund", QUESTIONS)
    team = original.answers["team"].probabilities
    assert copy.answers["team"].probabilities == pytest.approx(team, abs=0.01)
    assert copy.answers["refund"].noul == pytest.approx(original.answers["refund"].noul, abs=0.01)


@pytest.mark.parametrize(("options", "message"), [
    ({"target_bits": 4.5}, "no mixed precision"),
    ({"bits": 4, "quantize_output_embeddings": False}, "no output embeddings"),
])  # fmt: skip
def test_refused_options(release, tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        convert(release, tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_refuses_a_quantized_source(q8, tmp_path):
    with pytest.raises(ValueError, match="already quantized"):
        convert(q8, tmp_path / "again", bits=4)


def test_the_fit_check_counts_readout_and_vision_in_full(release, working_set):
    weights, quantizable = weight_bytes(release), memory.quantizable_bytes(release)
    readout = (release / "readout.safetensors").stat().st_size
    assert 0 < quantizable < weights - readout  # the vision tower stays as well
    rest = weights - quantizable
    working_set(round((rest + memory.quantized_size(quantizable, 4)) * 1.1))
    with pytest.raises(ModelTooLargeError) as error:
        check_fits(release, name="perplexity-ai/pplx-decider-v1.1-27b")
    assert "convert -m perplexity-ai/pplx-decider-v1.1-27b -q --bits 4 (4-bit" in str(error.value)


def test_no_convert_hint_for_a_converted_copy(q8, working_set):
    working_set(1)
    with pytest.raises(ModelTooLargeError) as error:
        check_fits(q8)
    assert "convert" not in str(error.value)
