"""Converting a Clef release into an MLX folder, on a tiny random Clef."""

import json

import mlx.nn as nn
import pytest
from typer.testing import CliRunner

import mlx_decision
from mlx_decision.cli import app
from mlx_decision.convert import convert
from mlx_decision.registry import MARKER_FILE
from tiny_models import write_clef

QUESTIONS = {
    "team": {"type": "choice", "criteria": {"billing": None, "technical": None, "sales": None}},
    "urgent": {"type": "noul", "instructions": "yes or no"},
    "level": {"type": "score", "criteria": ["no", "yes", "billing"]},
}


@pytest.fixture
def release(tmp_path):
    return write_clef(tmp_path / "tiny-clef")


def quantized(model):
    return {
        path
        for path, module in model.backend.backbone.named_modules()
        if isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding)
    }


def test_unquantized_copy_answers_like_the_release(release, tmp_path):
    out = convert(release, tmp_path / "copy")
    original = mlx_decision.load(release).decide("billing yes", QUESTIONS)
    copy = mlx_decision.load(out).decide("billing yes", QUESTIONS)
    assert copy.answers == original.answers
    assert copy.model == "copy"
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker == {
        "family": "clef",
        "format": 1,
        "source": str(release),
        "quantization": None,
        "quantized_output_embeddings": False,
    }


def test_folder_layout(release, tmp_path):
    out = convert(release, tmp_path / "copy", bits=8)
    names = {path.name for path in out.iterdir()}
    assert names == {
        "config.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "joint_head.safetensors",
        "joint_head_config.json",
        "tokenizer.json",
        "LICENSE",
        MARKER_FILE,
    }
    for name in ("joint_head.safetensors", "joint_head_config.json", "tokenizer.json"):
        assert (out / name).read_bytes() == (release / name).read_bytes()
    config = json.loads((out / "config.json").read_text())
    assert config["quantization"] == {"group_size": 64, "bits": 8, "mode": "affine"}


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_copy_loads_and_answers(release, tmp_path, bits):
    out = convert(release, tmp_path / "q", bits=bits)
    model = mlx_decision.load(out)
    assert "language_model.lm_head" in quantized(model)
    result = model.decide("billing yes", QUESTIONS)
    assert sum(result.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    assert json.loads((out / MARKER_FILE).read_text())["quantization"]["bits"] == bits


def test_output_embeddings_can_stay_unquantized(release, tmp_path):
    out = convert(release, tmp_path / "q", bits=4, quantize_output_embeddings=False)
    model = mlx_decision.load(out)
    assert "language_model.lm_head" not in quantized(model)
    assert "language_model.model.embed_tokens" in quantized(model)
    assert json.loads((out / MARKER_FILE).read_text())["quantized_output_embeddings"] is False


def test_refuses_a_non_empty_output(release, tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        convert(release, tmp_path / "busy")


def test_refuses_a_quantized_source(release, tmp_path):
    out = convert(release, tmp_path / "q", bits=8)
    with pytest.raises(ValueError, match="already quantized"):
        convert(out, tmp_path / "again", bits=4)


def test_family_without_converter(fake_model_path, tmp_path):
    with pytest.raises(ValueError, match="cannot be converted"):
        convert(fake_model_path, tmp_path / "out")


def test_cli(release, tmp_path):
    runner = CliRunner()
    out = tmp_path / "cli"
    args = ["convert", "-m", str(release), "-o", str(out), "-q", "--bits", "8"]
    result = runner.invoke(app, [*args, "--keep-output-embeddings"])
    assert result.exit_code == 0, result.output
    assert f"wrote {out}" in result.stdout
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["quantization"]["bits"] == 8
    assert marker["quantized_output_embeddings"] is False

    again = runner.invoke(app, args)
    assert again.exit_code == 1
    assert "not empty" in again.stderr
    bad_bits = runner.invoke(
        app, ["convert", "-m", str(release), "-o", str(tmp_path / "x"), "-q", "--bits", "7"]
    )
    assert bad_bits.exit_code == 2


@pytest.fixture
def short_calibration(monkeypatch):
    """Two short requests instead of the built-in set, to keep the sweep quick."""
    from mlx_decision.models.clef import convert as clef_convert

    requests = [
        {"state": "billing yes", "questions": QUESTIONS},
        {"state": {"a": ["technical", "no"]}, "questions": QUESTIONS},
    ]
    monkeypatch.setattr(clef_convert, "calibration_requests", lambda: requests)


@pytest.mark.usefixtures("short_calibration")
def test_mixed_precision(release, tmp_path):
    from mlx_decision.models.clef.convert import SENSITIVITY_FILE

    seen = []
    out = convert(
        release,
        tmp_path / "mixed",
        target_bits=4.5,
        group_size=32,
        progress=lambda done, total, unit, value: seen.append(done),
    )
    report = json.loads((out / SENSITIVITY_FILE).read_text())
    assert report["base_bits"] == 4
    assert report["average_bits"] <= 4.5
    assert len(report["units"]) == len(seen)
    assert {unit["bits"] for unit in report["units"]} <= {4, 5, 6, 8}
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["mixed"]["target_bits"] == 4.5
    assert marker["quantization"] == {"group_size": 32, "bits": 4, "mode": "affine"}

    model = mlx_decision.load(out)
    modules = dict(model.backend.backbone.named_modules())
    for unit in report["units"]:
        layers = [
            m for p, m in modules.items() if p == unit["name"] or p.startswith(unit["name"] + ".")
        ]
        assert {layer.bits for layer in layers if hasattr(layer, "bits")} == {unit["bits"]}
    result = model.decide("billing yes", QUESTIONS)
    assert sum(result.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-5)


@pytest.mark.usefixtures("short_calibration")
def test_mixed_precision_cli(release, tmp_path):
    out = tmp_path / "cli-mixed"
    args = [
        "convert",
        "-m",
        str(release),
        "-o",
        str(out),
        "--target-bits",
        "5",
        "--group-size",
        "32",
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "sensitivity 1/" in result.stderr
    assert json.loads((out / MARKER_FILE).read_text())["mixed"]["target_bits"] == 5
