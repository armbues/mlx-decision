"""Converting a Clef release into an MLX folder, on a tiny random Clef."""

import json
from pathlib import Path

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
        "source": release.name,  # a local folder by name only, never its full path
        "quantization": None,
        "quantized_output_embeddings": False,
        "vision": False,
    }


def picture(color) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_vision_tower_is_kept_unquantized(tmp_path):
    release = write_clef(tmp_path / "tiny-vision", vision=True)
    body = {"state": "billing", "questions": QUESTIONS, "images": [picture((9, 99, 199))]}
    original = mlx_decision.load(release).decide_request(body)

    copy = mlx_decision.load(convert(release, tmp_path / "copy")).decide_request(body)
    assert copy.answers == original.answers

    out = convert(release, tmp_path / "q8", bits=8)
    assert {"model-vision.safetensors", "processor_config.json"} <= {p.name for p in out.iterdir()}
    assert json.loads((out / MARKER_FILE).read_text())["vision"] is True
    model = mlx_decision.load(out, vision=True)
    tower = {
        p for p, m in model.backend.vision.named_modules() if isinstance(m, nn.QuantizedLinear)
    }
    assert tower == set()
    assert quantized(model)  # the text model is quantized
    assert model.decide_request(body).usage.input_tokens == original.usage.input_tokens


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


@pytest.mark.parametrize(
    ("model", "bits", "target", "name"),
    [
        ("Cloudflare/clef-flash", 8, None, "clef-flash-q8"),
        ("models/clef-flash/", 4, None, "clef-flash-q4"),
        ("clef-flash", None, 5.0, "clef-flash-mq5"),
        ("clef-flash", None, 4.5, "clef-flash-mq4.5"),
        ("clef-flash", None, None, "clef-flash-mlx"),
    ],
)
def test_default_output_names(model, bits, target, name):
    from mlx_decision.convert import default_output

    assert default_output(model, bits, target) == Path(name)


def test_cli_default_output(release, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["convert", "-m", str(release), "-q"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tiny-clef-q8" / MARKER_FILE).exists()


def test_bits_imply_quantize_and_flags_are_checked(release, tmp_path):
    runner = CliRunner()
    out = tmp_path / "q4"
    result = runner.invoke(app, ["convert", "-m", str(release), "-o", str(out), "--bits", "4"])
    assert result.exit_code == 0, result.output
    assert json.loads((out / MARKER_FILE).read_text())["quantization"]["bits"] == 4
    for extra, message in [
        (["--group-size", "100"], "must be 32, 64 or 128"),
        (["--keep-output-embeddings"], "only applies when quantizing"),
        (["--bits", "4", "--target-bits", "4.5"], "either --bits or --target-bits"),
    ]:
        bad = runner.invoke(app, ["convert", "-m", str(release), "-o", str(tmp_path / "x"), *extra])
        assert bad.exit_code == 2, extra
        assert message in bad.output


def test_a_failed_conversion_leaves_nothing_behind(release, tmp_path, monkeypatch):
    from mlx_decision.models.clef import convert as clef_convert

    def broken(*args, **kwargs):
        raise OSError("disk full")

    # Fails after the weights are written, while copying the release's files.
    monkeypatch.setattr(clef_convert.shutil, "copy2", broken)
    with pytest.raises(OSError, match="disk full"):
        convert(release, tmp_path / "out")
    assert [p.name for p in tmp_path.iterdir()] == [release.name]


def test_no_quantizable_layer_is_an_error(release):
    from mlx_decision.backbones.qwen3_5.load import load_text_model, quantize_text_model

    with pytest.raises(ValueError, match="no layer can be quantized with group size 100"):
        quantize_text_model(load_text_model(release), bits=4, group_size=100)


def hf_release(folder: Path, shards: int = 3) -> Path:
    """The tiny Clef with its backbone in the Hugging Face layout, as Cloudflare ships it.

    Release names, conv kernels as [C, 1, K], norm weights stored minus one,
    an MTP weight (dropped on load), spread over ``shards`` files.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_decision.backbones.qwen3_5.load import load_text_model

    path = write_clef(folder)
    model = load_text_model(path)
    norms = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight")
    norms += (".q_norm.weight", ".k_norm.weight")
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith("conv1d.weight"):
            value = value.moveaxis(1, 2)
        elif name.endswith(norms):
            value = value - 1.0
        name = name.replace("language_model.model.", "model.language_model.")
        weights[name.replace("language_model.lm_head", "lm_head")] = value
    weights["mtp.fc.weight"] = mx.zeros((4, 4))
    for file in path.glob("model*.safetensors"):
        file.unlink()
    names = sorted(weights)
    size = -(-len(names) // shards)
    weight_map = {}
    for index in range(shards):
        file = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        part = {name: weights[name] for name in names[index * size : (index + 1) * size]}
        mx.save_safetensors(str(path / file), part, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(part, file))
    index_file = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (path / "model.safetensors.index.json").write_text(json.dumps(index_file))
    return path


@pytest.mark.parametrize("layout", ["release", "mlx"])
@pytest.mark.parametrize("bits", [None, 8, 4])
def test_streaming_matches_the_whole_model_path(tmp_path, monkeypatch, layout, bits):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import mlx_decision.backbones.qwen3_5.load as load

    source = hf_release(tmp_path / "src") if layout == "release" else write_clef(tmp_path / "src")
    whole = load.load_text_model(source)
    if bits is not None:
        load.quantize_text_model(whole, bits, 64)
    expected = dict(tree_flatten(whole.parameters()))

    # Small output shards, so the tiny model is written as several.
    monkeypatch.setattr(load, "SHARD_BYTES", 2**16)
    convert(source, tmp_path / "out", bits=bits)
    out = tmp_path / "out"
    files = sorted(out.glob("model-0*.safetensors"))
    assert len(files) > 1
    assert files[0].name == f"model-00001-of-{len(files):05d}.safetensors"
    assert not list(out.glob(".model-*"))
    written = {}
    for file in files:
        written.update(mx.load(str(file)))
    assert written.keys() == expected.keys()
    for name, value in expected.items():
        assert written[name].dtype == value.dtype, name
        assert mx.array_equal(written[name], value).item(), name
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(written)
    config = json.loads((source / "config.json").read_text())
    if bits is not None:
        config["quantization"] = {"group_size": 64, "bits": bits, "mode": "affine"}
    assert json.loads((out / "config.json").read_text()) == config
    assert index["metadata"]["total_size"] == sum(v.nbytes for v in written.values())
    mlx_decision.load(out).decide("refund please", {"q": {"type": "noul"}})


def test_streaming_reads_each_shard_once_when_it_is_handed_over(tmp_path, monkeypatch):
    import mlx.core as mx

    import mlx_decision.backbones.qwen3_5.load as load

    source = hf_release(tmp_path / "src", shards=4)
    events = []

    def shards():
        for file in load.source_shards(source):
            events.append(("handed", file.name))
            yield file

    real_load = mx.load

    def recording_load(file):
        events.append(("read", Path(file).name))
        return real_load(file)

    monkeypatch.setattr(load.mx, "load", recording_load)
    config = json.loads((source / "config.json").read_text())
    load.convert_text_weights(config, tmp_path / "out", shards(), bits=8)
    names = [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    assert events == [(kind, name) for name in names for kind in ("handed", "read")]
