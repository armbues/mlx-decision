"""The memory check before loading, on tiny models with the working set patched."""

import json
from pathlib import Path

import mlx.core as mx
import pytest
from typer.testing import CliRunner

import mlx_decision
import mlx_decision.memory as memory
from mlx_decision.cli import app
from mlx_decision.memory import ModelTooLargeError, check_fits, tensor_bytes, weight_bytes
from tiny_models import write_clef, write_marker


@pytest.fixture
def working_set(monkeypatch):
    """Set the working set the check compares against."""

    def set_to(size: int) -> None:
        monkeypatch.setattr(memory, "working_set", lambda: size)

    return set_to


def test_tensor_bytes_reads_the_header(tmp_path):
    file = tmp_path / "w.safetensors"
    tensors = {
        "a": mx.zeros((3, 4), dtype=mx.float32),
        "b": mx.zeros((5,), dtype=mx.bfloat16),
        "c": mx.zeros((2, 2), dtype=mx.uint32),
    }
    mx.save_safetensors(str(file), tensors)
    assert tensor_bytes(file) == 12 * 4 + 5 * 2 + 4 * 4
    # Floating-point tensors take the load dtype's size; others stay as stored.
    assert tensor_bytes(file, "float16") == 12 * 2 + 5 * 2 + 4 * 4


def test_clef_counts_its_weights_and_head(tmp_path):
    path = write_clef(tmp_path / "clef")
    files = sorted(path.glob("model*.safetensors")) + [path / "joint_head.safetensors"]
    assert weight_bytes(path) == sum(tensor_bytes(file) for file in files)


def test_laya_counts_its_weights_in_the_load_dtype(tmp_path):
    path = write_marker(tmp_path / "laya", "laya")
    stored = tensor_bytes(path / "model.safetensors")
    assert weight_bytes(path, {"dtype": "float32"}) == tensor_bytes(
        path / "model.safetensors", "float32"
    )
    assert weight_bytes(path) == tensor_bytes(path / "model.safetensors", "float16")
    assert weight_bytes(path, {"dtype": None}) == stored


def test_julia_counts_the_weights_file_its_config_names(tmp_path):
    path = write_marker(tmp_path / "julia", "julia")
    root = path / "config.json"
    config = json.loads(root.read_text()) if root.exists() else {}
    (path / "model.safetensors").rename(path / "weights.safetensors")
    root.write_text(json.dumps({**config, "weights_file": "weights.safetensors"}))
    assert weight_bytes(path) == tensor_bytes(path / "weights.safetensors", "float16")


def test_a_model_that_fits_passes(tmp_path, working_set):
    path = write_clef(tmp_path / "clef")
    required = round(weight_bytes(path) * 1.1)
    working_set(required)
    assert check_fits(path) == required


def test_a_model_that_does_not_fit_is_refused_with_sizes_and_a_hint(tmp_path, working_set):
    path = write_clef(tmp_path / "clef")
    working_set(round(weight_bytes(path) * 1.1) - 1)
    with pytest.raises(ModelTooLargeError) as error:
        check_fits(path, name="Cloudflare/clef")
    message = str(error.value)
    assert message.startswith("Cloudflare/clef needs about ")
    assert "plus 10%); this Mac's GPU working set is " in message
    assert "mlx-decision convert -m Cloudflare/clef -q" in message
    assert "--no-memory-check" in message
    assert error.value.required > error.value.available


@pytest.mark.parametrize(("bits", "hint"), [
    (8, "convert -m clef -q (8-bit, needs about"),
    (4, "convert -m clef -q --bits 4 (4-bit, needs about"),
    (None, "Even a 4-bit copy would need about"),
])  # fmt: skip
def test_the_hint_names_a_copy_that_fits(tmp_path, working_set, bits, hint):
    path = write_clef(tmp_path / "clef")
    weights, quantizable = weight_bytes(path), memory.quantizable_bytes(path)
    assert 0 < quantizable < weights  # the head stays as it is
    rest = weights - quantizable
    needs = {b: round((rest + memory.quantized_size(quantizable, b)) * 1.1) for b in (8, 4)}
    working_set({8: needs[8], 4: needs[8] - 1, None: needs[4] - 1}[bits])
    with pytest.raises(ModelTooLargeError) as error:
        check_fits(path)
    assert hint in str(error.value)
    assert "--no-memory-check" in str(error.value)


def test_a_budget_replaces_the_working_set(tmp_path, working_set):
    path = write_clef(tmp_path / "clef")
    working_set(10**12)
    with pytest.raises(ModelTooLargeError, match="the memory budget is"):
        check_fits(path, budget=1)


def test_no_convert_hint_for_a_converted_copy_or_a_family_without_converter(tmp_path, working_set):
    working_set(1)
    clef = write_clef(tmp_path / "clef")
    (clef / "mlx_decision.json").write_text(json.dumps({"family": "clef"}))
    laya = write_marker(tmp_path / "laya", "laya")
    for path in (clef, laya):
        with pytest.raises(ModelTooLargeError) as error:
            check_fits(path)
        assert "convert" not in str(error.value)


def test_load_checks_before_reading_the_weights(tmp_path, working_set, monkeypatch):
    path = write_clef(tmp_path / "clef")
    working_set(1)
    read = []
    monkeypatch.setattr(mx, "load", lambda *args, **kwargs: read.append(args))
    with pytest.raises(ModelTooLargeError):
        mlx_decision.load(path)
    assert read == []


def test_load_without_the_check(tmp_path, working_set):
    path = write_clef(tmp_path / "clef")
    working_set(1)
    assert mlx_decision.load(path, check_memory=False).name == "clef"


@pytest.mark.parametrize("command", [["run", "-s", "x", "--noul", "q"], ["benchmark"]])
def test_cli_refuses_and_can_skip_the_check(tmp_path: Path, working_set, command):
    path = write_clef(tmp_path / "clef")
    working_set(1)
    runner = CliRunner()
    refused = runner.invoke(app, [command[0], "-m", str(path), *command[1:]])
    assert refused.exit_code == 1
    assert "needs about" in refused.stderr
    if command[0] == "run":
        skipped = runner.invoke(app, ["run", "-m", str(path), "--no-memory-check", *command[1:]])
        assert skipped.exit_code == 0, skipped.output
