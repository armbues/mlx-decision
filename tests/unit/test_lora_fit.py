"""The ``lora`` kind end to end on the tiny d3 and d1 LFM3 models: fits on text
and images, the fitted folder served as trained, soft labels, a cancelled run,
a fitted base's readout as the start, and the refusals."""

import json
from contextlib import contextmanager

import mlx.core as mx
import pytest
from test_image_data import PET, image_folder
from test_readout_fit import labelled
from test_soft_labels import fits_both_ways

import mlx_decision
from mlx_decision.fitted import FIT_FILE
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import Cancelled, evaluate, fit
from tiny_models import write_d1lfm3, write_d3, write_marker

WRITERS = {"d3": write_d3, "d1lfm3": write_d1lfm3}


@pytest.fixture(params=list(WRITERS))
def base(request, tmp_path):
    return WRITERS[request.param](tmp_path / f"tiny-{request.param}")


@contextmanager
def on_cpu():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


def test_a_lora_fit_is_served_as_it_was_trained(base, tmp_path):
    data, validation = labelled(40), labelled(12)
    events = []
    # On the CPU: the GPU's float32 matmuls round fused and unfused weights apart.
    with on_cpu():
        result = fit(
            base, data, output=tmp_path / "fitted", method="lora", validation=validation,
            on_event=events.append,
        )  # fmt: skip
        scored = evaluate(tmp_path / "fitted", validation).models[0].questions
    assert result.kinds == ["lora", "calibrate"]
    training = result.report["training"]
    assert training["rank"] == 8 and training["scale"] == 2.0
    assert training["learning_rate"] == 1e-4 and training["targets"] == "mlp"
    assert len(training["epochs"]) <= 2 and 1 <= training["best_epoch"] <= 2
    assert training["trainable_parameters"] > training["adapter_parameters"] > 0
    assert training["seconds_per_example"] > 0
    assert result.report["options"]["rank"] == 8
    assert result.report["calibration"]["questions"] == {"team": "temperature",
                                                         "urgent": "temperature"}  # fmt: skip
    assert any(e.stage == "train" and e.total for e in events)

    marker = json.loads((tmp_path / "fitted" / MARKER_FILE).read_text())
    assert marker["fit"]["kinds"] == ["lora", "calibrate"]
    assert marker["fit"]["lora"] == {"rank": 8, "scale": 2.0, "targets": "mlp"}
    tensors = mx.load(str(tmp_path / "fitted" / FIT_FILE))
    assert {name.split(".")[0] for name in tensors} == {"lora", "readout", "calibration"}
    assert tensors["lora.scale"].item() == 2.0

    # The served folder (adapters fused) answers the report's questions as the
    # run scored them.
    for question_id, entry in result.metrics.items():
        after = entry["after"]
        assert scored[question_id]["accuracy"] == after["accuracy"]
        assert scored[question_id]["log_loss"] == pytest.approx(after["log_loss"], abs=2e-4)


def test_a_lora_fit_changes_the_network(base, tmp_path):
    fit(base, labelled(40), output=tmp_path / "fitted", method="lora", epochs=1,
        calibration="none")  # fmt: skip
    request = {"state": "please refund", "questions": {"team": labelled(1)[0].request[
        "questions"]["team"]}}  # fmt: skip
    plain = mlx_decision.load(base).backend
    fitted = mlx_decision.load(tmp_path / "fitted").backend
    # The base network's hidden states and the fitted one's differ.
    checked = mlx_decision.load(base).check(request)
    before = plain.features(checked)["team"].hidden
    after = fitted.base.features(checked)["team"].hidden
    assert not mx.allclose(before, after, atol=1e-6).item()


def test_lora_fits_images(base, tmp_path):
    folder = image_folder(tmp_path / "pets", 12)
    result = fit(base, folder, output=tmp_path / "fitted", method="lora", epochs=1)
    assert result.kinds == ["lora", "calibrate"]
    assert result.report["training"]["examples"] > 0
    model = mlx_decision.load(tmp_path / "fitted")
    red, blue = (
        model.decide("a picture", {"pet": PET}, images=[str(folder / "images" / name)])
        .answers["pet"]
        .probabilities
        for name in ("0.png", "1.png")
    )
    assert red != blue


def test_lora_fits_soft_labels(base, tmp_path):
    result = fits_both_ways(tmp_path, base, labelled(40), method="lora", epochs=1)
    assert result.kinds == ["lora", "calibrate"]


def test_a_cancelled_lora_fit_writes_nothing(base, tmp_path):
    output = tmp_path / "fitted"
    seen = []

    def on_event(event):
        seen.append(event)

    def cancel():
        return any(e.stage == "train" and e.step for e in seen)

    with pytest.raises(Cancelled):
        fit(base, labelled(40), output=output, method="lora", on_event=on_event, cancel=cancel)
    assert not output.exists()
    assert not any(path.name.startswith(".fitted") for path in tmp_path.iterdir())


def test_a_lora_fit_starts_from_a_fitted_base_readout(base, tmp_path, monkeypatch):
    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 100)
    data = labelled(40)
    fit(base, data, output=tmp_path / "readout", method="readout")
    readout = mx.load(str(tmp_path / "readout" / FIT_FILE))
    # A learning rate too small to move anything: the rows stay the readout fit's.
    fit(tmp_path / "readout", data, output=tmp_path / "lora", method="lora", epochs=1,
        learning_rate=1e-12)  # fmt: skip
    marker = json.loads((tmp_path / "lora" / MARKER_FILE).read_text())
    assert marker["base"] == json.loads((tmp_path / "readout" / MARKER_FILE).read_text())["base"]
    fitted = mx.load(str(tmp_path / "lora" / FIT_FILE))
    first = json.loads((tmp_path / "readout" / MARKER_FILE).read_text())["fit"]["questions"]
    second = marker["fit"]["questions"]
    for i, question in enumerate(first):
        if f"readout.{i}.rows" not in readout:
            continue
        j = second.index(question)
        assert mx.allclose(fitted[f"readout.{j}.rows"], readout[f"readout.{i}.rows"], atol=1e-6)
        assert mx.allclose(fitted[f"readout.{j}.bias"], readout[f"readout.{i}.bias"], atol=1e-6)


def test_lora_settings_are_refused_where_they_do_not_apply(tmp_path):
    base = write_d3(tmp_path / "tiny-d3")
    with pytest.raises(ValueError, match="apply to lora"):
        fit(base, labelled(20), output=tmp_path / "x", method="readout", rank=4)
    with pytest.raises(ValueError, match="rank must be"):
        fit(base, labelled(20), output=tmp_path / "x", method="lora", rank=0)
    laya = write_marker(tmp_path / "tiny-laya", "laya")
    with pytest.raises(ValueError, match="cannot be fitted with lora"):
        fit(laya, labelled(20), output=tmp_path / "x", method="lora")


def test_a_lora_fit_on_a_quantized_copy_is_refused(tmp_path):
    from mlx_decision.convert import convert

    base = write_d3(tmp_path / "tiny-d3")
    copy = tmp_path / "tiny-d3-8bit"
    convert(base, copy, bits=8)
    with pytest.raises(ValueError, match="quantized copy"):
        fit(copy, labelled(20), output=tmp_path / "x", method="lora")
    assert not (tmp_path / "x").exists()


def test_the_summary_names_auto_s_pick_and_the_lora_path():
    from mlx_decision.training.console import method_lines

    assert method_lines("readout", "auto", True) == [
        "  auto picked readout: trains the answer rows on the model's frozen features, seconds",
        "  --method lora adapts the network itself: minutes",
    ]
    assert method_lines("lora", "lora", True) == ["  lora: adapts the network itself: minutes"]
    assert method_lines("finetune", "auto", False) == [
        "  auto picked finetune: trains the model's own weights"
    ]
    assert method_lines("calibrate", "calibrate", True)[-1].startswith("  --method lora")


def test_training_fit_says_which_method_ran(tmp_path):
    from typer.testing import CliRunner

    from mlx_decision.cli import app
    from mlx_decision.data.format import write_examples

    base = write_d1lfm3(tmp_path / "tiny-d1lfm3")
    write_examples(tmp_path / "data.jsonl", labelled(40))
    args = ["training", "fit", "-m", str(base), "--data", str(tmp_path / "data.jsonl")]
    result = CliRunner().invoke(app, [*args, "-o", str(tmp_path / "out"), "--method", "lora",
                                      "--rank", "4", "--epochs", "1", "--yes"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "  lora: adapts the network itself: minutes" in result.stderr
    assert "method: lora, then calibrate (rank 4, learning rate 0.0001;" in result.stdout
    shown = CliRunner().invoke(app, ["training", "fit", "--help"], terminal_width=200).output
    help_text = " ".join(shown.replace("│", " ").split())
    assert "lora (d3, d1 LFM3" in help_text and "--rank" in help_text
    assert "--learning-rate" in help_text


def test_the_lora_estimate_counts_adapters_the_longest_pass_and_kept_images(base, tmp_path):
    from mlx_decision.data.dataset import read_for
    from mlx_decision.data.format import training_splits
    from mlx_decision.training import lora, plan

    model = mlx_decision.load(base)
    assert plan.lora_parameters(base) == lora.add_adapters(model.backend)
    folder = image_folder(tmp_path / "pets", 12)
    examples = read_for(folder).examples
    # Image tokens from the headers, as the loaded model counts them.
    count = plan.lora_image_tokens(model.backend.family, base, {})
    for example in examples[:2]:
        passes = model.backend.training_passes(model.check(example.request))
        frozen = passes[0].frozen
        rows = frozen[0] if model.backend.family == "d3" else frozen[1]
        assert count(example.request["images"][0]) == rows.shape[0]
    splits = training_splits(examples, holdout=0.2)
    longest, kept = plan.lora_tokens(splits.fit + splits.dev, base, {})
    assert kept == sum(count(e.request["images"][0]) for e in splits.fit + splits.dev)
    hidden, _, layers = plan.lora_sizes(base)
    lora_estimate = plan.estimate(["lora", "calibrate"], base, {}, splits)
    calibrate = plan.estimate(["calibrate"], base, {}, splits)
    every, _ = plan.lora_tokens(splits.holdout + splits.fit + splits.dev, base, {})
    extra = (
        plan.lora_parameters(base) * plan.LORA_PARAMETER_BYTES
        + plan.LORA_STEP_BYTES
        + plan.LORA_TOKEN_BYTES * every * hidden * layers
        + kept * hidden * plan.FEATURE_BYTES
    )
    assert lora_estimate.peak_bytes == calibrate.peak_bytes + extra
    assert longest <= every


def test_a_lora_fit_over_the_working_set_names_the_readout_fit(tmp_path, monkeypatch):
    from mlx_decision.memory import ModelTooLargeError
    from mlx_decision.training.fitting import prepare

    base = write_d3(tmp_path / "tiny-d3")
    prepared = prepare(base, labelled(40))
    readout = prepared.estimate(["readout", "calibrate"], False).peak_bytes
    assert prepared.estimate(["lora", "calibrate"], False).peak_bytes > readout
    monkeypatch.setattr("mlx_decision.memory.working_set", lambda: readout)
    with pytest.raises(ModelTooLargeError) as error:
        fit(base, labelled(40), output=tmp_path / "fit", method="lora")
    assert "What would fit: a readout fit (--method readout, about" in str(error.value)
    assert not (tmp_path / "fit").exists()
