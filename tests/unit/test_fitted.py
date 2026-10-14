"""Fitted models: the folder, its base, the fingerprint and the fitted answers."""

import json
import logging
import math
import shutil
from pathlib import Path

import huggingface_hub
import mlx.core as mx
import pytest

import mlx_decision
from fake_backend import write_model, write_sized_model
from mlx_decision import fitted
from mlx_decision.fitted import (
    BaseModelError,
    Calibration,
    Fit,
    FitQuestion,
    base_reference,
    calibrate,
    fingerprint,
)
from mlx_decision.memory import weight_bytes
from mlx_decision.models.h2o.model import commit_noul
from mlx_decision.models.readout import Readout
from mlx_decision.registry import MARKER_FILE
from mlx_decision.types import parse_request
from tiny_models import write_h2o

TEAM = {"type": "choice", "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}
URGENT = {"type": "noul", "instructions": "Is this urgent?"}
CALIBRATION = Calibration(2.0, [0.0, 1.0, -1.0])


def team_fit(**meta) -> Fit:
    question = FitQuestion("team", "choice", ["billing", "tech", "sales"])
    return Fit(["calibrate"], [question], {0: CALIBRATION}, meta={"seed": 0, **meta})


def write_fitted(folder: Path, base: Path, fit: Fit, **options) -> Path:
    return fitted.write(
        folder,
        family="fake",
        base=base_reference(base, base),
        base_fingerprint=fingerprint(base),
        fit=fit,
        **options,
    )


@pytest.fixture
def base(tmp_path):
    return write_model(tmp_path / "base")


def test_a_calibrated_question_is_answered_through_the_fit(tmp_path, base):
    folder = write_fitted(tmp_path / "support-fit", base, team_fit(metrics={"accuracy": 0.9}))
    model = mlx_decision.load(folder)
    plain = mlx_decision.load(base)
    want = calibrate(plain.decide("x", {"team": TEAM}).answers["team"].probabilities, CALIBRATION)
    got = model.decide("x", {"team": TEAM}).answers["team"].probabilities
    assert got == pytest.approx(want, abs=1e-4)
    assert got != pytest.approx(plain.decide("x", {"team": TEAM}).answers["team"].probabilities)
    info = model.info()
    assert (info["name"], info["family"]) == ("support-fit", "fake")
    assert info["base"] == {"path": str(base.absolute())}
    assert info["fit"] == {
        "kinds": ["calibrate"],
        "questions": ["team"],
        "metrics": {"accuracy": 0.9},
        "seed": 0,
    }


def test_calibrate_by_hand():
    p = {"a": 0.5, "b": 0.25, "c": 0.25}
    got = calibrate(p, Calibration(2.0, [0.0, 1.0, 0.0]))
    weights = [math.sqrt(0.5), math.sqrt(0.25) * math.e, math.sqrt(0.25)]
    assert list(got.values()) == pytest.approx([w / sum(weights) for w in weights])
    assert calibrate({"a": 1.0, "b": 0.0}, Calibration(0.5)) == {"a": 1.0, "b": 0.0}


def test_other_questions_are_answered_by_the_base_with_one_warning(tmp_path, base, caplog):
    model = mlx_decision.load(write_fitted(tmp_path / "fit", base, team_fit()))
    plain = mlx_decision.load(base)
    other_options = {**TEAM, "criteria": {"billing": "Money", "tech": "Bugs"}}
    questions = {"urgent": URGENT, "team": other_options}
    with caplog.at_level(logging.WARNING, logger="mlx_decision.fitted"):
        first = model.decide("x", questions)
        model.decide("x", questions)
    assert first.to_wire()["answers"] == plain.decide("x", questions).to_wire()["answers"]
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2  # once per question id
    assert all("is not one it was fitted on" in w for w in warnings)
    assert {"'urgent'" in w for w in warnings} == {True, False}


def test_the_folder_is_written_whole_or_not_at_all(tmp_path, base, monkeypatch):
    folder = write_fitted(tmp_path / "fit", base, team_fit(), report={"before": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base", "fit"]
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [MARKER_FILE, "fit.safetensors", "training.json"]
    )
    marker = json.loads((folder / MARKER_FILE).read_text())
    assert marker["fit"]["questions"] == [
        {"id": "team", "type": "choice", "options": ["billing", "tech", "sales"]}
    ]
    with pytest.raises(FileExistsError):
        write_fitted(folder, base, team_fit())

    def disk_full(*args):
        raise OSError("disk full")

    monkeypatch.setattr(mx, "save_safetensors", disk_full)
    with pytest.raises(OSError, match="disk full"):
        write_fitted(folder, base, team_fit(seed=3), overwrite=True)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base", "fit"]  # the old one kept
    assert json.loads((folder / MARKER_FILE).read_text())["fit"]["seed"] == 0
    write_fitted(folder, base, team_fit(seed=7), overwrite=True)
    assert json.loads((folder / MARKER_FILE).read_text())["fit"]["seed"] == 7
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base", "fit"]


def test_a_moved_base_is_refused_naming_it(tmp_path, base):
    folder = write_fitted(tmp_path / "fit", base, team_fit())
    base.rename(tmp_path / "elsewhere")
    with pytest.raises(BaseModelError, match=f"fit: its base model {base.absolute()} is missing"):
        mlx_decision.load(folder)


def test_a_changed_base_is_refused(tmp_path):
    base = write_sized_model(tmp_path / "base", 64)
    folder = write_fitted(tmp_path / "fit", base, team_fit())
    mlx_decision.load(folder)
    mx.save_safetensors(str(base / "model.safetensors"), {"w": mx.ones(64, mx.uint8)})
    mlx_decision.load(folder)  # other weight bytes, same tensors: not read, not noticed
    mx.save_safetensors(str(base / "model.safetensors"), {"w": mx.zeros(65, mx.uint8)})
    with pytest.raises(BaseModelError, match="its base model .* has changed since the fit"):
        mlx_decision.load(folder)


def test_the_fingerprint_reads_configs_names_and_headers(tmp_path):
    base = write_sized_model(tmp_path / "base", 64)
    copy = shutil.copytree(base, tmp_path / "copy")
    assert fingerprint(copy) == fingerprint(base)
    (copy / ".cache").mkdir()
    (copy / ".cache" / "x.json").write_text("{}")
    assert fingerprint(copy) == fingerprint(base)  # hidden folders are not the model's
    (copy / MARKER_FILE).write_text(json.dumps({"family": "fake", "max_input_tokens": 9}))
    assert fingerprint(copy) != fingerprint(base)


def test_the_memory_check_counts_the_base_and_the_fit(tmp_path):
    base = write_sized_model(tmp_path / "base", 4096)
    folder = write_fitted(tmp_path / "fit", base, team_fit())
    assert weight_bytes(folder) == weight_bytes(base) + 4 + 3 * 4  # temperature and biases


def test_a_hub_base_resolves_at_its_commit_without_the_network(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    revision = "a" * 40
    snapshot = write_model(cache / "models--org--decider" / "snapshots" / revision)
    assert base_reference("org/decider", snapshot) == {
        "repo_id": "org/decider",
        "revision": revision,
    }
    assert base_reference(snapshot, snapshot) == {"repo_id": "org/decider", "revision": revision}
    folder = write_fitted(tmp_path / "fit", snapshot, team_fit())
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    def offline(*args, **kwargs):
        raise AssertionError("asked the Hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline)
    assert mlx_decision.load(folder).info()["base"]["revision"] == revision
    shutil.rmtree(snapshot)

    def unreachable(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unreachable)
    with pytest.raises(BaseModelError, match=f"org/decider at {revision[:12]} cannot be found"):
        mlx_decision.load(folder)


def test_a_readout_fit_needs_a_family_with_readout_hooks(tmp_path, base):
    fit = team_fit()
    fit.kinds = ["readout"]
    fit.readouts = {0: Readout(["billing", "tech", "sales"], mx.zeros((3, 4)), mx.zeros(3))}
    folder = write_fitted(tmp_path / "fit", base, fit)
    with pytest.raises(ValueError, match="a readout fit needs a family that reads out"):
        mlx_decision.load(folder)


def test_h2o_readout_and_calibration_come_before_the_floor(tmp_path):
    base = write_h2o(tmp_path / "tiny-h2o")
    plain = mlx_decision.load(base).backend
    request = parse_request(
        {"state": "please refund", "questions": {"team": TEAM, "urgent": URGENT}}
    )
    with mx.stream(mx.cpu):  # one device throughout: the prefix is kept between calls
        team = plain.features(request)["team"]
    readout = Readout(team.option_ids, team.rows * 0.5, mx.array([0.0, 0.5, -0.5]))
    questions = [FitQuestion.of(qid, q) for qid, q in request.questions.items()]
    urgent = Calibration(0.25, [0.3, -0.3])
    fit = Fit(["readout", "calibrate"], questions, {0: CALIBRATION, 1: urgent}, {0: readout})
    folder = fitted.write(
        tmp_path / "h2o-fit",
        family="h2o",
        base=base_reference(base, base),
        base_fingerprint=fingerprint(base),
        fit=fit,
    )
    model = mlx_decision.load(folder)
    assert (model.name, model.info()["family"]) == ("h2o-fit", "h2o")
    with mx.stream(mx.cpu):
        got = model.backend.score(request).probabilities
        raw = plain.score(request, readouts={"team": readout}, finish=False).probabilities
    assert got["team"] == pytest.approx(calibrate(raw["team"], CALIBRATION), abs=1e-6)
    p_true = commit_noul(calibrate(raw["urgent"], urgent)["true"], plain.noul_floor)
    assert got["urgent"] == pytest.approx({"true": p_true, "false": 1 - p_true}, abs=1e-6)


def test_run_answers_with_a_fitted_model(tmp_path, base):
    from typer.testing import CliRunner

    from mlx_decision.cli import app

    folder = write_fitted(tmp_path / "fit", base, team_fit())
    result = CliRunner().invoke(
        app, ["run", "-m", str(folder), "-s", "x", "--choice", "team=billing,tech,sales", "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["model"] == "fit"


def test_run_warns_once_about_questions_the_fit_does_not_cover(tmp_path, base):
    from typer.testing import CliRunner

    from mlx_decision.cli import app

    folder = write_fitted(tmp_path / "fit", base, team_fit())
    states = tmp_path / "states.jsonl"
    states.write_text('{"state": "a"}\n{"state": "b"}\n')
    flags = ["--choice", "team=billing,tech,sales", "--noul", "urgent"]
    result = CliRunner().invoke(app, ["run", "-m", str(folder), "--states", str(states), *flags])
    assert result.exit_code == 0, result.output
    warnings = [line for line in result.stderr.splitlines() if line.startswith("warning:")]
    assert warnings == [
        "warning: fit: question 'noul_1' is not one it was fitted on (same id and options); "
        "the base answers it"
    ]  # before answering, once, and not again from the log
    assert len(result.stdout.splitlines()) == 2


def test_mcp_tells_agents_which_questions_were_fitted(tmp_path, base):
    from mlx_decision.mcp_server import family_hints

    model = mlx_decision.load(write_fitted(tmp_path / "fit", base, team_fit()))
    hints = " ".join(family_hints(model.info(), allow_image_paths=False))
    assert "fitted on these questions: `team`." in hints
    assert "fitted on" not in " ".join(family_hints(mlx_decision.load(base).info(), False))


def test_questions_outside_a_weight_fit_are_answered_by_its_weights(base):
    from mlx_decision.mcp_server import family_hints

    fit = Fit(["finetune", "calibrate"], team_fit().questions, {0: CALIBRATION})
    backend = fitted.FittedBackend("tuned", mlx_decision.load(base).backend, fit, {})
    questions = parse_request({"state": "", "questions": {"urgent": {"type": "noul"}}}).questions
    assert backend.unfitted(questions) == [
        "tuned: question 'urgent' is not one it was fitted on (same id and options); "
        "its fitted weights answer it without the calibration"
    ]
    info = {**mlx_decision.load(base).info(), "fit": {"kinds": fit.kinds, "questions": ["team"]}}
    hints = " ".join(family_hints(info, allow_image_paths=False))
    assert "other questions are answered without the fit's calibration." in hints


def test_mcp_checks_options_against_the_base_family(tmp_path):
    from typer.testing import CliRunner

    from mlx_decision.cli import app
    from tiny_models import write_marker

    laya = write_marker(tmp_path / "laya", "laya")
    folder = write_fitted(tmp_path / "laya-fit", laya, team_fit())
    result = CliRunner().invoke(app, ["mcp", "-m", str(folder), "--noul-floor", "0.5"])
    assert result.exit_code == 1
    assert "laya models do not take noul_floor" in result.stderr


def test_a_shared_h2o_base_answers_as_before_beside_its_readout_fit(tmp_path):
    from mlx_decision.pool import ModelPool, discover

    root = tmp_path / "models"
    base = write_h2o(root / "tiny-h2o")
    request = parse_request({"state": "please refund", "questions": {"team": TEAM}})
    plain = mlx_decision.load(base)
    with mx.stream(mx.cpu):  # one device throughout: the prefix is kept between calls
        team = plain.backend.features(request)["team"]
        want = plain.decide_request(request).to_wire()["answers"]
    readout = Readout(team.option_ids, -team.rows, mx.zeros(3))
    fit = Fit(
        ["readout"], [FitQuestion.of("team", request.questions["team"])], readouts={0: readout}
    )
    fitted.write(
        root / "h2o-fit",
        family="h2o",
        base=base_reference(base, base),
        base_fingerprint=fingerprint(base),
        fit=fit,
    )
    pool = ModelPool(discover([root]), budget=10**12)
    with mx.stream(mx.cpu):
        got_fit = pool.get("h2o-fit").decide_request(request)
        got_base = pool.get("tiny-h2o").decide_request(request)
    assert pool.get("h2o-fit").backend.base is pool.get("tiny-h2o").backend
    assert got_base.to_wire()["answers"] == want
    assert got_fit.answers["team"].choice != got_base.answers["team"].choice
