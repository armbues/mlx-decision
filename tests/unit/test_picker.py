"""The model picker for run and chat without a model, and the Hub cache's models."""

import json
from pathlib import Path

import huggingface_hub.constants
import pytest

from fake_backend import write_model, write_sized_model
from mlx_decision.config import Config
from mlx_decision.hub import cached_models
from mlx_decision.picker import choices
from terminal import DOWN, ENTER, Terminal
from tiny_models import write_d1, write_marker


def cache_repo(cache: Path, repo_id: str, write, *, drop: str | None = None) -> Path:
    """A repo in a Hub cache laid out as huggingface_hub does: files in blobs,
    the snapshot of refs/main made of links to them; ``drop`` leaves a link
    to a missing blob (a download stopped before that file)."""
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    commit = "0123abcd"
    staged = write(folder / "staged")
    snapshot = folder / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (folder / "blobs").mkdir()
    for file in sorted(staged.rglob("*")):
        if file.is_file():
            blob = folder / "blobs" / f"{abs(hash(str(file)))}"
            file.rename(blob)
            link = snapshot / file.relative_to(staged)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(blob)
            if file.name == drop:
                blob.unlink()
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text(commit)
    return snapshot


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "hub"
    root.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(root))
    return root


def indexed(folder: Path) -> Path:
    """A fake model whose weights are split, with the index listing both parts."""
    write_sized_model(folder, 1000)
    (folder / "model.safetensors").rename(folder / "model-00001-of-00002.safetensors")
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"x")
    weight_map = {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return folder


def test_cached_models_are_complete_snapshots_a_family_loads(cache):
    cache_repo(cache, "org/whole", lambda f: write_sized_model(f, 1000))
    cache_repo(cache, "org/split", indexed)
    cache_repo(cache, "org/stopped", indexed, drop="model-00002-of-00002.safetensors")
    cache_repo(cache, "org/no-weights", write_model)
    cache_repo(cache, "org/other", lambda f: (f.mkdir(parents=True), f)[1])
    found = dict(cached_models())
    assert sorted(found) == ["org/split", "org/whole"]
    assert found["org/whole"] == cache / "models--org--whole/snapshots/0123abcd"


def test_without_a_cache_there_are_no_cached_models(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "none"))
    assert cached_models() == []


def test_the_picker_lists_models_dir_models_and_the_cache_once_each(tmp_path, cache):
    models_dir = tmp_path / "Decision"
    write_sized_model(models_dir / "beta", 1000)
    write_sized_model(models_dir / "Alpha", 2000)
    (models_dir / "notes").mkdir()
    write_model(tmp_path / "many/one")
    single = write_model(tmp_path / "single")
    cache_repo(cache, "org/cached", lambda f: write_sized_model(f, 1000))
    models = (
        str(single),
        str(tmp_path / "many"),
        str(models_dir / "beta"),
        "org/cached",
        "org/far",
    )
    config = Config(tmp_path / "config.toml", models=models, models_dir=models_dir)
    listed = [(c.name, c.place, c.family) for c in choices(config)]
    assert listed == [
        ("Alpha", "models_dir", "fake"),
        ("beta", "models_dir", "fake"),
        ("one", "models", "fake"),
        ("org/cached", "models", "fake"),
        ("single", "models", "fake"),
        ("org/far", "models", ""),  # not downloaded: no family or size yet, so last
    ]
    by_name = {c.name: c for c in choices(config)}
    assert by_name["Alpha"].reference == str(models_dir / "Alpha")
    assert by_name["Alpha"].size > by_name["beta"].size > 1000
    assert by_name["org/far"].size is None
    alone = Config(tmp_path / "config.toml")
    assert [(c.name, c.place) for c in choices(alone)] == [("org/cached", "Hugging Face cache")]


def test_without_a_terminal_the_error_lists_the_names(mlx_decision_home, tmp_path):
    from typer.testing import CliRunner

    from mlx_decision.cli import app

    models_dir = tmp_path / "Decision"
    write_model(models_dir / "alpha")
    (mlx_decision_home / "config.toml").write_text(f'models_dir = "{models_dir}"\n')
    result = CliRunner().invoke(app, ["run", "-s", "x", "--noul", "q=Is it?"])
    assert result.exit_code == 1
    assert "pass -m MODEL (or set default_model in " in result.stderr
    assert "; models: alpha" in result.stderr


def test_run_without_a_model_offers_the_picker_in_a_terminal(
    mlx_decision_home, tmp_path, monkeypatch
):
    models_dir = tmp_path / "Decision"
    write_marker(models_dir / "tiny-laya", "laya")
    write_d1(models_dir / "tiny-d1")
    (mlx_decision_home / "config.toml").write_text(f'models_dir = "{models_dir}"\n')
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    term = Terminal(
        ["run", "-s", "The app crashes", "--noul", "q=Is it a bug?", "--json"], tmp_path
    )
    screen = term.expect("Model:")
    term.expect("tiny-laya")
    screen = term.text()
    # d1 before Laya, as the families are listed everywhere.
    assert screen.index("tiny-d1") < screen.index("tiny-laya")
    assert "d1" in screen and "Laya" in screen and "models_dir" in screen
    term.send(DOWN + ENTER)
    assert term.wait() == 0, term.text()
    assert '"q"' in term.text()


def test_leaving_the_picker_cancels(mlx_decision_home, tmp_path, monkeypatch):
    models_dir = tmp_path / "Decision"
    write_marker(models_dir / "tiny-laya", "laya")
    (mlx_decision_home / "config.toml").write_text(f'models_dir = "{models_dir}"\n')
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    term = Terminal(["run", "-s", "x", "--noul", "q=Is it?"], tmp_path)
    term.expect("tiny-laya")
    term.send("\x03")  # Ctrl-C
    assert term.wait() == 1
    assert "cancelled" in term.text()
