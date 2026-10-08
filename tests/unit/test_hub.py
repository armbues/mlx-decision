"""Model references: local folders, Hub repo ids, and one-line errors for both."""

import httpx
import huggingface_hub
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from typer.testing import CliRunner

import mlx_decision
from fake_backend import write_model
from mlx_decision.cli import app
from mlx_decision.hub import ModelNotFoundError, is_repo_id, resolve_model_path


def http_error(kind):
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return kind("hub says no", response=httpx.Response(404, request=request))


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (http_error(RepositoryNotFoundError), "not found on the Hugging Face Hub"),
        (http_error(GatedRepoError), "gated on the Hugging Face Hub"),
        (HFValidationError("bad id"), "not a local folder or a valid Hugging Face repo id"),
        (LocalEntryNotFoundError("offline"), "cannot be reached"),
        (httpx.ConnectError("no route"), "cannot download from the Hugging Face Hub: no route"),
    ],
)
def test_hub_failures_become_one_line_errors(monkeypatch, error, reason):
    def snapshot_download(repo_id, allow_patterns=None):
        raise error

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(ModelNotFoundError) as caught:
        resolve_model_path("org/model")
    assert str(caught.value).startswith("org/model: ")
    assert reason in str(caught.value)
    assert "\n" not in str(caught.value)


def test_local_paths_are_never_looked_up_on_the_hub(tmp_path):
    assert not is_repo_id(tmp_path)
    assert not is_repo_id("./missing")
    assert is_repo_id("Cloudflare/clef-flash")
    with pytest.raises(ModelNotFoundError, match="model folder not found"):
        resolve_model_path("./missing")


def test_a_model_loaded_by_repo_id_is_named_after_the_repo(monkeypatch, tmp_path):
    snapshot = write_model(tmp_path / "models--org--decider" / "snapshots" / ("0" * 40))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, allow_patterns=None: str(snapshot)
    )
    assert mlx_decision.load("org/decider").name == "decider"
    assert mlx_decision.load(snapshot).name == "fake"  # a folder keeps the backend's name


@pytest.mark.parametrize("command", [["run", "-s", "x", "--noul", "q?"], ["server"], ["benchmark"]])
def test_commands_report_an_unknown_repo_in_one_line(monkeypatch, command):
    def snapshot_download(repo_id, allow_patterns=None):
        raise http_error(RepositoryNotFoundError)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    result = CliRunner().invoke(app, [command[0], "-m", "nobody/nothing", *command[1:]])
    assert result.exit_code == 1, result.output
    assert "error: nobody/nothing: not a local folder, and not found" in result.stderr
    assert "Traceback" not in result.output


def test_a_repo_id_fetches_only_the_files_its_family_needs(monkeypatch, tmp_path):
    from tiny_models import write_marker

    snapshot = write_marker(tmp_path / "snapshot", "laya")
    calls = []

    def snapshot_download(repo_id, allow_patterns=None):
        calls.append(allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert mlx_decision.load("org/laya").name == "laya"
    assert calls[0] == ["*.json"]
    assert "rl_agent_config.json" in calls[1] and "encoder/*" in calls[1]


def test_only_a_marker_written_by_download_counts_as_a_stored_model(monkeypatch, tmp_path):
    from mlx_decision.hub import stored_snapshot

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--decider"
    snapshot = repo / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0" * 40)
    assert stored_snapshot("org/decider") is None
    # A converted model on the Hub: its marker arrives as a link to a blob,
    # and the rest of its files may still be missing.
    (repo / "blobs").mkdir()
    (repo / "blobs" / "abc").write_text('{"family": "fake"}')
    (snapshot / "mlx_decision.json").symlink_to(repo / "blobs" / "abc")
    assert stored_snapshot("org/decider") is None
    (snapshot / "mlx_decision.json").unlink()
    (snapshot / "mlx_decision.json").write_text('{"family": "fake"}')
    assert stored_snapshot("org/decider") == snapshot
