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


def test_hub_2s_connection_errors_become_one_line_errors(monkeypatch):
    # huggingface_hub 2.x raises httpx2's errors instead of httpx's.
    httpx2 = pytest.importorskip("httpx2")

    def snapshot_download(repo_id, allow_patterns=None):
        raise httpx2.ConnectError("no route")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(ModelNotFoundError, match="cannot download from the Hugging Face Hub"):
        resolve_model_path("org/model")


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


def test_a_repo_id_in_a_format_no_family_knows_fetches_no_weights(monkeypatch, tmp_path):
    from mlx_decision.registry import UnknownFormat

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "decision_config.json").write_text('{"format_version": 2}')
    (snapshot / "joint_config.json").write_text('{"template": "joint-question-inline-readout-v9"}')
    calls = []

    def snapshot_download(repo_id, allow_patterns=None):
        calls.append(allow_patterns)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(UnknownFormat, match="^org/decider: joint_config.json names the request"):
        mlx_decision.load("org/decider")
    assert calls == [["*.json"]]


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


def test_downloading_by_repo_id_logs_the_familys_licence_note(monkeypatch, tmp_path, caplog):
    import logging

    from huggingface_hub.errors import LocalEntryNotFoundError

    from tiny_models import write_d1

    snapshot = write_d1(tmp_path / "snapshot")

    def not_cached(repo_id, allow_patterns=None, local_files_only=False):
        if local_files_only:
            raise LocalEntryNotFoundError("not in the cache")
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", not_cached)
    with caplog.at_level(logging.WARNING, logger="mlx_decision.hub"):
        assert mlx_decision.load("org/d1").name == "d1"
    assert "org/d1: d1's weights are under the LFM Open License v1.0" in caplog.text

    caplog.clear()
    cached = lambda repo_id, allow_patterns=None, local_files_only=False: str(snapshot)  # noqa: E731
    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    with caplog.at_level(logging.WARNING, logger="mlx_decision.hub"):
        mlx_decision.load("org/d1")
    assert "LFM" not in caplog.text


PIN = "e" * 40


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """``org/model`` pinned to PIN, in an empty Hub cache."""
    import mlx_decision.registry as registry

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(registry, "pinned_revision", lambda r: PIN if r == "org/model" else None)
    return tmp_path / "hub" / "models--org--model" / "snapshots"


def test_h2o_lightning_is_pinned_to_the_checked_release():
    from mlx_decision.registry import pinned_revision

    checked = "839c5c9523764f0b68c2785944f5541ee66a92b8"  # tag v1.4.2
    assert pinned_revision("h2oai/h2o-lightning-4b") == checked
    assert pinned_revision("H2OAI/h2o-lightning-4b") == checked
    assert pinned_revision("Cloudflare/clef-flash") is None


def test_a_pinned_repo_resolves_to_its_checked_commit_without_the_network(pinned, monkeypatch):
    snapshot = write_model(pinned / PIN)
    write_model(pinned / ("0" * 40))  # a later main, also cached

    def no_network(*args, **kwargs):
        raise AssertionError("the Hub was asked")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    assert resolve_model_path("org/model") == snapshot


def test_a_pinned_repo_is_fetched_at_its_checked_commit(pinned, monkeypatch):
    asked = []

    def snapshot_download(repo_id, allow_patterns=None, revision=None):
        asked.append(revision)
        return str(write_model(pinned / revision))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert resolve_model_path("org/model") == pinned / PIN
    assert set(asked) == {PIN}


def test_offline_an_earlier_cached_release_of_a_pinned_repo_serves(pinned, monkeypatch):
    earlier = write_model(pinned / ("0" * 40))

    def snapshot_download(repo_id, allow_patterns=None, revision=None):
        if revision is not None:
            raise LocalEntryNotFoundError("offline")
        return str(earlier)  # the cache's main

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert resolve_model_path("org/model") == earlier


def test_a_pinned_repo_stored_quantized_earlier_is_not_fetched_again(pinned, monkeypatch):
    import mlx_decision.hub as hub

    stored = write_model(pinned / ("0" * 40))
    monkeypatch.setattr(hub, "stored_snapshot", lambda repo_id: stored)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("fetch"))
    assert resolve_model_path("org/model") == stored
