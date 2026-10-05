"""``mlx-decision download`` with the Hub replaced by fakes."""

from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from typer.testing import CliRunner

import mlx_decision.download as download
from mlx_decision.cli import app, pick_repo
from mlx_decision.registry import known_models

REPOS = {
    "Cloudflare/clef-flash": {"joint_head_config.json": "{}", "config.json": "{}"},
    "Qwen/Qwen3.5-9B": {"config.json": "{}"},
}


@pytest.fixture
def hub(monkeypatch, tmp_path):
    calls = []

    class Api:
        def model_info(self, repo_id, files_metadata=False):
            if repo_id not in REPOS:
                request = httpx.Request("GET", "https://huggingface.co/api/models/x")
                raise RepositoryNotFoundError(
                    "missing", response=httpx.Response(404, request=request)
                )
            return SimpleNamespace(siblings=[SimpleNamespace(size=19_100_000_000)])

    def snapshot_download(repo_id, allow_patterns=None, local_dir=None):
        calls.append((repo_id, allow_patterns, local_dir))
        folder = local_dir or tmp_path / "cache" / repo_id
        folder = type(tmp_path)(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in REPOS[repo_id].items():
            (folder / name).write_text(text)
        return str(folder)

    monkeypatch.setattr(download, "HfApi", Api)
    monkeypatch.setattr(download, "snapshot_download", snapshot_download)
    return calls


def invoke(*args):
    return CliRunner().invoke(app, ["download", *args])


def test_a_supported_model_is_downloaded(hub):
    result = invoke("Cloudflare/clef-flash")
    assert result.exit_code == 0, result.output
    assert "downloading Cloudflare/clef-flash (19.1 GB)" in result.stderr
    assert "try it: mlx-decision chat -m Cloudflare/clef-flash" in result.stderr
    # First only the JSON files, then everything.
    assert [(repo, patterns) for repo, patterns, _ in hub] == [
        ("Cloudflare/clef-flash", ["*.json"]),
        ("Cloudflare/clef-flash", None),
    ]


def test_into_a_local_folder(hub, tmp_path):
    result = invoke("Cloudflare/clef-flash", "--local-dir", str(tmp_path / "here"))
    assert result.exit_code == 0, result.output
    assert hub[-1][2] == tmp_path / "here"
    assert f"chat -m {tmp_path / 'here'}" in result.stderr


def test_an_unsupported_model_needs_yes(hub):
    result = invoke("Qwen/Qwen3.5-9B")
    assert result.exit_code == 1
    assert "not a supported decision model" in result.stderr
    assert len(hub) == 1  # only the JSON files
    assert invoke("Qwen/Qwen3.5-9B", "--yes").exit_code == 0


def test_a_missing_repo(hub):
    result = invoke("nobody/nothing")
    assert result.exit_code == 1
    assert "not found on the Hub (or private: set HF_TOKEN)" in result.stderr


def test_the_menu_needs_a_terminal(hub):
    result = invoke()
    assert result.exit_code == 1
    assert "give a repo id" in result.stderr


def test_the_menu_lists_known_models_and_other():
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    assert [m.repo_id for m in known_models()] == ["Cloudflare/clef-flash"]
    for typed, expected in [("\r", "Cloudflare/clef-flash"), ("2\rorg/model\r", "org/model")]:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(typed)
            assert pick_repo() == expected
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x03")
        assert pick_repo() is None
