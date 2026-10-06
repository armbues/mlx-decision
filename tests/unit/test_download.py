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
    "convaiinnovations/laya": {"rl_agent_config.json": "{}"},
}
LAYA_FILES = {  # the root checkpoint, plus a second one in a sub-folder
    "model.safetensors": 843_000_000,
    "rl_agent_config.json": 1_000,
    "encoder/config.json": 2_000,
    "tokenizer/tokenizer.json": 3_600_000,
    "multilingual/model.safetensors": 644_000_000,
    "assets/logo.png": 300_000,
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
            if repo_id == "convaiinnovations/laya":
                files = LAYA_FILES.items()
                return SimpleNamespace(
                    siblings=[SimpleNamespace(rfilename=n, size=size) for n, size in files]
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

    assert [m.repo_id for m in known_models()] == [
        "Cloudflare/clef-flash",
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
        "SupersonicLabs/Julia-1",
    ]
    for typed, expected in [("\r", "Cloudflare/clef-flash"), ("6\rorg/model\r", "org/model")]:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(typed)
            assert pick_repo() == expected
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x03")
        assert pick_repo() is None


def ask(typed):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_decision.cli import ask_models_dir

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(typed)
        return ask_models_dir()


def test_the_folder_prompt(tmp_path):
    from pathlib import Path

    import typer

    assert ask(f"{tmp_path}\r") == tmp_path
    assert ask("~/Models\r") == Path.home() / "Models"
    assert ask("\r") is None  # the Hub cache
    with pytest.raises(typer.Exit):
        ask("\x03")


@pytest.fixture
def terminal(monkeypatch):
    """A terminal whose folder answers and picked repo are given; records the order of questions."""
    import mlx_decision.cli as cli

    state = SimpleNamespace(folders=[], asked=[])

    def ask_models_dir():
        state.asked.append("folder")
        return state.folders.pop(0)

    def pick_repo():
        state.asked.append("model")
        return "Cloudflare/clef-flash"

    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr(cli, "ask_models_dir", ask_models_dir)
    monkeypatch.setattr(cli, "pick_repo", pick_repo)
    return state


def test_a_terminal_asks_for_the_folder_before_the_model(hub, terminal, tmp_path):
    terminal.folders.append(tmp_path / "models")
    result = invoke()
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["folder", "model"]
    target = tmp_path / "models" / "clef-flash"
    assert hub[-1][2] == target
    assert f"(19.1 GB) to {target} ..." in result.stderr
    assert f"chat -m {target}" in result.stderr


def test_an_empty_folder_keeps_the_cache(hub, terminal):
    terminal.folders.append(None)
    result = invoke("Cloudflare/clef-flash")
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["folder"]
    assert hub[-1][2] is None
    assert "(19.1 GB) ..." in result.stderr
    assert "chat -m Cloudflare/clef-flash" in result.stderr


def test_local_dir_skips_the_question(hub, terminal, tmp_path):
    result = invoke("Cloudflare/clef-flash", "--local-dir", str(tmp_path / "here"))
    assert result.exit_code == 0, result.output
    assert terminal.asked == []
    assert hub[-1][2] == tmp_path / "here"


def test_an_invalid_repo_id(hub, monkeypatch):
    from huggingface_hub.errors import HFValidationError

    class Api:
        def model_info(self, repo_id, files_metadata=False):
            raise HFValidationError(
                "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"
            )

    monkeypatch.setattr(download, "HfApi", Api)
    result = invoke("not a valid/id")
    assert result.exit_code == 1
    assert "error: not a valid/id: Repo id must be" in result.stderr


def test_a_laya_repo_fetches_only_its_root_checkpoint(hub):
    result = invoke("convaiinnovations/laya")
    assert result.exit_code == 0, result.output
    assert "downloading convaiinnovations/laya (0.8 GB)" in result.stderr
    patterns = hub[-1][1]
    assert "model.safetensors" in patterns and "encoder/*" in patterns
    assert not any(p.startswith(("multilingual", "assets", "*")) for p in patterns)
