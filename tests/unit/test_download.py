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
CLEF_FILES = {  # the weights and head count; the reference code does not
    "config.json": 3_000,
    "model-00001-of-00002.safetensors": 9_000_000_000,
    "model-00002-of-00002.safetensors": 9_830_000_000,
    "joint_head.safetensors": 243_000_000,
    "tokenizer.json": 20_000_000,
    "joint_schema_model.py": 23_000,
    "chat_template.jinja": 7_000,
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
            files = {
                "convaiinnovations/laya": LAYA_FILES,
                "Cloudflare/clef-flash": CLEF_FILES,
            }.get(repo_id, {"model.safetensors": 19_100_000_000})
            return SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=n, size=size) for n, size in files.items()]
            )

        def get_safetensors_metadata(self, repo_id, revision=None):
            tensors = {  # bf16: 17.83 GB of text backbone, 1 GB of vision tower
                "model.language_model.embed_tokens.weight": [8_915_000_000, 1],
                "model.visual.blocks.0.mlp.weight": [500_000_000, 1],
            }
            info = {n: SimpleNamespace(dtype="BF16", shape=shape) for n, shape in tensors.items()}
            files = {"model.safetensors": SimpleNamespace(tensors=info)}
            return SimpleNamespace(files_metadata=files)

    def snapshot_download(repo_id, allow_patterns=None, local_dir=None, revision=None):
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
    # First only the JSON files, then the family's files.
    assert [(repo, patterns) for repo, patterns, _ in hub] == [
        ("Cloudflare/clef-flash", ["*.json"]),
        (
            "Cloudflare/clef-flash",
            ["*.json", "model*.safetensors", "joint_head.safetensors", "LICENSE*", "README.md"],
        ),
    ]


def test_clef_skips_the_reference_code(hub):
    check = download.check_repo("Cloudflare/clef-flash")
    skipped = CLEF_FILES["joint_schema_model.py"] + CLEF_FILES["chat_template.jinja"]
    assert check.size_bytes == sum(CLEF_FILES.values()) - skipped


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
        "Cloudflare/clef",
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
        "SupersonicLabs/Julia-1",
    ]
    for typed, expected in [("\r", "Cloudflare/clef-flash"), ("7\rorg/model\r", "org/model")]:
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

    def pick_bits(check):
        state.asked.append("size")
        return 0  # full size

    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: True)
    monkeypatch.setattr(cli, "ask_models_dir", ask_models_dir)
    monkeypatch.setattr(cli, "pick_repo", pick_repo)
    monkeypatch.setattr(cli, "pick_bits", pick_bits)
    return state


def test_a_terminal_asks_for_the_folder_before_the_model(hub, terminal, tmp_path):
    terminal.folders.append(tmp_path / "models")
    result = invoke()
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["folder", "model", "size"]
    target = tmp_path / "models" / "clef-flash"
    assert hub[-1][2] == target
    assert f"(19.1 GB) to {target} ..." in result.stderr
    assert f"chat -m {target}" in result.stderr


def test_an_empty_folder_keeps_the_cache(hub, terminal):
    terminal.folders.append(None)
    result = invoke("Cloudflare/clef-flash")
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["folder", "size"]
    assert hub[-1][2] is None
    assert "(19.1 GB) ..." in result.stderr
    assert "chat -m Cloudflare/clef-flash" in result.stderr


def test_local_dir_skips_the_question(hub, terminal, tmp_path):
    result = invoke("Cloudflare/clef-flash", "--local-dir", str(tmp_path / "here"))
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["size"]
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


# Storing a model quantized: a tiny Clef release stands in for the repository.

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def remote(monkeypatch, tmp_path):
    """Fake Hub functions serving a tiny Clef with vision, its text in several shards.

    ``events`` records each weight file fetched with the weight files then in
    the download folder, and each file's deletion is visible there.
    """
    import shutil

    import mlx_decision.backbones.qwen3_5.load as load
    from tiny_models import write_clef

    with monkeypatch.context() as patch:
        patch.setattr(load, "SHARD_BYTES", 2**16)
        release = write_clef(tmp_path / "release", vision=True)
    (release / "README.md").write_text("readme\n")
    (release / "joint_schema_model.py").write_text("# reference code\n")
    names = sorted(p.name for p in release.iterdir())
    state = SimpleNamespace(release=release, events=[], fail_on=None)

    class Api:
        def model_info(self, repo_id, files_metadata=False):
            siblings = [
                SimpleNamespace(rfilename=n, size=(release / n).stat().st_size) for n in names
            ]
            return SimpleNamespace(siblings=siblings, sha=SHA)

        def get_safetensors_metadata(self, repo_id, revision=None):
            raise download.NotASafetensorsRepoError("no index")

    def snapshot_download(
        repo_id, allow_patterns=None, local_dir=None, revision=None, ignore_patterns=None
    ):
        assert revision == SHA and local_dir is not None
        picked = filter_repo_objects(names, allow_patterns=allow_patterns)
        picked = filter_repo_objects(picked, ignore_patterns=ignore_patterns)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in picked:
            shutil.copy2(release / name, Path(local_dir) / name)
        return str(local_dir)

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None):
        assert revision == SHA
        present = sorted(p.name for p in Path(local_dir).glob("model*.safetensors"))
        state.events.append((filename, present))
        if filename == state.fail_on:
            raise OSError("connection reset")
        shutil.copy2(release / filename, Path(local_dir) / filename)
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(download, "HfApi", Api)
    monkeypatch.setattr(download, "snapshot_download", snapshot_download)
    monkeypatch.setattr(download, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(download.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    return state


from pathlib import Path  # noqa: E402

from huggingface_hub.utils import filter_repo_objects  # noqa: E402


def tensors(folder):
    import mlx.core as mx

    found = {}
    for file in sorted(folder.glob("*.safetensors")):
        found.update(mx.load(str(file)))
    return found


def assert_same_model(stored, converted):
    import mlx.core as mx

    expected = tensors(converted)
    got = tensors(stored)
    assert got.keys() == expected.keys()
    for name, value in expected.items():
        assert mx.array_equal(got[name], value).item(), name
    for name in ("config.json", "tokenizer.json", "processor_config.json", "LICENSE"):
        assert (stored / name).read_bytes() == (converted / name).read_bytes(), name


def test_a_quantized_download_equals_a_converted_copy(remote, tmp_path):
    import json

    from mlx_decision.convert import convert
    from mlx_decision.registry import MARKER_FILE

    check = download.check_repo("org/tiny-clef")
    assert check.convertible and check.revision == SHA
    fetched = []
    out = download.download_quantized(
        check, 8, tmp_path / "models" / "tiny-clef", lambda i, n, name: fetched.append((i, n))
    )
    assert out == tmp_path / "models" / "tiny-clef"
    assert_same_model(out, convert(remote.release, tmp_path / "converted", bits=8))
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["source"] == "org/tiny-clef" and marker["revision"] == SHA
    assert marker["quantization"]["bits"] == 8 and marker["vision"] is True
    # One weight file at a time: each fetch finds the previous one deleted.
    weights = download._weight_files(remote.release)
    assert len(weights) > 2
    assert remote.events == [(name, []) for name in weights]
    assert fetched == [(i, len(weights)) for i in range(1, len(weights) + 1)]
    assert sorted(p.name for p in out.parent.iterdir()) == ["tiny-clef"]
    assert not (out / "joint_schema_model.py").exists()


def test_a_quantized_download_into_the_cache_loads_by_repo_id(remote, tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import scan_cache_dir

    import mlx_decision
    from mlx_decision.hub import resolve_model_path

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    out = download.download_quantized(download.check_repo("org/tiny-clef"), 4)
    repo = tmp_path / "hub" / "models--org--tiny-clef"
    assert out == repo / "snapshots" / SHA
    assert (repo / "refs" / "main").read_text() == SHA
    assert sorted(p.name for p in repo.iterdir()) == ["refs", "snapshots"]

    def no_network(*args, **kwargs):
        raise AssertionError("the Hub was asked")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    assert resolve_model_path("org/tiny-clef") == out
    model = mlx_decision.load("org/tiny-clef")
    assert model.name == "tiny-clef"
    model.decide("refund please", {"q": {"type": "noul"}})

    # The Hub's cache tools see one revision of files and can delete it.
    cache = scan_cache_dir(tmp_path / "hub")
    assert not cache.warnings
    (cached,) = cache.repos
    (revision,) = cached.revisions
    assert revision.commit_hash == SHA and revision.refs == frozenset({"main"})
    assert revision.size_on_disk == sum(p.stat().st_size for p in out.iterdir())
    cache.delete_revisions(SHA).execute()
    assert not (repo / "snapshots" / SHA).exists()


def test_a_stored_model_is_not_overwritten(remote, tmp_path):
    check = download.check_repo("org/tiny-clef")
    out = download.download_quantized(check, 8)
    with pytest.raises(FileExistsError, match="cache already has"):
        download.download_quantized(check, 4)
    with pytest.raises(FileExistsError, match="already stored quantized"):
        download.download(check.repo_id, files=check.files, revision=SHA)
    assert (out / "mlx_decision.json").exists()


def test_a_failed_quantized_download_leaves_nothing_behind(remote, tmp_path):
    check = download.check_repo("org/tiny-clef")
    remote.fail_on = download._weight_files(remote.release)[1]
    with pytest.raises(OSError, match="connection reset"):
        download.download_quantized(check, 8, tmp_path / "models" / "tiny-clef")
    assert list((tmp_path / "models").iterdir()) == []
    with pytest.raises(OSError, match="connection reset"):
        download.download_quantized(check, 8)
    repo = tmp_path / "hub" / "models--org--tiny-clef"
    assert not (repo / "snapshots").exists() and not (repo / "refs").exists()
    assert list(repo.iterdir()) == []


def test_not_enough_disk_space(remote, tmp_path, monkeypatch):
    check = download.check_repo("org/tiny-clef")
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(free=1_000))
    with pytest.raises(OSError, match="of free disk space"):
        download.download_quantized(check, 8, tmp_path / "x")
    assert remote.events == []


def test_quantized_sizes_and_fit():
    check = download.RepoCheck(
        "Cloudflare/clef", 55_400_000_000, "clef", weight_bytes=55_100_000_000
    )
    assert check.quantized_bytes(8) == pytest.approx(29.6e9, rel=0.01)
    assert check.quantized_bytes(4) == pytest.approx(15.8e9, rel=0.01)
    assert not check.fits(budget=55_700_000_000)
    assert check.fits(budget=64e9)
    assert check.fits(8, budget=33e9) and not check.fits(8, budget=32e9)
    # What the converter leaves unquantized (vision tower, head) counts in full.
    check = download.RepoCheck(
        "Cloudflare/clef", 55_400_000_000, "clef", weight_bytes=55_100_000_000,
        quantizable_bytes=54_000_000_000,
    )  # fmt: skip
    assert check.quantized_bytes(8) == pytest.approx(30.0e9, rel=0.01)


def test_the_check_reads_what_the_converter_quantizes(hub):
    check = download.check_repo("Cloudflare/clef-flash")
    assert check.quantizable_bytes == 17_830_000_000
    assert check.quantized_bytes(8) == pytest.approx(
        check.size_bytes - 17_830_000_000 * 7.5 / 16, rel=1e-6
    )
    assert not download.RepoCheck("org/laya", 1, "laya").convertible


def test_bits_on_the_command_line(remote, tmp_path):
    out = tmp_path / "here"
    result = invoke("org/tiny-clef", "--bits", "8", "--local-dir", str(out))
    assert result.exit_code == 0, result.output
    assert "as 8-bit (about" in result.stderr and "1/" in result.stderr
    assert (out / "mlx_decision.json").exists()
    result = invoke("org/tiny-clef", "--bits", "6", "--local-dir", str(tmp_path / "x"))
    assert result.exit_code == 2 and "must be 8 or 4" in result.output


def test_bits_need_a_family_that_converts(hub):
    result = invoke("convaiinnovations/laya", "--bits", "8")
    assert result.exit_code == 1
    assert "laya models cannot be stored quantized" in result.stderr


def test_a_model_too_large_gets_a_hint_without_a_terminal(hub, monkeypatch):
    monkeypatch.setattr(download, "working_set", lambda: 12_000_000_000)
    result = invoke("Cloudflare/clef-flash")
    assert result.exit_code == 0, result.output
    assert "does not fit in this Mac's GPU working set at full size; --bits 8" in result.stderr
    monkeypatch.setattr(download, "working_set", lambda: 7_000_000_000)
    result = invoke("Cloudflare/clef-flash")
    assert "at full size; --bits 4 stores it quantized" in result.stderr
    monkeypatch.setattr(download, "working_set", lambda: 1_000_000_000)
    result = invoke("Cloudflare/clef-flash")
    assert "at full size; not even at 4 bits" in result.stderr


@pytest.mark.parametrize(("budget", "expected"), [
    (64e9, 0), (12e9, 8), (10e9, 4), (1e9, None),
])  # fmt: skip
def test_the_largest_size_that_fits(monkeypatch, budget, expected):
    monkeypatch.setattr(download, "working_set", lambda: budget)
    check = download.RepoCheck("org/m", 19e9, "clef", weight_bytes=18.8e9)
    assert check.fitting_bits() == expected


@pytest.mark.parametrize(("budget", "typed", "expected"), [
    (64e9, "\r", 0),  # fits: full size first
    (12e9, "\r", 8),  # does not fit: the largest that does
    (10e9, "\r", 4),
    (1e9, "\r", 4),  # nothing fits: the smallest
    (12e9, "3\r", 4),
])  # fmt: skip
def test_the_size_menu(monkeypatch, budget, typed, expected):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from mlx_decision.cli import pick_bits

    monkeypatch.setattr(download, "working_set", lambda: budget)
    check = download.RepoCheck("org/m", 19e9, "clef", weight_bytes=18.8e9)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(typed)
        assert pick_bits(check) == expected


def test_a_terminal_asks_for_the_size_after_the_model(remote, terminal, monkeypatch, tmp_path):
    import mlx_decision.cli as cli

    monkeypatch.setattr(cli, "pick_repo", lambda: terminal.asked.append("model") or "org/tiny")
    monkeypatch.setattr(cli, "pick_bits", lambda check: terminal.asked.append("size") or 4)
    terminal.folders.append(tmp_path / "models")
    result = invoke()
    assert result.exit_code == 0, result.output
    assert terminal.asked == ["folder", "model", "size"]
    marker = (tmp_path / "models" / "tiny" / "mlx_decision.json").read_text()
    assert '"bits": 4' in marker
