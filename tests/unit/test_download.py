"""``mlx-decision download`` with the Hub replaced by fakes."""

import io
import sys
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
    "LiquidAI/d1-3B": {
        "config.json": '{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1Model"}}'
    },
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

    def snapshot_download(
        repo_id, allow_patterns=None, local_dir=None, revision=None, tqdm_class=None
    ):
        assert tqdm_class is not None  # every download's bars are ours to close
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
    assert "known families: clef, pplx, d3, h2o, d1, laya, julia" in result.stderr
    assert len(hub) == 1  # only the JSON files
    result = invoke("Qwen/Qwen3.5-9B", "--yes")
    assert result.exit_code == 0
    assert "downloaded Qwen/Qwen3.5-9B" in result.stderr
    assert "try it" not in result.stderr


def test_d1_notes_its_licence_and_downloads_without_a_terminal(hub):
    result = invoke("LiquidAI/d1-3B")
    assert result.exit_code == 0, result.output
    assert "note: d1's weights are under the LFM Open License v1.0" in result.stderr
    assert "less than $10 million in annual revenue" in result.stderr
    assert len(hub) == 2  # the JSON files, then the model


@pytest.mark.parametrize("answer", [True, False])
def test_d1_asks_about_its_licence_in_a_terminal(hub, terminal, monkeypatch, answer):
    terminal.folders.append(None)
    asked = []
    monkeypatch.setattr(
        "prompt_toolkit.shortcuts.confirm", lambda message: asked.append(message) or answer
    )
    result = invoke("LiquidAI/d1-3B")
    assert asked == ["Download it under these terms?"]
    assert "LFM Open License" in result.stderr
    if answer:
        assert result.exit_code == 0, result.output
        assert len(hub) == 2
    else:
        assert result.exit_code == 1
        assert "not downloaded" in result.stderr
        assert len(hub) == 1  # only the JSON files


def test_yes_skips_the_licence_question(hub, terminal, monkeypatch):
    terminal.folders.append(None)
    monkeypatch.setattr("prompt_toolkit.shortcuts.confirm", lambda message: pytest.fail(message))
    result = invoke("LiquidAI/d1-3B", "--yes")
    assert result.exit_code == 0, result.output
    assert "note: d1's weights" in result.stderr


def test_other_families_note_nothing(hub):
    assert "note:" not in invoke("Cloudflare/clef-flash").stderr


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
        "perplexity-ai/pplx-decider-v1.1-27b",
        "vllm-sr/d3-flash",
        "vllm-sr/d3",
        "vllm-sr/d3-mini",
        "vllm-sr/d3-nano",
        "vllm-sr/d3-lite",
        "vllm-sr/d3-edge",
        "h2oai/h2o-lightning-4b",
        "LiquidAI/d1-3B",
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
        "SupersonicLabs/Julia-1",
    ]
    # Number keys reach the first nine entries; "Other..." is the tenth.
    other = "\x1b[B" * len(known_models()) + "\rorg/model\r"
    for typed, expected in [("\r", "Cloudflare/clef-flash"), (other, "org/model")]:
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
    import mlx_decision.backbones.qwen3_5.load as load
    from tiny_models import write_clef

    with monkeypatch.context() as patch:
        patch.setattr(load, "SHARD_BYTES", 2**16)
        release = write_clef(tmp_path / "release", vision=True)
    (release / "README.md").write_text("readme\n")
    (release / "joint_schema_model.py").write_text("# reference code\n")
    return serve(monkeypatch, tmp_path, release)


@pytest.fixture
def remote_pplx(monkeypatch, tmp_path):
    """The same for a tiny pplx release, with the release's extra files."""
    from tiny_models import write_pplx

    release = write_pplx(tmp_path / "release")
    for name in ("README.md", "NOTICE", "chat_template.jinja", "release-manifest.json"):
        (release / name).write_text("x\n")
    for name in ("source/src/autojev/model.py", "training/config.json"):
        (release / name).parent.mkdir(parents=True, exist_ok=True)
        (release / name).write_text("{}\n")
    return serve(monkeypatch, tmp_path, release)


@pytest.fixture
def remote_h2o(monkeypatch, tmp_path):
    """The same for a tiny H2O-Lightning release: one model.safetensors, no index."""
    from tiny_models import write_h2o

    release = write_h2o(tmp_path / "release")
    for name in ("README.md", "h2o_lightning_shim.py", "serve.sh", "adapter/adapter_config.json"):
        (release / name).parent.mkdir(parents=True, exist_ok=True)
        (release / name).write_text("x\n")
    return serve(monkeypatch, tmp_path, release)


def serve(monkeypatch, tmp_path, release):
    import shutil

    names = sorted(str(p.relative_to(release)) for p in release.rglob("*") if p.is_file())
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
        repo_id,
        allow_patterns=None,
        local_dir=None,
        revision=None,
        ignore_patterns=None,
        tqdm_class=None,
    ):
        assert revision == SHA and local_dir is not None and tqdm_class is not None
        picked = filter_repo_objects(names, allow_patterns=allow_patterns)
        picked = filter_repo_objects(picked, ignore_patterns=ignore_patterns)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in picked:
            (Path(local_dir) / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(release / name, Path(local_dir) / name)
        return str(local_dir)

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None, tqdm_class=None):
        assert revision == SHA and tqdm_class is not None
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
        if (converted / name).exists():  # the tiny pplx has no processor config
            assert (stored / name).read_bytes() == (converted / name).read_bytes(), name
        else:
            assert not (stored / name).exists(), name


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


def test_a_pplx_download_fetches_only_what_it_loads(remote_pplx, tmp_path):
    check = download.check_repo("perplexity-ai/tiny-pplx")
    assert check.family == "pplx" and check.convertible
    out = download.download(check.repo_id, tmp_path / "full", check.files, check.revision)
    assert sorted(p.name for p in out.iterdir()) == [
        "LICENSE",
        "NOTICE",
        "README.md",
        "config.json",
        "decision_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "readout.safetensors",
        "tokenizer.json",
    ]
    weights = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    assert check.weight_bytes == weights
    assert check.size_bytes == sum(p.stat().st_size for p in out.iterdir())


def test_a_quantized_pplx_download_equals_a_converted_copy(remote_pplx, tmp_path):
    import json

    import mlx_decision
    from mlx_decision.convert import convert
    from mlx_decision.registry import MARKER_FILE

    check = download.check_repo("perplexity-ai/tiny-pplx")
    out = download.download_quantized(check, 4, tmp_path / "models" / "tiny-pplx")
    converted = convert(remote_pplx.release, tmp_path / "converted", bits=4)
    assert_same_model(out, converted)
    for name in ("readout.safetensors", "decision_config.json", "NOTICE"):
        assert (out / name).read_bytes() == (remote_pplx.release / name).read_bytes(), name
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["family"] == "pplx" and marker["revision"] == SHA
    assert marker["quantization"]["bits"] == 4 and marker["vision"] is True
    weights = download._weight_files(remote_pplx.release)
    assert remote_pplx.events == [(name, []) for name in weights]
    assert not (out / "source").exists() and not (out / "chat_template.jinja").exists()
    result = mlx_decision.load(out).decide("refund please", {"q": {"type": "noul"}})
    assert 0 <= result.answers["q"].noul <= 1


def test_a_quantized_h2o_download_streams_its_single_weight_file(remote_h2o, tmp_path):
    import json

    import mlx_decision
    from mlx_decision.convert import convert
    from mlx_decision.registry import MARKER_FILE

    check = download.check_repo("h2oai/tiny-h2o")
    assert check.family == "h2o" and check.convertible
    assert check.model_files == ("model.safetensors",)
    out = download.download_quantized(check, 8, tmp_path / "models" / "tiny-h2o")
    assert remote_h2o.events == [("model.safetensors", [])]
    assert_same_model(out, convert(remote_h2o.release, tmp_path / "converted", bits=8))
    assert sorted(p.name for p in out.iterdir() if not p.name.endswith(".safetensors")) == [
        "LICENSE",
        "config.json",
        "mlx_decision.json",
        "model.safetensors.index.json",
        "serve_config.json",
        "tokenizer.json",
    ]
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["family"] == "h2o" and marker["revision"] == SHA and marker["vision"]
    result = mlx_decision.load(out).decide("refund please", {"q": {"type": "noul"}})
    assert 0 <= result.answers["q"].noul <= 1


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


@pytest.mark.parametrize("client", ["httpx", "httpx2"])
def test_a_lost_connection_is_a_one_line_error(monkeypatch, client):
    module = pytest.importorskip(client)  # huggingface_hub 1.x raises httpx's, 2.x httpx2's

    def check_repo(repo_id):
        raise module.ConnectError("no route")

    monkeypatch.setattr(download, "check_repo", check_repo)
    result = CliRunner().invoke(app, ["download", "Cloudflare/clef-flash", "--yes"])
    assert result.exit_code == 1
    assert "Cloudflare/clef-flash: no route" in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)


# Progress bars, through the installed Hub's own snapshot_download with only
# the file listing and the transfer faked.


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def real_snapshot(monkeypatch, tmp_path):
    """Runs ``snapshot_download``; returns the terminal its bars wrote to."""
    import importlib

    hub = importlib.import_module("huggingface_hub._snapshot_download")
    if not hasattr(hub, "read_tree_cache"):
        pytest.skip("this huggingface_hub lists files another way")
    terminal = Terminal()

    class Eager(download.hf_tqdm):  # draws every update, not at most every 0.1 s
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("mininterval", 0)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(download, "hf_tqdm", Eager)
    failure = {}

    def transfer(*args, **kwargs):
        with kwargs["tqdm_class"](total=10) as bar:
            if "xet" in failure:
                # Xet: bytes received come in early, the file is written at the end.
                bar.update_transfer(6)
                failure["xet"] = terminal.getvalue()
                bar.update_transfer(2)  # less than the file: deduplicated
                bar.update(10)
                return "unused"
            bar.update(4)
            if "error" in failure:
                # A fresh one: the parameter's own would keep its traceback,
                # and with it the bars, alive.
                error = failure["error"]
                raise type(error)(*error.args)
            bar.update(6)
        return "unused"

    monkeypatch.setattr(hub, "read_tree_cache", lambda *a: {"config.json": object()})
    monkeypatch.setattr(hub, "hf_hub_download", transfer)

    def run(tqdm_class, error=None, xet=False):
        # Here, not at setup: pytest sets its own stderr again for the test.
        monkeypatch.setattr(sys, "stderr", terminal)
        if error is not None:
            failure["error"] = error
        if xet:
            failure["xet"] = None
        hub.snapshot_download(
            "example/model", revision="a" * 40, local_dir=tmp_path, tqdm_class=tqdm_class
        )

    run.terminal = terminal
    run.seen = failure
    return run


def collected_output(terminal) -> str:
    """What the bars write when they are garbage collected."""
    import gc

    before = terminal.getvalue()
    gc.collect()
    return terminal.getvalue()[len(before) :]


def test_a_labelled_download_shows_one_bar_and_closes_it(real_snapshot):
    with download.progress("downloading") as bars:
        real_snapshot(bars)
    shown = real_snapshot.terminal.getvalue()
    assert shown.endswith("\n")  # closed: a bar ends its line when it closes
    assert "downloading: 100%" in shown
    # Not the Hub's file count, transfer bar or its renamed descriptions.
    assert "Fetching" not in shown and "Download complete" not in shown
    assert "Reconstruct" not in shown
    assert collected_output(real_snapshot.terminal) == ""


def test_the_bar_follows_the_bytes_received_before_the_file_is_written(real_snapshot):
    with download.progress("downloading") as bars:
        real_snapshot(bars, xet=True)
    assert "downloading:  60%" in real_snapshot.seen["xet"]  # before anything was written
    assert "downloading: 100%" in real_snapshot.terminal.getvalue()


def test_a_quiet_download_shows_nothing(real_snapshot):
    with download.progress() as quiet:
        real_snapshot(quiet)
    assert real_snapshot.terminal.getvalue() == ""
    assert collected_output(real_snapshot.terminal) == ""


@pytest.mark.parametrize("error", [OSError("connection reset"), KeyboardInterrupt()])
def test_bars_close_when_a_download_fails(real_snapshot, error):
    try:
        with download.progress("downloading") as bars:
            real_snapshot(bars, error)
    except type(error):
        pass  # no reference kept: its traceback would keep the bars alive
    else:
        pytest.fail("the download should fail")
    shown = real_snapshot.terminal.getvalue()
    assert "downloading:  40%" in shown and shown.endswith("\n")
    assert collected_output(real_snapshot.terminal) == ""


def test_bars_stay_off_when_the_user_turned_them_off(real_snapshot):
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    disable_progress_bars()
    try:
        with download.progress("downloading") as bars:
            real_snapshot(bars)
    finally:
        enable_progress_bars()
    assert real_snapshot.terminal.getvalue() == ""
