"""Shared fixtures. Tests that need model weights are skipped when none are found.

Weights are looked up in the folder named by ``MLX_DECISION_MODELS`` (one
sub-folder per model, e.g. ``clef-flash``), then in the Hugging Face cache.
"""

import os
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

CLEF_FLASH = ("clef-flash", "Cloudflare/clef-flash")
LAYA = ("laya", "convaiinnovations/laya")
LAYA_MULTILINGUAL = ("laya-multilingual", "convaiinnovations/laya-multilingual")
LAYA_TYPED = ("laya-typed-decisions", "convaiinnovations/laya-typed-decisions")
JULIA = ("Julia-1", "SupersonicLabs/Julia-1")


@pytest.fixture
def working_set(monkeypatch):
    """Set the working set the memory check compares against."""
    import mlx_decision.memory as memory

    def set_to(size: int) -> None:
        monkeypatch.setattr(memory, "working_set", lambda: size)

    return set_to


def find_model(name: str, repo_id: str) -> Path | None:
    root = os.environ.get("MLX_DECISION_MODELS")
    if root and (Path(root) / name).is_dir():
        return Path(root) / name
    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None


@pytest.fixture(scope="session")
def clef_path() -> Path:
    path = find_model(*CLEF_FLASH)
    if path is None:
        pytest.skip("clef-flash weights not found (set MLX_DECISION_MODELS)")
    return path


@pytest.fixture(scope="session")
def laya_path() -> Path:
    path = find_model(*LAYA)
    if path is None:
        pytest.skip("laya weights not found (set MLX_DECISION_MODELS)")
    return path


@pytest.fixture(scope="session")
def laya_multilingual_path() -> Path:
    path = find_model(*LAYA_MULTILINGUAL)
    if path is None:
        pytest.skip("laya-multilingual weights not found (set MLX_DECISION_MODELS)")
    return path


@pytest.fixture(scope="session")
def laya_typed_path() -> Path:
    path = find_model(*LAYA_TYPED)
    if path is None:
        pytest.skip("laya-typed-decisions weights not found (set MLX_DECISION_MODELS)")
    return path


@pytest.fixture(scope="session")
def julia_path() -> Path:
    path = find_model(*JULIA)
    if path is None:
        pytest.skip("Julia-1 weights not found (set MLX_DECISION_MODELS)")
    return path


@pytest.fixture(scope="session")
def clef(clef_path: Path):
    """clef-flash, loaded once for every test that needs it (19 GB)."""
    import mlx_decision

    # The parity reference has no image cap beyond the processor's own.
    return mlx_decision.load(clef_path, max_image_pixels=None)


@pytest.fixture
def fake_model_path(tmp_path: Path) -> Path:
    from fake_backend import write_model

    return write_model(tmp_path / "fake-model")


@pytest.fixture
def fake_model(fake_model_path: Path):
    import mlx_decision

    return mlx_decision.load(fake_model_path)
