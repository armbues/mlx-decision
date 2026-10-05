"""Fetch models from the Hugging Face Hub, checking first that a family can load them."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .registry import detect_family


@dataclass(frozen=True)
class RepoCheck:
    repo_id: str
    size_bytes: int
    family: str | None  # None: no supported family recognises the files
    reason: str = ""


def check_repo(repo_id: str) -> RepoCheck:
    """The download size, and the family that would load the model.

    Only the repository's JSON files are fetched for this (configs, the
    marker of a converted model); the weights are not touched.
    """
    info = HfApi().model_info(repo_id, files_metadata=True)
    size = sum(sibling.size or 0 for sibling in info.siblings or [])
    with tempfile.TemporaryDirectory() as folder:
        snapshot_download(repo_id, allow_patterns=["*.json"], local_dir=folder)
        try:
            family = detect_family(Path(folder)).name
        except ValueError as error:
            return RepoCheck(repo_id, size, None, str(error).split(": ", 1)[-1])
    return RepoCheck(repo_id, size, family)


def download(repo_id: str, local_dir: Path | None = None) -> Path:
    """Download the whole repository into the Hub cache, or into ``local_dir``."""
    return Path(snapshot_download(repo_id, local_dir=local_dir))


def format_size(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.0f} MB"
