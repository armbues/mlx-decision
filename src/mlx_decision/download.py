"""Fetch models from the Hugging Face Hub, checking first that a family can load them."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

from .registry import detect_family


@dataclass(frozen=True)
class RepoCheck:
    repo_id: str
    size_bytes: int  # of the files a download fetches
    family: str | None  # None: no supported family recognises the files
    reason: str = ""
    files: tuple[str, ...] | None = None  # Hub patterns; None: everything


def check_repo(repo_id: str) -> RepoCheck:
    """The download size, and the family that would load the model.

    Only the repository's JSON files are fetched for this (configs, the
    marker of a converted model); the weights are not touched. The size
    counts the files the family needs, or the whole repository.
    """
    info = HfApi().model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    with tempfile.TemporaryDirectory() as folder:
        snapshot_download(repo_id, allow_patterns=["*.json"], local_dir=folder)
        try:
            family = detect_family(Path(folder))
        except ValueError as error:
            size = sum(sibling.size or 0 for sibling in siblings)
            return RepoCheck(repo_id, size, None, str(error).split(": ", 1)[-1])
    fetched = siblings
    if family.files is not None:
        fetched = filter_repo_objects(
            siblings, allow_patterns=family.files, key=lambda sibling: sibling.rfilename
        )
    size = sum(sibling.size or 0 for sibling in fetched)
    return RepoCheck(repo_id, size, family.name, files=family.files)


def download(
    repo_id: str, local_dir: Path | None = None, files: tuple[str, ...] | None = None
) -> Path:
    """Download ``files`` (Hub patterns; None: the whole repository) into the
    Hub cache, or into ``local_dir``."""
    patterns = list(files) if files is not None else None
    return Path(snapshot_download(repo_id, local_dir=local_dir, allow_patterns=patterns))


def format_size(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.0f} MB"
