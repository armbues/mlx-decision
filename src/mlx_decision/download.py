"""Fetch models from the Hugging Face Hub, checking first that a family can load them."""

import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import filter_repo_objects

from .hub import stored_snapshot
from .memory import MARGIN, working_set
from .registry import detect_family, get_family, resolve

QUANTIZED_BITS = (8, 4)
# Affine quantization with groups of 64 adds a bf16 scale and bias per group.
GROUP_OVERHEAD_BITS = 0.5


@dataclass(frozen=True)
class RepoCheck:
    repo_id: str
    size_bytes: int  # of the files a download fetches
    family: str | None  # None: no supported family recognises the files
    reason: str = ""
    files: tuple[str, ...] | None = None  # Hub patterns; None: everything
    revision: str | None = None  # the commit the check saw
    weight_bytes: int = 0  # of the *.safetensors files among them
    largest_weight_file: int = 0

    @property
    def convertible(self) -> bool:
        """Whether the family can store the model quantized."""
        return self.family is not None and get_family(self.family).converter is not None

    def fits(self, budget: int | None = None) -> bool:
        """Whether the model fits in memory at full size (the fit check's rule)."""
        return self.weight_bytes * MARGIN <= (working_set() if budget is None else budget)

    def quantized_bytes(self, bits: int) -> int:
        """About the size of the model stored with ``bits``: every weight file
        counted as quantized, which the small unquantized parts barely change."""
        weights = self.weight_bytes * (bits + GROUP_OVERHEAD_BITS) / 16
        return round(weights) + self.size_bytes - self.weight_bytes


def check_repo(repo_id: str) -> RepoCheck:
    """The download size, and the family that would load the model.

    Only the repository's JSON files are fetched for this (configs, the
    marker of a converted model); the weights are not touched. The size
    counts the files the family needs, or the whole repository.
    """
    info = HfApi().model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    revision = getattr(info, "sha", None)
    with tempfile.TemporaryDirectory() as folder:
        snapshot_download(repo_id, allow_patterns=["*.json"], local_dir=folder, revision=revision)
        try:
            family = detect_family(Path(folder))
        except ValueError as error:
            size = sum(sibling.size or 0 for sibling in siblings)
            return RepoCheck(repo_id, size, None, str(error).split(": ", 1)[-1], revision=revision)
    fetched = siblings
    if family.files is not None:
        fetched = list(
            filter_repo_objects(
                siblings, allow_patterns=family.files, key=lambda sibling: sibling.rfilename
            )
        )
    weights = [s.size or 0 for s in fetched if s.rfilename.endswith(".safetensors")]
    return RepoCheck(
        repo_id,
        sum(sibling.size or 0 for sibling in fetched),
        family.name,
        files=family.files,
        revision=revision,
        weight_bytes=sum(weights),
        largest_weight_file=max(weights, default=0),
    )


def download(
    repo_id: str,
    local_dir: Path | None = None,
    files: tuple[str, ...] | None = None,
    revision: str | None = None,
) -> Path:
    """Download ``files`` (Hub patterns; None: the whole repository) into the
    Hub cache, or into ``local_dir``."""
    if local_dir is None and (stored := stored_snapshot(repo_id)) is not None:
        # The full-size files would be written into the quantized folder.
        raise FileExistsError(f"already stored quantized in the Hugging Face cache: {stored}")
    patterns = list(files) if files is not None else None
    path = snapshot_download(
        repo_id, local_dir=local_dir, allow_patterns=patterns, revision=revision
    )
    return Path(path)


def download_quantized(
    check: RepoCheck,
    bits: int,
    local_dir: Path | None = None,
    on_file: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Store the model quantized with ``bits``, fetching one weight file at a time.

    Each weight file is downloaded just before the converter reads it and
    deleted after, so the full-size weights are never all on disk. The
    model lands in ``local_dir``, or in the Hub cache as the snapshot of
    the checked revision (plain files, no blobs), where
    ``resolve_model_path`` finds it by its marker. ``on_file(i, n, name)``
    is called before each weight file is fetched.
    """
    if not check.convertible:
        raise ValueError(f"{check.family} models cannot be stored quantized; they load as they are")
    if bits not in QUANTIZED_BITS:
        raise ValueError(f"bits must be one of {', '.join(map(str, QUANTIZED_BITS))}")
    repo_id, revision = check.repo_id, check.revision
    if local_dir is None:
        repo = Path(constants.HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")
        destination, work = repo / "snapshots" / revision, repo
    else:
        destination = Path(local_dir).expanduser()
        work = destination.parent
    if destination.exists() and any(destination.iterdir()):
        where = "the Hugging Face cache already has" if local_dir is None else "not empty:"
        raise FileExistsError(f"{where} {destination}")
    _check_disk(work, check.quantized_bytes(bits) + check.largest_weight_file)

    # Hidden working folders next to the destination, on the same disk, so
    # the finished model is renamed into place.
    source = work / f".{destination.name}.download"
    partial = work / f".{destination.name}.partial"
    for folder in (source, partial):
        shutil.rmtree(folder, ignore_errors=True)
    try:
        snapshot_download(repo_id, revision=revision, local_dir=source, allow_patterns=["*.json"])
        names = _weight_files(source)
        snapshot_download(
            repo_id,
            revision=revision,
            local_dir=source,
            allow_patterns=list(check.files) if check.files is not None else None,
            ignore_patterns=names,
        )

        def fetch() -> Iterator[Path]:
            for number, name in enumerate(names, 1):
                if on_file is not None:
                    on_file(number, len(names), name)
                file = Path(hf_hub_download(repo_id, name, revision=revision, local_dir=source))
                yield file
                file.unlink()

        converter = resolve(get_family(check.family).converter)
        converter(source, partial, source=repo_id, revision=revision, bits=bits, shards=fetch())
        if destination.exists():
            destination.rmdir()  # empty, checked above
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial.rename(destination)
        if local_dir is None:
            # The cache's pointer from the branch to the commit, as a download writes it.
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs" / "main").write_text(revision)
    finally:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(partial, ignore_errors=True)
    return destination


def _weight_files(folder: Path) -> list[str]:
    """The weight files the index lists, in its order."""
    index = folder / "model.safetensors.index.json"
    if not index.exists():
        raise ValueError(
            "the repository has no model.safetensors.index.json, so it cannot be stored "
            "quantized; download it at full size and use convert"
        )
    return list(dict.fromkeys(json.loads(index.read_text())["weight_map"].values()))


def _check_disk(folder: Path, needed: int) -> None:
    existing = folder
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < needed:
        raise OSError(
            f"needs about {format_size(needed)} of free disk space "
            f"(the quantized model plus one full-size weight file); {format_size(free)} free"
        )


def format_size(size: int) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e8 else f"{size / 1e6:.0f} MB"
