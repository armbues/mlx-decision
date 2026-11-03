"""Fetch models from the Hugging Face Hub, checking first that a family can load them."""

import json
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, constants, hf_hub_download, snapshot_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    NotASafetensorsRepoError,
    SafetensorsParsingError,
)
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    filter_repo_objects,
)
from huggingface_hub.utils import tqdm as hf_tqdm

from .hub import stored_snapshot
from .memory import DTYPE_BYTES, MARGIN, QUANTIZED_BITS, element_count, quantized_size, working_set
from .registry import detect_family, get_family, resolve


@contextmanager
def progress(
    label: str | None = None,
    on_bytes: Callable[[int], None] | None = None,
    draw: bool = True,
) -> Iterator[type]:
    """A ``tqdm_class`` for the Hub's download functions whose bars all close on exit.

    The Hub leaves some of its bars open after a download; they close when
    garbage collected and draw their last line again then, over whatever the
    terminal shows by that time (a notice, a prompt). With ``label``, one bar
    of bytes is shown under it and the Hub's others (files counted, bytes
    over the network) are hidden; without, none is shown. Being the Hub's own
    class underneath, it keeps the Hub's rules: no bars without a terminal,
    none when the user turned them off.

    The shown bar counts the bytes written or, when further along, the bytes
    received: a Xet download writes a large file in a few big pieces (one
    update in 15 s, measured) while it reports the bytes received ten times
    a second. Written bytes reach the total at the end either way.

    ``on_bytes(n)`` hears each advance of the shown bar (from the Hub's
    threads); ``draw=False`` counts without drawing it.
    """
    bars: list[hf_tqdm] = []
    shown: list[hf_tqdm] = []  # the last is on screen
    lock = threading.Lock()

    class Bar(hf_tqdm):
        def __init__(self, *args, **kwargs):
            name = kwargs.get("name") or ""
            self.transfer = name.endswith(".transfer")
            self.label = label if kwargs.get("unit") == "B" and not self.transfer else None
            self.written = self.received = self.counted = 0
            if self.label is None or not draw:
                kwargs["disable"] = True
            if self.label is not None:
                # One line: the Hub's hidden bars must not push it down a line.
                kwargs.update(desc=self.label, position=0)
            super().__init__(*args, **kwargs)
            bars.append(self)
            if self.label is not None:
                shown.append(self)

        def update(self, n=1):
            if self.transfer:  # bytes received, for the bar on screen
                if shown:
                    shown[-1].count(received=n or 0)
            elif self.label is not None:
                self.count(written=n or 0)
            else:
                super().update(n)

        def count(self, written=0, received=0):
            with lock:
                self.written += written
                self.received += received
                ahead = max(self.written, self.received) - self.counted
                if ahead > 0:
                    self.counted += ahead
                    super().update(ahead)
                    if on_bytes is not None:
                        on_bytes(ahead)

        # The Hub renames its bars as it goes ("Reconstructing ..."); a shown
        # bar keeps the label.
        def set_description(self, desc=None, refresh=True):
            if self.label is None:
                super().set_description(desc, refresh)

        def set_description_str(self, desc=None, refresh=True):
            if self.label is None:
                super().set_description_str(desc, refresh)

        def close(self):
            if self.label is not None and not self.counted:
                self.leave = False  # nothing fetched (all there already): no empty bar left
            super().close()

    try:
        yield Bar
    finally:
        for bar in bars:
            bar.close()


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
    quantizable_bytes: int | None = None  # of the weights the converter quantizes; None: all
    # The backbone's weight files (model*.safetensors) among them, for a
    # repository without an index (a single model.safetensors).
    model_files: tuple[str, ...] = ()

    @property
    def convertible(self) -> bool:
        """Whether the family can store the model quantized."""
        return self.family is not None and get_family(self.family).converter is not None

    def fits(self, bits: int = 0, budget: int | None = None) -> bool:
        """Whether the model fits in memory at full size (``bits`` 0) or
        stored with ``bits`` (the fit check's rule)."""
        weights = self.quantized_weight_bytes(bits) if bits else self.weight_bytes
        return weights * MARGIN <= (working_set() if budget is None else budget)

    def fitting_bits(self) -> int | None:
        """The largest size that fits: 0 for full size, else the bits; None when none fits."""
        budget = working_set()
        return next((bits for bits in (0, *QUANTIZED_BITS) if self.fits(bits, budget)), None)

    def quantized_weight_bytes(self, bits: int) -> int:
        quantizable = (
            self.weight_bytes if self.quantizable_bytes is None else self.quantizable_bytes
        )
        return self.weight_bytes - quantizable + quantized_size(quantizable, bits)

    def quantized_bytes(self, bits: int) -> int:
        """About the size of the model stored with ``bits``."""
        return self.quantized_weight_bytes(bits) + self.size_bytes - self.weight_bytes


def check_repo(repo_id: str) -> RepoCheck:
    """The download size, and the family that would load the model.

    Only the repository's JSON files are fetched for this (configs, the
    marker of a converted model); the weights are not touched. The size
    counts the files the family needs, or the whole repository. A known
    model pinned to a checked commit is checked (and downloaded) there,
    others at the Hub's latest.
    """
    from .registry import pinned_revision

    pinned = pinned_revision(repo_id)
    info = HfApi().model_info(
        repo_id, files_metadata=True, **({"revision": pinned} if pinned else {})
    )
    siblings = info.siblings or []
    revision = getattr(info, "sha", None)
    with tempfile.TemporaryDirectory() as folder, progress() as quiet:
        snapshot_download(
            repo_id,
            allow_patterns=["*.json"],
            local_dir=folder,
            revision=revision,
            tqdm_class=quiet,
        )
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
    model_files = tuple(
        sorted(s.rfilename for s in fetched if fnmatch(s.rfilename, "model*.safetensors"))
    )
    quantizable = None
    if family.converter is not None and family.quantizable is not None:
        quantizable = _quantizable_on_hub(repo_id, revision, family.quantizable)
    return RepoCheck(
        repo_id,
        sum(sibling.size or 0 for sibling in fetched),
        family.name,
        files=family.files,
        revision=revision,
        weight_bytes=sum(weights),
        largest_weight_file=max(weights, default=0),
        quantizable_bytes=quantizable,
        model_files=model_files,
    )


def _quantizable_on_hub(repo_id: str, revision: str | None, rule) -> int | None:
    """Bytes of the tensors ``rule`` picks, from the weight files' headers on
    the Hub (a few small range requests); None when they cannot be read."""
    quiet = are_progress_bars_disabled()
    disable_progress_bars()  # its bar has no name to turn off on its own
    try:
        metadata = HfApi().get_safetensors_metadata(repo_id, revision=revision)
    except (HfHubHTTPError, NotASafetensorsRepoError, SafetensorsParsingError, OSError):
        return None
    finally:
        if not quiet:
            enable_progress_bars()
    return sum(
        element_count(tensor.shape) * DTYPE_BYTES[tensor.dtype]
        for file in metadata.files_metadata.values()
        for name, tensor in file.tensors.items()
        if rule(name, tensor.shape)
    )


def download(
    repo_id: str,
    local_dir: Path | None = None,
    files: tuple[str, ...] | None = None,
    revision: str | None = None,
    on_bytes: Callable[[int], None] | None = None,
    draw: bool = True,
) -> Path:
    """Download ``files`` (Hub patterns; None: the whole repository) into the
    Hub cache, or into ``local_dir``.

    A new (or empty) ``local_dir`` is written as ``.NAME.partial`` beside it
    and renamed when complete, so a stopped download leaves no half model
    there; an existing one is updated in place. ``on_bytes`` and ``draw`` as
    for ``progress``.
    """
    if local_dir is None and (stored := stored_snapshot(repo_id)) is not None:
        # The full-size files would be written into the quantized folder.
        raise FileExistsError(f"already stored quantized in the Hugging Face cache: {stored}")
    patterns = list(files) if files is not None else None
    target = Path(local_dir).expanduser() if local_dir is not None else None
    into = _partial_target(target)
    if into is not None:
        shutil.rmtree(into, ignore_errors=True)
    try:
        with progress("downloading", on_bytes, draw) as bars:
            path = snapshot_download(
                repo_id,
                local_dir=into or target,
                allow_patterns=patterns,
                revision=revision,
                tqdm_class=bars,
            )
    except BaseException:
        if into is not None:
            shutil.rmtree(into, ignore_errors=True)
        raise
    if into is None:
        return Path(path)
    if target.exists():
        target.rmdir()  # empty, checked before
    target.parent.mkdir(parents=True, exist_ok=True)
    into.rename(target)
    return target


# Beside a model downloaded into a folder: which repo (and commit) it holds,
# so another repo of the same short name is not taken for it.
SOURCE_FILE = ".download.json"


def record_source(folder: Path, repo_id: str, revision: str | None) -> None:
    """Note ``repo_id`` in ``folder`` (best effort: a model loads without it)."""
    import json

    body = json.dumps({"repo": repo_id, "revision": revision}) + "\n"
    with suppress(OSError):
        (folder / SOURCE_FILE).write_text(body)


def recorded_repo(folder: Path) -> str | None:
    """The repo a folder was downloaded from, as ``record_source`` noted it."""
    import json

    try:
        repo = json.loads((folder / SOURCE_FILE).read_text()).get("repo")
    except (OSError, ValueError, AttributeError):
        return None
    return repo if isinstance(repo, str) else None


def _partial_target(target: Path | None) -> Path | None:
    """Where a download into ``target`` goes first: ``.NAME.partial`` beside a
    new or empty folder; None for the cache or a folder with files (updated
    in place)."""
    if target is None or (target.exists() and any(target.iterdir())):
        return None
    return target.parent / f".{target.name}.partial"


def leftovers(
    repo_id: str,
    local_dir: Path | None = None,
    bits: int | None = None,
    revision: str | None = None,
) -> list[Path]:
    """What a download stopped mid-way leaves behind that has to go: its
    working folders, and its folder in the Hub cache when this download
    creates it, so no half model is listed. The Hub's unfinished files in
    a folder that was there before, and in the cache's shared blob store
    (which other downloads may be writing to at the same time), stay for
    the Hub to resume. Taken before the download starts."""
    if local_dir is None:
        cache = Path(constants.HF_HUB_CACHE)
        repo = cache / repo_folder_name(repo_id=repo_id, repo_type="model")
        found = [] if repo.exists() else [repo, cache / ".locks" / repo.name]
        if bits is not None and revision is not None:
            found += [repo / f".{revision}.download", repo / f".{revision}.partial"]
        return found
    target = Path(local_dir).expanduser()
    if bits is not None:
        return [target.parent / f".{target.name}.{end}" for end in ("download", "partial")]
    partial = _partial_target(target)
    return [partial] if partial is not None else []


def download_quantized(
    check: RepoCheck,
    bits: int,
    local_dir: Path | None = None,
    on_file: Callable[[int, int, str], None] | None = None,
    on_bytes: Callable[[int], None] | None = None,
    draw: bool = True,
) -> Path:
    """Store the model quantized with ``bits``, fetching one weight file at a time.

    Each weight file is downloaded just before the converter reads it and
    deleted after, so the full-size weights are never all on disk. The
    model lands in ``local_dir``, or in the Hub cache as the snapshot of
    the checked revision (plain files, no blobs), where
    ``resolve_model_path`` finds it by its marker. ``on_file(i, n, name)``
    is called before each weight file is fetched; ``on_bytes`` and ``draw``
    as for ``progress``.
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
        with progress() as quiet:
            snapshot_download(
                repo_id,
                revision=revision,
                local_dir=source,
                allow_patterns=["*.json"],
                tqdm_class=quiet,
            )
        names = _weight_files(source, check.model_files)
        with progress("other files", on_bytes, draw) as bars:
            snapshot_download(
                repo_id,
                revision=revision,
                local_dir=source,
                allow_patterns=list(check.files) if check.files is not None else None,
                ignore_patterns=names,
                tqdm_class=bars,
            )

        def fetch() -> Iterator[Path]:
            for number, name in enumerate(names, 1):
                if on_file is not None:
                    on_file(number, len(names), name)
                with progress("downloading", on_bytes, draw) as bars:
                    file = hf_hub_download(
                        repo_id, name, revision=revision, local_dir=source, tqdm_class=bars
                    )
                yield Path(file)
                Path(file).unlink()

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


def _weight_files(folder: Path, model_files: tuple[str, ...] = ()) -> list[str]:
    """The weight files the index lists, in its order; without an index, the
    repository's ``model*.safetensors`` (``model_files``, from the check)."""
    index = folder / "model.safetensors.index.json"
    if index.exists():
        return list(dict.fromkeys(json.loads(index.read_text())["weight_map"].values()))
    if model_files:
        return list(model_files)
    raise ValueError(
        "the repository has neither model.safetensors.index.json nor model*.safetensors "
        "files, so it cannot be stored quantized; download it at full size and use convert"
    )


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
