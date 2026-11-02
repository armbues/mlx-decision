"""Deleting a model from the Models page: three kinds.

- "studio": a model the Studio made, in the workspace's ``models/``.
- "cache": a repo in the Hugging Face cache, every revision of it, through
  ``huggingface_hub``'s cache API (other tools share the cache; a blob
  another repo still uses stays).
- "models": a model folder in the config's ``models_dir``; a link there
  loses only the link.

A model named itself (with ``-m`` or in the config's ``models``) is never
deleted, nor anything elsewhere. A model is refused while it is the default,
while fitted models are made from it, while a job that has not ended names
it or while a batch keeps it loaded. A loaded model is unloaded first. The
checks, the unload and taking it out of the pool run on the GPU worker, so
no request loads it in between; a folder is moved aside there and removed
afterwards, a cache repo removed afterwards (a rescan lists it again if that
fails). The catalog message that follows tells every tab.
"""

import asyncio
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from ..config import Config
from ..gpu import Gpu
from ..hub import snapshot_revision
from ..pool import ModelSpec
from .catalog import Catalog
from .jobs import Jobs

logger = logging.getLogger(__name__)


class RemovalError(Exception):
    """A model that cannot be deleted: ``status`` 403 (not the Studio's), 404
    (unknown) or 409 (in use)."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class ModelRemoval:
    """``delete(name)`` for the Models page and a fit's delete.

    ``catalog`` tells which references were named (as opposed to added by a
    download); ``config`` gives the config file as last read (its ``models``
    and ``models_dir``); ``rescan`` looks again after a cache repo that could
    not be removed.
    """

    def __init__(
        self,
        gpu: Gpu,
        jobs: Jobs,
        models_root: Path,
        on_removed: Callable[[list[str], list[str]], None] = lambda added, removed: None,
        catalog: Catalog | None = None,
        config: Callable[[], Config | None] = lambda: None,
        rescan: Callable[[], None] = lambda: None,
    ):
        self.gpu = gpu
        self.jobs = jobs
        self.models_root = models_root
        self.on_removed = on_removed
        self.catalog = catalog
        self.config = config
        self.rescan = rescan

    def in_workspace(self, path: Path) -> bool:
        """Whether ``path`` is a model folder in the workspace's ``models/``."""
        try:
            return path.resolve().parent == self.models_root.resolve()
        except OSError:
            return False

    def kind(self, name: str) -> str | None:
        """How model ``name`` is deleted: "studio", "cache" or "models"; None
        when it may not be (unknown, named itself, or elsewhere). Only its
        place counts here, not whether it is in use."""
        spec = self.gpu.pool.specs.get(name)
        return None if spec is None else self._kind(spec)[0]

    def _kind(self, spec: ModelSpec) -> tuple[str | None, str]:
        """The kind, and why a model without one stays."""
        path = spec.path
        if self.in_workspace(path):
            return "studio", ""
        cached = _cache_repo(path)
        if self._named(spec, cached[0] if cached else None):
            where = "the Hugging Face cache" if cached else "Finder"
            return None, (
                f"{spec.name} was named for the Studio (with -m or in the config file's "
                f"models): delete it in {where} if you want it gone"
            )
        if cached is not None:
            return "cache", ""
        config = self.config()
        folder = config.models_folder if config is not None else None
        if folder is not None and _parent_is(path, folder):
            return "models", ""
        return None, (
            f"{spec.name} is not in models_dir or the Hugging Face cache: "
            "delete it in Finder if you want it gone"
        )

    def _named(self, spec: ModelSpec, repo: str | None) -> bool:
        """Whether a reference names the model itself (a folder of models
        holding it does not): ``-m``'s or the config's ``models``."""
        references = self.catalog.named() if self.catalog is not None else []
        config = self.config()
        if config is not None:
            references += [str(reference) for reference in config.models]
        target = _resolved(spec.path)
        for reference in references:
            if repo is not None and reference == repo:
                return True
            path = Path(reference).expanduser()
            if path.exists() and _resolved(path) == target:
                return True
        return False

    def name_of(self, path: Path) -> str | None:
        """The pool's name for the model in ``path``, if it lists one."""
        try:
            wanted = path.resolve()
        except OSError:
            return None
        for name, spec in self.gpu.pool.specs.items():
            if _resolved(spec.path) == wanted:
                return name
        return None

    async def delete(self, name: str) -> str:
        """Delete model ``name`` (unloaded first); returns its folder's path.
        Raises ``RemovalError`` and ``Busy`` (a GPU job, or a batch's pin)."""
        busy = self._used_by_job(name)
        if busy is not None:
            raise RemovalError(f"{busy} uses {name}: delete it when the job has ended")
        spec = self.gpu.pool.specs.get(name)
        cached = _cache_repo(spec.path) if spec is not None else None
        revisions = None
        if cached is not None and self._kind(spec)[0] == "cache":
            # Read before anything changes: a repo the cache API cannot read stays.
            revisions = await asyncio.to_thread(_cached_revisions, cached[0])
        path, aside = await self.gpu.run(self._take_out, name, revisions is not None)
        self.on_removed([], [name])
        if revisions is not None:
            await self._delete_cached(cached[0], revisions)
        elif aside is not None:
            await asyncio.to_thread(shutil.rmtree, aside, ignore_errors=True)
        logger.info("deleted the model %s (%s)", name, path)
        return str(path)

    def _take_out(self, name: str, cache: bool = False) -> tuple[Path, Path | None]:
        """On the worker: check, unload, take it out of the pool; a folder is
        moved aside (returned), a link removed, a cache repo left for later."""
        pool = self.gpu.pool
        spec = pool.specs.get(name)
        if spec is None:
            raise RemovalError(f"no model is named {name}", status=404)
        kind, reason = self._kind(spec)
        if kind is None or (kind == "cache") != cache:
            raise RemovalError(reason or f"{name} cannot be deleted here", status=403)
        if name == pool.default:
            raise RemovalError(f"{name} is the default model: choose another default first")
        path = spec.path
        goes = _resolved(path)
        if kind == "cache":
            goes = goes.parent.parent  # the repo's folder: every revision
        children = [
            other
            for other, item in pool.specs.items()
            if item.fit is not None and _resolved(item.fit.base).is_relative_to(goes)
        ]
        if children:
            raise RemovalError(
                f"fitted models are made from {name} ({', '.join(children)}): delete them first"
            )
        others = [
            other
            for other, item in pool.specs.items()
            if kind == "cache" and other != name and _resolved(item.path).is_relative_to(goes)
        ]
        if others:  # another revision of the same repo, named itself
            raise RemovalError(f"{', '.join(others)} is in the same cached repo: it would go too")
        pool.unload(name)  # Busy while a batch keeps it loaded
        aside = None
        if kind == "models" and path.is_symlink():
            path.unlink()  # the link only; first, so one that cannot go stays listed
        elif kind != "cache":
            aside = path.with_name(f".{path.name}.{os.getpid()}.removing")
            path.rename(aside)  # first: a folder that cannot move stays listed
        pool.remove(name)
        return path, aside

    async def _delete_cached(self, repo: str, revisions: list[str]) -> None:
        """Every revision of ``repo`` out of the cache. Raises ``RemovalError``
        (500) when some of it stays, after asking for a rescan."""
        try:
            left = await asyncio.to_thread(_delete_revisions, repo, revisions)
        except Exception as error:  # the cache API's own errors, or the disk's
            logger.warning("deleting %s from the Hugging Face cache: %s", repo, error)
            left = True
        if left:
            self.rescan()  # listed again while it is still complete
            raise RemovalError(
                f"{repo} is still in the Hugging Face cache, at least in part (see the "
                f"Studio's log): hf cache rm model/{repo} removes the rest",
                status=500,
            )

    async def delete_all(self) -> list[str]:
        """Delete every model the Studio made, and anything else in its
        ``models/`` folder; returns the models' names. Refused as a whole
        (nothing deleted) when one of them cannot go: the default, a job
        that has not ended naming one or writing there, a fitted model
        elsewhere made from one, a batch keeping one loaded (``Busy``)."""
        for job in self.jobs.jobs.values():
            if job.state != "ended" and job.output is not None and self.in_workspace(job.output):
                raise RemovalError(f"{job.title} writes into the models folder: wait until it ends")
        names = [name for name, spec in self.gpu.pool.specs.items() if self.in_workspace(spec.path)]
        for name in names:
            busy = self._used_by_job(name)
            if busy is not None:
                raise RemovalError(f"{busy} uses {name}: clear the folder when the job has ended")
        asides = await self.gpu.run(self._take_out_all, names)
        # Links to folders elsewhere stay, as Delete refuses them (not made here).
        others = [
            path
            for path in _entries(self.models_root)
            if not path.name.endswith(".removing") and not path.is_symlink()
        ]
        await asyncio.to_thread(_remove_all, [*asides, *others])
        if names:
            logger.info("deleted the models %s", ", ".join(names))
            self.on_removed([], names)
        return names

    def _take_out_all(self, names: list[str]) -> list[Path]:
        pool = self.gpu.pool
        paths = {_resolved(pool.specs[name].path) for name in names}
        if pool.default in names:
            raise RemovalError(f"{pool.default} is the default model: choose another default first")
        for other, item in pool.specs.items():
            if other not in names and item.fit is not None and _resolved(item.fit.base) in paths:
                raise RemovalError(f"{other} is fitted on a model in the folder: delete it first")
        for name in names:
            holder = pool.pinned_by(name)
            if holder is not None:
                from ..gpu import Busy

                raise Busy(holder)
        for name in names:  # all first: a fitted model holds its base's weights
            pool.unload(name)
        moved: list[tuple[Path, Path]] = []
        try:  # every folder aside first; one that cannot move puts the others back
            for name in names:
                path = pool.specs[name].path
                aside = path.with_name(f".{path.name}.{os.getpid()}.removing")
                path.rename(aside)
                moved.append((path, aside))
        except OSError:
            for path, aside in reversed(moved):
                aside.rename(path)
            raise
        for name in names:
            pool.remove(name)
        return [aside for _, aside in moved]

    def _used_by_job(self, name: str) -> str | None:
        """The title of a job that has not ended and names the model."""
        spec = self.gpu.pool.specs.get(name)
        wanted = {name}
        if spec is not None:
            wanted.add(str(spec.path))
            cached = snapshot_revision(spec.path)
            if cached is not None:
                wanted.add(cached[0])  # a download of the repo, or -m REPO
        for job in self.jobs.jobs.values():
            if job.state == "ended":
                continue
            if wanted & set(job.argv):
                return job.title
        return None


def _entries(folder: Path) -> list[Path]:
    try:
        return list(folder.iterdir())
    except OSError:
        return []


def _remove_all(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _cache_root() -> Path:
    from huggingface_hub import constants

    return _resolved(Path(constants.HF_HUB_CACHE).expanduser())


def _cache_repo(path: Path) -> tuple[str, str] | None:
    """The repo id and commit of a snapshot in the Hugging Face cache this
    process uses (``.../models--org--name/snapshots/<commit>``), else None."""
    cached = snapshot_revision(path)
    parents = _resolved(path).parents
    if cached is None or len(parents) < 3:
        return None
    return cached if parents[2] == _cache_root() else None


def _cached_revisions(repo: str) -> list[str]:
    """The commits of ``repo`` the cache API lists; ``RemovalError`` (409) when
    it cannot read the repo, so nothing is deleted."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(_cache_root())
    for item in info.repos:
        if item.repo_type == "model" and item.repo_id == repo:
            return sorted(revision.commit_hash for revision in item.revisions)
    reasons = [str(warning) for warning in info.warnings if repo.replace("/", "--") in str(warning)]
    raise RemovalError(
        f"the Hugging Face cache cannot read {repo}"
        + (f": {reasons[0]}" if reasons else "")
        + "; hf cache ls tells more"
    )


def _delete_revisions(repo: str, revisions: list[str]) -> bool:
    """Delete ``revisions`` of ``repo`` (all of them) through the cache API;
    whether anything of the repo is left."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(_cache_root())
    strategy = info.delete_revisions(*revisions)
    strategy.execute()
    return any(path.exists() for path in strategy.repos | strategy.snapshots)


def _parent_is(path: Path, folder: Path) -> bool:
    """Whether ``path`` (a link left as it is) lies right in ``folder``."""
    try:
        return _resolved(Path(path).absolute().parent) == _resolved(folder.expanduser())
    except OSError:
        return False


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)
