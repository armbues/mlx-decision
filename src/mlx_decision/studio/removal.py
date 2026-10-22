"""Deleting a model the Studio made: a converted or fitted model in the
workspace's ``models/``.

Downloaded models (the Hugging Face cache, ``models_dir``, ``-m`` folders) are
never deleted here: other tools may use them. A model is refused while it is
the default, while fitted models are made from it, while a job that has not
ended names it or while a batch keeps it loaded. A loaded model is unloaded
first. The checks, the unload and taking it out of the pool run on the GPU
worker, so no request loads it in between; its folder is moved aside there
and removed afterwards. The catalog message that follows tells every tab.
"""

import asyncio
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from ..gpu import Gpu
from .jobs import Jobs

logger = logging.getLogger(__name__)


class RemovalError(Exception):
    """A model that cannot be deleted: ``status`` 403 (not the Studio's), 404
    (unknown) or 409 (in use)."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


class ModelRemoval:
    """``delete(name)`` for the Models page and a fit's delete."""

    def __init__(
        self,
        gpu: Gpu,
        jobs: Jobs,
        models_root: Path,
        on_removed: Callable[[list[str], list[str]], None] = lambda added, removed: None,
    ):
        self.gpu = gpu
        self.jobs = jobs
        self.models_root = models_root
        self.on_removed = on_removed

    def deletable(self, path: Path) -> bool:
        """Whether ``path`` is a model folder in the workspace's ``models/``."""
        try:
            return path.resolve().parent == self.models_root.resolve()
        except OSError:
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
        path, aside = await self.gpu.run(self._take_out, name)
        await asyncio.to_thread(shutil.rmtree, aside, ignore_errors=True)
        logger.info("deleted the model %s (%s)", name, path)
        self.on_removed([], [name])
        return str(path)

    def _take_out(self, name: str) -> tuple[Path, Path]:
        """On the worker: check, unload, take it out of the pool, move it aside."""
        pool = self.gpu.pool
        spec = pool.specs.get(name)
        if spec is None:
            raise RemovalError(f"no model is named {name}", status=404)
        path = spec.path
        if not self.deletable(path):
            raise RemovalError(
                f"{name} was not made by the Studio: delete it in Finder if you want it gone",
                status=403,
            )
        if name == pool.default:
            raise RemovalError(f"{name} is the default model: choose another default first")
        children = [
            other
            for other, item in pool.specs.items()
            if item.fit is not None and _resolved(item.fit.base) == _resolved(path)
        ]
        if children:
            raise RemovalError(
                f"fitted models are made from {name} ({', '.join(children)}): delete them first"
            )
        pool.unload(name)  # Busy while a batch keeps it loaded
        pool.remove(name)
        aside = path.with_name(f".{path.name}.{os.getpid()}.removing")
        path.rename(aside)
        return path, aside

    async def delete_all(self) -> list[str]:
        """Delete every model the Studio made, and anything else in its
        ``models/`` folder; returns the models' names. Refused as a whole
        (nothing deleted) when one of them cannot go: the default, a job
        that has not ended naming one or writing there, a fitted model
        elsewhere made from one, a batch keeping one loaded (``Busy``)."""
        for job in self.jobs.jobs.values():
            if job.state != "ended" and job.output is not None and self.deletable(job.output):
                raise RemovalError(f"{job.title} writes into the models folder: wait until it ends")
        names = [name for name, spec in self.gpu.pool.specs.items() if self.deletable(spec.path)]
        for name in names:
            busy = self._used_by_job(name)
            if busy is not None:
                raise RemovalError(f"{busy} uses {name}: clear the folder when the job has ended")
        asides = await self.gpu.run(self._take_out_all, names)
        others = [
            path for path in _entries(self.models_root) if not path.name.endswith(".removing")
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
        asides = []
        for name in names:
            path = pool.specs[name].path
            pool.remove(name)
            aside = path.with_name(f".{path.name}.{os.getpid()}.removing")
            path.rename(aside)
            asides.append(aside)
        return asides

    def _used_by_job(self, name: str) -> str | None:
        """The title of a job that has not ended and names the model."""
        spec = self.gpu.pool.specs.get(name)
        path = str(spec.path) if spec is not None else None
        for job in self.jobs.jobs.values():
            if job.state == "ended":
                continue
            words = set(job.argv)
            if name in words or (path is not None and path in words):
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


def _resolved(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)
