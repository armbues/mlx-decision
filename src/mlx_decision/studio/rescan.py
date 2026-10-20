"""Rescans: the models catalog looked at again while the Studio runs (D135).

A download, convert or fit that ends ``done`` writes a model; so may the CLI
behind the Studio's back (the page asks for a rescan when it regains focus).
A rescan adds the models whose folder is new to the pool (``pool.add``)
and takes out those whose folder is gone, unless loaded or the default
(they stay until unloaded). Models are matched by folder, so a named model
the scan does not list is never touched. A change becomes a ``catalog``
message, and every tab reloads its list.
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from ..gpu import Gpu
from ..pool import ModelSpec
from .catalog import Catalog

logger = logging.getLogger(__name__)


class Rescans:
    """``run()`` scans and applies; one at a time. ``on_change(added, removed)``
    is called on the loop after a scan that changed the pool."""

    def __init__(
        self,
        gpu: Gpu,
        catalog: Catalog,
        on_change: Callable[[list[str], list[str]], None] = lambda added, removed: None,
    ):
        self.gpu = gpu
        self.catalog = catalog
        self.on_change = on_change
        self.lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def run(self) -> dict[str, list[str]]:
        """Scan (in a thread) and apply (on the worker, which the pool belongs to):
        ``{added, removed}``, the names."""
        async with self.lock:
            found = await asyncio.to_thread(self.catalog.scan, _skipped)
            added, removed = await self.gpu.run(self._apply, found, check=False)
        if added or removed:
            logger.info("models added: %s; removed: %s", added, removed)
            self.on_change(added, removed)
        return {"added": added, "removed": removed}

    def soon(self) -> None:
        """``run`` in the background (after a job wrote a model)."""
        task = asyncio.create_task(self._quietly(), name="rescan")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _quietly(self) -> None:
        try:
            await self.run()
        except RuntimeError:  # the worker has shut down: the Studio is stopping
            pass
        except Exception:
            logger.exception("the models could not be listed again")

    def _apply(self, found: list[ModelSpec]) -> tuple[list[str], list[str]]:
        pool = self.gpu.pool
        known = {_folder(spec.path) for spec in pool.specs.values()}
        added = []
        for spec in found:
            if _folder(spec.path) in known:
                continue
            if spec.name in pool.specs:
                logger.info("skipped %s: a model named %s is listed already", spec.path, spec.name)
                continue
            pool.add(spec)
            known.add(_folder(spec.path))
            added.append(spec.name)
        removed = []
        for name, spec in list(pool.specs.items()):
            if spec.path.is_dir():
                continue
            try:
                pool.remove(name)
            except ValueError:
                continue  # loaded or the default: it stays until unloaded
            removed.append(name)
        return added, removed


def _folder(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _skipped(path: Path, reason: str) -> None:
    logger.info("skipped %s", reason)
