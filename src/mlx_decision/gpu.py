"""The GPU: the model pool, the one thread that runs model work, and who holds it.

All model work (loads, unloads, answers) runs on one worker thread, in the
order it was submitted, so MLX sees one thread and requests wait their
turn. Something that needs the GPU to itself (a job that loads its own
model in another process) calls ``acquire``: from then on model work is
refused with ``Busy``, both when a request arrives (``check``) and again
when its turn on the worker comes, so work queued before the handover
cannot load a model after the pool has been emptied. ``release`` ends it;
models load again on their next request.

The holder is set and cleared on the event loop and read on the worker;
a plain attribute is enough for that. ``on_change``, when set, is called
on the event loop after the holder changed.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from .pool import ModelPool

# Seconds a busy answer tells clients to wait before trying again.
RETRY_AFTER = 10

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Busy(Exception):
    """Model work refused because ``holder`` has the GPU."""

    def __init__(self, holder: str):
        super().__init__(f"the GPU is busy with {holder}; try again when it is done")
        self.holder = holder


class Gpu:
    """The pool, its worker thread and the holder; one per process."""

    def __init__(self, pool: ModelPool):
        self.pool = pool
        self.holder: str | None = None
        self.on_change: Callable[[], None] | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision")

    def check(self) -> None:
        """Raise ``Busy`` while something holds the GPU."""
        holder = self.holder
        if holder is not None:
            raise Busy(holder)

    async def run(self, function: Callable[..., T], *args, check: bool = True) -> T:
        """``function(*args)`` on the worker, after the work queued before it.

        With ``check`` (the default) it is refused with ``Busy`` if the GPU
        is held when its turn comes; without, it runs anyway (for the pool's
        bookkeeping, which loads nothing).
        """

        def task():
            if check:
                self.check()
            return function(*args)

        return await asyncio.get_running_loop().run_in_executor(self._worker, task)

    async def start(self) -> None:
        """Load the pool's default model, if it has one."""
        if self.pool.default is not None:
            await self.run(self.pool.get, self.pool.default)
            logger.info("model %s loaded", self.pool.default)

    async def acquire(self, holder: str) -> None:
        """Take the GPU for ``holder``: refuse new model work, then unload every model.

        The unload runs on the worker after the work it is on and the work
        queued before it, which is refused from now on. Raises ``Busy`` when
        something else holds the GPU.
        """
        self.check()
        self.holder = holder
        self._changed()
        try:
            await self.run(self.pool.unload_all, check=False)
        except BaseException:
            self.holder = None
            self._changed()
            raise

    def release(self, holder: str) -> None:
        """Give the GPU back; does nothing unless ``holder`` holds it."""
        if self.holder == holder:
            self.holder = None
            self._changed()

    def _changed(self) -> None:
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                logger.exception("the GPU's change callback failed")

    def close(self) -> None:
        """Stop the worker after the work queued on it."""
        self._worker.shutdown(wait=True)
