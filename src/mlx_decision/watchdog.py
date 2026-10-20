"""Stop a run whose parent is gone.

The Studio runs a job as a subprocess with a pipe on stdin that it never
writes to. When the Studio dies, the pipe closes and reading it ends: the
run's cancel flag is set, so it stops at its next step as after Ctrl-C. A
run still going ``GRACE_S`` later (stuck in a long step) removes what it
had half written and exits at once with 130.

Only a pipe (or socket) on stdin is watched. A terminal, ``/dev/null`` or a
file is left alone, so a run started by hand or from a script behaves as
before.

A run that cannot stop from inside (a download: the Hub's threads write
until their files are done) does its work with ``in_thread`` and, once
cancelled, ends with ``quit_now``.
"""

import os
import shutil
import stat
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, NoReturn

from .events import Cancelled

GRACE_S = 4.0


@contextmanager
def watch_stdin(
    cancel: threading.Event,
    leftovers: Iterable[str | Path] = (),
    *,
    grace: float = GRACE_S,
    stdin: IO | None = None,
) -> Iterator[bool]:
    """Watch stdin while the block runs: when it closes, set ``cancel``; if the
    block has not ended ``grace`` seconds later, remove ``leftovers`` (folders or
    files the run writes before renaming them into place) and exit with 130.

    Yields whether stdin is watched (only a pipe or a socket is). Leaving the
    block ends the watch: from then on a closed stdin changes nothing.
    """
    fd = _pipe(sys.stdin if stdin is None else stdin)
    if fd is None:
        yield False
        return
    ended = threading.Event()
    paths = [Path(path) for path in leftovers]

    def watch() -> None:
        try:
            while os.read(fd, 65536):
                pass  # nobody should write here; whatever comes is dropped
        except OSError:
            pass  # a broken pipe counts as closed
        if ended.is_set():
            return
        cancel.set()
        _say("stdin closed: stopping after the current step")
        if ended.wait(grace):
            return
        for path in paths:
            remove(path)
        _say("stdin closed and the run did not stop: quitting")
        os._exit(130)

    threading.Thread(target=watch, name="stdin watchdog", daemon=True).start()
    try:
        yield True
    finally:
        ended.set()


def _pipe(stream: IO) -> int | None:
    """The file descriptor of ``stream`` when it is a pipe or a socket."""
    try:
        fd = stream.fileno()
        mode = os.fstat(fd).st_mode
    except (AttributeError, OSError, ValueError):
        return None  # no real file behind it (a test runner's stand-in)
    return fd if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) else None


def in_thread[T](work: Callable[[], T], cancel: threading.Event) -> T:
    """``work()`` on a thread of its own: its result or its exception, or
    ``Cancelled`` as soon as ``cancel`` is set or Ctrl-C interrupts the wait.
    A cancelled ``work`` keeps running: end the process with ``quit_now``."""
    outcome: dict[str, object] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            outcome["result"] = work()
        except BaseException as error:  # noqa: BLE001 - handed to the caller
            outcome["error"] = error
        finally:
            finished.set()

    threading.Thread(target=run, name="work", daemon=True).start()
    try:
        while not finished.wait(0.05):
            if cancel.is_set():
                raise Cancelled()
    except KeyboardInterrupt:
        raise Cancelled() from None
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["result"]  # type: ignore[return-value]


def quit_now(code: int, leftovers: Iterable[str | Path] = ()) -> NoReturn:
    """Remove ``leftovers`` and end the process at once, threads and all (they
    would keep it alive until their work is done)."""
    paths = [Path(path) for path in leftovers]
    for path in paths:
        remove(path)
    for stream in (sys.stdout, sys.stderr):
        with suppress(OSError, ValueError):
            stream.flush()
    for path in paths:  # a thread may have started a new file meanwhile
        remove(path)
    os._exit(code)


def remove(path: Path) -> None:
    """Remove ``path``, first moving it aside, so that the run cannot rename it
    into place half removed (if the run renamed it first, it is complete and
    stays)."""
    aside = path.with_name(f"{path.name}.{os.getpid()}.removing")
    try:
        path.rename(aside)
    except OSError:
        return
    if aside.is_dir():
        shutil.rmtree(aside, ignore_errors=True)
    else:
        aside.unlink(missing_ok=True)


def _say(message: str) -> None:
    """A line on stderr, unless stderr is gone too."""
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
