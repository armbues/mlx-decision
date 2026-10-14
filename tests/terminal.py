"""A command run in a pseudo-terminal, its screen kept by pyte, for testing menus.

prompt_toolkit asks the terminal where its cursor is (``ESC [6n``) and waits
for the answer; the harness answers from pyte's cursor, as a terminal would.
"""

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
import warnings

import pyte

CURSOR_REQUEST = b"\x1b[6n"
ENTER = "\r"
DOWN = "\x1b[B"


class Terminal:
    """``python -m mlx_decision ARGS`` in a pty of ``rows`` x ``cols`` under ``cwd``."""

    def __init__(self, args: list[str], cwd, rows: int = 50, cols: int = 140):
        self.screen = pyte.HistoryScreen(cols, rows, history=2000)
        self.stream = pyte.ByteStream(self.screen)
        self.log = b""
        self.status: int | None = None
        with warnings.catch_warnings():
            # The child only sizes the terminal and changes folder before exec,
            # so the threads of the test process cannot leave it deadlocked.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid, fd = pty.fork()
        if pid == 0:  # the child: size the terminal, then become the command
            fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            os.chdir(cwd)
            env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(cols)}
            os.execve(sys.executable, [sys.executable, "-m", "mlx_decision", *args], env)
        self.pid, self.fd = pid, fd

    def _read(self, timeout: float) -> bool:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return False
        try:
            data = os.read(self.fd, 65536)
        except OSError:  # the child closed the terminal
            data = b""
        if not data:
            self._reap(block=True)
            return False
        self.log += data
        self.stream.feed(data)
        if CURSOR_REQUEST in data:
            y, x = self.screen.cursor.y + 1, self.screen.cursor.x + 1
            os.write(self.fd, f"\x1b[{y};{x}R".encode())
        return True

    def _reap(self, block: bool = False) -> None:
        if self.status is None:
            pid, status = os.waitpid(self.pid, 0 if block else os.WNOHANG)
            if pid:
                self.status = os.waitstatus_to_exitcode(status)

    def text(self) -> str:
        """Everything shown so far: the scrolled-off lines and the screen."""
        history = ["".join(char.data for char in line.values()) for line in self.screen.history.top]
        return "\n".join(line.rstrip() for line in [*history, *self.screen.display])

    def expect(self, wanted: str, timeout: float = 60) -> str:
        """Wait until ``wanted`` is on the screen (or was); the screen's text."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if wanted in self.text():
                return self.text()
            if self.status is not None:
                break
            self._read(0.2)
            self._reap()
        raise AssertionError(f"{wanted!r} never showed; the screen:\n{self.text()}")

    def send(self, keys: str, pause: float = 0.3) -> None:
        """Type ``keys``, then let the screen settle."""
        os.write(self.fd, keys.encode())
        end = time.monotonic() + pause
        while time.monotonic() < end:
            self._read(0.05)

    def wait(self, timeout: float = 120) -> int:
        """The exit code, reading output until the command ends."""
        deadline = time.monotonic() + timeout
        while self.status is None and time.monotonic() < deadline:
            self._read(0.2)
            self._reap()
        if self.status is None:
            os.kill(self.pid, signal.SIGKILL)
            self._reap(block=True)
            raise AssertionError(f"still running after {timeout} s; the screen:\n{self.text()}")
        return self.status
