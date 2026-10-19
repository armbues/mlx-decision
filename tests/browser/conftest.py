"""Browser tests: the Studio in Chrome, driven by Playwright.

Each module is marked ``browser`` (``pytestmark``), which ``make test``
leaves out; ``make test-browser`` runs them. They drive the installed Chrome
(``channel="chrome"``, no browser download) and are skipped without
Playwright or Chrome.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import fake_studio
import httpx
import pytest

from fake_backend import write_model

TESTS = Path(__file__).resolve().parents[1]

# `mlx-decision web` with the fake model family registered first (it is found
# only through tests/fake_backend.py), as its own process.
LAUNCHER = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); import fake_backend; "
    "from mlx_decision.cli import app; app(prog_name='mlx-decision')"
)


@pytest.fixture(scope="session")
def chrome():
    """The installed Chrome, headless."""
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except sync_api.Error as error:
            pytest.skip(f"Chrome cannot be started: {error.message.splitlines()[0]}")
        yield browser
        browser.close()


@pytest.fixture
def page(chrome):
    """A fresh tab (its own cookies and storage) at the Studio's design size; a
    script error or a ``console.error`` in the page fails the test (not the
    browser's own line for a request answered 4xx, which tests cause on purpose)."""
    context = chrome.new_context(viewport={"width": 1440, "height": 900})
    tab = context.new_page()
    errors = []
    tab.on("pageerror", lambda error: errors.append(str(error)))

    def console(message):
        if message.type == "error" and not message.text.startswith("Failed to load resource"):
            errors.append(f"console.error: {message.text}")

    tab.on("console", console)
    yield tab
    context.close()
    assert errors == [], f"script errors in the page: {errors}"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Studio:
    url: str  # with the session token
    port: int
    workspace: Path


@pytest.fixture
def fake_web(tmp_path):
    """``mlx-decision web`` serving one fake model, in its own process with an
    empty home folder; stopped with Ctrl-C at the end."""
    models = tmp_path / "models"
    write_model(models / "fake")
    home = tmp_path / "home"
    port = free_port()
    env = {**os.environ, "MLX_DECISION_HOME": str(home), "HF_HUB_CACHE": str(tmp_path / "hf-cache")}
    command = [sys.executable, "-c", LAUNCHER, str(TESTS), "web", "-m", str(models)]
    process = subprocess.Popen(
        [*command, "--no-browser", "--port", str(port)],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url, lines = None, []
        deadline = time.monotonic() + 60
        while url is None:
            assert time.monotonic() < deadline, "".join(lines)
            line = process.stderr.readline()
            assert line or process.poll() is None, "".join(lines)
            lines.append(line)
            if line.startswith("Studio: "):
                url = line.removeprefix("Studio: ").strip()
        deadline = time.monotonic() + 30
        while True:
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except httpx.ConnectError:
                assert time.monotonic() < deadline, "".join(lines)
                time.sleep(0.1)
        yield Studio(url=url, port=port, workspace=home)
        process.send_signal(signal.SIGINT)
        assert process.wait(15) in (0, 130, -signal.SIGINT)  # 130: stopped by Ctrl-C
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.fixture
def make_studio(tmp_path):
    """Start a fake Studio in this process (``fake_studio.serve``): call it with
    ``sizes=True`` and / or ``slow=True`` for those models; one per test."""
    running = []

    def start(**options) -> fake_studio.Served:
        assert not running, "one fake Studio per test"
        context = fake_studio.serve(tmp_path / "studio", **options)
        running.append(context)
        return context.__enter__()

    yield start
    for context in running:
        context.__exit__(None, None, None)


@pytest.fixture
def studio(make_studio) -> fake_studio.Served:
    """A fake Studio with its three plain models and two sample sets."""
    return make_studio()
