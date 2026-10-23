"""``mlx-decision web``: the command, the page and its static files, the models it serves."""

import asyncio
import base64
import hashlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import mlx_decision.studio as studio_module
from fake_backend import write_model
from mlx_decision import cli
from mlx_decision.cli import app as cli_app
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import catalog, create_studio
from mlx_decision.studio.page import DEV_PREFIX, STATIC, Page, static_id, watch

HERE = "http://127.0.0.1:8000"
VENDORED = [
    "preact.js",
    "preact-hooks.js",
    "htm.js",
    "htm-preact.js",
    "signals.js",
    "signals-core.js",
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def logged_in(tmp_path):
    write_model(tmp_path / "models" / "fake")
    app = create_studio(ModelPool(discover([tmp_path / "models"])), workspace=tmp_path / "ws")
    client = TestClient(app, base_url=HERE, follow_redirects=False)
    client.get(f"/?token={app.state.session.token}")
    return client


# The page and its static files


def test_the_page_names_its_static_files_and_allows_only_this_server(logged_in):
    response = logged_in.get("/")
    assert response.status_code == 200
    page = response.text
    prefix = f"/static/{static_id()}"
    assert "__STATIC__" not in page and f'src="{prefix}/app/main.js"' in page
    assert response.headers["cache-control"] == "no-store"
    policy = response.headers["content-security-policy"]
    [import_map] = re.findall(r'<script type="importmap">(.*?)</script>', page, re.S)
    digest = base64.b64encode(hashlib.sha256(import_map.encode()).digest()).decode()
    assert f"script-src 'self' 'sha256-{digest}'" in policy
    assert "default-src 'self'" in policy and "connect-src 'self'" in policy
    assert "http" not in policy  # no other host
    for module in ("preact", "preact/hooks", "htm/preact", "@preact/signals"):
        assert f'"{module}": "{prefix}/vendor/' in import_map


def test_static_files_are_immutable_and_need_no_session(logged_in, tmp_path):
    prefix = f"/static/{static_id()}"
    anonymous = TestClient(logged_in.app, base_url=HERE)
    for name in ["app/main.js", "studio.css", *(f"vendor/{v}" for v in VENDORED)]:
        response = anonymous.get(f"{prefix}/{name}")
        assert response.status_code == 200, name
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert response.headers["x-content-type-options"] == "nosniff"
    assert (
        anonymous.get(f"{prefix}/app/main.js").headers["content-type"].startswith("text/javascript")
    )
    assert anonymous.get(f"{prefix}/missing.js").status_code == 404
    assert anonymous.get("/static/000000000000/app/main.js").status_code == 404
    assert anonymous.get(f"{prefix}/../pyproject.toml").status_code == 404


# web --dev: page files served uncached, open tabs told to reload on a change


def test_the_dev_page_serves_its_files_uncached(tmp_path):
    write_model(tmp_path / "models" / "fake")
    app = create_studio(
        ModelPool(discover([tmp_path / "models"])), workspace=tmp_path / "ws", dev=True
    )
    with TestClient(app, base_url=HERE, follow_redirects=False) as client:
        client.get(f"/?token={app.state.session.token}")
        assert f'src="{DEV_PREFIX}/app/main.js"' in client.get("/").text
        script = client.get(f"{DEV_PREFIX}/app/main.js")
        assert script.status_code == 200 and script.headers["cache-control"] == "no-store"
        assert "etag" not in script.headers


def test_a_dev_page_sees_its_files_change(tmp_path):
    folder = tmp_path / "static"
    shutil.copytree(STATIC, folder)
    page = Page(folder, dev=True)
    assert page.prefix == DEV_PREFIX and page.changed() is False
    (folder / "studio.css").write_text((folder / "studio.css").read_text() + "\n/* edit */\n")
    assert page.changed() is True and page.changed() is False
    index = folder / "index.html"
    index.write_text(index.read_text().replace("<title>", "<title>Edited "))
    assert page.changed() is True and "<title>Edited " in page.html
    (folder / "app" / "new.js").write_text("export {};\n")
    assert page.changed() is True
    assert Page(folder).prefix != DEV_PREFIX  # without dev: hashed and cached


def test_watch_asks_open_tabs_to_reload(tmp_path):
    folder = tmp_path / "static"
    shutil.copytree(STATIC, folder)
    page = Page(folder, dev=True)
    reloads = []

    async def run():
        task = asyncio.create_task(watch(page, lambda: reloads.append(1), interval=0.01))
        await asyncio.sleep(0.05)
        assert reloads == []
        (folder / "app" / "main.js").write_text("// edited\n")
        for _ in range(100):
            if reloads:
                break
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(run())
    assert reloads == [1]


def test_the_page_has_its_icon(logged_in):
    prefix = f"/static/{static_id()}"
    assert f'<link rel="icon" type="image/svg+xml" href="{prefix}/favicon.svg">' in (
        logged_in.get("/").text
    )
    icon = TestClient(logged_in.app, base_url=HERE).get(f"{prefix}/favicon.svg")
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/svg+xml")


def test_the_static_id_follows_the_contents(tmp_path):
    folder = tmp_path / "static"
    (folder / "app").mkdir(parents=True)
    (folder / "index.html").write_text('<script type="importmap">{}</script>')
    (folder / "app" / "main.js").write_text("one")
    first = static_id(folder)
    assert static_id(folder) == first
    (folder / "app" / "main.js").write_text("two")
    assert static_id(folder) != first
    assert Page(folder).prefix == f"/static/{static_id(folder)}"


def test_the_vendored_modules_import_each_other_by_the_mapped_names():
    imports = set()
    for name in VENDORED:
        imports |= set(re.findall(r'from"([^"]+)"', (STATIC / "vendor" / name).read_text()))
    mapped = set(re.findall(r'"([^"]+)": "__STATIC__', (STATIC / "index.html").read_text()))
    assert imports <= mapped, imports - mapped
    assert (
        (STATIC / "vendor" / "htm.js").read_text().startswith("/**\n * Copyright 2018 Google Inc.")
    )


def test_the_default_model_loads_only_when_asked(tmp_path):
    write_model(tmp_path / "models" / "fake")
    for load_default, loaded in ((True, ["fake"]), (False, [])):
        pool = ModelPool(discover([tmp_path / "models"]))
        app = create_studio(pool, workspace=tmp_path / "ws", load_default=load_default)
        with TestClient(app, base_url=HERE):
            assert pool.loaded() == loaded


# The models web serves


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Named models, workspace models and a fake Hugging Face cache."""
    named = write_model(tmp_path / "named" / "laya")
    workspace_models = tmp_path / "ws" / "models"
    write_model(workspace_models / "fitted-one")
    write_model(workspace_models / "laya")  # clashes with the named one
    cache = tmp_path / "cache"
    cached = {
        "org/edge": write_model(cache / "edge"),
        "other/laya": write_model(cache / "laya"),
        "broken/thing": cache / "missing",
    }
    monkeypatch.setattr(catalog, "cached_models", lambda: list(cached.items()))
    return named, workspace_models, cached


def test_studio_models_names_and_clashes(places):
    named, workspace_models, cached = places
    skipped = []
    specs = catalog.studio_models(
        [str(named)], workspace_models, on_skip=lambda path, reason: skipped.append(reason)
    )
    by_name = {spec.name: spec.path for spec in specs}
    assert list(by_name) == ["laya", "fitted-one", "edge", "other/laya"]
    assert by_name["laya"] == named
    assert by_name["other/laya"] == cached["other/laya"]
    assert any("a model named laya is listed already" in reason for reason in skipped)
    assert any("broken/thing" in reason for reason in skipped)


def test_studio_models_without_named_ones_or_the_cache(places):
    _, workspace_models, _ = places
    names = [spec.name for spec in catalog.studio_models([], workspace_models, include_cache=False)]
    assert names == ["fitted-one", "laya"]
    assert catalog.studio_models([], None, include_cache=False) == []
    assert catalog.studio_models([], workspace_models.parent / "empty", include_cache=False) == []


def test_a_model_listed_twice_is_listed_once(places):
    named, workspace_models, cached = places
    specs = catalog.studio_models([str(cached["org/edge"])], workspace_models)
    assert [spec.name for spec in specs].count("edge") == 1


# The command


class Served:
    """Stands in for the uvicorn server: records what web would serve."""

    def __init__(self, app, **config):
        self.app, self.config = app, config
        self.started = True
        self.should_exit = False
        self.ran_with = None

    def run(self, sockets):
        self.ran_with = sockets


@pytest.fixture
def web(monkeypatch, tmp_path):
    """``web(*args)`` -> (result, the server it would run, the browser's URL or None)."""
    monkeypatch.setattr(catalog, "cached_models", lambda: [])
    seen = {}

    def fake_server(app, **config):
        seen["served"] = Served(app, **config)
        return seen["served"]

    monkeypatch.setattr(studio_module, "server", fake_server)
    monkeypatch.setattr(cli, "open_when_started", lambda served, url: seen.update(browser=url))

    def run(*args):
        seen.clear()
        result = CliRunner().invoke(cli_app, ["web", "--port", str(free_port()), *args])
        return result, seen.get("served"), seen.get("browser")

    return run


def test_web_serves_the_studio(web, tmp_path):
    model = write_model(tmp_path / "models" / "fake")
    write_model(tmp_path / "ws" / "models" / "mine")
    result, served, browser = web("-m", str(model), "--workspace", str(tmp_path / "ws"))
    assert result.exit_code == 0, result.output
    app = served.app
    assert app.state.gpu.pool.names == ["fake", "mine"]
    assert app.state.workspace.root == (tmp_path / "ws").resolve()
    token = app.state.session.token
    assert f"/?token={token}" in result.stderr
    assert browser.endswith(f"/?token={token}")
    assert "Jev API: http://127.0.0.1:" in result.stderr
    [sock] = served.ran_with
    assert sock.fileno() == -1  # closed after the server ran


def test_web_logs_no_line_per_request(web):
    result, served, _ = web("--no-browser")
    assert result.exit_code == 0, result.output
    assert served.config["access_log"] is False


@pytest.mark.parametrize(
    ("host", "family"),
    [
        ("localhost", socket.AF_INET),  # as uvicorn binds it for `server`
        ("127.0.0.1", socket.AF_INET),
        ("::1", socket.AF_INET6),
    ],
)
def test_web_binds_a_name_as_server_does(web, host, family):
    result, served, _ = web("--host", host, "--no-browser")
    assert result.exit_code == 0, result.output
    [sock] = served.ran_with
    assert sock.family == family


def test_web_with_no_models_starts(web, mlx_decision_home):
    result, served, _ = web("--no-browser")
    assert result.exit_code == 0, result.output
    assert served.app.state.gpu.pool.names == []
    assert "no models yet" in result.stderr
    assert served.app.state.workspace.root == mlx_decision_home.resolve()


def test_web_no_browser(web):
    result, served, browser = web("--no-browser")
    assert result.exit_code == 0 and browser is None


def test_web_refuses_another_host_without_a_key(web):
    result, served, _ = web("--host", "0.0.0.0", "--no-browser")
    assert result.exit_code == 1
    assert "needs an API key" in result.stderr and served is None
    result, served, _ = web("--host", "0.0.0.0", "--api-key", "k", "--no-browser")
    assert result.exit_code == 0, result.output
    assert "log in with the key" in result.stderr


def test_web_refuses_a_port_in_use(monkeypatch, tmp_path):
    monkeypatch.setattr(catalog, "cached_models", lambda: [])
    for taken_on in ("127.0.0.1", "0.0.0.0"):
        with socket.socket() as taken:
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            taken.bind((taken_on, 0))
            taken.listen()
            port = taken.getsockname()[1]
            result = CliRunner().invoke(cli_app, ["web", "--port", str(port), "--no-browser"])
        assert result.exit_code == 1
        assert f"port {port} is in use" in result.stderr and "--port" in result.stderr


def test_web_workspace_must_be_a_folder(web, tmp_path):
    (tmp_path / "file").write_text("x")
    result, _, _ = web("--workspace", str(tmp_path / "file"), "--no-browser")
    assert result.exit_code != 0


def test_web_refuses_a_workspace_of_a_newer_format(web, tmp_path):
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "workspace.json").write_text('{"format": 2}')
    result, served, _ = web("--workspace", str(tmp_path / "ws"), "--no-browser")
    assert result.exit_code == 1 and served is None
    assert "workspace.json: the workspace's format is 2" in result.stderr
    assert not (tmp_path / "ws" / "README.md").exists()


def test_web_default_model(web, tmp_path):
    write_model(tmp_path / "models" / "a")
    write_model(tmp_path / "models" / "b")
    result, served, _ = web("-m", str(tmp_path / "models"), "--default-model", "b", "--no-browser")
    assert result.exit_code == 0, result.output
    assert served.app.state.gpu.pool.default == "b"
    assert "default b" in result.stderr
    result, _, _ = web("-m", str(tmp_path / "models"), "--default-model", "c", "--no-browser")
    assert result.exit_code == 1 and "c" in result.stderr


def test_web_end_to_end(tmp_path):
    """The real command in a process: it prints the address, serves the page with
    the token and stops on Ctrl-C."""
    port = free_port()
    env = {
        **os.environ,
        "MLX_DECISION_HOME": str(tmp_path / "home"),
        "HF_HUB_CACHE": str(tmp_path / "hf-cache"),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "mlx_decision", "web", "--no-browser", "--port", str(port)],
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
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            deadline = time.monotonic() + 30
            while True:
                try:
                    page = client.get(url)
                    break
                except httpx.ConnectError:
                    assert time.monotonic() < deadline
                    time.sleep(0.1)
            assert page.status_code == 200 and "importmap" in page.text
            prefix = re.search(r"/static/[0-9a-f]+", page.text)[0]
            assert client.get(f"http://127.0.0.1:{port}{prefix}/app/main.js").status_code == 200
            status = client.get(f"http://127.0.0.1:{port}/studio/api/status").json()
            assert status["models"] == [] and Path(status["workspace"]).name == "home"
        assert (tmp_path / "home" / "sets").is_dir()
        process.send_signal(signal.SIGINT)
        assert process.wait(15) in (0, 130, -signal.SIGINT)  # 130: stopped by Ctrl-C
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
