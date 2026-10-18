"""The Studio: the Jev API, the Studio's pages and its own API in one app.

Needs the ``web`` extra (FastAPI, uvicorn). One ``Gpu`` serves both APIs,
so the Studio's runs and the Jev API's requests share one worker and one
model pool. ``security`` holds what every request goes through;
``workspace`` the files the Studio keeps; ``stream`` the messages that
keep every open tab up to date.
"""

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import __version__
from ..gpu import Gpu
from ..pool import ModelPool
from ..server import MAX_BODY_BYTES, add_jev_api, gpu_lifespan
from . import security
from .security import NO_STORE, Session
from .stream import Hub, stream_router
from .workspace import Workspace, root_folder

# Stands in for the Studio's page until its files ship.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mlx-decision Studio</title></head>
<body><p>mlx-decision Studio</p></body></html>
"""


def create_studio(
    gpu: Gpu | ModelPool,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    session: Session | None = None,
    workspace: Path | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> FastAPI:
    """The Studio's app for a process serving ``host:port``.

    It owns ``gpu`` as ``server.create_app`` does. ``app.state.session``
    holds the session, whose token goes into the address ``web`` opens;
    ``app.state.workspace`` the workspace (``workspace``, else ``home()``),
    created and cleaned up here; ``app.state.hub`` the stream's hub. Raises
    ``ValueError`` for an address beyond this machine without an API key.
    """
    if isinstance(gpu, ModelPool):
        gpu = Gpu(gpu)
    session = session or Session(port)
    security.check_bind(host, api_key)  # before the lifespan could load a model
    files = Workspace(workspace if workspace is not None else root_folder())
    files.open()
    status = status_reader(gpu)
    hub = Hub(lambda: {"status": status()})
    app = FastAPI(
        title="mlx-decision Studio",
        lifespan=studio_lifespan(gpu, files, hub, status),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.gpu = gpu
    app.state.session = session
    app.state.workspace = files
    app.state.hub = hub
    security.install(app, session, host, port, api_key)
    add_jev_api(app, gpu, api_key, max_body_bytes)
    app.include_router(page_router(session, api_key))
    app.include_router(api_router(gpu))
    app.include_router(stream_router(hub))
    return app


def server(app: FastAPI, **config: Any):
    """A uvicorn server for the Studio's ``app`` (``config`` as ``uvicorn.Config``).

    It ends the open streams when it starts shutting down: uvicorn waits for
    open responses before the app's own shutdown, so a tab left open would
    otherwise keep the Studio from stopping.
    """
    import uvicorn

    class StudioServer(uvicorn.Server):
        async def shutdown(self, sockets=None) -> None:
            app.state.hub.close()
            await super().shutdown(sockets=sockets)

    return StudioServer(uvicorn.Config(app, **config))


def status_reader(gpu: Gpu) -> Callable[[], dict[str, Any]]:
    """What the ``status`` message says: who holds the GPU (None when no one
    does), the loaded models (least recently used first) and the memory they
    count against the budget, in bytes. Safe to call from any thread."""
    pool = gpu.pool
    budget = None

    def status() -> dict[str, Any]:
        nonlocal budget
        if budget is None:  # the working set does not change: read it once
            budget = pool.limit()
        return {
            "busy": gpu.holder,
            "loaded": pool.loaded(),
            "memory": {"used": pool.used_bytes(), "budget": budget},
        }

    return status


def studio_lifespan(gpu: Gpu, files: Workspace, hub: Hub, status: Callable[[], dict[str, Any]]):
    """``gpu_lifespan`` plus the stream: changes to the GPU's holder, the pool and
    the workspace become messages; the streams end when the Studio stops."""

    def publish_status() -> None:
        hub.publish_changed("status", status())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.attach(asyncio.get_running_loop())
        publish_status()  # the baseline; later ones are changes
        gpu.on_change = publish_status
        # The pool and the workspace change on other threads; the status is read
        # on the loop, so messages never arrive out of order.
        gpu.pool.on_change = lambda: hub.soon(publish_status)
        files.on_change = lambda change: hub.soon(hub.publish, "workspace", change)
        try:
            async with gpu_lifespan(gpu)(app):
                yield
        finally:
            hub.close()
            gpu.on_change = gpu.pool.on_change = files.on_change = None

    return lifespan


def page_router(session: Session, api_key: str | None) -> APIRouter:
    """``/``: the page, and the exchange of the address's token for the cookie."""
    router = APIRouter()

    @router.get("/")
    async def page(request: Request, token: str | None = None):
        to_page = RedirectResponse("/", status_code=303, headers=NO_STORE)  # drops the token
        if session.has_cookie(request.cookies):
            return to_page if token is not None else HTMLResponse(PAGE, headers=NO_STORE)
        if token is not None and session.matches_token(token):
            client = request.client.host if request.client else None
            # With a key, the address alone does not let in another machine.
            if not api_key or security.is_loopback(client):
                return session.set_cookie(to_page)
        return security.page_without_session(api_key)

    return router


def api_router(gpu: Gpu) -> APIRouter:
    """The Studio's own API under ``/studio/api`` (the guard requires the session)."""
    router = APIRouter(prefix="/studio/api")

    @router.get("/status")
    async def status():
        return {"version": __version__, "busy": gpu.holder}

    return router
