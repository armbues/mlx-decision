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

from .. import memory
from ..config import Config
from ..config import load as load_config
from ..gpu import Gpu
from ..pool import ModelPool
from ..server import MAX_BODY_BYTES, add_jev_api
from . import routes, security
from .catalog import Catalog
from .jobs import Jobs
from .page import Page, watch
from .rescan import Rescans
from .routes.data import data_router
from .routes.datasets import datasets_router
from .routes.downloads import downloads_router
from .routes.jobs import JobMessages, jobs_router
from .routes.models import models_router
from .routes.run import Models, Recorder, run_router
from .routes.sets import sets_router
from .routes.settings import settings_router
from .routes.status import status_router
from .routes.training import training_router
from .security import NO_STORE, Session
from .settings import Settings
from .stream import Hub, stream_router
from .workspace import Workspace, root_folder


def create_studio(
    gpu: Gpu | ModelPool,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    session: Session | None = None,
    workspace: Path | None = None,
    config: Config | None = None,
    load_default: bool = True,
    max_body_bytes: int = MAX_BODY_BYTES,
    dev: bool = False,
    catalog: Catalog | None = None,
    sources: dict[str, str] | None = None,
) -> FastAPI:
    """The Studio's app for a process serving ``host:port``.

    It owns ``gpu`` as ``server.create_app`` does. ``app.state.session``
    holds the session, whose token goes into the address ``web`` opens;
    ``app.state.workspace`` the workspace (``workspace``, else ``home()``),
    created and cleaned up here; ``app.state.hub`` the stream's hub;
    ``app.state.jobs`` the jobs, stopped when the Studio stops; a fresh
    stream's snapshot carries this process's jobs. ``app.state.settings``
    holds the config file as last read (downloads go where its
    ``models_dir`` says) and changes it (the Settings page).
    ``catalog`` is where the pool's models came from (``web``'s); the Studio
    looks there again after a job wrote a model (by default only the
    workspace's ``models/``), as ``app.state.rescans``. ``sources`` says
    where each setting came from ("flag", "config", "default"; ``web``'s),
    for the config the Models page shows.
    ``config`` is the config file the process read (read here when not
    given), shown by the status route. The pool's default model loads at
    startup unless ``load_default`` is False. With ``dev`` the page's files
    are served uncached and open tabs reload when one changes. Raises ``ValueError`` for an address
    beyond this machine without an API key.
    """
    if isinstance(gpu, ModelPool):
        gpu = Gpu(gpu)
    session = session or Session(port)
    security.check_bind(host, api_key)  # before the lifespan could load a model
    config = config if config is not None else load_config()
    files = Workspace(workspace if workspace is not None else root_folder(config))
    files.open()
    page = Page(dev=dev)
    status = status_reader(gpu)
    jobs = Jobs(gpu, files)
    hub = Hub(lambda: {"status": status(), "jobs": jobs.current()})
    models = Models(gpu, files.root / "models", files)
    catalog = catalog or Catalog(workspace_models=files.root / "models", include_cache=False)

    def catalog_changed(added: list[str], removed: list[str]) -> None:
        for name in added + removed:
            models.forget(name)
        hub.publish("catalog", {"added": added, "removed": removed})

    rescans = Rescans(gpu, catalog, catalog_changed)
    settings = Settings(
        config,
        gpu,
        catalog,
        files,
        host,
        port,
        rescans,
        sources,
        on_change=lambda key: hub.publish("settings", {"key": key}),
        api_key=bool(api_key),
    )
    app = FastAPI(
        title="mlx-decision Studio",
        lifespan=studio_lifespan(gpu, files, hub, status, load_default, page, jobs, rescans),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.gpu = gpu
    app.state.session = session
    app.state.workspace = files
    app.state.hub = hub
    app.state.jobs = jobs
    app.state.settings = settings
    app.state.rescans = rescans
    security.install(app, session, host, port, api_key)
    recorder = Recorder(files, models)
    routes.install_errors(app)
    add_jev_api(app, gpu, api_key, max_body_bytes, on_answer=recorder.answered)
    page.mount(app)
    app.include_router(page_router(page, session, api_key))
    current = lambda: settings.config  # noqa: E731 - read at each request: Settings changes it
    app.include_router(status_router(gpu, files, models, current, host, port, api_key, status))
    app.include_router(sets_router(files, max_body_bytes))
    app.include_router(data_router(files, jobs, max_body_bytes))
    app.include_router(training_router(gpu, files, max_body_bytes))
    app.include_router(datasets_router(max_body_bytes))
    app.include_router(run_router(gpu, models, recorder, max_body_bytes))
    app.include_router(models_router(gpu, max_body_bytes, rescans))
    app.include_router(jobs_router(jobs, gpu, files, current, max_body_bytes))
    app.include_router(
        downloads_router(
            gpu,
            current,
            max_body_bytes,
            set_config=lambda updated: setattr(settings, "config", updated),
            catalog=catalog,
        )
    )
    app.include_router(settings_router(gpu, files, catalog, settings, max_body_bytes))
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
    does), the loaded models (least recently used first), the groups they
    load in (``ModelPool.groups``), when each was last used (Unix seconds)
    and the memory they count against the budget, in bytes. Safe to call
    from any thread."""
    pool = gpu.pool
    limits: dict[int | None, int] = {}
    working: list[int] = []  # the GPU's working set, read once

    def status() -> dict[str, Any]:
        # The working set does not change: read once per budget (Settings
        # may change the budget).
        if pool.budget not in limits:
            limits[pool.budget] = pool.limit()
        budget = limits[pool.budget]
        if not working:
            working.append(memory.working_set())
        return {
            "busy": gpu.holder,
            "loaded": pool.loaded(),
            "groups": pool.groups(),
            # Whole seconds: a model used twice in a second sends one message.
            "used": {name: int(when) for name, when in pool.last_used().items()},
            "memory": {"used": pool.used_bytes(), "budget": budget, "working_set": working[0]},
        }

    return status


# Jobs whose output is a model.
WRITES_MODELS = ("download", "convert", "fit")


def studio_lifespan(
    gpu: Gpu,
    files: Workspace,
    hub: Hub,
    status: Callable[[], dict[str, Any]],
    load_default: bool = True,
    page: Page | None = None,
    jobs: Jobs | None = None,
    rescans: Rescans | None = None,
):
    """Start the GPU (its default model loads, with ``load_default``) and close it
    at the end, as ``server.gpu_lifespan`` does, plus the stream: changes to the
    GPU's holder, the pool, the workspace and the ``jobs`` become messages; the
    streams end when the Studio stops. Running jobs are stopped first. A job
    that wrote a model and ended ``done`` starts a rescan."""

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
        gpu.pool.on_use = lambda: hub.soon(publish_status)  # last used
        files.on_change = lambda change: hub.soon(hub.publish, "workspace", change)
        if jobs is not None:  # they change on the loop
            messages = JobMessages(hub.publish)

            def job_changed(job, event) -> None:
                messages(job, event)
                ended = event is None and job.state == "ended" and job.outcome == "done"
                if ended and job.kind == "download" and job.output and rescans:
                    rescans.catalog.add_download(job.output)  # listed even with web -m
                if ended and job.kind in WRITES_MODELS and rescans is not None:
                    rescans.soon()
                if ended and job.kind == "data":
                    # New datasets: every tab's list again.
                    for output in job.outputs:  # a dataset folder (or a plain file)
                        files.changed("data", output.name.removesuffix(".jsonl"), "saved")

            jobs.on_change = job_changed
        watcher = None
        if page is not None and page.dev:  # open tabs reload when a page file changes
            watcher = asyncio.create_task(watch(page, lambda: hub.publish("reload", {})))
        try:
            if load_default:
                await gpu.start()
            yield
        finally:  # also when the default model fails to load
            if watcher is not None:
                watcher.cancel()
            if jobs is not None:
                await jobs.shutdown()
            hub.close()
            gpu.close()
            gpu.on_change = gpu.pool.on_change = gpu.pool.on_use = files.on_change = None
            if jobs is not None:
                jobs.on_change = None

    return lifespan


def page_router(page: Page, session: Session, api_key: str | None) -> APIRouter:
    """``/``: the page, and the exchange of the address's token for the cookie."""
    router = APIRouter()

    @router.get("/")
    async def serve_page(request: Request, token: str | None = None):
        to_page = RedirectResponse("/", status_code=303, headers=NO_STORE)  # drops the token
        if session.has_cookie(request.cookies):
            if token is not None:
                return to_page
            return HTMLResponse(page.html, headers=page.headers)
        if token is not None and session.matches_token(token):
            client = request.client.host if request.client else None
            # With a key, the address alone does not let in another machine.
            if not api_key or security.is_loopback(client):
                return session.set_cookie(to_page)
        return security.page_without_session(api_key)

    return router
