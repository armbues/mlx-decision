"""An HTTP server with an API compatible with Jev's: ``POST /v1/systemone``.

Needs the ``server`` extra (FastAPI, uvicorn). All model work, loading
included, runs on one worker thread; requests wait their turn there. With
an API key, ``/v1/*`` requires ``Authorization: Bearer <key>``.
"""

import asyncio
import hmac
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import DecisionError
from .images import is_data_url
from .model import DecisionModel

TRUNCATED_HEADER = "X-MLX-Decision-Truncated"
# Room for several large images as data URLs.
MAX_BODY_BYTES = 64 * 2**20

logger = logging.getLogger(__name__)


def error_response(
    status: int, message: str, kind: str, param: str | None = None, code: str | None = None
) -> JSONResponse:
    """An error in the API's error body shape."""
    error = {"message": message, "type": kind, "param": param}
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


def check_data_urls(body) -> None:
    """Images over HTTP must be data URLs: the server reads no files and fetches nothing."""
    images = body.get("images") if isinstance(body, dict) else None
    if not isinstance(images, list):
        return
    for index, image in enumerate(images):
        if not is_data_url(image):
            raise DecisionError(
                "expected a data URL (data:image/png;base64,...); "
                "the server does not read files or fetch URLs",
                param=f"images.{index}",
            )


async def read_body(request: Request, limit: int) -> bytes | None:
    """The request body, or None when it is larger than ``limit`` bytes."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def create_app(
    load_model: Callable[[], DecisionModel],
    api_key: str | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> FastAPI:
    """The app; ``load_model`` runs on the worker thread at startup."""
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision")

    async def on_worker(function, *args):
        return await asyncio.get_running_loop().run_in_executor(worker, function, *args)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = await on_worker(load_model)
        logger.info("model %s loaded", app.state.model.name)
        yield
        worker.shutdown(wait=True)

    app = FastAPI(title="mlx-decision", lifespan=lifespan)

    if api_key:
        expected = f"Bearer {api_key}".encode()

        @app.middleware("http")
        async def check_api_key(request: Request, call_next):
            if request.url.path.startswith("/v1/"):
                given = request.headers.get("authorization", "").encode()
                if not hmac.compare_digest(given, expected):
                    return error_response(401, "missing or invalid API key", "authentication_error")
            return await call_next(request)

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        raw = await read_body(request, max_body_bytes)
        if raw is None:
            return error_response(
                413,
                f"the request body is larger than {max_body_bytes / 2**20:.0f} MB",
                "invalid_request_error",
            )
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            return error_response(
                400, f"the body is not valid JSON: {error}", "invalid_request_error"
            )
        except RecursionError:
            return error_response(400, "the body is nested too deeply", "invalid_request_error")
        model: DecisionModel = request.app.state.model
        try:
            check_data_urls(body)
            result = await on_worker(model.decide_request, body)
        except DecisionError as error:
            return error_response(
                422, error.message, "invalid_request_error", error.param, error.code
            )
        except Exception:
            logger.exception("request failed")
            return error_response(500, "the model failed to answer the request", "api_error")
        headers = {TRUNCATED_HEADER: "true"} if result.truncated else None
        return JSONResponse(result.to_wire(), headers=headers)

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "model": request.app.state.model.name}

    @app.get("/v1/models")
    async def models(request: Request):
        name = request.app.state.model.name
        return {
            "models": [
                {
                    "name": name,
                    "description": f"{name}, served locally by mlx-decision",
                    # Unknown for a local model; the field is required.
                    "release_date": "",
                }
            ]
        }

    return app
