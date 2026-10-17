"""An HTTP server with an API compatible with Jev's: ``POST /v1/systemone``.

Needs the ``server`` extra (FastAPI, uvicorn). The models come from a
``ModelPool``; the request's ``model`` field picks one. All model work,
loading included, runs on the ``Gpu``'s one worker thread; requests wait
their turn there, and get a 503 while something holds the GPU (never
under the ``server`` command). With an API key, the ``/v1`` routes require ``Authorization: Bearer
<key>``; the check belongs to the routes, so it holds wherever the app is
mounted (a mount or a proxy's root path).
"""

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import DecisionError
from .gpu import RETRY_AFTER, Busy, Gpu
from .images import check_data_urls
from .memory import ModelTooLargeError
from .pool import ModelPool
from .types import decode_json, parse_request

TRUNCATED_HEADER = "X-MLX-Decision-Truncated"
# Room for several large images as data URLs.
MAX_BODY_BYTES = 64 * 2**20

logger = logging.getLogger(__name__)


class _Unauthorized(Exception):
    """A request to the API without the configured key."""


def error_response(
    status: int, message: str, kind: str, param: str | None = None, code: str | None = None
) -> JSONResponse:
    """An error in the API's error body shape."""
    error = {"message": message, "type": kind, "param": param}
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


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


def busy_response(error: Busy) -> JSONResponse:
    """The 503 for model work refused while something holds the GPU."""
    response = error_response(503, str(error), "api_error", code="busy")
    response.headers["Retry-After"] = str(RETRY_AFTER)
    return response


def create_app(
    gpu: Gpu | ModelPool, api_key: str | None = None, max_body_bytes: int = MAX_BODY_BYTES
) -> FastAPI:
    """The app, which owns ``gpu`` (a pool gets a ``Gpu`` of its own) and closes it
    when it stops; the default model, if there is one, loads at startup."""
    if isinstance(gpu, ModelPool):
        gpu = Gpu(gpu)
    pool = gpu.pool

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await gpu.start()
            yield
        finally:  # also when the default model fails to load
            gpu.close()

    app = FastAPI(title="mlx-decision", lifespan=lifespan)
    guards = []

    if api_key:
        expected = f"Bearer {api_key}".encode()

        async def check_api_key(request: Request) -> None:
            given = request.headers.get("authorization", "").encode()
            if not hmac.compare_digest(given, expected):
                raise _Unauthorized

        @app.exception_handler(_Unauthorized)
        async def unauthorized(request: Request, error: _Unauthorized):
            return error_response(401, "missing or invalid API key", "authentication_error")

        guards.append(Depends(check_api_key))

    # The key check runs before an endpoint reads the body.
    api = APIRouter(prefix="/v1", dependencies=guards)

    @api.post("/systemone")
    async def systemone(request: Request):
        try:
            gpu.check()  # before reading the body
        except Busy as error:
            return busy_response(error)
        raw = await read_body(request, max_body_bytes)
        if raw is None:
            return error_response(
                413,
                f"the request body is larger than {max_body_bytes / 2**20:.0f} MB",
                "invalid_request_error",
            )
        try:
            body = decode_json(raw, "the body")
        except DecisionError as error:
            return error_response(400, error.message, "invalid_request_error")
        name = body.get("model") if isinstance(body, dict) else None

        def answer():
            # What needs no model first, so a bad body loads and unloads nothing.
            chosen = pool.resolve(name if isinstance(name, str) else None)
            parse_request(body)
            return pool.get(chosen).decide_request(body)

        try:
            check_data_urls(body)
            result = await gpu.run(answer)
        except Busy as error:
            return busy_response(error)
        except DecisionError as error:
            return error_response(
                422, error.message, "invalid_request_error", error.param, error.code
            )
        except ModelTooLargeError as error:
            return error_response(422, str(error), "invalid_request_error", "model")
        except Exception:
            logger.exception("request failed")
            return error_response(500, "the model failed to answer the request", "api_error")
        headers = {TRUNCATED_HEADER: "true"} if result.truncated else None
        return JSONResponse(result.to_wire(), headers=headers)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": pool.default, "loaded": pool.loaded()}

    @api.get("/models")
    async def models():
        return {
            "models": [
                {
                    "name": name,
                    "description": f"{name}, served locally by mlx-decision",
                    # Unknown for a local model; the field is required.
                    "release_date": "",
                }
                for name in pool.names
            ]
        }

    app.include_router(api)
    return app
