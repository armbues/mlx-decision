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
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import DecisionError
from .gpu import RETRY_AFTER, Busy, Gpu
from .images import check_data_urls
from .memory import ModelTooLargeError
from .pool import ModelPool
from .types import Result, decode_json, parse_request

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


@dataclass(frozen=True)
class Answered:
    """A request answered: by which model, how long answering took (without
    the load) and the load's time when the request loaded the model."""

    model: str
    result: Result
    seconds: float
    load_seconds: float | None


def answer(
    pool: ModelPool,
    name: str | None,
    body: Any,
    check: Callable[[str, Any], None] | None = None,
) -> Answered:
    """Answer ``body`` with the model ``name`` resolves to; run it on the worker.

    What needs no model comes first (``parse_request``, then ``check`` with the
    chosen model's name), so a bad body loads and unloads nothing.
    """
    chosen = pool.resolve(name)
    parse_request(body)
    if check is not None:
        check(chosen, body)
    load_seconds = None
    if chosen not in pool.loaded():
        started = time.perf_counter()
        model = pool.get(chosen)
        load_seconds = time.perf_counter() - started
    else:
        model = pool.get(chosen)
    started = time.perf_counter()
    result = model.decide_request(body)
    return Answered(chosen, result, time.perf_counter() - started, load_seconds)


def busy_response(error: Busy) -> JSONResponse:
    """The 503 for model work refused while something holds the GPU."""
    response = error_response(503, str(error), "api_error", code="busy")
    response.headers["Retry-After"] = str(RETRY_AFTER)
    return response


def gpu_lifespan(gpu: Gpu):
    """A lifespan that starts ``gpu`` (its default model loads) and closes it at the end."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await gpu.start()
            yield
        finally:  # also when the default model fails to load
            gpu.close()

    return lifespan


def create_app(
    gpu: Gpu | ModelPool, api_key: str | None = None, max_body_bytes: int = MAX_BODY_BYTES
) -> FastAPI:
    """The app, which owns ``gpu`` (a pool gets a ``Gpu`` of its own) and closes it
    when it stops; the default model, if there is one, loads at startup."""
    if isinstance(gpu, ModelPool):
        gpu = Gpu(gpu)
    app = FastAPI(title="mlx-decision", lifespan=gpu_lifespan(gpu))
    add_jev_api(app, gpu, api_key, max_body_bytes)
    return app


def add_jev_api(
    app: FastAPI,
    gpu: Gpu,
    api_key: str | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
    on_answer: Callable[[Answered], None] | None = None,
) -> None:
    """Add the Jev API (``/v1/systemone``, ``/v1/models``) and ``/health`` to ``app``.

    ``on_answer``, when given, is called on the event loop with each request
    answered, before the response is sent; it must not block.
    """
    pool = gpu.pool
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

        try:
            check_data_urls(body)
            answered = await gpu.run(answer, pool, name if isinstance(name, str) else None, body)
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
        if on_answer is not None:
            try:
                on_answer(answered)
            except Exception:
                logger.exception("the answer callback failed")
        result = answered.result
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
