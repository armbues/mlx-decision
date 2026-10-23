"""An HTTP server with an API compatible with Jev's: ``POST /v1/systemone``.

Needs the ``server`` extra (FastAPI, uvicorn). The models come from a
``ModelPool``; the request's ``model`` field picks one. All model work,
loading included, runs on the ``Gpu``'s one worker thread; requests wait
their turn there, and get a 503 while something holds the GPU (never
under the ``server`` command). With an API key, the ``/v1`` routes require ``Authorization: Bearer
<key>``; the check belongs to the routes, so it holds wherever the app is
mounted (a mount or a proxy's root path).
"""

import asyncio
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

try:  # Python 3.14's standard library, else the backport (the server extra's)
    from compression import zstd
except ImportError:
    try:
        from backports import zstd
    except ImportError:
        zstd = None

TRUNCATED_HEADER = "X-MLX-Decision-Truncated"
# Room for several large images as data URLs; for a compressed body, the decoded size.
MAX_BODY_BYTES = 64 * 2**20
# How a body may be encoded: as it is, or zstd-compressed, as the Jev SDK sends
# bodies of 1 KB or more on Python 3.14 or with its zstd extra.
ENCODINGS = ("identity", "zstd")
# A compressed body holds at most this many zstd frames (the SDK sends one):
# each frame costs a copy of the rest of the body, so many tiny ones would
# take quadratic time.
MAX_ZSTD_FRAMES = 16
# Wrong API keys a client may send in a row before it has to wait, and the
# longest wait (doubling from one second).
FREE_KEY_FAILURES = 10
MAX_KEY_WAIT_S = 60

logger = logging.getLogger(__name__)


class _Unauthorized(Exception):
    """A request to the API without the configured key; its argument, when
    given, is the seconds the client must wait before its next attempt."""


def error_response(
    status: int, message: str, kind: str, param: str | None = None, code: str | None = None
) -> JSONResponse:
    """An error in the API's error body shape."""
    error = {"message": message, "type": kind, "param": param}
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


class KeyAttempts:
    """Wrong API keys per client address, shared by the Jev API and the Studio's
    login: after ``FREE_KEY_FAILURES`` in a row a client waits, one second
    doubling up to ``MAX_KEY_WAIT_S``, before its next attempt is checked. Each
    wrong key is logged with the address; the right one clears the count."""

    def __init__(self, limit: int = 1024):
        self.limit = limit
        self.failures: dict[str, tuple[int, float]] = {}  # address -> (count, until)

    def wait(self, client: str | None) -> float:
        """Seconds ``client`` must still wait before its key is checked."""
        _, until = self.failures.get(client or "", (0, 0.0))
        return max(0.0, until - time.monotonic())

    def failed(self, client: str | None) -> None:
        key = client or ""
        count = self.failures.pop(key, (0, 0.0))[0] + 1
        until = 0.0
        if count > FREE_KEY_FAILURES:
            until = time.monotonic() + min(2 ** (count - FREE_KEY_FAILURES - 1), MAX_KEY_WAIT_S)
        self.failures[key] = (count, until)
        while len(self.failures) > self.limit:  # the oldest go first
            self.failures.pop(next(iter(self.failures)))
        logger.warning("wrong API key from %s (%d in a row)", key or "an unknown address", count)

    def succeeded(self, client: str | None) -> None:
        self.failures.pop(client or "", None)


def too_many_attempts(seconds: float) -> JSONResponse:
    response = error_response(
        429,
        "too many wrong API keys from this address; try again later",
        "rate_limit_error",
    )
    response.headers["Retry-After"] = str(max(1, round(seconds)))
    return response


class UnsupportedEncoding(Exception):
    """A request body in a ``Content-Encoding`` the server does not decode."""

    def __init__(self, encoding: str):
        super().__init__(
            f"Content-Encoding {encoding} is not supported: send the body as it is or "
            "zstd-compressed"
        )


def decode_zstd(data: bytes, limit: int) -> bytes | None:
    """The bytes of one or more zstd frames (at most ``MAX_ZSTD_FRAMES``), or None
    when they decode to more than ``limit`` bytes (decoding stops there). Raises
    ``DecisionError`` for data that is not zstd, is cut short or has more frames."""
    if zstd is None:
        raise UnsupportedEncoding("zstd (pip install backports.zstd)")
    parts, size, rest, frames = [], 0, data, 0
    try:
        while rest:
            frames += 1
            if frames > MAX_ZSTD_FRAMES:
                raise DecisionError(f"the body's zstd data has more than {MAX_ZSTD_FRAMES} frames")
            decompressor = zstd.ZstdDecompressor()
            part = decompressor.decompress(rest, max_length=limit - size + 1)
            while True:
                size += len(part)
                if size > limit:
                    return None
                parts.append(part)
                if decompressor.eof:
                    break
                if decompressor.needs_input:
                    raise DecisionError("the body's zstd data is cut short")
                part = decompressor.decompress(b"", max_length=limit - size + 1)
            rest = decompressor.unused_data
    except zstd.ZstdError as error:
        raise DecisionError(f"the body is not valid zstd data: {error}") from None
    return b"".join(parts)


async def read_body(request: Request, limit: int) -> bytes | None:
    """The request body, decoded when it is zstd-compressed; None when it is
    larger than ``limit`` bytes, sent or decoded.

    Raises ``UnsupportedEncoding`` (before reading) for another
    ``Content-Encoding``, and ``DecisionError`` for zstd data that does not decode.
    """
    encoding = request.headers.get("content-encoding", "").strip().lower() or "identity"
    if encoding not in ENCODINGS:
        raise UnsupportedEncoding(encoding)
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        return None
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    body = b"".join(chunks)
    if encoding == "zstd":  # off the event loop: decoding takes a while
        return await asyncio.to_thread(decode_zstd, body, limit)
    return body


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
    gpu: Gpu | ModelPool,
    api_key: str | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
    *,
    host: str | None = None,
    port: int | None = None,
) -> FastAPI:
    """The app, which owns ``gpu`` (a pool gets a ``Gpu`` of its own) and closes it
    when it stops; the default model, if there is one, loads at startup.

    With the address it serves (``host``, ``port``), requests must name it in
    ``Host`` and may not come from another site's pages (``Origin``), as in the
    Studio: a web page cannot use the server, nor read it by DNS rebinding.
    """
    if isinstance(gpu, ModelPool):
        gpu = Gpu(gpu)
    app = FastAPI(title="mlx-decision", lifespan=gpu_lifespan(gpu))
    add_jev_api(app, gpu, api_key, max_body_bytes)
    if host is not None and port is not None:
        from .studio.security import Guard, allowed_hosts

        app.add_middleware(Guard, hosts=allowed_hosts(host, port), session=None)
    return app


def add_jev_api(
    app: FastAPI,
    gpu: Gpu,
    api_key: str | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
    on_answer: Callable[[Answered], None] | None = None,
    attempts: KeyAttempts | None = None,
) -> None:
    """Add the Jev API (``/v1/systemone``, ``/v1/models``) and ``/health`` to ``app``.

    ``on_answer``, when given, is called on the event loop with each request
    answered, before the response is sent; it must not block. ``attempts``
    counts wrong keys (shared with the Studio's login).
    """
    pool = gpu.pool
    guards = []

    if api_key:
        expected = f"Bearer {api_key}".encode()
        attempts = attempts if attempts is not None else KeyAttempts()

        async def check_api_key(request: Request) -> None:
            client = request.client.host if request.client else None
            wait = attempts.wait(client)
            if wait > 0:
                raise _Unauthorized(wait)
            given = request.headers.get("authorization", "").encode()
            if not hmac.compare_digest(given, expected):
                attempts.failed(client)
                raise _Unauthorized
            attempts.succeeded(client)

        @app.exception_handler(_Unauthorized)
        async def unauthorized(request: Request, error: _Unauthorized):
            if error.args:
                return too_many_attempts(error.args[0])
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
        content_type = request.headers.get("content-type", "").strip().lower()
        if content_type.startswith("multipart/"):
            return error_response(
                415,
                "multipart bodies (the Jev API's image uploads) are not supported yet; send "
                "JSON, with images as data URLs in images",
                "invalid_request_error",
                code="unsupported_media",
            )
        try:
            raw = await read_body(request, max_body_bytes)
        except UnsupportedEncoding as error:
            return error_response(415, str(error), "invalid_request_error")
        except DecisionError as error:
            return error_response(400, error.message, "invalid_request_error")
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
        except ModelTooLargeError as error:  # a DecisionError, named by the model
            return error_response(422, str(error), "invalid_request_error", "model")
        except DecisionError as error:
            return error_response(
                422, error.message, "invalid_request_error", error.param, error.code
            )
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
