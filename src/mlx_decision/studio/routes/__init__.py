"""The Studio's own API under ``/studio/api``: what the pages call.

Errors have the Jev API's error body (``{"error": {"message", "type",
"param", ...}}``): 400 for a bad body, 404 for a missing set or run, 409
for a set changed on disk or a name taken (with ``current``, the hash of
what is on disk), 422 for a request the model refuses, 503 while a job
holds the GPU. Bodies are JSON (the guard refuses anything else).
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ...errors import DecisionError
from ...gpu import Busy
from ...server import UnsupportedEncoding, busy_response, error_response, read_body
from ...types import decode_json
from ..workspace import Conflict, WorkspaceError


class ApiError(Exception):
    """An error answer from a Studio route."""

    def __init__(
        self,
        status: int,
        message: str,
        param: str | None = None,
        code: str | None = None,
        kind: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.status, self.message, self.param, self.code, self.kind = (
            status,
            message,
            param,
            code,
            kind,
        )


def install_errors(app: FastAPI) -> None:
    """Turn the routes' exceptions into error bodies."""

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError):
        return error_response(error.status, error.message, error.kind, error.param, error.code)

    @app.exception_handler(WorkspaceError)
    async def workspace_error(request: Request, error: WorkspaceError):
        body = {"message": error.message, "type": "invalid_request_error", "param": None}
        if isinstance(error, Conflict):
            body["current"] = error.current
        return JSONResponse({"error": body}, status_code=error.status)

    @app.exception_handler(DecisionError)
    async def decision_error(request: Request, error: DecisionError):
        return error_response(422, error.message, "invalid_request_error", error.param, error.code)

    @app.exception_handler(Busy)
    async def busy(request: Request, error: Busy):
        return busy_response(error)


async def json_body(request: Request, limit: int) -> Any:
    """The request's JSON body (zstd-compressed or not); ``ApiError`` 413 above
    ``limit`` bytes, 415 for another encoding, 400 when it is not JSON."""
    try:
        raw = await read_body(request, limit)
    except UnsupportedEncoding as error:
        raise ApiError(415, str(error)) from None
    except DecisionError as error:
        raise ApiError(400, error.message) from None
    if raw is None:
        raise ApiError(413, f"the request body is larger than {limit / 2**20:.0f} MB")
    try:
        return decode_json(raw, "the body")
    except DecisionError as error:
        raise ApiError(400, error.message) from None


def field(body: Any, key: str, kind: type | tuple[type, ...], required: bool = True) -> Any:
    """``body[key]``, checked to be of ``kind``; ``ApiError`` 400 otherwise."""
    if not isinstance(body, dict):
        raise ApiError(400, "the body must be a JSON object")
    if key not in body or body[key] is None:
        if required:
            raise ApiError(400, f"{key} is required", param=key)
        return None
    value = body[key]
    if not isinstance(value, kind) or isinstance(value, bool) and bool not in _types(kind):
        raise ApiError(400, f"{key} has the wrong type", param=key)
    return value


def _types(kind: type | tuple[type, ...]) -> tuple[type, ...]:
    return kind if isinstance(kind, tuple) else (kind,)
