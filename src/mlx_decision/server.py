"""An HTTP server with the Jev API: ``POST /v1/systemone``.

Needs the ``server`` extra (FastAPI, uvicorn). All model work, loading
included, runs on one worker thread; requests wait their turn there.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .errors import DecisionError
from .model import DecisionModel

TRUNCATED_HEADER = "X-MLX-Decision-Truncated"

logger = logging.getLogger(__name__)


def error_response(
    status: int, message: str, kind: str, param: str | None = None, code: str | None = None
) -> JSONResponse:
    """An error in the body shape the Jev service uses."""
    error = {"message": message, "type": kind, "param": param}
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


def create_app(load_model: Callable[[], DecisionModel]) -> FastAPI:
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

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            body = json.loads(await request.body())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            return error_response(
                400, f"the body is not valid JSON: {error}", "invalid_request_error"
            )
        model: DecisionModel = request.app.state.model
        try:
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
