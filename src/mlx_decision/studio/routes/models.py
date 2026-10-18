"""Load and Unload: a model put into memory or taken out of it by hand.

Both run on the GPU worker like the answers, so they wait for the request
in progress and are refused (503) while a job holds the GPU. Unloading a
model frees everything that shares its weights (a base and the fitted
models on it), as making room does. The status message that follows the
pool's change keeps every open tab current.
"""

import logging
import time

from fastapi import APIRouter, Request

from ...gpu import Busy, Gpu
from ...memory import ModelTooLargeError
from . import ApiError, field, json_body

logger = logging.getLogger(__name__)


def models_router(gpu: Gpu, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/models")
    pool = gpu.pool

    async def named(request: Request) -> str:
        gpu.check()  # before reading the body
        name = field(await json_body(request, max_body_bytes), "model", str)
        if name not in pool.specs:
            raise ApiError(404, f"no model is named {name}", param="model")
        return name

    @router.post("/load")
    async def load(request: Request):
        """``{model}`` -> ``{model, loaded, unloaded, load_ms}``: ``loaded`` the
        models in memory afterwards, ``unloaded`` the ones that made room,
        ``load_ms`` None when it was loaded already."""
        name = await named(request)

        def run() -> dict:
            before = pool.loaded()
            started = time.perf_counter()
            pool.get(name)
            seconds = time.perf_counter() - started
            after = pool.loaded()
            return {
                "model": name,
                "loaded": after,
                "unloaded": [other for other in before if other not in after],
                "load_ms": None if name in before else round(seconds * 1000, 3),
            }

        try:
            return await gpu.run(run)
        except Busy:
            raise
        except ModelTooLargeError as error:
            raise ApiError(422, str(error), param="model") from None
        except Exception:
            logger.exception("loading %s failed", name)
            raise ApiError(500, f"{name} could not be loaded", kind="api_error") from None

    @router.post("/unload")
    async def unload(request: Request):
        """``{model}`` -> ``{model, unloaded}``: the names unloaded with it
        (none when it was not loaded)."""
        name = await named(request)
        return {"model": name, "unloaded": await gpu.run(pool.unload, name)}

    return router
