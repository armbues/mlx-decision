"""Load and Unload: a model put into memory or taken out of it by hand.

Both run on the GPU worker like the answers, so they wait for the request
in progress and are refused (503) while a job holds the GPU. ``rescan``
looks for models that came or went (``rescan.Rescans``). Unloading a
model frees everything that shares its weights (a base and the fitted
models on it), as making room does. The status message that follows the
pool's change keeps every open tab current.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Request

from ...gpu import Busy, Gpu
from ...memory import (
    MARGIN,
    QUANTIZED_BITS,
    ModelTooLargeError,
    quantizable_bytes,
    quantized_size,
    weight_bytes,
    working_set,
)
from ...pool import ModelSpec
from ...registry import MARKER_FILE
from ..rescan import Rescans
from . import ApiError, field, json_body

logger = logging.getLogger(__name__)


def models_router(gpu: Gpu, max_body_bytes: int, rescans: Rescans | None = None) -> APIRouter:
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

    @router.get("/convert")
    async def convert_plan(model: str):
        """``?model=NAME`` -> what Convert offers for it (``convert_options``);
        404 for an unknown model."""
        if model not in pool.specs:
            raise ApiError(404, f"no model is named {model}", param="model")
        spec = pool.specs[model]
        reason = not_convertible(spec)
        if reason is not None:
            return {"model": model, "convertible": False, "reason": reason, "options": []}

        def plan() -> dict[str, Any]:
            return convert_options(spec, pool.limit(), working_set())

        try:
            return {
                "model": model,
                "convertible": True,
                "reason": None,
                **await asyncio.to_thread(plan),
            }
        except (OSError, ValueError, KeyError) as error:
            raise ApiError(422, f"cannot read {model}'s weights: {error}", param="model") from None

    @router.post("/rescan")
    async def rescan():
        """``{added, removed}``: the models looked for again (the page asks when it
        regains focus: the CLI may have downloaded or deleted one). Also while
        a job holds the GPU: nothing is loaded."""
        if rescans is None:
            return {"added": [], "removed": []}
        return await rescans.run()

    return router


def not_convertible(spec: ModelSpec) -> str | None:
    """Why Convert does not offer ``spec``, or None when it does."""
    if spec.fit is not None:
        return "a fitted model is not converted: convert its base, then fit the copy"
    if spec.family.converter is None:
        title = spec.family.listing.title if spec.family.listing else spec.family.name
        return f"{title} models are not stored quantized"
    if (spec.path / MARKER_FILE).exists():
        return "it is a converted copy already: convert the original"
    return None


def convert_options(spec: ModelSpec, budget: int, mac: int) -> dict[str, Any]:
    """The sizes Convert shows (estimates, as ``convert``'s memory hint has them):
    ``source_bytes`` (the weights now) and per ``bits`` the copy's ``bytes`` on
    disk, ``memory_bytes`` to run (plus 10%) and ``fit`` ("ok", "over_budget",
    "too_large", as Download Model has it)."""
    weights = weight_bytes(spec.path, spec.options)
    quantizable = quantizable_bytes(spec.path)
    options = []
    for bits in QUANTIZED_BITS:
        size = weights - quantizable + quantized_size(quantizable, bits)
        memory = round(size * MARGIN)
        fit = "ok" if memory <= budget else "over_budget" if memory <= mac else "too_large"
        options.append({"bits": bits, "bytes": size, "memory_bytes": memory, "fit": fit})
    return {"source_bytes": weights, "options": options}
