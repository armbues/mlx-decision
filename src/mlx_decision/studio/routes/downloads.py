"""Download Model: what the sheet offers and the check before a download.

``GET /studio/api/downloads`` lists the families ``download``'s menu offers,
in its order, with their models, their licence notes and which of those
models are on this Mac already, plus where a download goes.
``POST /studio/api/downloads/check`` runs ``download``'s own check on the
Hub (a repo's JSON files, no weights): the family that would load it, or
the reason none does, and per way of storing it (full size, 8-bit, 4-bit)
the size and whether it fits. The download itself is a job
(``POST /studio/api/jobs`` with ``kind`` "download").

The fit check is ``download``'s (the weights plus 10% against this Mac's
GPU working set), with one step more: a model that fits the Mac but not
the Studio's memory budget cannot be loaded here. Quantized sizes are
estimates and say so.
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ...config import HF_CACHE, Config
from ...download import RepoCheck, check_repo
from ...gpu import Gpu
from ...hub import http_errors, snapshot_revision
from ...memory import MARGIN, QUANTIZED_BITS, working_set
from ...registry import families, get_family
from . import ApiError, field, json_body
from .jobs import REPO_ID

logger = logging.getLogger(__name__)


def downloads_router(gpu: Gpu, config: Callable[[], Config], max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/downloads")
    pool = gpu.pool

    def destination() -> dict[str, Any]:
        """Where a download goes: models_dir's folder, else the Hub cache."""
        models_dir = config().models_dir
        if isinstance(models_dir, Path):
            return {"kind": "folder", "path": str(models_dir.expanduser()), "setting": "models_dir"}
        from huggingface_hub import constants

        setting = "models_dir" if models_dir == HF_CACHE else None
        return {"kind": "cache", "path": str(constants.HF_HUB_CACHE), "setting": setting}

    def on_this_mac() -> Callable[[str], str | None]:
        """The name a repo's model is listed under: a snapshot of it in the Hub
        cache, or a folder of its name in models_dir (where ``download`` puts
        it); None when neither is listed."""
        cached, in_folder = {}, {}
        models_dir = config().models_dir
        folder = models_dir.expanduser() if isinstance(models_dir, Path) else None
        for name, spec in pool.specs.items():
            if spec.fit is not None:
                continue
            snapshot = snapshot_revision(spec.path)
            if snapshot is not None:
                cached[snapshot[0]] = name
            elif folder is not None and spec.path.parent == folder:
                in_folder[spec.path.name] = name
        return lambda repo: cached.get(repo) or in_folder.get(repo.rsplit("/", 1)[-1])

    @router.get("")
    async def offered():
        """``{families, destination}``: each family (``name``, ``title``,
        ``maker``, ``summary``, ``licence``) with its ``models`` (``repo``,
        ``description``, ``size`` as listed, ``on_this_mac``: the name it is
        listed under, else None)."""
        present = on_this_mac()
        listed = []
        for family in families():
            if not family.listing or not family.known_models:
                continue
            listed.append(
                {
                    "name": family.name,
                    "title": family.listing.title,
                    "maker": family.listing.maker,
                    "summary": family.listing.summary,
                    "licence": family.licence,
                    "models": [
                        {
                            "repo": model.repo_id,
                            "description": model.description,
                            "size": model.size,
                            "on_this_mac": present(model.repo_id),
                        }
                        for model in family.known_models
                    ],
                }
            )
        return {"families": listed, "destination": destination()}

    @router.post("/check")
    async def check(request: Request):
        """``{repo}`` -> what ``download`` would fetch (see ``describe``) and
        ``on_this_mac`` (the name it is listed under, else None). 404 for
        a repo the Hub does not have (or will not show without a token), 502
        when the Hub cannot be asked."""
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

        repo = field(await json_body(request, max_body_bytes), "repo", str).strip()
        if not REPO_ID.fullmatch(repo) or ".." in repo:
            raise ApiError(400, f"not a Hugging Face repo id: {repo!r}", param="repo")
        try:
            found = await asyncio.to_thread(check_repo, repo)
            budget = pool.limit()
            mac = await asyncio.to_thread(working_set)
        except RepositoryNotFoundError:
            raise ApiError(
                404, f"{repo}: not found on the Hub (or private: set HF_TOKEN)", param="repo"
            ) from None
        except (HfHubHTTPError, *http_errors(), OSError, ValueError) as error:
            message = str(error).splitlines()[0] if str(error) else type(error).__name__
            raise ApiError(502, f"{repo}: {message}", param="repo", kind="api_error") from None
        return describe(found, budget, mac) | {"on_this_mac": on_this_mac()(repo)}

    return router


def describe(check: RepoCheck, budget: int, mac: int) -> dict[str, Any]:
    """The check as the sheet shows it: ``family`` (None with ``reason`` when no
    family loads it), ``licence``, and ``options``: full size (``bits`` 0,
    exact) and, for a family that can store it quantized, 8-bit and 4-bit
    (``estimate``: about); each with ``bytes`` on disk, ``memory_bytes`` to
    run (an estimate: the weights plus 10%) and ``fit``: "ok" (within the
    Studio's memory budget), "over_budget" (fits this Mac, not the budget) or
    "too_large". ``default_bits``: the largest that fits the budget, else the
    Mac, else the smallest."""
    family = get_family(check.family) if check.family is not None else None
    options = []
    if family is not None:
        for bits in (0, *QUANTIZED_BITS) if check.convertible else (0,):
            weights = check.quantized_weight_bytes(bits) if bits else check.weight_bytes
            memory = round(weights * MARGIN)
            fit = "ok" if memory <= budget else "over_budget" if memory <= mac else "too_large"
            options.append(
                {
                    "bits": bits,
                    "bytes": check.quantized_bytes(bits) if bits else check.size_bytes,
                    "estimate": bool(bits),
                    "memory_bytes": memory,
                    "fit": fit,
                }
            )
    default = next((o for o in options if o["fit"] == "ok"), None)
    default = default or next((o for o in options if o["fit"] == "over_budget"), None)
    default = default or (options[-1] if options else None)
    return {
        "repo": check.repo_id,
        "revision": check.revision,
        "family": check.family,
        "family_title": family.listing.title if family and family.listing else check.family,
        "reason": check.reason or None,
        "licence": family.licence if family is not None else None,
        "size_bytes": check.size_bytes,
        "options": options,
        "default_bits": default["bits"] if default else None,
        "budget": budget,
    }
