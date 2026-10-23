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
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ...config import HF_CACHE, Config, ConfigError, set_models_dir
from ...config import load as load_config
from ...download import RepoCheck, check_repo, recorded_repo
from ...gpu import Gpu
from ...hub import http_errors, snapshot_revision
from ...memory import MARGIN, QUANTIZED_BITS, working_set
from ...registry import families, get_family
from ..catalog import Catalog
from ..security import on_this_mac as browser_on_this_mac
from . import ApiError, field, json_body
from .jobs import REPO_ID

logger = logging.getLogger(__name__)


def downloads_router(
    gpu: Gpu,
    config: Callable[[], Config],
    max_body_bytes: int,
    set_config: Callable[[Config], None] = lambda config: None,
    catalog: Catalog | None = None,
    choose: random.Random | None = None,
) -> APIRouter:
    """``config`` returns the config file as it is now; ``set_config`` takes
    it after the first start wrote ``models_dir``, and ``catalog`` gets the
    folder chosen. ``choose`` picks the first start's models (random)."""
    router = APIRouter(prefix="/studio/api/downloads")
    choose = choose or random.Random()
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
                in_folder[spec.path.name] = (name, recorded_repo(spec.path))

        def found(repo: str) -> str | None:
            if repo in cached:
                return cached[repo]
            name, recorded = in_folder.get(repo.rsplit("/", 1)[-1], (None, None))
            return name if recorded in (None, repo) else None  # another repo of that name

        return found

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

    @router.get("/first")
    async def first(request: Request):
        """``{picks, models_dir_set, elsewhere, config_path, suggestion}`` for the
        first start: one small, one medium and one large model that fit the
        Studio's memory budget at full size (``first_picks``), whether the config
        file says where models go, whether the browser is on another machine
        (which cannot set it), and the folder offered when it does not."""
        budget = await asyncio.to_thread(pool.limit)
        current = config()
        return {
            "picks": first_picks(budget, choose),
            "models_dir_set": current.models_dir is not None,
            "elsewhere": not browser_on_this_mac(request),
            "config_path": str(current.path),
            "suggestion": "~/Models",
        }

    @router.post("/destination")
    async def set_destination(request: Request):
        """``{place: "hf-cache"}`` or ``{place: "folder", path}`` -> ``{line,
        config_path, destination}``: where models go, added to the config file
        as ``models_dir`` (the first start's question, asked once). The folder
        must be under the home folder; it is created. 409 when the file says
        already; 403 for a browser elsewhere (the config file changes only from
        this Mac)."""
        if not browser_on_this_mac(request):
            raise ApiError(
                403,
                "The config file changes only from a browser on this Mac",
                kind="permission_error",
            )
        body = await json_body(request, max_body_bytes)
        place = field(body, "place", str)
        current = config()
        if current.models_dir is not None:
            raise ApiError(409, f"{current.path} sets models_dir already", param="place")
        if place == "hf-cache":
            folder = None
        elif place == "folder":
            folder = models_folder(field(body, "path", str))
        else:
            raise ApiError(400, 'place must be "hf-cache" or "folder"', param="place")
        try:
            if folder is not None:
                await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            line = await asyncio.to_thread(set_models_dir, folder, current.path)
            updated = await asyncio.to_thread(load_config, current.path)
        except ConfigError as error:
            raise ApiError(409, str(error), param="place") from None
        except OSError as error:
            raise ApiError(400, f"{folder}: {error.strerror or error}", param="path") from None
        set_config(updated)
        if folder is not None and catalog is not None:
            catalog.models_dir = folder
        logger.info("added %s to %s", line, current.path)
        return {"line": line, "config_path": str(current.path), "destination": destination()}

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


def models_folder(text: str) -> Path:
    """The folder the first start names for models: a path under the home folder
    (``~`` allowed), not the home folder itself and not a file (SEC-4: the one
    path the page sends, MOD-7). Raises ``ApiError`` 400."""
    text = text.strip()
    home = Path.home().resolve()
    path = Path(text).expanduser()
    if not text or not path.is_absolute():
        raise ApiError(400, "give a folder under your home folder, e.g. ~/Models", param="path")
    folder = path.resolve()
    if folder == home or not folder.is_relative_to(home):
        raise ApiError(400, f"{text}: not a folder under your home folder ({home})", param="path")
    if folder.exists() and not folder.is_dir():
        raise ApiError(400, f"{text}: a file is there", param="path")
    return folder


# Listed sizes ("843 MB", "4.4 GB") in bytes.
UNITS = {"MB": 10**6, "GB": 10**9}


def listed_bytes(size: str) -> int:
    number, unit = size.split()
    return round(float(number) * UNITS[unit])


# The first start's size classes, by download size: small below the first,
# large above the second.
SMALL_BELOW = 2 * 10**9
LARGE_ABOVE = 8 * 10**9


def first_picks(budget: int, choose: random.Random) -> list[dict[str, Any]]:
    """Three models for a first download, from the known models that fit
    ``budget`` at full size (their listed size plus 10%): one chosen at random
    among the small ones (below 2 GB), one among the medium ones and one among
    the large ones (above 8 GB), the large first, then the others preferring
    families not chosen yet. When a class has none that fits (a small Mac),
    what fits is cut in thirds by size instead; with fewer than four that fit,
    each is a pick."""
    fitting = []
    for family in families():
        if not family.listing:
            continue
        for model in family.known_models:
            size = listed_bytes(model.size)
            if size * MARGIN <= budget:
                fitting.append((size, family, model))
    fitting.sort(key=lambda item: item[0])
    names = ("small", "medium", "large")
    classes = [
        [item for item in fitting if item[0] < SMALL_BELOW],
        [item for item in fitting if SMALL_BELOW <= item[0] <= LARGE_ABOVE],
        [item for item in fitting if item[0] > LARGE_ABOVE],
    ]
    if all(classes):
        tiers = list(zip(names, classes, strict=True))
    elif len(fitting) <= 3:
        tiers = [(names[i], [item]) for i, item in enumerate(fitting)]
    else:
        count = len(fitting)
        cut = [0, count // 3, 2 * count // 3, count]
        tiers = [(name, fitting[cut[i] : cut[i + 1]]) for i, name in enumerate(names)]
    chosen: dict[str, tuple] = {}
    taken: set[str] = set()
    for name, items in reversed(tiers):
        fresh = [item for item in items if item[1].name not in taken]
        item = choose.choice(fresh or items)
        chosen[name] = item
        taken.add(item[1].name)
    picks = []
    for name, _ in tiers:
        size, family, model = chosen[name]
        picks.append(
            {
                "tier": name,
                "repo": model.repo_id,
                "family": family.name,
                "family_title": family.listing.title,
                "maker": family.listing.maker,
                "description": model.description,
                "size": model.size,
                "bytes": size,
                "memory_bytes": round(size * MARGIN),
            }
        )
    return picks
