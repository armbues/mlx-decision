"""The Settings page's API: where models live, and the config file.

``GET /studio/api/places`` lists the places the Studio's models come from:
the config's ``models_dir`` folder, the Hugging Face cache, the
workspace's ``models/`` (what the Studio made) and the folders named with
``-m`` or the config's ``models``, each with its path, how many listed
models are in it and whether downloads go there; and the config file with
each key's value in use, its source and what the file says.

``PUT /studio/api/settings/{key}`` ``{value}`` changes a key in the file
(``value`` null takes it out) and applies what the running Studio can
(``settings.Settings``). ``POST /studio/api/places/reveal`` shows a place
in Finder by its id (SEC-4: the page names a place, never a path). Both
only for a browser on this Mac; elsewhere the page is read-only.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response

from ...config import ConfigError
from ...gpu import Gpu
from ...memory import working_set
from ..catalog import Catalog
from ..clearing import FOLDERS as CLEARABLE
from ..clearing import Clearing
from ..removal import RemovalError
from ..security import on_this_mac
from ..settings import KEYS, READ_ONLY, Settings
from ..workspace import Workspace
from . import ApiError, field, json_body
from .sets import reveal


def settings_router(
    gpu: Gpu,
    files: Workspace,
    catalog: Catalog,
    settings: Settings,
    max_body_bytes: int,
    clearing: Clearing | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/studio/api")
    pool = gpu.pool

    def places() -> list[dict[str, Any]]:
        """Each place: ``id``, ``kind``, ``path`` (and ``display``, with ``~``),
        ``exists``, ``models`` (how many listed models are in it) and
        ``downloads`` (downloads go there). ``models_dir`` first, then the
        cache, the workspace's ``models/``, the named folders."""
        from huggingface_hub import constants

        models_dir = settings.config.models_dir
        folder = models_dir.expanduser() if isinstance(models_dir, Path) else None
        found = [
            {
                "id": "cache",
                "kind": "cache",
                "path": Path(constants.HF_HUB_CACHE),
                "downloads": folder is None,
            },
            {"id": "studio", "kind": "studio", "path": files.root / "models", "downloads": False},
        ]
        if folder is not None:
            found.insert(
                0, {"id": "models_dir", "kind": "models_dir", "path": folder, "downloads": True}
            )
        taken = {_resolved(place["path"]) for place in found}
        for reference in catalog.references:
            path = Path(reference).expanduser()
            if not path.is_dir() or _resolved(path) in taken:
                continue  # a repo id (in the cache) or listed already
            taken.add(_resolved(path))
            found.append(
                {"id": f"named-{len(found)}", "kind": "named", "path": path, "downloads": False}
            )
        specs = [_resolved(spec.path) for spec in pool.specs.values()]
        for place in found:
            root = _resolved(place["path"])
            place["exists"] = place["path"].is_dir()
            place["models"] = sum(1 for path in specs if path == root or path.is_relative_to(root))
            place["path"] = str(place["path"])
            place["display"] = tilde(place["path"])
        return found

    def find(place_id: str) -> Path:
        if place_id == "config":
            path = settings.config.path
            return path if path.exists() else path.parent
        if place_id == "workspace":
            return files.root
        if place_id == "data":  # the datasets' folder
            return files.root / "data"
        if place_id.startswith("folder:"):  # a workspace folder, as Settings lists them
            folder = place_id.removeprefix("folder:")
            if folder in CLEARABLE:
                return files.root / folder
        for place in places():
            if place["id"] == place_id:
                return Path(place["path"])
        raise ApiError(404, f"no place {place_id!r}", param="place")

    def only_on_this_mac(request: Request, what: str) -> None:
        if not on_this_mac(request):
            raise ApiError(403, f"{what} only from a browser on this Mac", kind="permission_error")

    @router.get("/places")
    async def list_places(request: Request):
        """``{places, config, editable, home, working_set_gb, api_key}``
        (``api_key``: whether this run needs one, not the key): ``config`` has ``path``,
        ``display``, ``exists``, ``warnings`` and ``settings`` (per key:
        ``key``, ``value`` in use, ``source``, ``in_file``, ``live``);
        ``editable``: this browser may change them (it is on this Mac)."""
        config = settings.config
        return {
            "places": await asyncio.to_thread(places),
            "config": {
                "path": str(config.path),
                "display": tilde(str(config.path)),
                "exists": config.path.is_file(),
                "warnings": list(config.warnings),
                "settings": await asyncio.to_thread(settings.listed),
            },
            "editable": on_this_mac(request),
            "home": str(Path.home()),  # the page writes paths under it with ~
            # The memory budget's slider goes up to the GPU's working set.
            "working_set_gb": round(await asyncio.to_thread(working_set) / 1e9, 1),
            "api_key": settings.api_key,  # whether this run needs one (host's note)
        }

    @router.post("/places/reveal", status_code=204)
    async def reveal_place(request: Request):
        """``{place}``: show it in Finder (a place's id from the list, "config"
        for the config file or, before it exists, its folder, "workspace" or
        "data", the workspace's datasets)."""
        only_on_this_mac(request, "Finder opens")
        place_id = field(await json_body(request, max_body_bytes), "place", str)
        path = await asyncio.to_thread(find, place_id)
        if not path.exists():
            raise ApiError(404, f"{path} does not exist yet", param="place")
        if sys.platform == "darwin":
            await asyncio.to_thread(reveal, path)
        return Response(status_code=204)

    @router.get("/workspace/folders")
    async def workspace_folders(request: Request):
        """``{folders, editable}``: the workspace's folders (``Clearing.folders``)
        and whether this browser may clear them (it is on this Mac)."""
        listed = await asyncio.to_thread(clearing.folders) if clearing is not None else []
        for entry in listed:
            entry["display"] = tilde(entry["path"])
        return {"folders": listed, "editable": on_this_mac(request)}

    @router.post("/workspace/folders/{folder}/clear")
    async def clear_folder(folder: str, request: Request):
        """``{folder, cleared, kept}``: everything in it deleted, as deleting each
        entry would (``Clearing.clear``). 404 for another folder, 409 while
        something in it is in use (nothing cleared), 403 off this Mac."""
        only_on_this_mac(request, "The workspace's folders are cleared")
        if clearing is None:
            raise ApiError(404, f"no folder {folder} to clear", param="folder")
        try:
            return await clearing.clear(folder)
        except KeyError:
            raise ApiError(
                404, f"{folder} is not one of the workspace's folders", param="folder"
            ) from None
        except RemovalError as error:
            raise ApiError(error.status, str(error), param="folder") from None

    @router.put("/settings/{key}")
    async def change(key: str, request: Request):
        """``{value}`` -> ``{setting, line, applies, note}``: the key written to
        the config file (``value`` null: taken out of it), ``applies`` "now"
        or "next start". 400 for a value the file's reader refuses, 409 for a
        file it cannot change safely, 403 off this Mac."""
        only_on_this_mac(request, "The config file changes")
        if key not in KEYS:
            raise ApiError(404, f"{key} is not a key of the config file", param="key")
        if key in READ_ONLY:
            raise ApiError(
                403,
                f"{key} is not changed from the Studio: pass --workspace, or edit the config file",
                param="key",
                kind="permission_error",
            )
        body = await json_body(request, max_body_bytes)
        if not isinstance(body, dict) or "value" not in body:
            raise ApiError(400, "value is required (null takes the key out)", param="value")
        try:
            result = await settings.change(key, body["value"])
        except ConfigError as error:
            message = str(error)
            status = (
                409 if "safely" in message or "cannot add" in message or "TOML" in message else 400
            )
            raise ApiError(status, message, param="value") from None
        setting = next(item for item in settings.listed() if item["key"] == key)
        return {"setting": setting, **result}

    return router


def tilde(path: str) -> str:
    """``path`` with the home folder written as ``~``, as people type it."""
    home = str(Path.home())
    return "~" + path[len(home) :] if path == home or path.startswith(home + "/") else path


def _resolved(path: Path) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path)
