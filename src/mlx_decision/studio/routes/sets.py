"""Question sets and their history: the Playground's files in the workspace.

A set is saved with the hash it was loaded with (``loaded``), so a set
changed on disk since is not overwritten (409 with the hash now on disk).
Import takes a file's text as the browser read it (a ``-q`` questions map
or a whole request body) and saves it under a free name. Templates are
the sets that ship with the Studio; one is copied into the workspace under
the name the page gives. Nothing here takes a path: only names inside the
workspace.
"""

import asyncio
import subprocess
import sys
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from ...errors import DecisionError
from ...types import decode_json
from ..security import is_loopback
from ..templates import UnknownTemplate, list_templates, template_set
from ..workspace import Conflict, NotFound, Workspace, check_name
from . import ApiError, field, json_body

# Import tries ``name``, then ``name-2``, ``name-3``, ...
IMPORT_TRIES = 100


def as_set(data: Any) -> dict[str, Any]:
    """A file's JSON as a set, as ``run -q`` reads it: a body with ``questions``
    as it is, anything else as the questions."""
    if isinstance(data, dict) and "questions" in data:
        return dict(data)
    return {"questions": data}


def sets_router(files: Workspace, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api")

    def run(function, *args):
        return asyncio.to_thread(function, *args)

    @router.get("/sets")
    async def list_sets():
        return {"sets": await run(files.list_sets)}

    @router.get("/sets/{name}")
    async def load_set(name: str):
        body, digest = await run(files.load_set, name)
        return {"name": name, "body": body, "sha256": digest}

    @router.put("/sets/{name}")
    async def save_set(name: str, request: Request):
        """``{body, loaded}``: ``loaded`` is the hash the set was loaded with, or
        null for a new set (the name must be free)."""
        data = await json_body(request, max_body_bytes)
        body = field(data, "body", dict)
        loaded = field(data, "loaded", str, required=False)
        digest = await run(files.save_set, name, body, loaded)
        return {"name": name, "sha256": digest}

    @router.post("/sets/{name}/rename")
    async def rename_set(name: str, request: Request):
        """``{name}``: the new name; the set's history moves with it."""
        new_name = field(await json_body(request, max_body_bytes), "name", str)
        await run(files.rename_set, name, new_name)
        return {"name": new_name}

    @router.delete("/sets/{name}", status_code=204)
    async def delete_set(name: str):
        await run(files.delete_set, name)
        return Response(status_code=204)

    @router.post("/sets/import")
    async def import_set(request: Request):
        """``{name, text}``: a file's name (without ``.json``) and its text; saved
        under ``name`` or the first free ``name-N``."""
        data = await json_body(request, max_body_bytes)
        name = check_name(field(data, "name", str), "a set's name")
        text = field(data, "text", str)
        try:
            body = as_set(decode_json(text, "the file"))
        except DecisionError as error:
            raise ApiError(400, error.message, param="text") from None

        def save() -> tuple[str, str]:
            for attempt in range(1, IMPORT_TRIES + 1):
                candidate = name if attempt == 1 else f"{name[:76]}-{attempt}"
                try:
                    return candidate, files.save_set(candidate, body)
                except Conflict:
                    continue
            raise ApiError(409, f"no free name for {name!r}", param="name")

        saved, digest = await run(save)
        return {"name": saved, "sha256": digest, "body": body}

    @router.get("/templates")
    async def templates():
        return {"templates": list_templates()}

    @router.post("/sets/{name}/from-template")
    async def from_template(name: str, request: Request):
        """``{template}``: a new set ``name`` (which must be free) holding the
        template's questions and examples."""
        template = field(await json_body(request, max_body_bytes), "template", str)
        try:
            body = template_set(template)
        except UnknownTemplate:
            raise NotFound(f"no template named {template!r}") from None
        digest = await run(files.save_set, name, body, None)
        return {"name": name, "sha256": digest, "body": body}

    @router.get("/sets/{name}/export")
    async def export_set(name: str):
        """The set's file, as a download."""
        path = files.set_path(name)
        if not path.is_file():
            raise NotFound(f"no set named {name!r}")
        return FileResponse(path, media_type="application/json", filename=path.name)

    @router.post("/sets/{name}/reveal", status_code=204)
    async def reveal_set(name: str, request: Request):
        """Show the set's file in Finder (only for a browser on this Mac)."""
        client = request.client.host if request.client else None
        if not is_loopback(client):
            raise ApiError(
                403, "Finder opens only for a browser on this Mac", kind="permission_error"
            )
        path = files.set_path(name)
        if not path.is_file():
            raise NotFound(f"no set named {name!r}")
        if sys.platform == "darwin":
            await run(reveal, path)
        return Response(status_code=204)

    @router.get("/history/{name}")
    async def history(name: str):
        """The set's runs, oldest first; images as ``{"sha256", "type"}`` (fetch
        them from ``/images/<sha256>``)."""
        return {"runs": await run(files.read_history, name)}

    @router.get("/history/{name}/{run_id}")
    async def history_run(name: str, run_id: str):
        """One run, its request's images as data URLs again (for Restore)."""

        def find() -> dict[str, Any]:
            for entry in files.read_history(name):
                if entry.get("id") == run_id:
                    request = entry.get("request")
                    if isinstance(request, dict):
                        entry = {**entry, "request": files.restore_images(request)}
                    return entry
            raise NotFound(f"no run {run_id!r} in the history of {name!r}")

        return await run(find)

    @router.delete("/history/{name}", status_code=204)
    async def clear_history(name: str):
        await run(files.clear_history, name)
        return Response(status_code=204)

    @router.get("/images/{digest}")
    async def image(digest: str):
        path = files.image_file(digest)
        if path is None:
            raise NotFound("no such image")
        media_type = {"jpg": "image/jpeg", "svg": "image/svg+xml"}.get(
            path.suffix[1:], f"image/{path.suffix[1:]}"
        )
        headers = {
            "Cache-Control": "private, max-age=31536000, immutable",
            # Shown as an image only: a stored SVG's scripts never run.
            "Content-Security-Policy": "sandbox",
            "X-Content-Type-Options": "nosniff",
        }
        return FileResponse(path, media_type=media_type, headers=headers)

    return router


def reveal(path) -> None:
    subprocess.run(["open", "-R", str(path)], check=False, timeout=10)
