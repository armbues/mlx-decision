"""The Studio's page and its static files.

The files ship in the package (``static/``): the page, its scripts and
styles, and the vendored Preact, htm and signals modules. They are served
under ``/static/<id>/``, ``<id>`` a hash of their contents, so the browser
may keep them for good (``immutable``) and still gets new ones after an
update; the page itself is never cached. Its Content-Security-Policy lets
it load and connect to this server only, which keeps the Studio working
offline and its page from reaching other hosts.
"""

import base64
import hashlib
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

STATIC = Path(__file__).parent / "static"
PLACEHOLDER = "__STATIC__"
IMMUTABLE = "public, max-age=31536000, immutable"
_IMPORT_MAP = re.compile(r'<script type="importmap">(.*?)</script>', re.S)


def static_id(folder: Path = STATIC) -> str:
    """A short hash of every file's path and bytes under ``folder``."""
    sha = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        sha.update(path.relative_to(folder).as_posix().encode() + b"\0")
        sha.update(path.read_bytes() + b"\0")
    return sha.hexdigest()[:12]


class Page:
    """The page's HTML and headers for one set of static files."""

    def __init__(self, folder: Path = STATIC):
        self.folder = folder
        self.prefix = f"/static/{static_id(folder)}"
        self.html = (folder / "index.html").read_text(encoding="utf-8")
        self.html = self.html.replace(PLACEHOLDER, self.prefix)
        script_hashes = [
            "'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'"
            for text in _IMPORT_MAP.findall(self.html)
        ]
        self.headers = {
            "Cache-Control": "no-store",
            "Content-Security-Policy": "; ".join(
                [
                    "default-src 'self'",
                    f"script-src 'self' {' '.join(script_hashes)}".rstrip(),
                    "style-src 'self' 'unsafe-inline'",
                    "img-src 'self' data: blob:",
                    "connect-src 'self'",
                    "object-src 'none'",
                    "base-uri 'none'",
                    "form-action 'self'",
                    "frame-ancestors 'none'",
                ]
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        }

    def mount(self, app: FastAPI) -> None:
        app.mount(self.prefix, ImmutableFiles(directory=self.folder), name="static")


class ImmutableFiles(StaticFiles):
    """Static files the browser keeps (their path changes with their contents)."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start" and message["status"] == 200:
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", IMMUTABLE.encode()))
                headers.append((b"x-content-type-options", b"nosniff"))
                message = {**message, "headers": headers}
            await send(message)

        await super().__call__(scope, receive, send_with_headers)
