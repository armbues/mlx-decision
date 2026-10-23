"""The Studio's protections: Host and Origin checks, the session cookie, the login.

Every request, the Jev API's included, must name an address the Studio
serves in ``Host`` (against DNS rebinding) and, when it sends an
``Origin``, the page's own origin; anything else gets 403 before a route
runs, so it does no model work. Clients that send no ``Origin`` (the Jev
SDK, curl) are unaffected. No CORS headers are ever sent.

The Studio's pages and API need a session: a random token per process,
part of the address ``web`` prints, is exchanged for an HttpOnly,
SameSite=Strict cookie on the first visit. With an API key the token is
accepted only from this machine; a browser elsewhere logs in with the
key instead. The Jev API keeps its own check (the key, when set). A client that sends
wrong keys in a row has to wait before trying again (``KeyAttempts``,
shared by the login and the Jev API).

Requests to the Studio other than GET and HEAD must be JSON, which with
the Origin check and SameSite closes cross-site form posts.
"""

import hmac
import ipaddress
import json
import secrets

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.datastructures import Headers
from starlette.requests import cookie_parser
from starlette.types import ASGIApp, Receive, Scope, Send

from ..server import KeyAttempts, error_response, read_body, too_many_attempts

LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
# Bind addresses that serve every address of the machine.
WILDCARDS = ("0.0.0.0", "::", "")
# Paths that belong to the Studio rather than the Jev API.
STUDIO_PREFIX = "/studio/"
API_PREFIX = "/studio/api/"
LOGIN_PATH = "/login"
SAFE_METHODS = ("GET", "HEAD")
MAX_LOGIN_BYTES = 4096
NO_STORE = {"Cache-Control": "no-store"}


def is_loopback(address: str | None) -> bool:
    """Whether ``address`` (a bind address or a client's IP) is this machine."""
    if address is None:
        return False
    address = address.strip("[]")
    if address == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _address(text: str):
    """``text`` as an IP address (IPv4-mapped IPv6 as IPv4), or None."""
    try:
        ip = ipaddress.ip_address(text.strip("[]"))
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def on_this_mac(request: Request) -> bool:
    """Whether the request comes from a browser on this machine: from a loopback
    address, or from the very address it reached the server on (a browser here
    opening the Studio by this Mac's network address; another machine's
    address differs)."""
    client = request.client.host if request.client else None
    if is_loopback(client):
        return True
    local = (request.scope.get("server") or (None,))[0]
    if client is None or local is None:
        return False
    return _address(client) is not None and _address(client) == _address(str(local))


def check_bind(host: str, api_key: str | None) -> None:
    """Refuse to serve beyond this machine without an API key (``ValueError``)."""
    if not api_key and not is_loopback(host):
        raise ValueError(
            f"--host {host or '(all addresses)'} serves the Studio beyond this machine; "
            "that needs an API key (--api-key or MLX_DECISION_API_KEY)"
        )


def allowed_hosts(host: str, port: int) -> frozenset[str] | None:
    """The ``Host`` header values the Studio answers, or None for any.

    Those are the loopback names and the bind address, with the port (and
    without it on port 80, where browsers leave it out). A wildcard bind
    serves every name the machine has, so the header is not checked; that
    needs an API key (``check_bind``), and the key then guards both the
    Jev API and the Studio's login.
    """
    if host in WILDCARDS:
        return None
    names = set(LOOPBACK_NAMES)
    try:
        if isinstance(ipaddress.ip_address(host.strip("[]")), ipaddress.IPv6Address):
            host = f"[{host.strip('[]')}]"
    except ValueError:
        pass
    names.add(host.lower())
    values = {f"{name}:{port}" for name in names}
    if port == 80:
        values |= names
    return frozenset(values)


class Session:
    """The process's session token (in the address) and cookie value."""

    def __init__(self, port: int, token: str | None = None):
        self.token = token or secrets.token_urlsafe(32)
        self._secret = secrets.token_urlsafe(32)
        # Cookies ignore the port: a name per port keeps two Studios apart.
        self.cookie = f"mlxd_session_{port}"

    def matches_token(self, token: str) -> bool:
        return hmac.compare_digest(token.encode(), self.token.encode())

    def has_cookie(self, cookies: dict[str, str]) -> bool:
        given = cookies.get(self.cookie, "")
        return hmac.compare_digest(given.encode(), self._secret.encode())

    def set_cookie(self, response: Response) -> Response:
        response.set_cookie(self.cookie, self._secret, path="/", httponly=True, samesite="strict")
        return response


def _cookies(headers: Headers) -> dict[str, str]:
    # Starlette's lenient parser (as ``request.cookies``): cookies ignore the
    # port, so another app on this machine may set ones that the standard
    # library's strict parser rejects, and it would then drop ours with them.
    return cookie_parser(headers.get("cookie", ""))


class Guard:
    """ASGI middleware: Host, Origin, JSON-only Studio writes, the API's session.
    Without a ``session`` (the ``server`` command) only Host and Origin."""

    def __init__(self, app: ASGIApp, hosts: frozenset[str] | None, session: Session | None):
        self.app = app
        self.hosts = hosts
        self.session = session

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        refusal = self.refusal(scope)
        if refusal is None:
            await self.app(scope, receive, send)
        else:
            await refusal(scope, receive, send)

    def refusal(self, scope: Scope) -> Response | None:
        headers = Headers(scope=scope)
        host = headers.get("host", "").lower()
        if self.hosts is not None and host not in self.hosts:
            return error_response(
                403,
                "the Host header names an address this server does not serve",
                "permission_error",
            )
        origin = headers.get("origin")
        scheme = "https" if scope.get("scheme") in ("https", "wss") else "http"
        if origin is not None and origin.lower() != f"{scheme}://{host}":
            return error_response(
                403, "requests from another site's pages are refused", "permission_error"
            )
        if self.session is None:
            return None
        path = scope.get("path", "")
        studio = path.startswith(STUDIO_PREFIX) or path == LOGIN_PATH
        if studio and scope.get("method", "GET") not in SAFE_METHODS:
            media = headers.get("content-type", "").split(";")[0].strip().lower()
            if media != "application/json":
                return error_response(
                    415,
                    "the Studio's API takes JSON (Content-Type: application/json)",
                    "invalid_request_error",
                )
        if path.startswith(API_PREFIX) and not self.session.has_cookie(_cookies(headers)):
            return error_response(
                401,
                "no Studio session; open the address mlx-decision web printed when it started",
                "authentication_error",
            )
        return None


def install(
    app: FastAPI,
    session: Session,
    host: str,
    port: int,
    api_key: str | None,
    attempts: KeyAttempts | None = None,
) -> None:
    """Add the guard middleware and the login routes to ``app``."""
    app.add_middleware(Guard, hosts=allowed_hosts(host, port), session=session)
    app.include_router(login_router(session, api_key, attempts or KeyAttempts()))


def login_router(session: Session, api_key: str | None, attempts: KeyAttempts) -> APIRouter:
    """``/login``: the page and the key check; only with an API key."""
    router = APIRouter()
    if not api_key:
        return router
    expected = api_key.encode()

    @router.get(LOGIN_PATH)
    async def login_page():
        return HTMLResponse(LOGIN_PAGE, headers=NO_STORE)

    @router.post(LOGIN_PATH)
    async def login(request: Request):
        client = request.client.host if request.client else None
        wait = attempts.wait(client)
        if wait > 0:
            return too_many_attempts(wait)
        raw = await read_body(request, MAX_LOGIN_BYTES)
        try:
            given = json.loads(raw)["api_key"] if raw is not None else None
        except (ValueError, TypeError, KeyError):
            given = None
        if not isinstance(given, str):
            return error_response(
                400, 'the body must be {"api_key": "..."}', "invalid_request_error", "api_key"
            )
        if not hmac.compare_digest(given.encode(), expected):
            attempts.failed(client)
            return error_response(401, "wrong API key", "authentication_error", "api_key")
        attempts.succeeded(client)
        return session.set_cookie(Response(status_code=204, headers=NO_STORE))

    return router


def page_without_session(api_key: str | None) -> Response:
    """The answer to the page without a session: the login, or how to get in."""
    if api_key:
        return RedirectResponse(LOGIN_PATH, status_code=303, headers=NO_STORE)
    return HTMLResponse(NO_SESSION_PAGE, status_code=401, headers=NO_STORE)


_STYLE = """
:root { color-scheme: light dark; --bg: #f5f5f7; --panel: #fff; --text: #1d1d1f;
  --muted: #6e6e73; --line: #d2d2d7; --accent: #0071e3; --bad: #d70015; }
@media (prefers-color-scheme: dark) { :root { --bg: #1c1c1e; --panel: #2c2c2e;
  --text: #f5f5f7; --muted: #98989d; --line: #3a3a3c; --accent: #0a84ff; --bad: #ff453a; } }
body { margin: 0; min-height: 100vh; display: grid; place-items: center;
  background: var(--bg); color: var(--text); font: 13px/1.4 -apple-system, system-ui, sans-serif; }
main { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 24px; width: min(320px, calc(100vw - 32px)); box-sizing: border-box; }
h1 { font-size: 15px; margin: 0 0 4px; } p { color: var(--muted); margin: 0 0 16px; }
input { width: 100%; box-sizing: border-box; padding: 6px 8px; font: inherit;
  border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--text); }
button { margin-top: 12px; width: 100%; padding: 6px; font: inherit; border: 0;
  border-radius: 6px; background: var(--accent); color: #fff; }
#error { color: var(--bad); margin: 8px 0 0; min-height: 1.4em; }
code { font: 12px ui-monospace, monospace; }
"""

LOGIN_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mlx-decision Studio</title><style>{_STYLE}</style></head>
<body><main><form id="login">
<h1>mlx-decision Studio</h1>
<p>Enter the API key this Studio was started with.</p>
<input id="key" type="password" autocomplete="current-password" autofocus
  aria-label="API key" placeholder="API key">
<button type="submit">Log In</button>
<p id="error" role="alert"></p>
</form></main>
<script>
document.getElementById("login").addEventListener("submit", async (event) => {{
  event.preventDefault();
  const error = document.getElementById("error");
  error.textContent = "";
  const response = await fetch("/login", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{api_key: document.getElementById("key").value}}),
  }});
  if (response.ok) location.replace("/");
  else error.textContent = response.status === 401 ? "Wrong API key." : "Could not log in.";
}});
</script></body></html>
"""

NO_SESSION_PAGE = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mlx-decision Studio</title><style>{_STYLE}</style></head>
<body><main><h1>mlx-decision Studio</h1>
<p>Open the address that <code>mlx-decision web</code> printed when it started.
It changes each time the Studio starts.</p></main></body></html>
"""
