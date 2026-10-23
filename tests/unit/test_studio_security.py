"""The Studio's protections: Host, Origin, the session cookie and the login."""

import pytest
from fastapi.testclient import TestClient

from fake_backend import FakeBackend, write_model
from mlx_decision import DecisionModel
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.security import allowed_hosts, check_bind, is_loopback

BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}
HERE = "http://127.0.0.1:8000"
LOOPBACK_CLIENT = ("127.0.0.1", 50000)
OTHER_CLIENT = ("192.168.1.20", 50000)
KEY = "s3cret"


class Loads:
    """A pool's load function that counts loads and answers."""

    def __init__(self):
        self.loads = 0
        self.answers = 0

    def __call__(self, spec):
        self.loads += 1
        loads = self

        class Counting(FakeBackend):
            def score(self, request):
                loads.answers += 1
                return super().score(request)

        return DecisionModel(Counting())


@pytest.fixture
def make(tmp_path):
    """``make(**studio_options)`` -> (app, load counter) for one fake model."""
    write_model(tmp_path / "models" / "fake")

    def make(**options):
        loads = Loads()
        pool = ModelPool(discover([tmp_path / "models"]), budget=10**9, load=loads)
        app = create_studio(pool, **options)

        @app.post("/studio/api/echo")  # stands in for the Studio's POST routes
        async def echo(body: dict):
            return body

        return app, loads

    return make


def client(app, base_url=HERE, address=LOOPBACK_CLIENT) -> TestClient:
    return TestClient(app, base_url=base_url, client=address, follow_redirects=False)


def sign_in(browser: TestClient, app) -> None:
    response = browser.get("/", params={"token": app.state.session.token})
    assert response.status_code == 303, response.text
    assert response.headers["location"] == "/"


# Origin and Host


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
def test_a_foreign_origin_gets_403_on_the_jev_api_with_no_model_work(make, content_type):
    app, loads = make()
    with client(app) as browser:
        app.state.gpu.pool.unload_all()  # the default model loaded at startup
        response = browser.post(
            "/v1/systemone",
            content=b'{"state": "x", "questions": {"q": {"type": "noul"}}}',
            headers={"Origin": "https://evil.example", "Content-Type": content_type},
        )
        assert response.status_code == 403
        assert response.json()["error"]["type"] == "permission_error"
        assert (loads.loads, loads.answers) == (1, 0)
        assert app.state.gpu.pool.loaded() == []


def test_a_foreign_origin_gets_403_on_every_route(make):
    app, _ = make()
    with client(app) as browser:
        sign_in(browser, app)
        foreign = {"Origin": "http://localhost:8000"}  # another origin of the same machine
        for path in ("/", "/health", "/v1/models", "/studio/api/status"):
            assert browser.get(path, headers=foreign).status_code == 403, path
        echo = browser.post("/studio/api/echo", json={"a": 1}, headers=foreign)
        assert echo.status_code == 403
        assert browser.get("/", headers={"Origin": "null"}).status_code == 403


def test_the_pages_own_origin_and_no_origin_pass(make):
    app, loads = make()
    with client(app) as browser:
        assert browser.post("/v1/systemone", json=BODY, headers={"Origin": HERE}).status_code == 200
        assert browser.post("/v1/systemone", json=BODY).status_code == 200  # the SDK, curl
        assert loads.answers == 2


def test_no_cors_headers(make):
    app, _ = make()
    with client(app) as browser:
        responses = [
            browser.post("/v1/systemone", json=BODY, headers={"Origin": HERE}),
            browser.options(
                "/v1/systemone",
                headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
            ),
        ]
        for response in responses:
            assert not [name for name in response.headers if name.startswith("access-control-")]


@pytest.mark.parametrize(
    "host", ["evil.example:8000", "localhost.example:8000", "127.0.0.1:8001", "127.0.0.1", ""]
)
def test_an_unknown_host_gets_403_with_no_model_work(make, host):
    app, loads = make()
    with client(app) as browser:
        app.state.gpu.pool.unload_all()  # the default model loaded at startup
        for method, path in (("POST", "/v1/systemone"), ("GET", "/health"), ("GET", "/")):
            response = browser.request(method, path, json=BODY, headers={"Host": host})
            assert response.status_code == 403, (method, path)
        assert (loads.loads, loads.answers) == (1, 0)
        assert app.state.gpu.pool.loaded() == []


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000", "LOCALHOST:8000"]
)
def test_the_loopback_names_are_served(make, host):
    app, _ = make()
    with client(app, base_url=f"http://{host}") as browser:
        assert browser.get("/health").status_code == 200
        assert browser.get("/health", headers={"Origin": f"http://{host}"}).status_code == 200


def test_allowed_hosts():
    assert allowed_hosts("127.0.0.1", 8000) == {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"}
    assert "[fd00::5]:8000" in allowed_hosts("fd00::5", 8000)
    assert "studio.lan:8000" in allowed_hosts("Studio.lan", 8000)
    assert {"localhost", "localhost:80"} <= allowed_hosts("localhost", 80)  # browsers drop :80
    assert allowed_hosts("0.0.0.0", 8000) is None
    assert allowed_hosts("::", 8000) is None


def test_a_wildcard_bind_takes_any_host_and_relies_on_the_key(make):
    app, _ = make(host="0.0.0.0", api_key=KEY)
    with client(app, base_url="http://my-mac.local:8000", address=OTHER_CLIENT) as browser:
        assert browser.get("/health").status_code == 200
        assert browser.post("/v1/systemone", json=BODY).status_code == 401
        assert browser.get("/studio/api/status").status_code == 401
        evil = {"Origin": "https://evil.example"}
        assert browser.get("/health", headers=evil).status_code == 403


# Session


def test_the_studios_api_needs_the_session_cookie(make):
    app, _ = make()
    with client(app) as browser:
        response = browser.get("/studio/api/status")
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"
        browser.cookies.set(app.state.session.cookie, "guessed")
        assert browser.get("/studio/api/status").status_code == 401


def test_the_token_becomes_a_strict_http_only_cookie(make):
    app, _ = make()
    with client(app) as browser:
        assert browser.get("/").status_code == 401  # no session: how to get in
        response = browser.get("/", params={"token": app.state.session.token})
        assert response.status_code == 303 and response.headers["location"] == "/"
        cookie = response.headers["set-cookie"]
        assert cookie.startswith(f"{app.state.session.cookie}=")
        for part in ("HttpOnly", "SameSite=strict", "Path=/"):
            assert part.lower() in cookie.lower()
        assert app.state.session.token not in cookie
        page = browser.get("/")
        assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
        status = browser.get("/studio/api/status")
        assert status.status_code == 200 and status.json()["busy"] is None


@pytest.mark.parametrize("other", ['prefs={"theme":"dark"}', "note=two words", 'x="unclosed'])
def test_another_apps_cookie_does_not_hide_the_session(make, other):
    """Cookies ignore the port: another app on this machine may set ones the
    strict parser of the standard library rejects."""
    app, _ = make()
    with client(app) as browser:
        sign_in(browser, app)
        name = app.state.session.cookie
        header = f"{other}; {name}={browser.cookies.get(name)}"
        browser.cookies.clear()
        response = browser.get("/studio/api/status", headers={"Cookie": header})
        assert response.status_code == 200, response.text


def test_a_wrong_token_sets_no_cookie(make):
    app, _ = make()
    with client(app) as browser:
        response = browser.get("/", params={"token": "wrong"})
        assert response.status_code == 401
        assert "set-cookie" not in response.headers
        assert browser.get("/studio/api/status").status_code == 401


def test_a_session_is_per_process(make):
    first, _ = make()
    second, _ = make()
    assert first.state.session.token != second.state.session.token
    with client(first) as browser, client(second) as other:
        sign_in(browser, first)
        other.cookies = browser.cookies
        assert other.get("/studio/api/status").status_code == 401


def test_the_jev_api_needs_no_cookie_and_the_cookie_is_no_key(make):
    app, _ = make(api_key=KEY)
    with client(app) as browser:
        sign_in(browser, app)
        assert browser.post("/v1/systemone", json=BODY).status_code == 401
        headers = {"Authorization": f"Bearer {KEY}"}
        assert browser.post("/v1/systemone", json=BODY, headers=headers).status_code == 200


# Login (with an API key)


def test_with_a_key_the_token_works_on_this_machine(make):
    app, _ = make(api_key=KEY)
    with client(app) as browser:
        sign_in(browser, app)
        assert browser.get("/studio/api/status").status_code == 200


def test_with_a_key_the_token_is_refused_from_another_machine(make):
    app, _ = make(host="0.0.0.0", api_key=KEY)
    with client(app, address=OTHER_CLIENT) as browser:
        response = browser.get("/", params={"token": app.state.session.token})
        assert response.status_code == 303 and response.headers["location"] == "/login"
        assert "set-cookie" not in response.headers
        assert browser.get("/studio/api/status").status_code == 401


def test_login_with_the_key(make):
    app, _ = make(host="0.0.0.0", api_key=KEY)
    with client(app, address=OTHER_CLIENT) as browser:
        assert browser.get("/").headers["location"] == "/login"
        page = browser.get("/login")
        assert page.status_code == 200 and "API key" in page.text
        wrong = browser.post("/login", json={"api_key": "guess"})
        assert wrong.status_code == 401 and "set-cookie" not in wrong.headers
        assert browser.post("/login", json={"key": KEY}).status_code == 400
        assert browser.post("/login", content=b"{", headers=JSON).status_code == 400
        right = browser.post("/login", json={"api_key": KEY})
        assert right.status_code == 204
        assert "httponly" in right.headers["set-cookie"].lower()
        assert browser.get("/").status_code == 200
        assert browser.get("/studio/api/status").status_code == 200


def test_without_a_key_there_is_no_login(make):
    app, _ = make()
    with client(app) as browser:
        assert browser.get("/login").status_code == 404
        assert browser.post("/login", json={"api_key": ""}).status_code == 404


# JSON-only writes

JSON = {"Content-Type": "application/json"}


@pytest.mark.parametrize(
    "content_type",
    [None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"],
)
def test_studio_writes_must_be_json(make, content_type):
    app, _ = make(api_key=KEY)
    with client(app) as browser:
        sign_in(browser, app)
        headers = {"Content-Type": content_type} if content_type else {}
        for path in ("/studio/api/echo", "/login"):
            response = browser.post(path, content=b'{"api_key": "s3cret"}', headers=headers)
            assert response.status_code == 415, path
        ok = browser.post("/studio/api/echo", content=b'{"a": 1}', headers=JSON)
        assert ok.json() == {"a": 1}
        charset = {"Content-Type": "application/json; charset=utf-8"}
        assert browser.post("/studio/api/echo", content=b"{}", headers=charset).status_code == 200


# Binding beyond this machine


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "my-mac.local"])
def test_beyond_loopback_needs_a_key(make, host):
    with pytest.raises(ValueError, match="needs an API key"):
        make(host=host)
    with pytest.raises(ValueError):
        check_bind(host, "")
    check_bind(host, KEY)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
def test_loopback_needs_no_key(host):
    check_bind(host, None)


def test_is_loopback():
    assert is_loopback("::ffff:127.0.0.1")
    for address in ("testclient", "192.168.1.20", "::ffff:10.0.0.1", "", None):
        assert not is_loopback(address)


# This Mac by its network address, the destination, wrong keys


LAN = "10.20.1.145"


def test_with_a_key_the_printed_address_lets_in_this_macs_browser(make):
    """``web --host <this Mac's address>``: a browser here connects from that
    same address, so the token works and it counts as on this Mac."""
    app, _ = make(host=LAN, api_key=KEY)
    with client(app, base_url=f"http://{LAN}:8000", address=(LAN, 50000)) as here:
        sign_in(here, app)
        assert here.get("/studio/api/places").json()["editable"] is True
        with client(app, base_url=f"http://{LAN}:8000", address=("10.20.1.99", 50000)) as other:
            response = other.get("/", params={"token": app.state.session.token})
            assert response.headers["location"] == "/login"


def test_the_download_destination_is_set_only_from_this_mac(make, tmp_path):
    from mlx_decision.config import Config

    config = Config(tmp_path / "config.toml")
    app, _ = make(host="0.0.0.0", api_key=KEY, config=config)
    with client(app, address=OTHER_CLIENT) as remote:
        assert remote.post("/login", json={"api_key": KEY}).status_code == 204
        response = remote.post(
            "/studio/api/downloads/destination", json={"place": "folder", "path": "~/x"}
        )
        assert response.status_code == 403
        assert response.json()["error"]["type"] == "permission_error"
        assert remote.get("/studio/api/downloads/first").json()["elsewhere"] is True
    assert not (tmp_path / "config.toml").exists()


def test_wrong_keys_in_a_row_make_a_client_wait(make, monkeypatch, caplog):
    import mlx_decision.server as server

    now = [1000.0]
    monkeypatch.setattr(server.time, "monotonic", lambda: now[0])
    app, _ = make(host="0.0.0.0", api_key=KEY)
    with client(app, address=OTHER_CLIENT) as remote, caplog.at_level("WARNING"):
        for _ in range(server.FREE_KEY_FAILURES):
            assert remote.post("/login", json={"api_key": "guess"}).status_code == 401
        # One more wrong key: the client now waits, on the login and the Jev API alike.
        assert remote.post("/v1/systemone", json=BODY).status_code == 401
        waiting = remote.post("/login", json={"api_key": KEY})
        assert waiting.status_code == 429 and waiting.headers["Retry-After"] == "1"
        assert remote.post("/v1/systemone", json=BODY).status_code == 429
        with client(app) as here:  # another address does not wait
            assert here.post("/login", json={"api_key": KEY}).status_code == 204
        now[0] += 1.5
        assert remote.post("/login", json={"api_key": KEY}).status_code == 204
        assert remote.post("/login", json={"api_key": "guess"}).status_code == 401  # count reset
    warnings = [r.getMessage() for r in caplog.records if "wrong API key" in r.getMessage()]
    assert warnings[0] == f"wrong API key from {OTHER_CLIENT[0]} (1 in a row)"


def test_the_server_command_refuses_other_sites_and_other_hosts(tmp_path):
    from mlx_decision.server import create_app

    write_model(tmp_path / "models" / "fake")
    loads = Loads()
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9, load=loads)
    app = create_app(pool, host="127.0.0.1", port=8000)
    with client(app) as local:
        pool.unload_all()  # the default model loaded at startup
        loaded_at_start = loads.loads
        page = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
        response = local.post("/v1/systemone", content=b'{"state": "x"}', headers=page)
        assert response.status_code == 403
        rebound = local.get("/v1/models", headers={"Host": "rebind.evil.example:8000"})
        assert rebound.status_code == 403
        assert loads.loads == loaded_at_start  # no model work
        assert local.post("/v1/systemone", json=BODY).status_code == 200  # the SDK, curl
        same = local.post("/v1/systemone", json=BODY, headers={"Origin": HERE})
        assert same.status_code == 200
    wildcard = create_app(ModelPool(discover([tmp_path / "models"]), budget=10**9),
                          host="0.0.0.0", port=8000)  # fmt: skip
    with client(wildcard, base_url="http://mac.local:8000") as any_name:
        assert any_name.post("/v1/systemone", json=BODY).status_code == 200
        foreign = any_name.post("/v1/systemone", json=BODY, headers=page)
        assert foreign.status_code == 403
