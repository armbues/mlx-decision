"""The HTTP server, through the fake backend."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from fake_backend import FakeBackend, write_model, write_sized_model
from mlx_decision import DecisionModel
from mlx_decision.pool import ModelPool, discover
from mlx_decision.server import TRUNCATED_HEADER, create_app

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs", "billing": "Payments"},
    },
    "frustration": {"type": "score", "criteria": ["calm", "upset", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
}
BODY = {"model": "jev-latest", "state": "Stripe is down", "questions": QUESTIONS}


class RecordingBackend(FakeBackend):
    """Notes the threads it runs on and whether two requests ever overlap."""

    def __init__(self, **options):
        super().__init__(**options)
        self.loaded_on = threading.current_thread().name
        self.threads: set[str] = set()
        self.states: list[str] = []
        self.active = 0
        self.overlapped = False

    def score(self, request):
        self.active += 1
        self.overlapped |= self.active > 1
        self.threads.add(threading.current_thread().name)
        time.sleep(0.005)
        self.states.append(request.state)
        self.active -= 1
        return super().score(request)


def single_pool(folder, make_backend) -> ModelPool:
    """A pool of one fake model named "fake" whose backend ``make_backend()`` builds."""
    specs = discover([write_model(folder / "fake")])
    return ModelPool(specs, budget=10**9, load=lambda spec: DecisionModel(make_backend()))


@pytest.fixture
def backend_options():
    return {}


@pytest.fixture
def client(backend_options, tmp_path):
    holder = {}

    def make_backend():
        holder["backend"] = RecordingBackend(**backend_options)
        return holder["backend"]

    with TestClient(create_app(single_pool(tmp_path, make_backend))) as client:
        client.backend = holder["backend"]
        yield client


def test_answers_a_request_like_the_python_api(client):
    response = client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer x"})
    assert response.status_code == 200
    expected = DecisionModel(FakeBackend()).decide_request(BODY).to_wire()
    assert response.json() == expected
    assert response.json()["model"] == "fake"
    assert TRUNCATED_HEADER not in response.headers


def test_the_model_field_is_ignored(client):
    for model in ("jev-latest", "something-else", None):
        body = (
            {**BODY, "model": model} if model else {k: v for k, v in BODY.items() if k != "model"}
        )
        assert client.post("/v1/systemone", json=body).json()["model"] == "fake"


@pytest.mark.parametrize("backend_options", [{"max_input_tokens": 2}])
def test_truncation_header(client):
    response = client.post("/v1/systemone", json=BODY)
    assert response.headers[TRUNCATED_HEADER] == "true"
    assert set(response.json()) == {"model", "answers", "usage"}


def test_invalid_request_gets_a_jev_error_body(client):
    body = {"state": "x", "questions": {"q": {"type": "rank"}}}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "message": "unknown question type 'rank'; expected noul, choice or score",
            "type": "invalid_request_error",
            "param": "questions.q.type",
        }
    }
    assert client.backend.states == []


def test_images_are_rejected_with_a_code(client):
    response = client.post("/v1/systemone", json={**BODY, "images": ["data:image/png;base64,"]})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["param"] == "images"
    assert error["code"] == "unsupported_media"


@pytest.mark.parametrize("backend_options", [{"supports_images": True}])
def test_data_urls_are_accepted(client):
    body = {**BODY, "images": ["data:image/png;base64,iVBORw0KGgo="]}
    assert client.post("/v1/systemone", json=body).status_code == 200


@pytest.mark.parametrize("backend_options", [{"supports_images": True}])
@pytest.mark.parametrize("image", ["/etc/hosts", "https://example.com/cat.png", {"path": "x"}, 1])
def test_anything_but_a_data_url_is_refused(client, image):
    body = {**BODY, "images": ["data:image/png;base64,iVBORw0KGgo=", image]}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["param"] == "images.1"
    assert "does not read files or fetch URLs" in error["message"]
    assert client.backend.states == []


def test_malformed_json(client):
    response = client.post(
        "/v1/systemone", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_a_failing_model_gives_500(client, monkeypatch):
    def broken(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(client.backend, "score", broken)
    response = client.post("/v1/systemone", json=BODY)
    assert response.status_code == 500
    assert response.json()["error"]["type"] == "api_error"
    assert "boom" not in response.text


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok", "model": "fake", "loaded": ["fake"]}
    models = client.get("/v1/models").json()["models"]
    assert [model["name"] for model in models] == ["fake"]
    assert set(models[0]) == {"name", "description", "release_date"}


def test_concurrent_requests_queue_on_one_thread(client):
    bodies = [{**BODY, "state": f"request {i}"} for i in range(20)]
    with ThreadPoolExecutor(max_workers=20) as pool:
        responses = list(pool.map(lambda body: client.post("/v1/systemone", json=body), bodies))
    assert [response.status_code for response in responses] == [200] * 20
    for body, response in zip(bodies, responses, strict=True):
        expected = DecisionModel(FakeBackend()).decide_request(body).to_wire()
        assert response.json() == expected
    backend = client.backend
    assert not backend.overlapped
    assert backend.threads == {backend.loaded_on}
    assert sorted(backend.states) == sorted(body["state"] for body in bodies)


@pytest.fixture
def keyed_client(tmp_path):
    app = create_app(single_pool(tmp_path, FakeBackend), api_key="s3cret")
    with TestClient(app) as client:
        yield client


def test_api_key_is_required_when_configured(keyed_client):
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
        response = keyed_client.post("/v1/systemone", json=BODY, headers=headers)
        assert response.status_code == 401
        assert response.json() == {
            "error": {
                "message": "missing or invalid API key",
                "type": "authentication_error",
                "param": None,
            }
        }
    assert keyed_client.get("/v1/models").status_code == 401


def test_the_right_key_passes_and_health_stays_open(keyed_client):
    ok = keyed_client.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert keyed_client.get("/health").status_code == 200


def test_server_command_reads_the_key_from_the_environment(monkeypatch, fake_model_path):
    import uvicorn
    from typer.testing import CliRunner

    from mlx_decision.cli import app as cli_app

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    result = CliRunner().invoke(
        cli_app,
        ["server", "-m", str(fake_model_path)],
        env={"MLX_DECISION_API_KEY": "from-env"},
    )
    assert result.exit_code == 0, result.output
    with TestClient(seen["app"]) as client:
        assert client.post("/v1/systemone", json=BODY).status_code == 401
        headers = {"Authorization": "Bearer from-env"}
        assert client.post("/v1/systemone", json=BODY, headers=headers).status_code == 200


def test_oversized_bodies_get_413(tmp_path):
    app = create_app(single_pool(tmp_path, RecordingBackend), max_body_bytes=1000)
    with TestClient(app) as small:
        response = small.post("/v1/systemone", json={**BODY, "state": "x" * 2000})
        assert response.status_code == 413
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert small.post("/v1/systemone", json=BODY).status_code == 200


def test_deeply_nested_json_is_a_400(client):
    response = client.post("/v1/systemone", content=b"[" * 100_000 + b"]" * 100_000)
    assert response.status_code == 400
    assert "nested too deeply" in response.json()["error"]["message"]


# Several models: three fake models of 1,000 bytes each (1,100 counted).


@pytest.fixture
def models_folder(tmp_path):
    root = tmp_path / "models"
    for name in ("Alpha", "beta", "gamma"):
        write_sized_model(root / name, 1000)
    return root


def several(models_folder, **pool_options) -> TestClient:
    return TestClient(create_app(ModelPool(discover([models_folder]), **pool_options)))


def test_the_model_field_picks_the_model(models_folder):
    with several(models_folder, budget=10**6) as client:
        for name in ("beta", "Alpha", "beta"):
            response = client.post("/v1/systemone", json={**BODY, "model": name})
            assert response.status_code == 200
            assert response.json()["model"] == name
        assert client.get("/health").json() == {
            "status": "ok",
            "model": None,
            "loaded": ["Alpha", "beta"],
        }


def test_without_a_default_a_request_must_name_a_model(models_folder):
    with several(models_folder, budget=10**6) as client:
        assert client.get("/health").json()["loaded"] == []  # nothing loaded at startup
        for body in (BODY, {**BODY, "model": None}, {**BODY, "model": 3}):
            response = client.post("/v1/systemone", json=body)
            assert response.status_code == 422
            error = response.json()["error"]
            assert error["param"] == "model"
            assert "Alpha, beta, gamma" in error["message"]


def test_the_default_model_loads_at_startup_and_answers_unknown_names(models_folder):
    with several(models_folder, budget=10**6, default="gamma") as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model": "gamma",
            "loaded": ["gamma"],
        }
        assert client.post("/v1/systemone", json=BODY).json()["model"] == "gamma"
        assert client.post("/v1/systemone", json={**BODY, "model": "Alpha"}).json()["model"] == (
            "Alpha"
        )


def test_models_lists_every_model_loaded_or_not(models_folder):
    with several(models_folder, budget=10**6) as client:
        names = [model["name"] for model in client.get("/v1/models").json()["models"]]
        assert names == ["Alpha", "beta", "gamma"]


def test_models_are_unloaded_to_fit_the_budget(models_folder):
    with several(models_folder, budget=2200) as client:
        for name in ("Alpha", "beta", "Alpha", "gamma"):
            assert client.post("/v1/systemone", json={**BODY, "model": name}).status_code == 200
        assert client.get("/health").json()["loaded"] == ["Alpha", "gamma"]


def test_a_model_larger_than_the_budget_gets_a_422_on_model(models_folder):
    write_sized_model(models_folder / "huge", 10**4)
    with several(models_folder, budget=2200) as client:
        response = client.post("/v1/systemone", json={**BODY, "model": "huge"})
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["param"] == "model"
        assert error["message"].startswith("huge needs about ")


@pytest.fixture
def serve(monkeypatch):
    """Run the server command with uvicorn replaced; returns (result, app)."""
    import uvicorn
    from typer.testing import CliRunner

    from mlx_decision.cli import app as cli_app

    def run(*args):
        seen = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app))
        result = CliRunner().invoke(cli_app, ["server", *map(str, args)])
        return result, seen.get("app")

    return run


def test_server_command_without_a_default_warns(serve, models_folder):
    (models_folder / "encoder-base").mkdir()
    result, app = serve("-m", models_folder)
    assert result.exit_code == 0, result.output
    assert "skipped " in result.stderr and "encoder-base" in result.stderr
    assert "models: Alpha, beta, gamma; no default model" in result.stderr
    assert "--default-model" in result.stderr
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=BODY).status_code == 422


def test_server_command_with_a_default_and_several_m(serve, models_folder, tmp_path):
    extra = write_model(tmp_path / "extra")
    result, app = serve("-m", extra, "-m", models_folder, "--default-model", "beta")
    assert result.exit_code == 0, result.output
    assert "default beta" in result.stderr
    with TestClient(app) as client:
        assert client.get("/health").json()["loaded"] == ["beta"]
        assert client.post("/v1/systemone", json={**BODY, "model": "extra"}).status_code == 200


def test_server_command_refuses_bad_setups(serve, models_folder, tmp_path):
    result, _ = serve("-m", models_folder, "--default-model", "delta")
    assert result.exit_code == 1
    assert "no model is named delta" in result.stderr
    result, _ = serve("-m", models_folder, "-m", models_folder)
    assert result.exit_code == 1
    assert "two models are named Alpha" in result.stderr
    # The default model is checked against the budget before the server starts.
    result, _ = serve("-m", models_folder, "--default-model", "beta", "--memory-budget", "0.000001")
    assert result.exit_code == 1
    assert "beta needs about" in result.stderr
    result, app = serve(
        "-m", models_folder, "--default-model", "beta", "--memory-budget", "0.000001",
        "--no-memory-check",
    )  # fmt: skip
    assert result.exit_code == 0, result.output


def test_server_command_passes_the_budget(serve, models_folder):
    result, app = serve("-m", models_folder, "--memory-budget", "0.0000022")
    assert result.exit_code == 0, result.output
    with TestClient(app) as client:
        for name in ("Alpha", "beta", "gamma"):
            client.post("/v1/systemone", json={**BODY, "model": name})
        assert client.get("/health").json()["loaded"] == ["beta", "gamma"]
