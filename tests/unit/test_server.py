"""The HTTP server, through the fake backend."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from fake_backend import FakeBackend
from mlx_decision import DecisionModel
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


@pytest.fixture
def backend_options():
    return {}


@pytest.fixture
def client(backend_options):
    holder = {}

    def load():
        holder["backend"] = RecordingBackend(**backend_options)
        return DecisionModel(holder["backend"])

    with TestClient(create_app(load)) as client:
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
    assert client.get("/health").json() == {"status": "ok", "model": "fake"}
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
def keyed_client():
    app = create_app(lambda: DecisionModel(FakeBackend()), api_key="s3cret")
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
