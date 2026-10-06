"""The MCP server, through the SDK's in-process client and the fake backend."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from fake_backend import FakeBackend  # noqa: E402
from mlx_decision import DecisionModel  # noqa: E402
from mlx_decision.mcp_server import create_server  # noqa: E402

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs", "billing": "Payments"},
    },
    "frustration": {"type": "score", "criteria": ["calm", "upset", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
}
ARGUMENTS = {"state": "Stripe is down", "questions": QUESTIONS}
PIXEL = "data:image/png;base64,iVBORw0KGgo="


class BrokenBackend(FakeBackend):
    def score(self, request):
        raise RuntimeError("out of memory")


def serve(backend, method: str, *args, allow_image_paths: bool = False):
    """Call ``method`` of a client connected to a server for ``backend``."""
    return serve_model(DecisionModel(backend), method, *args, allow_image_paths=allow_image_paths)


def serve_model(model, method: str, *args, allow_image_paths: bool = False):
    async def main():
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision") as worker:
            server = create_server(model, worker, allow_image_paths)
            async with Client(server) as client:
                return await getattr(client, method)(*args)

    return anyio.run(main)


def tools(backend=None, **options) -> dict:
    result = serve(backend or FakeBackend(), "list_tools", **options)
    return {tool.name: tool for tool in result.tools}


def decide(arguments, backend=None, **options):
    return serve(backend or FakeBackend(), "call_tool", "decide", arguments, **options)


def test_lists_decide_and_model_info_as_read_only_tools():
    listed = tools()
    assert set(listed) == {"decide", "model_info"}
    for tool in listed.values():
        assert tool.annotations.read_only_hint and tool.annotations.idempotent_hint
    schema = listed["decide"].input_schema
    assert schema["required"] == ["state", "questions"]
    assert "images" not in schema["properties"]
    assert "fake" in listed["decide"].description


def test_images_appear_only_for_models_that_read_them():
    schema = tools(FakeBackend(supports_images=True))["decide"].input_schema
    assert "data URL" in schema["properties"]["images"]["description"]
    assert "file path" not in schema["properties"]["images"]["description"]
    with_paths = tools(FakeBackend(supports_images=True), allow_image_paths=True)
    assert "file path" in with_paths["decide"].input_schema["properties"]["images"]["description"]


def test_descriptions_carry_the_model_limits():
    backend = FakeBackend(max_input_tokens=512, max_choice_options=20)
    backend.family = "julia"
    description = tools(backend)["decide"].description
    assert "512 tokens" in description
    assert "at most 20 options" in description
    assert "bare ids" in description
    assert "bare ids" not in tools()["decide"].description
    refusing = FakeBackend(max_input_tokens=512, truncates_input=False)
    assert "are refused" in tools(refusing)["decide"].description


def test_decide_answers_every_question_type():
    result = decide(ARGUMENTS)
    assert not result.is_error
    body = result.structured_content
    assert body["model"] == "fake"
    assert body["truncated"] is False
    assert body["answers"]["department"]["choice"] == "technical"
    assert body["answers"]["frustration"]["type"] == "score"
    assert body["answers"]["is_urgent"]["noul"] == 0.75
    assert json.loads(result.content[0].text) == body


def test_decide_ignores_a_model_argument():
    result = decide({**ARGUMENTS, "model": "jev-latest"})
    assert result.structured_content["model"] == "fake"


def test_truncation_is_part_of_the_result():
    result = decide(ARGUMENTS, FakeBackend(max_input_tokens=2))
    assert result.structured_content["truncated"] is True


def test_model_info():
    result = serve(FakeBackend(max_input_tokens=8), "call_tool", "model_info", {})
    assert result.structured_content["max_input_tokens"] == 8
    assert json.loads(result.content[0].text) == result.structured_content


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"state": "x"}, "questions"),
        ({"state": "x", "questions": {}}, "questions"),
        ({"state": "x", "questions": {"q": {"type": "maybe"}}}, "questions.q"),
        ({**ARGUMENTS, "images": [PIXEL]}, "images: fake does not support images"),
    ],
)
def test_invalid_requests_are_tool_errors_naming_the_field(arguments, expected):
    result = decide(arguments)
    assert result.is_error
    assert expected in result.content[0].text


def test_image_paths_are_refused_unless_allowed(tmp_path):
    arguments = {**ARGUMENTS, "images": [PIXEL, str(tmp_path / "cat.png")]}
    refused = decide(arguments, FakeBackend(supports_images=True))
    assert refused.is_error
    assert refused.content[0].text.startswith("images.1: expected a data URL")
    allowed = decide(arguments, FakeBackend(supports_images=True), allow_image_paths=True)
    assert not allowed.is_error


def test_failures_do_not_leak_details():
    result = decide(ARGUMENTS, BrokenBackend())
    assert result.is_error
    assert result.content[0].text == "the model failed to answer the request"


def test_unknown_tools_are_errors():
    result = serve(FakeBackend(), "call_tool", "guess", {})
    assert result.is_error


def test_calls_run_on_the_worker_thread():
    threads = []

    class Recording(FakeBackend):
        def score(self, request):
            threads.append(threading.current_thread().name)
            return super().score(request)

    decide(ARGUMENTS, Recording())
    assert threads and threads[0].startswith("mlx-decision")


def test_stdio_serves_the_cli_command_with_image_paths(tmp_path):
    """The real command in a subprocess: handshake, listing and a call with an image file."""
    from mcp import StdioServerParameters
    from PIL import Image

    from tiny_models import write_clef

    folder = write_clef(tmp_path / "tiny-clef", vision=True)
    image = tmp_path / "red.png"
    Image.new("RGB", (16, 16), (200, 30, 30)).save(image)
    command = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from mlx_decision.cli import app; app()", "mcp", "-m", str(folder)],
    )

    async def main():
        async with Client(command) as client:
            listed = await client.list_tools()
            arguments = {**ARGUMENTS, "images": [str(image)]}
            return listed, await client.call_tool("decide", arguments)

    listed, result = anyio.run(main)
    assert {tool.name for tool in listed.tools} == {"decide", "model_info"}
    assert "file path" in listed.tools[0].input_schema["properties"]["images"]["description"]
    assert not result.is_error, result.content[0].text
    assert set(result.structured_content["answers"]) == set(QUESTIONS)


def test_without_the_extra_the_command_says_what_to_install(tmp_path):
    script = "import sys\nsys.modules['mcp'] = None\nfrom mlx_decision.cli import app\napp()"
    result = subprocess.run(
        [sys.executable, "-c", script, "mcp", "-m", str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "pip install 'mlx-decision[mcp]'" in result.stderr


@pytest.fixture
def http_url():
    """Start a streamable HTTP server in a thread; yields a function of the API key."""
    import socket
    import time

    import uvicorn

    from mlx_decision.mcp_server import http_app

    servers = []
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision")

    def start(api_key=None):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        model = DecisionModel(FakeBackend(supports_images=True))
        app = http_app(create_server(model, worker), "127.0.0.1", api_key)
        server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        while not server.started:
            time.sleep(0.01)
        return f"http://127.0.0.1:{port}/mcp"

    yield start
    for server in servers:
        server.should_exit = True
    worker.shutdown()


def over_http(url, *calls, api_key=None):
    """Run ``(method, args...)`` calls with one HTTP client; returns their results."""
    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def main():
        async with (
            httpx2.AsyncClient(headers=headers) as http,
            Client(streamable_http_client(url, http_client=http)) as client,
        ):
            return [await getattr(client, method)(*args) for method, *args in calls]

    return anyio.run(main)


def test_http_serves_the_tools_and_refuses_image_paths(http_url, tmp_path):
    listed, answered, refused = over_http(
        http_url(),
        ("list_tools",),
        ("call_tool", "decide", {**ARGUMENTS, "images": [PIXEL]}),
        ("call_tool", "decide", {**ARGUMENTS, "images": [str(tmp_path / "secret.png")]}),
    )
    images = next(tool for tool in listed.tools if tool.name == "decide").input_schema
    assert "file path" not in images["properties"]["images"]["description"]
    assert not answered.is_error
    assert refused.is_error and refused.content[0].text.startswith("images.0: expected a data URL")


def test_http_clients_share_one_server(http_url):
    url = http_url()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: over_http(url, ("call_tool", "decide", ARGUMENTS)), range(8))
        )
    assert all(not result.is_error for (result,) in results)


def test_http_api_key(http_url):
    import httpx2

    url = http_url(api_key="secret")
    (result,) = over_http(url, ("call_tool", "decide", ARGUMENTS), api_key="secret")
    assert not result.is_error
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        response = httpx2.post(url, json={}, headers=headers)
        assert response.status_code == 401
        assert response.json() == {"error": "missing or invalid API key"}


TICKET = {
    "state": "The app crashes every time I open the settings page.",
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, invoices, refunds",
                "technical": "Bugs, crashes and outages",
                "sales": "Pricing and plans",
            },
        },
        "angry": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["Calm", "Annoyed", "Furious"],
        },
        "bug": {"type": "noul", "instructions": "Is this a bug report?"},
    },
}


@pytest.fixture(params=["clef", "laya", "laya_multilingual", "laya_typed", "julia"])
def real_model(request):
    """Each family's models with weights; Clef is the shared session copy."""
    if request.param == "clef":
        return request.getfixturevalue("clef")
    import mlx_decision

    return mlx_decision.load(request.getfixturevalue(f"{request.param}_path"))


def test_real_models_answer_through_mcp(real_model):
    result = serve_model(real_model, "call_tool", "decide", TICKET)
    assert not result.is_error, result.content[0].text
    answers = result.structured_content["answers"]
    assert answers["team"]["choice"] == "technical"
    assert answers["bug"]["noul"] > 0.5
    assert 0 <= answers["angry"]["score"] <= 2
    info = serve_model(real_model, "call_tool", "model_info", {}).structured_content
    assert info["name"] == real_model.name and info["precision"]
