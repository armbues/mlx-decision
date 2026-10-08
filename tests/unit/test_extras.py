"""Optional extras: each feature behind one says what to install when it is missing.

Every case runs in a fresh interpreter with the extra's modules blocked, so
nothing imported earlier in the test session hides a missing import.
"""

import base64
import io
import json
import subprocess
import sys

import pytest
from PIL import Image

from tiny_models import write_clef

HINT = "pip install 'mlx-decision[images]'"
QUESTIONS = {"q": {"type": "noul", "instructions": "Is it red?"}}


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_clef(tmp_path_factory.mktemp("models") / "tiny-clef", vision=True)


@pytest.fixture(scope="module")
def png(tmp_path_factory):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 30)).save(buffer, format="PNG")
    path = tmp_path_factory.mktemp("images") / "red.png"
    path.write_bytes(buffer.getvalue())
    return path


def run_without(blocked: list[str], code: str) -> dict:
    """Run ``code`` with ``blocked`` modules unimportable; it prints one JSON line."""
    block = "".join(f"sys.modules[{name!r}] = None\n" for name in blocked)
    script = f"import sys, json\n{block}{code}"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("blocked", [["PIL"], ["numpy"], ["PIL", "numpy"]])
def test_python_api_image_requests_get_the_hint(folder, png, blocked):
    out = run_without(
        blocked,
        f"""
import mlx_decision
model = mlx_decision.load({str(folder)!r})
try:
    model.decide("x", {QUESTIONS!r}, images=[{str(png)!r}])
    print(json.dumps({{"error": None}}))
except mlx_decision.DecisionError as error:
    print(json.dumps({{"error": error.message, "param": error.param, "code": error.code}}))
""",
    )
    assert HINT in out["error"]
    assert (out["param"], out["code"]) == ("images", "unsupported_media")


def test_server_answers_422_with_the_hint(folder, png):
    url = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()
    body = {"state": "x", "questions": QUESTIONS, "images": [url]}
    out = run_without(
        ["PIL", "numpy"],
        f"""
from fastapi.testclient import TestClient
from mlx_decision.pool import ModelPool, discover
from mlx_decision.server import create_app
with TestClient(create_app(ModelPool(discover([{str(folder)!r}])))) as client:
    response = client.post("/v1/systemone", json={body!r})
print(json.dumps({{"status": response.status_code, "body": response.json()}}))
""",
    )
    assert out["status"] == 422
    assert HINT in out["body"]["error"]["message"]


def cli(blocked: list[str], args: list[str]) -> dict:
    return run_without(
        blocked,
        f"""
from typer.testing import CliRunner
from mlx_decision.cli import app
result = CliRunner().invoke(app, {args!r})
print(json.dumps({{"code": result.exit_code, "err": result.stderr}}))
""",
    )


def test_run_with_image_names_the_extra_before_loading(folder, png):
    # A missing model folder would fail at load; the hint comes first.
    out = cli(["PIL"], ["run", "-m", "./no-model", "-s", "x", "--noul", "q?", "--image", str(png)])
    assert out["code"] == 1
    assert f"error: --image: images need Pillow and NumPy: {HINT}" in out["err"]


def test_server_command_names_the_extra(folder):
    out = cli(["fastapi", "uvicorn", "mlx_decision.server"], ["server", "-m", str(folder)])
    assert out["code"] == 1
    assert "pip install 'mlx-decision[server]'" in out["err"]


def test_convert_keeps_the_vision_tower_without_numpy(folder, png, tmp_path):
    out = run_without(
        ["numpy", "PIL"],
        f"""
from mlx_decision.convert import convert
print(json.dumps({{"out": str(convert({str(folder)!r}, {str(tmp_path / "q8")!r}, bits=8))}}))
""",
    )
    import mlx_decision

    copy = mlx_decision.load(out["out"])
    assert copy.backend.capabilities.supports_images
    result = copy.decide("x", QUESTIONS, images=[png])
    assert 0 <= result.answers["q"].noul <= 1


def test_the_base_install_imports_no_extra(folder):
    out = run_without(
        ["numpy", "PIL", "fastapi", "uvicorn"],
        f"""
import mlx_decision, mlx_decision.cli
model = mlx_decision.load({str(folder)!r})
model.decide("billing", {QUESTIONS!r})
print(json.dumps({{"ok": True}}))
""",
    )
    assert out["ok"]
