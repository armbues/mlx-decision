"""Optional extras: each feature behind one says what to install when it is missing,
and packages a feature needs are imported only when it is used.

Every case runs in a fresh interpreter with the modules blocked, so nothing
imported earlier in the test session hides a missing import.
"""

import io
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from tiny_models import write_clef

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
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


def test_data_from_the_hub_names_the_training_extra():
    out = cli(["datasets"], ["data", "hf:org/name", "--show", "1"])
    assert out["code"] == 1
    assert "needs the training extra: pip install 'mlx-decision[training]'" in out["err"]


def test_data_from_csv_and_training_need_no_extra(tmp_path):
    csv = tmp_path / "tickets.csv"
    csv.write_text("text,label\nrefund,billing\ncrash,tech\n")
    out = cli(
        ["datasets"], ["data", str(csv), "--question", "Team?", "-o", str(tmp_path / "t.jsonl")]
    )
    assert out["code"] == 0, out["err"]
    out = cli(
        ["datasets"], ["training", "eval", "-m", "./no-model", "--data", str(tmp_path / "t.jsonl")]
    )
    assert "datasets" not in out["err"] and "no-model" in out["err"]


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


def test_text_only_use_imports_no_image_server_or_training_package(folder):
    out = run_without(
        ["numpy", "PIL", "fastapi", "uvicorn", "datasets"],
        f"""
import mlx_decision, mlx_decision.cli, mlx_decision.training
model = mlx_decision.load({str(folder)!r})
model.decide("billing", {QUESTIONS!r})
print(json.dumps({{"ok": True}}))
""",
    )
    assert out["ok"]


def test_the_extras_now_in_the_base_install_still_exist():
    # Older install lines such as pip install "mlx-decision[web]" keep working
    # without pip's warning about an unknown extra.
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    assert {name: extras[name] for name in ("server", "web", "images")} == {
        "server": [],
        "web": [],
        "images": [],
    }
    assert extras["all"] == ["mlx-decision[mcp,training]"]
