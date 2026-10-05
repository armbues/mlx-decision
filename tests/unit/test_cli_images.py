"""Images in ``run`` and ``chat``, through a tiny random Clef with a vision tower."""

import json

import pytest
from PIL import Image
from test_interactive import drive, lines
from typer.testing import CliRunner

from mlx_decision import load
from mlx_decision.cli import app
from tiny_models import write_clef

QUESTIONS = '{"team": {"type": "choice", "criteria": {"billing": null, "sales": null}}}'


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_clef(tmp_path_factory.mktemp("models") / "tiny-clef", vision=True)


@pytest.fixture(scope="module")
def pictures(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pictures")
    small, large = folder / "small.png", folder / "large.png"
    Image.new("RGB", (16, 16), (200, 0, 0)).save(small)  # 4 tokens
    Image.new("RGB", (32, 32), (0, 0, 200)).save(large)  # 16 tokens
    return small, large


@pytest.fixture
def run(folder, tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(QUESTIONS)

    def invoke(*args, input=None):
        base = ["run", "-m", str(folder), "-q", str(questions)]
        return CliRunner().invoke(app, [*base, *args], input=input)

    return invoke


def tokens(result) -> int:
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["usage"]["input_tokens"]


def test_image_flags_add_their_tokens(run, pictures):
    small, large = pictures
    text = tokens(run("-s", "billing", "--json"))
    both = tokens(run("-s", "billing", "--image", str(small), "--image", str(large), "--json"))
    assert both == text + (4 + 2) + (16 + 2)


def test_states_lines_bring_images_or_take_the_flags(run, pictures):
    small, large = pictures
    data_url = "data:image/png;base64,iVBORw0KGgo="  # not a whole image: an error line
    requests = [
        {"state": "billing"},
        {"state": "billing", "images": [str(large)]},
        {"state": "billing", "images": []},
        {"state": "billing", "images": [data_url]},
    ]
    typed = "".join(json.dumps(r) + "\n" for r in requests)
    result = run("--states", "-", "--image", str(small), input=typed)
    assert result.exit_code == 0, result.output
    bodies = [json.loads(line) for line in result.stdout.splitlines()]
    flags, own, none = (b["usage"]["input_tokens"] for b in bodies[:3])
    assert flags - none == 4 + 2 and own - none == 16 + 2
    assert bodies[3]["error"]["param"] == "images.0" and bodies[3]["line"] == 4


def test_a_missing_image_file_fails_before_loading(run):
    result = run("-s", "x", "--image", "nope.png")
    assert result.exit_code == 1
    assert "--image: no such file: nope.png" in result.stderr


def test_bare_base64_gets_a_hint(run):
    body = '{"state": "x", "images": ["' + "iVBORw0KGgo" * 30 + '"]}\n'
    result = run("--states", "-", input=body)
    error = json.loads(result.stdout)["error"]
    assert "needs the data:image/...;base64, prefix" in error["message"]


def test_chat_attaches_and_clears_images(folder, pictures, capsys):
    small, _ = pictures
    model = load(folder)
    body = {"questions": json.loads(QUESTIONS)}
    typed = lines("billing", "", f"/image {small}", "/list", "billing", "", "/image clear")
    typed += lines("billing", "", "/image nope.png", "/image")
    session = drive(model, typed, body=body)
    out, err = capsys.readouterr()
    first, second, third = (json.loads(line)["usage"]["input_tokens"] for line in out.splitlines())
    assert second == first + 4 + 2 and third == first
    assert f"attached {small} (1 image)" in err
    assert "images: " in err and "/small.png" in err
    assert "images cleared" in err
    assert "error: nope.png: No such file or directory" in err
    assert "/image needs a file name" in err
    assert session.images == []


def test_chat_refuses_images_for_a_text_model(fake_model, pictures, capsys):
    drive(fake_model, lines(f"/image {pictures[0]}"), body={"questions": json.loads(QUESTIONS)})
    assert "fake does not support images" in capsys.readouterr().err


def test_max_image_mp_maps_to_load_options(run, pictures):
    from mlx_decision.cli import image_options

    assert image_options(None) == {}
    assert image_options(0) == {"max_image_pixels": None}
    assert image_options(2) == {"max_image_pixels": 2**21}
    result = run("-s", "billing", "--image", str(pictures[1]), "--max-image-mp", "0.0002")
    assert result.exit_code == 1
    assert "max_image_pixels must be at least 1024" in result.stderr
