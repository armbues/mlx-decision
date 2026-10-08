"""Image requests through a tiny random Clef with a vision tower (no weights needed)."""

import base64
import io

import pytest
from PIL import Image

import mlx_decision
from mlx_decision import DecisionError
from tiny_models import write_clef

QUESTIONS = {"team": {"type": "choice", "criteria": {"billing": None, "sales": None}}}


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_clef(tmp_path_factory.mktemp("models") / "tiny-clef", vision=True)


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def png(width, height, color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_text_requests_leave_the_vision_tower_unloaded(folder):
    model = mlx_decision.load(folder)
    assert model.backend.capabilities.supports_images
    model.decide("billing", QUESTIONS)
    assert model.backend.vision is None
    model.decide_request({"state": "billing", "questions": QUESTIONS, "images": [png(16, 16)]})
    assert model.backend.vision is not None


def test_the_vision_tower_can_be_loaded_up_front(folder):
    assert mlx_decision.load(folder, vision=True).backend.vision is not None


def test_images_add_their_tokens(model):
    text = model.decide("billing", QUESTIONS).usage.input_tokens
    url = "data:image/png;base64," + base64.b64encode(png(32, 16)).decode()
    image = Image.new("RGB", (8, 8))  # upscaled to the minimum: 8 x 8 pixels, one token
    result = model.decide_request(
        {"state": "billing", "questions": QUESTIONS, "images": [url, png(16, 16), image]}
    )
    # Each image adds start, pads and end; one newline token after them (none in this vocabulary).
    assert result.usage.input_tokens == text + (2 + 8) + (2 + 4) + (2 + 1)
    assert sum(result.answers["team"].probabilities.values()) == pytest.approx(1, abs=1e-5)


def test_decide_takes_images(model, tmp_path):
    path = tmp_path / "red.png"
    path.write_bytes(png(16, 16))
    with_path = model.decide("billing", QUESTIONS, images=[path])
    with_bytes = model.decide("billing", QUESTIONS, images=[png(16, 16)])
    assert with_path.answers == with_bytes.answers
    assert with_path.usage.input_tokens > model.decide("billing", QUESTIONS).usage.input_tokens


def test_image_features_change_the_answer(model):
    def answer(color):
        body = {"state": "billing", "questions": QUESTIONS, "images": [png(16, 16, color)]}
        return model.decide_request(body).answers["team"].probabilities["billing"]

    assert answer((255, 0, 0)) != answer((0, 0, 255))


def test_bad_images_name_their_index(model):
    with pytest.raises(DecisionError) as caught:
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": [png(8, 8), b"?"]})
    assert caught.value.param == "images.1"
    with pytest.raises(DecisionError) as caught:
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": [png(2010, 8)]})
    assert caught.value.param == "images.0"
    assert "aspect ratio" in caught.value.message


def test_images_that_do_not_fit_fail_naming_images(folder):
    model = mlx_decision.load(folder, max_input_tokens=120)  # the questions take 90
    body = {"state": "billing " * 50, "questions": QUESTIONS}
    assert model.decide_request(body).truncated
    with pytest.raises(DecisionError) as caught:
        model.decide_request({**body, "images": [png(64, 64)]})
    assert caught.value.param == "images"
    assert "the images and questions need" in caught.value.message


def test_a_folder_without_vision_weights_refuses_images(tmp_path):
    model = mlx_decision.load(write_clef(tmp_path / "text-only"))
    with pytest.raises(DecisionError, match="does not support images"):
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": [png(8, 8)]})
    with pytest.raises(ValueError, match="no vision weights"):
        mlx_decision.load(tmp_path / "text-only", vision=True)


def test_text_only_use_needs_neither_numpy_nor_pillow(tmp_path):
    """The images extra is optional: without it Clef still loads and answers text."""
    import subprocess
    import sys

    folder = write_clef(tmp_path / "text-only")
    script = (
        "import sys; sys.modules['numpy'] = None; sys.modules['PIL'] = None\n"
        "import mlx_decision\n"
        f"model = mlx_decision.load({str(folder)!r})\n"
        "print(model.decide('billing', {'q': {'type': 'noul'}}).usage.input_tokens)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_large_images_are_shrunk_to_the_cap(folder):
    body = {"state": "billing", "questions": QUESTIONS, "images": [png(64, 64)]}
    # The tiny processor's own maximum is 4,096 pixels: 64 tokens of 8 x 8 pixels.
    uncapped = mlx_decision.load(folder, max_image_pixels=None).decide_request(body)
    capped = mlx_decision.load(folder, max_image_pixels=1024).decide_request(body)
    assert uncapped.usage.input_tokens - capped.usage.input_tokens == 64 - 16
    with pytest.raises(ValueError, match="at least 1024"):
        mlx_decision.load(folder, max_image_pixels=100)


def test_the_default_cap_is_two_megapixels(folder):
    from mlx_decision.models.clef.model import DEFAULT_MAX_IMAGE_PIXELS

    assert DEFAULT_MAX_IMAGE_PIXELS == 2**21
    assert mlx_decision.load(folder).backend.max_image_pixels == 2**21


def test_images_over_the_limit_are_never_decoded(folder, monkeypatch):
    import mlx_decision.images

    def decode(*args):
        raise AssertionError("decoded an image of a request that cannot fit")

    monkeypatch.setattr(mlx_decision.images, "decode_image", decode)
    model = mlx_decision.load(folder, max_input_tokens=120)
    with pytest.raises(DecisionError) as caught:
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": [png(64, 64)] * 3})
    assert caught.value.param == "images"


def test_a_cap_below_the_minimum_lowers_the_minimum(tmp_path):
    from mlx_decision.models.qwen import image_config

    assert image_config(tmp_path, None).min_pixels == 65536  # the processor's default
    config = image_config(tmp_path, 1024)
    assert (config.min_pixels, config.max_pixels) == (1024, 1024)


def test_other_formats_are_refused_naming_the_image(model):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="TIFF")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(DecisionError) as caught:
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": [png(16, 16), url]})
    assert caught.value.param == "images.1"
    assert "PNG, JPEG or WebP" in caught.value.message


def test_missing_files_are_named_before_the_backbone_loads(folder, tmp_path):
    import shutil

    copy = shutil.copytree(folder, tmp_path / "copy")
    (copy / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json is missing"):
        mlx_decision.load(copy)
