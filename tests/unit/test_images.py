"""Reading images from data URLs, paths, bytes and PIL images."""

import base64
import io
import sys

import pytest
from PIL import Image

from mlx_decision.errors import DecisionError
from mlx_decision.images import load_image


def encoded(image, fmt="PNG", **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def data_url(image, fmt="PNG", mime="png") -> str:
    return f"data:image/{mime};base64," + base64.b64encode(encoded(image, fmt)).decode()


RED = Image.new("RGB", (8, 4), (255, 0, 0))


@pytest.mark.parametrize(("fmt", "mime"), [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp")])
def test_data_urls(fmt, mime):
    image = load_image(data_url(RED, fmt, mime), 0)
    assert image.mode == "RGB" and image.size == (8, 4)


def test_paths_bytes_and_pil_images(tmp_path):
    path = tmp_path / "red.png"
    RED.save(path)
    for value in (path, str(path), path.read_bytes(), RED):
        assert load_image(value, 0).getpixel((0, 0)) == (255, 0, 0)


def test_alpha_is_dropped_like_the_reference():
    rgba = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    assert load_image(rgba, 0).getpixel((0, 0)) == (10, 20, 30)


def test_exif_orientation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees clockwise
    raw = encoded(RED, "JPEG", exif=exif.tobytes())
    assert load_image(raw, 0).size == (4, 8)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("data:image/gif;base64,R0lG", "a data URL must look like"),
        ("data:image/png;base64,***", "not valid base64"),
        ("data:image/png;base64," + base64.b64encode(b"hello").decode(), "not a readable image"),
        (b"not an image", "not a readable image"),
        ("missing.png", "missing.png: No such file or directory"),
        (base64.b64encode(encoded(RED)).decode() * 3, "needs the data:image/...;base64, prefix"),
        (42, "expected a data URL"),
    ],
)
def test_errors_name_the_image(value, message):
    with pytest.raises(DecisionError) as caught:
        load_image(value, 3)
    assert caught.value.param == "images.3"
    assert message in caught.value.message


def test_paths_can_be_refused(tmp_path):
    path = tmp_path / "red.png"
    RED.save(path)
    with pytest.raises(DecisionError, match="expected a data URL"):
        load_image(str(path), 0, allow_paths=False)
    assert load_image(data_url(RED), 0, allow_paths=False).size == (8, 4)


def test_without_pillow_there_is_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(DecisionError) as caught:
        load_image(b"", 0)
    assert caught.value.code == "unsupported_media"
    assert "mlx-decision[images]" in caught.value.message
