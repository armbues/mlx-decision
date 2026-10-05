"""Images in requests: data URLs, file paths, bytes or PIL images, read into RGB.

Pillow comes with the ``images`` extra; it is imported only when a request
carries images. Nothing is fetched from the network.
"""

import base64
import binascii
import io
import re
from pathlib import Path
from typing import Any

from .errors import DecisionError

INSTALL_HINT = "images need Pillow: pip install 'mlx-decision[images]'"
DATA_URL = re.compile(r"data:image/(png|jpeg|jpg|webp);base64,(.*)", re.DOTALL)


def require_pillow():
    try:
        import PIL.Image
        import PIL.ImageOps
    except ImportError:
        raise DecisionError(INSTALL_HINT, param="images", code="unsupported_media") from None
    return PIL


def is_data_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("data:")


def load_image(value: Any, index: int, allow_paths: bool = True):
    """One image as an RGB ``PIL.Image``; errors name ``images.<index>``.

    Accepts a data URL (PNG, JPEG or WebP), raw image bytes, a PIL image and,
    with ``allow_paths``, a file path. Like the reference, the image is turned
    upright by its EXIF orientation and converted to RGB (alpha is dropped).
    """
    PIL = require_pillow()
    where = f"images.{index}"
    if isinstance(value, PIL.Image.Image):
        image = value
    else:
        data = _image_bytes(value, where, allow_paths)
        try:
            image = PIL.Image.open(io.BytesIO(data))
            image.load()
        except PIL.Image.DecompressionBombError as error:
            raise DecisionError(f"the image is too large: {error}", param=where) from None
        except (PIL.UnidentifiedImageError, OSError, ValueError):
            raise DecisionError("not a readable image", param=where) from None
    image = PIL.ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _image_bytes(value: Any, where: str, allow_paths: bool) -> bytes:
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value)
    if is_data_url(value):
        match = DATA_URL.fullmatch(value)
        if match is None:
            raise DecisionError(
                "a data URL must look like data:image/png;base64,... (png, jpeg or webp)",
                param=where,
            )
        try:
            return base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError):
            raise DecisionError("the data URL is not valid base64", param=where) from None
    if allow_paths and isinstance(value, str | Path):
        path = Path(value).expanduser()
        try:
            return path.read_bytes()
        except OSError as error:
            hint = ""
            if isinstance(value, str) and len(value) > 200:
                hint = "; base64 needs the data:image/...;base64, prefix"
            reason = error.strerror or "cannot be read"
            raise DecisionError(f"{_short(value)}: {reason}{hint}", param=where) from None
    expected = "a data URL (data:image/png;base64,...)"
    if allow_paths:
        expected += ", a file path, bytes or a PIL image"
    raise DecisionError(f"expected {expected}", param=where)


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."
