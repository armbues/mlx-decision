# Copyright H2O.ai (the H2O-Lightning authors).
#
# Ported from H2O-Lightning-4B (https://huggingface.co/h2oai/h2o-lightning-4b,
# revision acaf0d4ea251e54de928c75ef4352670d33192d3, tag v1.2.3), file
# h2o_lightning_shim.py (sniff_image, _b64_bytes, image_from_string,
# image_from_part, lift_images), under the Apache License 2.0. See
# LICENSES/h2o-lightning-Apache-2.0.txt.
#
# Modified for mlx-decision: images are returned as their bytes with the
# path in the state they came from; image URLs are refused (nothing is
# fetched) instead of passed on; errors name the offending field.
"""Images inside a request's state, lifted out into its images.

A request can carry an image in its state instead of in ``images``: a value
that is a data URI (``data:image/png;base64,...``), bare base64 of a PNG,
JPEG, GIF or WebP (one unbroken token of at least 64 characters, so text
never qualifies), or a chat-style image part holding one of these. Each one
is replaced by ``[image N]`` and joins the request's images after the
top-level ones, N counting all of them from 1.
"""

import base64
import binascii
import re
from typing import Any

from .errors import DecisionError

_DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S | re.I)
_URL = re.compile(r"https?://", re.I)
_B64_CHARS = re.compile(r"[A-Za-z0-9+/=\s]+")
_B64_URLSAFE = re.compile(r"[A-Za-z0-9_\-=\s]+")
_PART_TYPES = ("image_url", "image", "input_image")
# How base64 of a PNG, JPEG, GIF and WebP begins: the cheap test before
# anything in the state is decoded.
_B64_HEADS = ("iVBORw0KGgo", "/9j/", "R0lGOD", "UklGR")
_BARE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")


def sniff_image(raw: bytes) -> str | None:
    """The image type from its first bytes, or None."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _b64_bytes(text: str) -> bytes | None:
    """The decoded bytes of standard or URL-safe base64, or None."""
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    padding = "=" * (-len(text) % 4)
    try:
        if _B64_CHARS.fullmatch(text):
            return base64.b64decode(text + padding, validate=True)
        if _B64_URLSAFE.fullmatch(text):
            return base64.urlsafe_b64decode(text + padding)
    except (binascii.Error, ValueError):
        return None
    return None


def _from_string(value: str, where: str) -> bytes | None:
    """A string's image bytes, or None when it is not an image.

    A data URI is always an image (one without valid base64 is an error);
    bare base64 only as described in the module docstring.
    """
    value = value.strip()
    if match := _DATA_URI.match(value):
        raw = _b64_bytes(match.group(2))
        if not raw:
            raise DecisionError("an image data URI does not carry valid base64", param=where)
        return raw
    # The cheap tests first: most strings are text and fail the first one.
    if (
        not value.startswith(_B64_HEADS)
        or len(value) < 64
        or not _B64_CHARS.fullmatch(value)
        or any(c.isspace() for c in value)
    ):
        return None
    raw = _b64_bytes(value)
    return raw if raw and sniff_image(raw) in _BARE_TYPES else None


def _from_part(part: dict, where: str) -> bytes | None:
    """A chat-style image part's bytes, or None when it is record data.

    An object shaped like a part whose value is not an image (say
    ``{"type": "image", "image": "a sunset"}``) stays in the state.
    """
    kind = part.get("type")
    if kind == "base64" and isinstance(part.get("data"), str):
        value = f"data:{part.get('media_type') or 'image/png'};base64,{part['data']}"
    elif kind in _PART_TYPES:
        value = (
            part.get(kind) if kind != "input_image" else part.get("image_url", part.get("image"))
        )
        if isinstance(value, dict):
            value = value.get("url", value.get("data"))
        if not isinstance(value, str):
            return None
    else:
        return None
    if _URL.match(value.strip()):
        raise DecisionError("image URLs are not fetched; send the image as a data URL", param=where)
    return _from_string(value, where)


def lift_images(state: Any, first: int = 0) -> tuple[Any, list[tuple[bytes, str]]]:
    """``state`` with each image in it replaced by ``[image N]``, and the images.

    ``first`` is the number of images the request already has (numbered
    first). Each lifted image comes with the path where it was in the state
    (``state.attachments.0``), for errors. A state without images is
    returned as it is (the same object).
    """
    found: list[tuple[bytes, str]] = []

    def lift(value: Any, where: str) -> Any:
        image = None
        if isinstance(value, str):
            image = _from_string(value, where)
        elif isinstance(value, dict):
            image = _from_part(value, where)
            if image is None:
                out = {key: lift(item, f"{where}.{key}") for key, item in value.items()}
                changed = any(out[key] is not value[key] for key in value)
                return out if changed else value
        elif isinstance(value, list):
            out = [lift(item, f"{where}.{index}") for index, item in enumerate(value)]
            changed = any(a is not b for a, b in zip(out, value, strict=True))
            return out if changed else value
        if image is None:
            return value
        found.append((image, where))
        return f"[image {first + len(found)}]"

    return lift(state, "state"), found
