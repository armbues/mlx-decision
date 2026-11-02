"""Images of the parity set, opened the same way for the reference and the tests.

A request lists its images as file names in ``images/`` or as
``{"file": name, "size": [width, height]}``, which resizes the image when it
is opened (bicubic), so large inputs need not be stored.

Base Labs' image requests (``baselabs/image-requests.json``) may also hold
images inside the state, as ``{"image_file": spec}`` with ``spec`` as above;
``with_data_urls`` turns them (and the top-level images) into PNG data URLs
for a request body, and ``onejev_media`` into the releases' own convention:
``<image:N>`` placeholders and the images as PNG files in a media folder.
"""

import base64
import io
from pathlib import Path

IMAGES = Path(__file__).parent / "images"
FILE = "image_file"
# Chat-style image parts, which the state's image lifting takes as one image.
PART_TYPES = ("image_url", "image", "input_image")


def open_image(spec):
    from PIL import Image

    if isinstance(spec, str):
        spec = {"file": spec}
    image = Image.open(IMAGES / spec["file"]).convert("RGB")
    if "size" in spec:
        image = image.resize(tuple(spec["size"]), Image.Resampling.BICUBIC)
    return image


def png_bytes(spec) -> bytes:
    buffer = io.BytesIO()
    open_image(spec).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(spec) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(spec)).decode()


def _spec_in(value):
    """The first ``{"image_file": spec}``'s spec inside ``value``, or None."""
    if isinstance(value, dict):
        if FILE in value:
            return value[FILE]
        items = value.values()
    elif isinstance(value, list):
        items = value
    else:
        return None
    for item in items:
        found = _spec_in(item)
        if found is not None:
            return found
    return None


def _walk(value, image, parts: bool):
    """``value`` with every ``{"image_file": spec}`` replaced by ``image(spec)``;
    with ``parts`` an image part holding one is replaced as a whole."""
    if isinstance(value, dict):
        if FILE in value:
            return image(value[FILE])
        if parts and value.get("type") in PART_TYPES and _spec_in(value) is not None:
            return image(_spec_in(value))
        return {key: _walk(item, image, parts) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, image, parts) for item in value]
    return value


def with_data_urls(case: dict) -> dict:
    """The request body: state images and top-level images as PNG data URLs."""
    body = {
        "state": _walk(case.get("state"), data_url, parts=False),
        "questions": case["questions"],
    }
    if case.get("images"):
        body["images"] = [data_url(spec) for spec in case["images"]]
    return body


def onejev_media(case: dict, folder: Path) -> tuple[object, list[dict]]:
    """The state in the releases' convention and its ``media`` (D234): top-level
    images as placeholders at the start of the state block, before its text
    (then numbered first), images inside the state replaced in place; each
    image written to ``folder`` as PNG, with its size."""
    from qev.prompt import render_state

    media = []

    def add(spec) -> str:
        image = open_image(spec)
        name = f"{len(media) + 1}.png"
        image.save(folder / name, format="PNG")
        media.append({"type": "image", "path": name, "h": image.height, "w": image.width})
        return f"<image:{len(media)}>"

    leading = "".join(add(spec) for spec in case.get("images") or [])
    state = _walk(case.get("state"), add, parts=True)
    if leading:
        text = render_state(state)
        state = leading + ("\n" + text if text else "")
    return state, media
