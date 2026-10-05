"""Images of the parity set, opened the same way for the reference and the tests.

A request lists its images as file names in ``images/`` or as
``{"file": name, "size": [width, height]}``, which resizes the image when it
is opened (bicubic), so large inputs need not be stored.
"""

from pathlib import Path

IMAGES = Path(__file__).parent / "images"


def open_image(spec):
    from PIL import Image

    if isinstance(spec, str):
        spec = {"file": spec}
    image = Image.open(IMAGES / spec["file"]).convert("RGB")
    if "size" in spec:
        image = image.resize(tuple(spec["size"]), Image.Resampling.BICUBIC)
    return image
