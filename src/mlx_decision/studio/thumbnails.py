"""Small previews of images for the Studio's pages: a JPEG data URL a few
dozen pixels across, made with Pillow (imported only when there is one)."""

import base64
import io

# The longest side of a thumbnail, in pixels (shown at half that on a 2x screen).
THUMB_PIXELS = 160


def thumbnail(data: bytes | None, pixels: int = THUMB_PIXELS) -> str | None:
    """``data`` (an image file's bytes) as a ``data:image/jpeg;base64,...`` URL
    of at most ``pixels`` on its longest side; None when it does not read."""
    if not data:
        return None
    import PIL.Image
    import PIL.ImageOps

    try:
        with PIL.Image.open(io.BytesIO(data)) as image:
            image.draft("RGB", (pixels, pixels))  # JPEG: decode at a smaller scale
            small = PIL.ImageOps.exif_transpose(image).convert("RGB")
            small.thumbnail((pixels, pixels))
            buffer = io.BytesIO()
            small.save(buffer, "JPEG", quality=80)
    except (PIL.Image.DecompressionBombError, PIL.UnidentifiedImageError, OSError, ValueError):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
