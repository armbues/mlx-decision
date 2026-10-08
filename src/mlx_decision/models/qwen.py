"""What the Qwen3.5-based families (Clef, pplx) share: images and precision."""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..errors import DecisionError
from ..registry import MARKER_FILE

IMAGE_PAD = "<|image_pad|>"


def image_inputs(backbone, vision, tokenizer, input_ids, images, features=None):
    """Token embeddings with the image features in place, and per-axis positions.

    ``features`` are the vision tower's outputs for ``images`` when already
    computed (several sequences with the same images); otherwise computed here.
    """
    import numpy as np

    from ..backbones.qwen3_5.positions import position_ids

    grids = [image.grid for image in images]
    if features is None:
        features = image_features(vision, images)
    embed = backbone.language_model.model.embed_tokens
    embeddings = embed(mx.array(input_ids))
    pad = tokenizer.token_to_id(IMAGE_PAD)
    slots = mx.array(np.flatnonzero(np.asarray(input_ids) == pad).astype(np.int32))
    embeddings[slots] = features.astype(embeddings.dtype)
    positions = position_ids(input_ids, pad, grids, vision.args.spatial_merge_size)
    return embeddings, positions


def image_features(vision, images) -> mx.array:
    """The vision tower's merged embeddings of all ``images``, in order."""
    import numpy as np

    pixels = np.concatenate([image.pixel_values for image in images])
    return vision(mx.array(pixels), [image.grid for image in images])


def image_config(path, max_pixels: int | None):
    """The release's image settings; ``max_pixels`` lowers its maximum (None keeps it)."""
    from dataclasses import replace

    from ..images import require_pillow

    require_pillow()  # before the preprocessing module imports NumPy
    from ..backbones.qwen3_5.preprocess import ImageConfig

    config = ImageConfig.from_folder(path)
    if max_pixels is not None and max_pixels < config.max_pixels:
        config = replace(
            config, max_pixels=max_pixels, min_pixels=min(config.min_pixels, max_pixels)
        )
    return config


def open_images(values) -> list[tuple[int, object]]:
    """Each image opened (header read, not decoded), with its index."""
    from ..images import open_image

    return [(index, open_image(value, index)) for index, value in enumerate(values)]


def count_image_tokens(image, index: int, config) -> int:
    from ..backbones.qwen3_5.preprocess import image_tokens

    try:
        return image_tokens(image.height, image.width, config)
    except ValueError as error:
        raise DecisionError(str(error), param=f"images.{index}") from None


def prepare(image, index: int, config):
    """Decode one opened image and cut it into the vision tower's patches."""
    from ..backbones.qwen3_5.preprocess import prepare_image
    from ..images import decode_image

    return prepare_image(decode_image(image, index), config)


def prepare_images(values, path, max_pixels: int | None):
    """Each image read and cut into patches; errors name ``images.<index>``.

    ``max_pixels`` lowers the processor's own maximum (None keeps it).
    """
    config = image_config(path, max_pixels)
    opened = open_images(values)
    for index, image in opened:
        count_image_tokens(image, index, config)
    return [prepare(image, index, config) for index, image in opened]


def precision(path: Path, backbone) -> str:
    """How the backbone's weights are stored: "mixed 4-bit", "8-bit" or the dtype."""
    marker = path / MARKER_FILE
    mixed = json.loads(marker.read_text()).get("mixed") if marker.exists() else None
    if mixed:
        return f"mixed {mixed['target_bits']:g}-bit"
    quantization = json.loads((path / "config.json").read_text()).get("quantization")
    if quantization:
        return f"{quantization['bits']}-bit"
    return str(tree_flatten(backbone.parameters())[0][1].dtype).rsplit(".", 1)[-1]
