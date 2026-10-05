# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Ported from transformers (https://github.com/huggingface/transformers) 5.18.0,
# files models/qwen2_vl/image_processing_pil_qwen2_vl.py (smart_resize,
# patchify) and image_processing_backends.py (the PIL resize, rescale and
# normalize steps), under the Apache License 2.0. See
# LICENSES/transformers-Apache-2.0.txt.
#
# Modified for mlx-decision: one image at a time, Pillow and NumPy only, the
# settings read from the release's processor_config.json; images only (no
# video frames).
"""Turn an RGB image into the vision tower's patches, as the reference does."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_ASPECT_RATIO = 200


@dataclass(frozen=True)
class ImageConfig:
    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 16777216
    rescale_factor: float = 1 / 255
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @classmethod
    def from_folder(cls, path: str | Path) -> "ImageConfig":
        """The image settings of ``processor_config.json``, or the defaults without one."""
        file = Path(path) / "processor_config.json"
        if not file.exists():
            return cls()
        config = json.loads(file.read_text()).get("image_processor", {})
        size = config.get("size", {})
        return cls(
            patch_size=config.get("patch_size", cls.patch_size),
            merge_size=config.get("merge_size", cls.merge_size),
            temporal_patch_size=config.get("temporal_patch_size", cls.temporal_patch_size),
            min_pixels=size.get("shortest_edge", cls.min_pixels),
            max_pixels=size.get("longest_edge", cls.max_pixels),
            rescale_factor=config.get("rescale_factor", cls.rescale_factor),
            image_mean=tuple(config.get("image_mean", cls.image_mean)),
            image_std=tuple(config.get("image_std", cls.image_std)),
        )


def smart_resize(height: int, width: int, config: ImageConfig) -> tuple[int, int]:
    """The resized (height, width): multiples of the patch factor, area within the limits."""
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"the aspect ratio must be at most {MAX_ASPECT_RATIO}:1, "
            f"got {max(height, width) / min(height, width):.0f}:1"
        )
    factor = config.factor
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > config.max_pixels:
        beta = math.sqrt((height * width) / config.max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < config.min_pixels:
        beta = math.sqrt(config.min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(height: int, width: int, config: ImageConfig) -> int:
    """How many tokens an image of this size becomes."""
    h, w = smart_resize(height, width, config)
    return (h // config.factor) * (w // config.factor)


@dataclass(frozen=True)
class PreparedImage:
    pixel_values: np.ndarray  # (patches, channels * temporal * patch * patch), float32
    grid: tuple[int, int, int]  # (time, height, width) in patches
    tokens: int  # after the patch merger


def prepare_image(image, config: ImageConfig) -> PreparedImage:
    """Resize (bicubic), rescale, normalize and cut an RGB ``PIL.Image`` into patches."""
    from PIL import Image

    height, width = smart_resize(image.height, image.width, config)
    pixels = np.array(image.resize((width, height), resample=Image.Resampling.BICUBIC))
    pixels = pixels.transpose(2, 0, 1)  # channels first
    pixels = (pixels.astype(np.float64) * config.rescale_factor).astype(np.float32)
    mean = np.array(config.image_mean, dtype=np.float32)[:, None, None]
    std = np.array(config.image_std, dtype=np.float32)[:, None, None]
    pixels = (pixels - mean) / std

    p, m, t = config.patch_size, config.merge_size, config.temporal_patch_size
    channels = pixels.shape[0]
    grid_h, grid_w = height // p, width // p
    patches = pixels.reshape(channels, grid_h // m, m, p, grid_w // m, m, p)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)  # (gh, gw, mh, mw, C, ph, pw)
    # A still image is its own second frame.
    patches = np.broadcast_to(
        patches[:, :, :, :, :, None], (*patches.shape[:5], t, *patches.shape[5:])
    )
    flat = patches.reshape(grid_h * grid_w, channels * t * p * p)
    return PreparedImage(
        pixel_values=np.ascontiguousarray(flat),
        grid=(1, grid_h, grid_w),
        tokens=grid_h * grid_w // (m * m),
    )
