"""Image preprocessing against the reference's PIL processor (transformers, when installed)."""

import numpy as np
import pytest
from PIL import Image

from mlx_decision.backbones.qwen3_5.preprocess import (
    ImageConfig,
    image_tokens,
    prepare_image,
    smart_resize,
)

# clef-flash's processor_config.json
CONFIG = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_size": 2,
    "patch_size": 16,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "temporal_patch_size": 2,
}
SIZES = [(32, 32), (300, 200), (640, 480), (1001, 333), (2000, 10), (250, 4000), (4100, 4100)]


def picture(width, height, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, width)[None, :, None]
    y = np.linspace(0, 255, height)[:, None, None]
    noise = rng.normal(0, 25, (height, width, 3))
    pixels = np.clip((x * 0.6 + y * 0.4) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.fixture(scope="module")
def reference():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from transformers.models.qwen2_vl import image_processing_pil_qwen2_vl as module

    del transformers
    return module


def test_config_from_the_release_file(tmp_path):
    import json

    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": CONFIG}))
    assert ImageConfig.from_folder(tmp_path) == ImageConfig()
    assert ImageConfig.from_folder(tmp_path / "missing") == ImageConfig()


def test_smart_resize_matches_the_reference(reference):
    config = ImageConfig()
    for height in (1, 10, 31, 32, 33, 100, 255, 479, 1080, 3000, 5000, 9999):
        for width in (1, 17, 64, 333, 1920, 4096, 8000):
            if max(height, width) / min(height, width) > 200:
                continue
            expected = reference.smart_resize(height, width, 32, 65536, 16777216)
            assert smart_resize(height, width, config) == expected, (height, width)


def test_tokens_per_image():
    config = ImageConfig()
    assert image_tokens(480, 640, config) == 300
    assert image_tokens(10, 10, config) == 64
    assert image_tokens(4096, 4096, config) == 16384


def test_extreme_aspect_ratios_are_refused():
    with pytest.raises(ValueError, match="aspect ratio must be at most 200:1"):
        smart_resize(10, 2010, ImageConfig())


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_pixel_values_equal_the_reference(reference, width, height):
    image = picture(width, height)
    expected = reference.Qwen2VLImageProcessorPil(**CONFIG)(images=[image], return_tensors="np")
    prepared = prepare_image(image, ImageConfig())
    assert list(prepared.grid) == expected["image_grid_thw"][0].tolist()
    assert prepared.pixel_values.dtype == np.float32
    assert prepared.pixel_values.shape == expected["pixel_values"].shape
    np.testing.assert_array_equal(prepared.pixel_values, expected["pixel_values"])
    assert prepared.tokens == prepared.pixel_values.shape[0] // 4
