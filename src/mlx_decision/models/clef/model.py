"""The Clef backend: Qwen3.5 backbone (text and, on demand, vision) plus joint schema head."""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from ...backbones.qwen3_5.load import has_vision_weights, load_text_model
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...registry import MARKER_FILE
from ...types import Request
from .encode import DEFAULT_MAX_LENGTH, IMAGE_PAD, encode_request
from .head import JointSchemaHead

# Larger images are shrunk to about this many pixels (2,048 tokens): measured
# to keep answers while bounding time and room in the input limit.
DEFAULT_MAX_IMAGE_PIXELS = 2**21
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    "joint_head_config.json",
    "joint_head.safetensors",
)


class ClefBackend:
    def __init__(
        self,
        name,
        backbone,
        head,
        tokenizer,
        max_input_tokens=DEFAULT_MAX_LENGTH,
        path: Path | None = None,
        max_image_pixels: int | None = DEFAULT_MAX_IMAGE_PIXELS,
    ):
        self.name = name
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.path = path
        self.max_image_pixels = max_image_pixels
        self.vision = None
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            # The release documents no limits; these are the Jev API's.
            max_choice_options=255,
            max_score_levels=10,
            supports_images=path is not None and has_vision_weights(path),
        )

    def load_vision(self):
        """The vision tower, loaded on first use (it stays in its stored precision)."""
        if self.vision is None:
            # Imported here: the vision code needs numpy (the images extra).
            from ...backbones.qwen3_5.vision import load_vision_model

            self.vision = load_vision_model(self.path)
        return self.vision

    def score(self, request: Request) -> BackendOutput:
        # Image sizes come from the file headers, so the input limit is checked
        # before any image is decoded or cut into patches.
        opened, config = [], None
        if request.images:
            config = image_config(self.path, self.max_image_pixels)
            opened = open_images(request.images)
        encoded = encode_request(
            self.tokenizer,
            request,
            max_length=self.capabilities.max_input_tokens,
            image_tokens=[count_image_tokens(image, index, config) for index, image in opened],
        )
        images = [prepare(image, index, config) for index, image in opened]
        input_ids = mx.array(encoded.input_ids)
        if images:
            embeddings, positions = self._image_inputs(encoded.input_ids, images)
            hidden = self.backbone(input_ids[None], embeddings[None], positions)[0]
        else:
            hidden = self.backbone(input_ids[None])[0]
        logits = self.head(
            hidden,
            input_ids,
            encoded.questions,
            self.backbone.language_model.output_embedding_rows,
        )
        probabilities = [mx.softmax(values.astype(mx.float32), axis=-1) for values in logits]
        mx.eval(probabilities)
        return BackendOutput(
            probabilities={
                question.question_id: dict(zip(question.option_ids, values.tolist(), strict=True))
                for question, values in zip(encoded.questions, probabilities, strict=True)
            },
            input_tokens=len(encoded.input_ids),
            truncated=encoded.truncated,
        )

    def _image_inputs(self, input_ids, images):
        """Token embeddings with the image features in place, and per-axis positions."""
        import numpy as np

        from ...backbones.qwen3_5.positions import position_ids

        vision = self.load_vision()
        grids = [image.grid for image in images]
        pixels = np.concatenate([image.pixel_values for image in images])
        features = vision(mx.array(pixels), grids)
        embed = self.backbone.language_model.model.embed_tokens
        embeddings = embed(mx.array(input_ids))
        pad = self.tokenizer.token_to_id(IMAGE_PAD)
        slots = mx.array(np.flatnonzero(np.asarray(input_ids) == pad).astype(np.int32))
        embeddings[slots] = features.astype(embeddings.dtype)
        positions = position_ids(input_ids, pad, grids, vision.args.spatial_merge_size)
        return embeddings, positions


def image_config(path, max_pixels: int | None):
    """The release's image settings; ``max_pixels`` lowers its maximum (None keeps it)."""
    from dataclasses import replace

    from ...images import require_pillow

    require_pillow()  # before the preprocessing module imports NumPy
    from ...backbones.qwen3_5.preprocess import ImageConfig

    config = ImageConfig.from_folder(path)
    if max_pixels is not None and max_pixels < config.max_pixels:
        config = replace(
            config, max_pixels=max_pixels, min_pixels=min(config.min_pixels, max_pixels)
        )
    return config


def open_images(values) -> list[tuple[int, object]]:
    """Each image opened (header read, not decoded), with its index."""
    from ...images import open_image

    return [(index, open_image(value, index)) for index, value in enumerate(values)]


def count_image_tokens(image, index: int, config) -> int:
    from ...backbones.qwen3_5.preprocess import image_tokens

    try:
        return image_tokens(image.height, image.width, config)
    except ValueError as error:
        raise DecisionError(str(error), param=f"images.{index}") from None


def prepare(image, index: int, config):
    """Decode one opened image and cut it into the vision tower's patches."""
    from ...backbones.qwen3_5.preprocess import prepare_image
    from ...images import decode_image

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


def load(
    path: Path,
    max_input_tokens: int = DEFAULT_MAX_LENGTH,
    vision: bool = False,
    max_image_pixels: int | None = DEFAULT_MAX_IMAGE_PIXELS,
) -> ClefBackend:
    """Load Clef from ``path``.

    ``vision=True`` loads the vision tower now, not on first use. Images
    larger than ``max_image_pixels`` are shrunk (keeping their aspect
    ratio); None leaves only the processor's own maximum, as the reference.
    """
    if max_image_pixels is not None and max_image_pixels < 32 * 32:
        raise ValueError("max_image_pixels must be at least 1024 (one image token)")
    # Checked before the backbone, which takes a while to load.
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    backbone = load_text_model(path)
    head = JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_weights(str(path / "joint_head.safetensors"), strict=True)
    head.eval()
    mx.eval(head.parameters())
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    backend = ClefBackend(
        path.resolve().name, backbone, head, tokenizer, max_input_tokens, path, max_image_pixels
    )
    backend.precision = precision(path, backbone)
    if vision:
        if not backend.capabilities.supports_images:
            raise ValueError(f"{path} has no vision weights")
        backend.load_vision()
    return backend
