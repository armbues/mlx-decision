"""The Clef backend: Qwen3.5 backbone (text and, on demand, vision) plus joint schema head."""

import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.qwen3_5.load import has_vision_weights, load_text_model
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Request
from .encode import DEFAULT_MAX_LENGTH, IMAGE_PAD, encode_request
from .head import JointSchemaHead


class ClefBackend:
    def __init__(
        self,
        name,
        backbone,
        head,
        tokenizer,
        max_input_tokens=DEFAULT_MAX_LENGTH,
        path: Path | None = None,
    ):
        self.name = name
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.path = path
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
        images = prepare_images(request.images, self.path) if request.images else []
        encoded = encode_request(
            self.tokenizer,
            request,
            max_length=self.capabilities.max_input_tokens,
            image_tokens=[image.tokens for image in images],
        )
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


def prepare_images(values, path):
    """Each image read and cut into patches; errors name ``images.<index>``."""
    from ...backbones.qwen3_5.preprocess import ImageConfig, prepare_image
    from ...images import load_image

    config = ImageConfig.from_folder(path)
    prepared = []
    for index, value in enumerate(values):
        image = load_image(value, index)
        try:
            prepared.append(prepare_image(image, config))
        except ValueError as error:
            raise DecisionError(str(error), param=f"images.{index}") from None
    return prepared


def load(
    path: Path, max_input_tokens: int = DEFAULT_MAX_LENGTH, vision: bool = False
) -> ClefBackend:
    """Load Clef from ``path``; ``vision=True`` loads the vision tower now, not on first use."""
    backbone = load_text_model(path)
    head = JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_weights(str(path / "joint_head.safetensors"), strict=True)
    head.eval()
    mx.eval(head.parameters())
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    backend = ClefBackend(path.resolve().name, backbone, head, tokenizer, max_input_tokens, path)
    if vision:
        if not backend.capabilities.supports_images:
            raise ValueError(f"{path} has no vision weights")
        backend.load_vision()
    return backend
