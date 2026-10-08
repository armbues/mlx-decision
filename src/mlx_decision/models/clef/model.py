"""The Clef backend: Qwen3.5 backbone (text and, on demand, vision) plus joint schema head."""

import json
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import has_vision_weights, load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...types import Request
from ..qwen import count_image_tokens, image_config, image_inputs, open_images, precision, prepare
from .encode import DEFAULT_MAX_LENGTH, IMAGE_PAD, encode_request
from .head import JointSchemaHead
from .prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache, prefix_key

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
        prefix_cache_gb: float = 0.0,
    ):
        self.name = name
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.path = path
        self.max_image_pixels = max_image_pixels
        self.vision = None
        # None: every request runs in one pass, as without reuse.
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
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
        if self.prefix_cache is not None:
            hidden = self._reusing_prefix(encoded, images, config)
        elif images:
            embeddings, positions = image_inputs(
                self.backbone, self.load_vision(), self.tokenizer, encoded.input_ids, images
            )
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

    def _reusing_prefix(self, encoded, images, config):
        """Hidden states of the whole input, the prefix's taken from the cache if kept.

        The prefix (up to the state's last token) and the rest run as two
        passes, whether the prefix is new or kept, so a request gets the same
        answers the first time and when its state comes again.
        """
        ids, length = encoded.input_ids, encoded.prefix_length
        key = prefix_key(ids[:length], [image.pixel_values for image in images])
        entry = self.prefix_cache.get(key)
        positions = None
        if entry is None:
            cache = self.backbone.make_cache()
            prefix_ids = mx.array(ids[:length])[None]
            if images:
                embeddings, positions = image_inputs(
                    self.backbone, self.load_vision(), self.tokenizer, ids, images
                )
                hidden = self.backbone(
                    prefix_ids, embeddings[None, :length], positions[:, :, :length], cache=cache
                )
            else:
                hidden = self.backbone(prefix_ids, cache=cache)
            mx.eval(hidden, [c.state for c in cache])
            entry = self.prefix_cache.put(key, cache, hidden[0])
        elif images:
            from ...backbones.qwen3_5.positions import position_ids

            pad = self.tokenizer.token_to_id(IMAGE_PAD)
            grids = [image.grid for image in images]
            positions = position_ids(ids, pad, grids, config.merge_size)
        # The rest is text: its token embeddings, and per-axis positions
        # continuing after the images when there are any.
        rest = self.backbone(
            mx.array(ids[length:])[None],
            None,
            None if positions is None else positions[:, :, length:],
            cache=entry.cache,
        )
        return mx.concatenate([entry.hidden, rest[0]], axis=0)


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weights ``load`` reads (the vision tower's included), kept as stored."""
    files = sorted(path.glob("model*.safetensors")) + [path / "joint_head.safetensors"]
    return [(file, None) for file in files if file.exists()]


def load(
    path: Path,
    max_input_tokens: int = DEFAULT_MAX_LENGTH,
    vision: bool = False,
    max_image_pixels: int | None = DEFAULT_MAX_IMAGE_PIXELS,
    prefix_cache_gb: float = DEFAULT_PREFIX_CACHE_GB,
) -> ClefBackend:
    """Load Clef from ``path``.

    ``vision=True`` loads the vision tower now, not on first use. Images
    larger than ``max_image_pixels`` are shrunk (keeping their aspect
    ratio); None leaves only the processor's own maximum, as the reference.
    ``prefix_cache_gb`` is the memory for kept prefixes (the computed state
    after a request's state), so a request with a state seen recently and
    other questions computes only its questions; 0 turns reuse off.
    """
    if max_image_pixels is not None and max_image_pixels < 32 * 32:
        raise ValueError("max_image_pixels must be at least 1024 (one image token)")
    if prefix_cache_gb < 0:
        raise ValueError("prefix_cache_gb must be 0 (off) or more")
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
    tokenizer = load_tokenizer(path)
    backend = ClefBackend(
        path.resolve().name,
        backbone,
        head,
        tokenizer,
        max_input_tokens,
        path,
        max_image_pixels,
        prefix_cache_gb,
    )
    backend.precision = precision(path, backbone)
    if vision:
        if not backend.capabilities.supports_images:
            raise ValueError(f"{path} has no vision weights")
        backend.load_vision()
    return backend
