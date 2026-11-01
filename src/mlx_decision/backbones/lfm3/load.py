"""Load an LFM3-VL release's text model, or its vision tower and projector.

The release keeps the text model under ``model.language_model.*``, the
tower under ``model.vision_tower.*`` and the projector under
``model.multi_modal_projector.*``; ``config.json`` holds ``text_config``,
``vision_config`` and the projector's settings. Each part reads only its
own weights (the rest of a lazily loaded file is never read), in the
precision they are stored in.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .language import Model
from .lfm3_vl import Vision
from .vl_config import ModelConfig, TextConfig


def read_config(path: str | Path) -> ModelConfig:
    return ModelConfig.from_dict(json.loads((Path(path) / "config.json").read_text()))


def _weights(path: Path) -> dict[str, mx.array]:
    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors files in {path}")
    weights = {}
    for file in files:
        weights.update(mx.load(str(file)))
    return weights


def _load(model: nn.Module, weights: dict[str, mx.array]) -> nn.Module:
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_text_model(path: str | Path) -> Model:
    """The text model (tied output embeddings, no ``lm_head``)."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(TextConfig.from_dict(config["text_config"]))
    return _load(model, model.sanitize(_weights(path)))


def load_vision(path: str | Path) -> Vision:
    """The vision tower and projector."""
    path = Path(path)
    vision = Vision(read_config(path))
    return _load(vision, vision.sanitize(_weights(path)))
