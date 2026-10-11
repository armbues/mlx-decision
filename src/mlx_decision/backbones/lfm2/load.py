"""Load an LFM2 text model from a Hugging Face folder.

LFM2-VL releases (d1-3B) keep the text model under ``model.language_model.*``
next to a vision tower and a projector, and their ``config.json`` holds the
text model's settings in ``text_config``; plain LFM2 releases use ``model.*``
and a flat config. The vision weights are not read.
"""

import json
from pathlib import Path

import mlx.core as mx

from .lfm2 import Model, ModelArgs

LANGUAGE_MODEL = "model.language_model."
NOT_TEXT = ("model.vision_tower.", "model.multi_modal_projector.", "lm_head.")


def text_config(config: dict) -> dict:
    """The text model's settings of an LFM2 or LFM2-VL ``config.json``."""
    return config.get("text_config", config)


def text_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The text model's weights, named as ``Model`` expects (``model.*``)."""
    if any(name.startswith(LANGUAGE_MODEL) for name in weights):
        return {
            "model." + name[len(LANGUAGE_MODEL) :]: value
            for name, value in weights.items()
            if name.startswith(LANGUAGE_MODEL)
        }
    return {name: value for name, value in weights.items() if not name.startswith(NOT_TEXT)}


def load_text_model(path: str | Path) -> Model:
    """Build the text model described by ``config.json`` and load its weights.

    Weights keep the precision they are stored in. The output embeddings are
    the input embeddings (tied), so there is no separate ``lm_head``.
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(text_config(config)))
    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors files in {path}")
    weights = {}
    for file in files:
        weights.update(mx.load(str(file)))
    weights = model.sanitize(text_weights(weights))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
