"""Load a Qwen3.5 text model from a Hugging Face or MLX model folder."""

import json
from pathlib import Path

import mlx.core as mx

from .qwen3_5 import Model, ModelArgs


def load_text_model(path: str | Path) -> Model:
    """Build the text model described by ``config.json`` and load its weights.

    Vision weights in the folder are ignored. Weights keep the precision they
    are stored in.
    """
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))

    weight_files = sorted(path.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no model*.safetensors files in {path}")
    weights = {}
    for file in weight_files:
        weights.update(mx.load(str(file)))

    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
