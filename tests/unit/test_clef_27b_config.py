"""Clef 27B's release layout against the models built from its config.

Needs only the release's JSON files in the Hugging Face cache (config and
weight index, a few hundred KB); skipped without them. Shapes were checked
once against the safetensors headers; the names are checked here.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from mlx_decision.backbones.qwen3_5.load import VISION_PREFIXES, sanitize_vision_weights
from mlx_decision.backbones.qwen3_5.qwen3_5 import Model, ModelArgs
from mlx_decision.backbones.qwen3_5.vision import VisionArgs, VisionModel
from mlx_decision.models.clef.head import JointSchemaHead


@pytest.fixture(scope="module")
def release() -> Path:
    try:
        path = Path(
            snapshot_download("Cloudflare/clef", allow_patterns=["*.json"], local_files_only=True)
        )
    except Exception:
        path = None
    if path is None or not (path / "model.safetensors.index.json").exists():
        pytest.skip("Clef 27B's JSON files are not in the Hugging Face cache")
    return path


def stand_ins(names) -> dict[str, mx.array]:
    """Unevaluated arrays for the names; conv kernels in the release's [C, 1, K] layout."""
    return {
        name: mx.zeros((1, 1, 4)) if name.endswith("conv1d.weight") else mx.zeros((1,))
        for name in names
    }


def parameter_names(module) -> set[str]:
    # Built lazily: MLX does not allocate the random initial weights until evaluated.
    return {name for name, _ in tree_flatten(module.parameters())}


def test_the_text_model_has_every_released_weight(release):
    config = json.loads((release / "config.json").read_text())
    names = json.loads((release / "model.safetensors.index.json").read_text())["weight_map"]
    model = Model(ModelArgs.from_dict(config))
    text = [name for name in names if not name.startswith(VISION_PREFIXES)]
    assert set(model.sanitize(stand_ins(text))) == parameter_names(model)
    assert model.language_model.args.hidden_size == 5120


def test_the_vision_tower_has_every_released_weight(release):
    config = json.loads((release / "config.json").read_text())
    names = json.loads((release / "model.safetensors.index.json").read_text())["weight_map"]
    vision = [name for name in names if name.startswith(VISION_PREFIXES)]
    model = VisionModel(VisionArgs.from_config(config))
    assert set(sanitize_vision_weights(stand_ins(vision))) == parameter_names(model)


def test_the_head_config_builds(release):
    config = json.loads((release / "joint_head_config.json").read_text())
    assert config["hidden_size"] == 5120
    JointSchemaHead(**config)
