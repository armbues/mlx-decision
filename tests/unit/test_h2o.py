"""H2O-Lightning on a tiny random model in the release's layout."""

import pytest

from mlx_decision.backbones.qwen3_5.load import has_vision_weights, load_text_model
from mlx_decision.backbones.qwen3_5.preprocess import ImageConfig
from mlx_decision.backbones.qwen3_5.vision import load_vision_model
from tiny_models import H2O_TEXT, PPLX_VISION, write_h2o


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o")


def test_the_single_file_release_loads_with_its_vision_tower(folder):
    assert not (folder / "model.safetensors.index.json").exists()
    backbone = load_text_model(folder, lm_head=False)
    assert backbone.language_model.args.tie_word_embeddings
    assert backbone.language_model.args.vocab_size == H2O_TEXT["vocab_size"]
    assert has_vision_weights(folder)
    vision = load_vision_model(folder)
    assert vision.patch_embed.proj.weight.shape[0] == PPLX_VISION["hidden_size"]


def test_the_flat_processor_config_is_read(folder, tmp_path):
    import json
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    size = {"shortest_edge": 65536, "longest_edge": 1048576}
    flat = {"patch_size": 16, "merge_size": 2, "size": size}
    (copy / "preprocessor_config.json").write_text(json.dumps(flat))
    assert ImageConfig.from_folder(copy).max_pixels == 1048576
