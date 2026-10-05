"""Rotary positions with images, against transformers' get_rope_index when installed."""

from types import SimpleNamespace

import numpy as np
import pytest

from mlx_decision.backbones.qwen3_5.positions import position_ids

PAD = 9


def sequence(*parts):
    """Text lengths (int) and image grids (tuple), as token ids with start/end markers."""
    ids, grids = [], []
    for part in parts:
        if isinstance(part, int):
            ids += [1] * part
        else:
            t, h, w = part
            ids += [7] + [PAD] * (t * h * w // 4) + [8]
            grids.append(part)
    return ids, grids


def test_text_only_is_ordinary_positions():
    ids, grids = sequence(5)
    assert np.array(position_ids(ids, PAD, grids, 2)).tolist() == [[[0, 1, 2, 3, 4]]] * 3


def test_an_image_spans_rows_and_columns():
    ids, grids = sequence(2, (1, 4, 6), 2)  # 2 x 3 tokens after merging
    positions = np.array(position_ids(ids, PAD, grids, 2))[:, 0]
    # text 0..1, vision_start 2, image at 3 (rows 3..4, columns 3..5), vision_end 6, ...
    assert positions[:, :3].tolist() == [[0, 1, 2]] * 3
    assert positions[0, 3:9].tolist() == [3] * 6
    assert positions[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
    assert positions[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
    assert positions[:, 9:].tolist() == [[6, 7, 8]] * 3


def test_grids_must_match_the_tokens():
    ids, _ = sequence((1, 4, 4))
    with pytest.raises(ValueError, match="needs 16"):
        position_ids(ids, PAD, [(1, 8, 8)], 2)
    with pytest.raises(ValueError, match="more image grids"):
        position_ids(ids, PAD, [(1, 4, 4), (1, 4, 4)], 2)


def test_equal_to_the_reference():
    pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    import torch

    ids, grids = sequence(36, (1, 30, 40), 1, (1, 16, 16), (1, 2, 228), 500)
    model = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    )
    cls = modeling.Qwen3_5Model
    model.get_vision_position_ids = cls.get_vision_position_ids.__get__(model)
    input_ids = torch.tensor([ids])
    expected, _ = cls.get_rope_index(
        model,
        input_ids,
        mm_token_type_ids=(input_ids == PAD).long(),
        image_grid_thw=torch.tensor(grids),
    )
    assert np.array(position_ids(ids, PAD, grids, 2)).tolist() == expected.tolist()
