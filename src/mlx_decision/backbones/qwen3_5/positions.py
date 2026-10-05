# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Ported from transformers (https://github.com/huggingface/transformers) 5.18.0,
# file models/qwen3_5/modeling_qwen3_5.py (get_rope_index and
# get_vision_position_ids), under the Apache License 2.0. See
# LICENSES/transformers-Apache-2.0.txt.
#
# Modified for mlx-decision: one sequence without padding, images only, image
# tokens found by their pad id instead of token type ids; NumPy and MLX.
"""Per-axis rotary positions (time, height, width) for a sequence with images."""

import mlx.core as mx
import numpy as np


def position_ids(
    input_ids: list[int] | tuple[int, ...],
    image_pad_id: int,
    grids: list[tuple[int, int, int]],
    merge_size: int,
) -> mx.array:
    """Positions ``[3, 1, L]`` for ``input_ids``.

    Text advances all three axes together. Each run of image pad tokens is
    the next image of ``grids`` (patch grid before merging): its tokens get
    the same time position and their row and column offsets; the text after
    it continues ``max(height, width)`` positions further.
    """
    ids = np.asarray(input_ids)
    is_image = ids == image_pad_id
    # Start and end of each run of equal token kinds.
    edges = np.flatnonzero(np.diff(is_image.astype(np.int8))) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(ids)]])

    pieces = []
    current = 0
    images = iter(grids)
    for start, end in zip(starts, ends, strict=True):
        if not is_image[start]:
            length = end - start
            pieces.append(np.broadcast_to(np.arange(length) + current, (3, length)))
            current += length
            continue
        t, h, w = next(images)
        h, w = h // merge_size, w // merge_size
        if t * h * w != end - start:
            raise ValueError(f"an image has {end - start} tokens, its grid needs {t * h * w}")
        time, row, col = np.meshgrid(np.arange(t), np.arange(h), np.arange(w), indexing="ij")
        pieces.append(np.stack([time, row, col]).reshape(3, -1) + current)
        current += max(h, w)
    if next(images, None) is not None:
        raise ValueError("more image grids than images in the input")
    return mx.array(np.concatenate(pieces, axis=1)[:, None, :].astype(np.int32))
