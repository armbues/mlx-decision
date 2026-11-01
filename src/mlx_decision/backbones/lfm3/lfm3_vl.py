# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0
#
# Copied from mlx-vlm (https://github.com/Blaizzy/mlx-vlm), file
# mlx_vlm/models/lfm3_vl/lfm3_vl.py at commit 527c513a0768245644a70266a22126d880767b18
# (v0.7.11), under the Apache License 2.0. See
# LICENSES/mlx-vlm-lfm3-Apache-2.0.txt.
#
# Modified for mlx-decision: only the vision side (tower and projector,
# ``Vision``); the text model loads on its own (``language.Model``), and the
# caller places the image features at the image tokens. Removed: the merge
# into input embeddings, videos, the language model and its cache.

import mlx.core as mx
import mlx.nn as nn

from .vision import VisionModel


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.vision_config.hidden_size * config.downsample_factor**2
        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size, eps=config.projector_layer_norm_eps
        )
        self.linear_1 = nn.Linear(
            self.input_dim, config.projector_hidden_size, bias=config.projector_bias
        )
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def __call__(self, x):
        x = self.layer_norm(x).reshape(-1, self.input_dim)
        return self.linear_2(nn.gelu(self.linear_1(x)))


class Vision(nn.Module):
    """The vision tower and projector: patches to the text model's embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)

    def __call__(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """One row per image token, the images' in order. ``pixel_values`` are the
        patches of every image grouped by merge block; ``grid_thw`` each image's
        (frames, height, width) in patches."""
        return self.multi_modal_projector(self.vision_tower(pixel_values, grid_thw))

    def sanitize(self, weights):
        """Release names (``model.vision_tower.*``, ``model.multi_modal_projector.*``,
        PyTorch's conv kernels) to these; other weights are left out."""
        result = {}
        for name, value in weights.items():
            if not name.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
                continue
            name = name[len("model.") :]
            if name.endswith("spatial_conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 3, 1)
            result[name] = value
        return result
