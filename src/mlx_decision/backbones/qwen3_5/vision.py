# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Ported from transformers (https://github.com/huggingface/transformers) 5.18.0,
# files models/qwen3_5/modeling_qwen3_5.py (Qwen3_5VisionModel and its
# patch embedding, rotary embedding, attention, blocks and patch merger) and
# vision_utils.py (vision position ids, bilinear interpolation of the learned
# position embeddings), under the Apache License 2.0. See
# LICENSES/transformers-Apache-2.0.txt.
#
# Modified for mlx-decision: written for MLX; inference only; images only (one
# frame each); the 3-D patch convolution is a linear layer over the flattened
# patch; attention runs image by image with MLX's fused attention.
"""The Qwen3.5 vision tower: image patches in, one embedding per merged token out."""

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .load import read_vision_weights, write_vision_weights


@dataclass
class VisionArgs:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    num_position_embeddings: int = 2304
    rope_theta: float = 10000.0

    @classmethod
    def from_config(cls, config: dict) -> "VisionArgs":
        vision = dict(config["vision_config"])
        rope = vision.pop("rope_parameters", None) or {}
        known = set(cls.__dataclass_fields__)
        args = cls(**{k: v for k, v in vision.items() if k in known})
        args.rope_theta = rope.get("rope_theta", args.rope_theta)
        return args

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class PatchEmbed(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        size = args.in_channels * args.temporal_patch_size * args.patch_size**2
        # A convolution whose kernel equals its stride is a linear map per patch.
        self.proj = nn.Linear(size, args.hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.proj(pixel_values.astype(self.proj.weight.dtype))


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class Attention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.num_heads = args.num_heads
        self.scale = args.head_dim**-0.5
        self.qkv = nn.Linear(args.hidden_size, args.hidden_size * 3)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, x: mx.array, lengths: list[int], cos: mx.array, sin: mx.array):
        n = x.shape[0]
        q, k, v = self.qkv(x).reshape(n, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        # The rotation is done in float32, as in the reference.
        dtype = q.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        q = (q * cos + rotate_half(q) * sin).astype(dtype)
        k = (k * cos + rotate_half(k) * sin).astype(dtype)
        # [heads, N, head_dim]; each image attends only to itself.
        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))
        outputs = []
        start = 0
        for length in lengths:
            part = slice(start, start + length)
            outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q[None, :, part], k[None, :, part], v[None, :, part], scale=self.scale
                )[0]
            )
            start += length
        out = mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]
        return self.proj(out.transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.linear_fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.linear_fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class Block(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(self, x, lengths, cos, sin):
        x = x + self.attn(self.norm1(x), lengths, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.merged_size = args.hidden_size * args.spatial_merge_size**2
        self.norm = nn.LayerNorm(args.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.linear_fc2 = nn.Linear(self.merged_size, args.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x).reshape(-1, self.merged_size)
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionModel(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbed(args)
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.hidden_size)
        self.blocks = [Block(args) for _ in range(args.depth)]
        self.merger = PatchMerger(args)
        spatial = args.head_dim // 2
        self._inv_freq = 1.0 / (
            args.rope_theta ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial)
        )

    def __call__(self, pixel_values: mx.array, grids: list[tuple[int, int, int]]) -> mx.array:
        """Patches ``[patches, C*T*P*P]`` of all images to merged embeddings ``[tokens, out]``."""
        rows, cols = patch_coordinates(grids, self.args.spatial_merge_size)
        x = self.patch_embed(pixel_values)
        x = x + self._position_embeddings(rows, cols, grids).astype(x.dtype)
        cos, sin = self._rotary(rows, cols)
        lengths = [t * h * w for t, h, w in grids]
        for block in self.blocks:
            x = block(x, lengths, cos, sin)
        return self.merger(x)

    def _position_embeddings(self, rows, cols, grids) -> mx.array:
        """The learned square grid resampled to each image (bilinear, corners aligned)."""
        side = int(self.args.num_position_embeddings**0.5)
        heights = np.concatenate([np.full(t * h * w, h) for t, h, w in grids])
        widths = np.concatenate([np.full(t * h * w, w) for t, h, w in grids])
        h_taps, h_weights = _bilinear_taps(rows, heights, side)
        w_taps, w_weights = _bilinear_taps(cols, widths, side)
        indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
        weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
        table = self.pos_embed(mx.array(indices.astype(np.int32)))
        return (table * mx.array(weights)[:, :, None]).sum(axis=1)

    def _rotary(self, rows, cols) -> tuple[mx.array, mx.array]:
        freqs = np.concatenate(
            [rows[:, None] * self._inv_freq, cols[:, None] * self._inv_freq], axis=-1
        )
        freqs = mx.array(np.concatenate([freqs, freqs], axis=-1).astype(np.float32))
        return mx.cos(freqs)[:, None, :], mx.sin(freqs)[:, None, :]


def patch_coordinates(grids, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of every patch, in the merge-block order of the pixel values."""
    rows, cols = [], []
    for t, h, w in grids:
        row, col = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        shape = (h // merge, merge, w // merge, merge)
        rows.append(np.tile(row.reshape(shape).transpose(0, 2, 1, 3).ravel(), t))
        cols.append(np.tile(col.reshape(shape).transpose(0, 2, 1, 3).ravel(), t))
    return np.concatenate(rows), np.concatenate(cols)


def _bilinear_taps(index: np.ndarray, size: np.ndarray, side: int):
    source = index.astype(np.float32) * (side - 1) / np.maximum(size - 1, 1).astype(np.float32)
    floor = np.floor(source)
    taps = floor.astype(np.int64)[:, None] + np.arange(2)
    weights = np.clip(1 - np.abs(source[:, None] - floor[:, None] - np.arange(2)), 0, None)
    return np.clip(taps, 0, side - 1), weights.astype(np.float32)


def load_vision_model(path: str | Path) -> VisionModel:
    """The vision tower of a Qwen3.5 folder, in the precision it is stored in."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    model = VisionModel(VisionArgs.from_config(config))
    model.load_weights(list(read_vision_weights(path).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def save_vision_model(model: VisionModel, path: str | Path) -> None:
    """Add the vision tower's weights to a folder written by ``save_text_model``."""
    write_vision_weights(dict(tree_flatten(model.parameters())), path)
