# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0
#
# Copied from mlx-vlm (https://github.com/Blaizzy/mlx-vlm), file
# mlx_vlm/models/lfm3/language.py at commit 527c513a0768245644a70266a22126d880767b18
# (v0.7.11), under the Apache License 2.0. See
# LICENSES/mlx-vlm-lfm3-Apache-2.0.txt.
#
# Modified for mlx-decision: the model returns the final hidden states, and
# ``output_embedding_rows`` gives rows of the tied output embeddings, so a
# readout computes only the logits it needs. A pass may take a ``SetLayout``
# (rows of packed question sets, right-padded): on full-attention layers the
# state's tokens see every real token of their row and a question's tokens see
# the state and their own question; sliding-window layers stay causal. The
# masks are built a block of queries at a time, so long rows never hold a
# tokens x tokens mask. Imports from the Qwen3.5 copy's ``base`` and the LFM2
# copy's ``ShortConv`` (mlx-vlm's ``base``, ``cache`` and LFM2 are not
# copied). Removed: caches (nothing is kept between passes) and the logits.

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..lfm2.lfm2 import ShortConv
from ..qwen3_5.base import SCORE_BUDGET, scaled_dot_product_attention

# Rows at most this long get their sliding-window mask whole; longer ones go
# in blocks of this many queries.
SLIDING_WHOLE = 4096
SLIDING_BLOCK = 1024


@dataclass
class SetLayout:
    """Rows of packed question sets, right-padded to a common length.

    ``lengths`` are each row's real tokens, ``state`` each row's state tokens
    (everything before its first question), and ``question`` ([rows, tokens],
    int32) the question each token belongs to, -1 on the state and padding.
    """

    lengths: list[int]
    state: list[int]
    question: mx.array


@dataclass
class SetMasks:
    """What a pass's full-attention layers need from a ``SetLayout``."""

    first: int  # queries before this are state tokens in every row
    keys: mx.array  # [rows, 1, 1, tokens]: the real tokens
    lengths: mx.array  # [rows]
    state: mx.array  # [rows]
    question: mx.array  # [rows, tokens]

    @classmethod
    def of(cls, layout: SetLayout) -> "SetMasks":
        tokens = layout.question.shape[1]
        lengths = mx.array(layout.lengths)
        keys = (mx.arange(tokens)[None] < lengths[:, None])[:, None, None, :]
        return cls(min(layout.state), keys, lengths, mx.array(layout.state), layout.question)

    def block(self, start: int, end: int) -> mx.array:
        """``[rows, 1, end - start, tokens]``: what queries ``start`` to ``end`` see.
        A state query sees every real token; any other query the state's tokens
        and those of its own question (padding: the state only)."""
        tokens = self.question.shape[1]
        queries = mx.arange(start, end)[None, :, None]
        keys = mx.arange(tokens)[None, None, :]
        state = self.state[:, None, None]
        own = self.question[:, start:end, None]
        same = (own == self.question[:, None, :]) & (own >= 0)
        seen = (queries < state) | (keys < state) | same
        return (seen & (keys < self.lengths[:, None, None]))[:, None]


def query_blocks(queries, keys, values, scale: float, start: int, mask_of) -> mx.array:
    """Attention for queries ``start`` onwards, a block of queries at a time:
    ``mask_of(first, last)`` gives ``(mask, key_start)``, the mask over keys from
    ``key_start``. Each block is evaluated before the next one is built."""
    batch, heads, length, _ = queries.shape
    block = max(64, SCORE_BUDGET // (batch * heads * keys.shape[-2]) // 64 * 64)
    outputs = []
    for first in range(start, length, block):
        last = min(length, first + block)
        mask, key_start = mask_of(first, last)
        output = mx.fast.scaled_dot_product_attention(
            queries[:, :, first:last],
            keys[:, :, key_start:],
            values[:, :, key_start:],
            scale=scale,
            mask=mask,
        )
        if last < length:
            mx.eval(output)
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def set_attention(queries, keys, values, scale: float, masks: SetMasks) -> mx.array:
    """Full attention under a set layout: the state's queries with a key-only mask
    (blockwise as needed), the rest with masks a block of queries at a time."""
    parts = []
    if masks.first > 0:
        parts.append(
            scaled_dot_product_attention(
                queries[:, :, : masks.first], keys, values, None, scale, masks.keys
            )
        )
    if masks.first < queries.shape[2]:
        parts.append(
            query_blocks(
                queries, keys, values, scale, masks.first, lambda a, b: (masks.block(a, b), 0)
            )
        )
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def sliding_attention(queries, keys, values, scale: float, window: int) -> mx.array:
    """Causal attention within ``window`` tokens (a query sees itself and the
    ``window - 1`` before it). Long rows go a block of queries at a time, each
    over only the keys its window reaches."""
    length = queries.shape[2]

    def mask_of(first, last):
        key_start = 0 if length <= SLIDING_WHOLE else max(0, first - window + 1)
        q = mx.arange(first, last)[:, None]
        k = mx.arange(key_start, last)[None, :]
        return (k <= q) & (k > q - window), key_start

    if length <= SLIDING_WHOLE:
        mask, _ = mask_of(0, length)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
    outputs = []
    block = SLIDING_BLOCK
    for first in range(0, length, block):
        last = min(length, first + block)
        mask, key_start = mask_of(first, last)
        output = mx.fast.scaled_dot_product_attention(
            queries[:, :, first:last],
            keys[:, :, key_start:last],
            values[:, :, key_start:last],
            scale=scale,
            mask=mask,
        )
        if last < length:
            mx.eval(output)
        outputs.append(output)
    return mx.concatenate(outputs, axis=2)


class RotaryEmbedding(nn.Module):
    """Rotate matching coordinates in both head halves, leaving NoPE pairs intact."""

    def __init__(self, dim, parameters):
        super().__init__()
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("LFM3 supports default rotary embeddings only")
        self.dim = dim
        self.base = parameters.get("rope_theta", 1e6)
        self.angles = int(dim * parameters.get("partial_rotary_factor", 1.0)) // 2
        if not 0 < self.angles <= dim // 2:
            raise ValueError("Invalid partial_rotary_factor")

    def __call__(self, x, offset=0, position_ids=None):
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offset = offset[:, None]
        positions = (
            mx.arange(x.shape[-2]) + offset if position_ids is None else position_ids
        )
        inv_freq = self.base ** (
            -mx.arange(self.angles, dtype=mx.float32) * 2 / self.dim
        )
        inv_freq = mx.pad(inv_freq, [(0, self.dim // 2 - self.angles)])
        angles = positions[..., None].astype(mx.float32) * inv_freq
        angles = mx.concatenate([angles, angles], axis=-1)
        if angles.ndim == 3:
            angles = angles[:, None]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate([-b, a], axis=-1) * sin


class Attention(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.window = config.sliding_window
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.gate_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_parameters[layer_type])

    def __call__(self, x, masks: SetMasks | None = None, position_ids=None):
        batch, length, _ = x.shape
        q = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q = self.rope(q.transpose(0, 2, 1, 3), 0, position_ids)
        k = self.rope(k.transpose(0, 2, 1, 3), 0, position_ids)
        if self.layer_type == "sliding_attention":
            out = sliding_attention(q, k, v, self.scale, self.window)
        elif masks is not None:
            out = set_attention(q, k, v, self.scale, masks)
        else:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal" if length > 1 else None
            )
        out = out.transpose(0, 2, 1, 3)
        out = out * mx.sigmoid(self.gate_proj(x))[..., None]
        return self.o_proj(out.reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_type = config.layer_types[index]
        self.is_attention_layer = self.layer_type != "conv"
        self.use_sandwich_norm = config.use_sandwich_norm
        if self.is_attention_layer:
            self.self_attn = Attention(config, self.layer_type)
        else:
            self.conv = ShortConv(config, index)
        self.feed_forward = MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = nn.RMSNorm(
                config.hidden_size, eps=config.norm_eps
            )
            self.post_ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, masks=None, position_ids=None):
        h = self.operator_norm(x)
        # Convolutions are causal; right padding after a row's last real token
        # changes none of its real tokens.
        h = self.self_attn(h, masks, position_ids) if self.is_attention_layer else self.conv(h)
        x = x + (self.post_operator_norm(h) if self.use_sandwich_norm else h)
        h = self.feed_forward(self.ffn_norm(x))
        return x + (self.post_ffn_norm(h) if self.use_sandwich_norm else h)


class Lfm3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self, inputs=None, input_embeddings=None, layout: SetLayout | None = None,
        position_ids=None,
    ):
        """Final hidden states for token ids ``[rows, tokens]`` (or their
        embeddings): causal, or under ``layout``'s set masks."""
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        masks = None if layout is None else SetMasks.of(layout)
        for layer in self.layers:
            h = layer(h, masks, position_ids)
        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("LFM3 decision models tie their output embeddings")
        self.config = config
        self.model_type = config.model_type
        self.model = Lfm3Model(config)

    def __call__(self, inputs=None, input_embeddings=None, layout=None, position_ids=None):
        return self.model(inputs, input_embeddings, layout, position_ids)

    def output_embedding_rows(self, ids: mx.array) -> mx.array:
        """Rows of the output embeddings (the input embeddings, tied)."""
        return self.model.embed_tokens.weight[ids]

    def sanitize(self, weights):
        """Release names (``model.language_model.*``, PyTorch's conv kernels) to these."""
        result = {}
        for name, value in weights.items():
            if not name.startswith("model.language_model."):
                continue
            name = "model." + name[len("model.language_model.") :]
            if name.endswith("conv.conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 1)
            result[name] = value
        return result

    @property
    def layers(self):
        return self.model.layers
