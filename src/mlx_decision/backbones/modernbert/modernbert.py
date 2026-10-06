# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc.
# team. All rights reserved.
#
# Ported from transformers (https://github.com/huggingface/transformers) 5.18.0,
# files models/modernbert/modeling_modernbert.py (ModernBertModel: embeddings,
# attention, MLP, encoder layers, rotary embedding) and
# configuration_modernbert.py, under the Apache License 2.0. See
# LICENSES/transformers-Apache-2.0.txt.
#
# Modified for mlx-decision: written for MLX; inference only (no dropout, no
# gradient checkpointing, no masked-language-model head); batches are padded
# and masked instead of unpadded; the sliding-window layers attend block by
# block instead of through a full band mask.
"""ModernBERT, the bidirectional encoder under the Laya and Julia families.

Covers ModernBERT and mmBERT checkpoints: every n-th layer attends globally,
the others within a sliding window of ``local_attention // 2`` tokens on each
side; each kind has its own rotary base.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    norm_eps: float = 1e-5
    local_attention: int = 128
    global_layers: tuple[bool, ...] = ()
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    pad_token_id: int = 0

    @classmethod
    def from_config(cls, config: dict) -> "ModelArgs":
        """Read a ModernBERT ``config.json``, in the old or the new transformers layout.

        Older releases name the layer pattern ``global_attn_every_n_layers`` and the
        rotary bases ``global_rope_theta`` / ``local_rope_theta``; newer ones write
        ``layer_types`` and ``rope_parameters``.
        """
        for key in ("attention_bias", "mlp_bias", "norm_bias"):
            if config.get(key, False):
                raise ValueError(f"ModernBERT with {key} is not supported")
        activation = config.get("hidden_activation", "gelu")
        if activation != "gelu":
            raise ValueError(f"ModernBERT with hidden_activation {activation!r} is not supported")
        layers = config["num_hidden_layers"]
        if config.get("layer_types"):
            global_layers = tuple(t == "full_attention" for t in config["layer_types"])
        else:
            every = config.get("global_attn_every_n_layers", 3)
            global_layers = tuple(i % every == 0 for i in range(layers))
        if len(global_layers) != layers:
            raise ValueError("ModernBERT config: layer_types does not match num_hidden_layers")
        rope = config.get("rope_parameters") or {}
        global_theta = config.get("global_rope_theta")
        local_theta = config.get("local_rope_theta")
        if global_theta is None:
            global_theta = rope.get("full_attention", {}).get("rope_theta", 160000.0)
        if local_theta is None:
            local_theta = rope.get("sliding_attention", {}).get("rope_theta", 10000.0)
        for kind in ("full_attention", "sliding_attention"):
            rope_type = rope.get(kind, {}).get("rope_type", "default")
            if rope_type != "default":
                raise ValueError(f"ModernBERT with rope_type {rope_type!r} is not supported")
        return cls(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_hidden_layers=layers,
            num_attention_heads=config["num_attention_heads"],
            norm_eps=config.get("norm_eps", 1e-5),
            local_attention=config.get("local_attention", 128),
            global_layers=global_layers,
            global_rope_theta=float(global_theta),
            local_rope_theta=float(local_theta),
            max_position_embeddings=config.get("max_position_embeddings", 8192),
            pad_token_id=config.get("pad_token_id", 0),
        )

    @property
    def window(self) -> int:
        """Tokens seen on each side by a sliding-window layer (inclusive)."""
        return self.local_attention // 2


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_global: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.is_global = is_global
        self.rope_theta = args.global_rope_theta if is_global else args.local_rope_theta
        self.Wqkv = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=False)
        self.Wo = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array, masks: "Masks") -> mx.array:
        B, L, _ = x.shape
        qkv = self.Wqkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = mx.fast.rope(
            q, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )
        k = mx.fast.rope(
            k, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )
        if self.is_global or masks.window_is_global:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=masks.full)
        else:
            o = _window_attention(q, k, v, self.scale, masks)
        return self.Wo(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class Masks:
    """Attention masks for one forward pass, built once and shared by all layers.

    Masks are boolean (True: may attend). A padded query keeps its whole window,
    padding included, so no row is fully masked: its output is unused but finite,
    and real tokens never attend to it.
    """

    def __init__(self, attention_mask: mx.array | None, length: int, window: int):
        self.window = window
        # Within one window of length every token sees every other one.
        self.window_is_global = length <= window + 1
        keep = None if attention_mask is None else attention_mask.astype(mx.bool_)
        self.full = None if keep is None else keep[:, None, None, :]
        if self.window_is_global:
            return
        w = window
        blocks = -(-length // w)
        self.blocks = blocks
        # Block b holds the queries [b*w, b*w + w) and the keys [b*w - w, b*w + 2w).
        qpos = (mx.arange(blocks)[:, None] * w + mx.arange(w)[None, :])[:, :, None]
        kpos = (mx.arange(blocks)[:, None] * w - w + mx.arange(3 * w)[None, :])[:, None, :]
        allowed = (mx.abs(qpos - kpos) <= w) & (kpos >= 0)  # (blocks, w, 3w)
        if keep is None:
            # Queries past the end (block padding) may see the zero keys there.
            self.windowed = (allowed & ((kpos < length) | (qpos >= length)))[:, None]
            return
        batch = keep.shape[0]
        padded = mx.pad(keep, [(0, 0), (w, blocks * w - length + w)])  # index: position + w
        key_kept = padded[:, (kpos + w).reshape(-1)].reshape(batch, blocks, 1, 3 * w)
        query_kept = padded[:, (qpos + w).reshape(-1)].reshape(batch, blocks, w, 1)
        windowed = allowed[None] & (key_kept | ~query_kept)
        self.windowed = windowed.reshape(batch * blocks, 1, w, 3 * w)


def _window_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float, masks: Masks
) -> mx.array:
    """Sliding-window attention by blocks of ``w`` queries against the ``3w`` keys around them."""
    w, blocks = masks.window, masks.blocks
    B, H, L, D = q.shape
    pad = blocks * w - L
    if pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
    k = mx.pad(k, [(0, 0), (0, 0), (w, pad + w), (0, 0)]).reshape(B, H, blocks + 2, w, D)
    v = mx.pad(v, [(0, 0), (0, 0), (w, pad + w), (0, 0)]).reshape(B, H, blocks + 2, w, D)
    k = mx.concatenate([k[:, :, :-2], k[:, :, 1:-1], k[:, :, 2:]], axis=3)
    v = mx.concatenate([v[:, :, :-2], v[:, :, 1:-1], v[:, :, 2:]], axis=3)

    def by_block(x, n):
        return x.transpose(0, 2, 1, 3, 4).reshape(B * blocks, H, n, D)

    q = by_block(q.reshape(B, H, blocks, w, D), w)
    o = mx.fast.scaled_dot_product_attention(
        q, by_block(k, 3 * w), by_block(v, 3 * w), scale=scale, mask=masks.windowed
    )
    o = o.reshape(B, blocks, H, w, D).transpose(0, 2, 1, 3, 4).reshape(B, H, blocks * w, D)
    return o[:, :, :L]


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.Wi = nn.Linear(args.hidden_size, 2 * args.intermediate_size, bias=False)
        self.Wo = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(x) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        # The first layer has no attention norm: the embeddings are normed already.
        if index > 0:
            self.attn_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps, bias=False)
        self.attn = Attention(args, args.global_layers[index])
        self.mlp_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps, bias=False)
        self.mlp = MLP(args)

    def __call__(self, x: mx.array, masks: Masks) -> mx.array:
        h = self.attn_norm(x) if "attn_norm" in self else x
        x = x + self.attn(h, masks)
        return x + self.mlp(self.mlp_norm(x))


class Embeddings(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class Model(nn.Module):
    """The encoder: token ids in, final hidden states (after the final norm) out."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embeddings = Embeddings(args)
        self.layers = [EncoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(args.hidden_size, eps=args.norm_eps, bias=False)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """``input_ids`` (B, L); ``attention_mask`` (B, L) of 1 for tokens, 0 for padding."""
        masks = Masks(attention_mask, input_ids.shape[1], self.args.window)
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, masks)
        return self.final_norm(x)

    @staticmethod
    def sanitize(weights: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
        """The encoder's weights from a checkpoint, with ``prefix`` (e.g. ``"encoder."``) removed.

        Other weights (a masked-language-model head, a decision head) are left out.
        """
        return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
