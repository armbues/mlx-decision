# Copyright Convai Innovations (the laya authors).
#
# Ported from laya (https://pypi.org/project/laya/) 0.3.28, file laya/common.py
# (DecisionModel: type embedding, head transformer layers, option scorer), under
# the Apache License 2.0. See LICENSES/laya-Apache-2.0.txt. Supersonic Labs'
# Julia 1 ships the same head (julia/model.py, JuliaDecisionModel).
#
# Modified for mlx-decision: written for MLX; inference only (no dropout, no
# act head); the last head layer computes only the option markers, which is
# exact since nothing else is read.
"""The marker head of Laya and Julia: one score per option marker.

Each option starts with a ``[MASK]`` marker token. The encoder's hidden
states get a question-type embedding, pass two pre-norm transformer layers
(PyTorch's ``nn.TransformerEncoderLayer``: ReLU, biases, LayerNorm with
bias), and a small MLP turns the hidden state at each marker into the
option's score. A softmax over a question's scores gives its probabilities.
"""

import re

import mlx.core as mx
import mlx.nn as nn

QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}


class HeadAttention(nn.Module):
    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj = nn.Linear(dims, 3 * dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, queries: mx.array, x: mx.array, mask: mx.array | None) -> mx.array:
        """``queries`` (B, Q, D) attend to ``x`` (B, L, D); ``mask`` (B, 1, 1, L) bool."""
        B, Q, D = queries.shape
        L = x.shape[1]
        head_dim = D // self.heads
        wq, wk, wv = mx.split(self.in_proj.weight, 3, axis=0)
        bq, bk, bv = mx.split(self.in_proj.bias, 3, axis=0)

        def split_heads(t, n):
            return t.reshape(B, n, self.heads, head_dim).transpose(0, 2, 1, 3)

        q = split_heads(queries @ wq.T + bq, Q)
        k = split_heads(x @ wk.T + bk, L)
        v = split_heads(x @ wv.T + bv, L)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5, mask=mask)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, Q, D))


class HeadLayer(nn.Module):
    """A pre-norm transformer layer, as ``nn.TransformerEncoderLayer(norm_first=True)``."""

    def __init__(self, dims: int, heads: int, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims, eps=eps)
        self.self_attn = HeadAttention(dims, heads)
        self.norm2 = nn.LayerNorm(dims, eps=eps)
        self.linear1 = nn.Linear(dims, 4 * dims)
        self.linear2 = nn.Linear(4 * dims, dims)

    def __call__(self, x: mx.array, mask: mx.array | None, rows: mx.array | None = None):
        """``rows`` (B, Q) limits the output to those positions; all positions stay keys."""
        normed = self.norm1(x)
        if rows is None:
            residual, queries = x, normed
        else:
            residual, queries = _gather(x, rows), _gather(normed, rows)
        h = residual + self.self_attn(queries, normed, mask)
        return h + self.linear2(nn.relu(self.linear1(self.norm2(h))))


def _gather(x: mx.array, rows: mx.array) -> mx.array:
    return mx.take_along_axis(x, rows[:, :, None], axis=1)


class MarkerHead(nn.Module):
    def __init__(self, dims: int, layers: int = 2):
        super().__init__()
        heads = max(1, dims // 64)
        self.type_emb = nn.Embedding(len(QUESTION_TYPES), dims)
        self.layers = [HeadLayer(dims, heads) for _ in range(layers)]
        self.scorer_norm = nn.LayerNorm(dims)
        self.scorer_in = nn.Linear(dims, dims)
        self.scorer_out = nn.Linear(dims, 1)

    def __call__(
        self,
        hidden: mx.array,
        attention_mask: mx.array | None,
        question_types: mx.array,
        markers: mx.array,
    ) -> mx.array:
        """Scores (B, K) in float32 for the marker positions ``markers`` (B, K).

        ``hidden`` (B, L, D) are the encoder's final hidden states,
        ``attention_mask`` (B, L) is 1 for tokens and 0 for padding, and
        ``question_types`` (B,) holds ``QUESTION_TYPES`` values. Rows with fewer
        markers repeat a valid one; the caller ignores those scores.
        """
        h = hidden + self.type_emb(question_types)[:, None, :]
        mask = None if attention_mask is None else attention_mask.astype(mx.bool_)[:, None, None]
        if not self.layers:
            h = _gather(h, markers)
        for index, layer in enumerate(self.layers):
            last = index == len(self.layers) - 1
            h = layer(h, mask, markers if last else None)
        scores = self.scorer_out(nn.gelu(self.scorer_in(self.scorer_norm(h))))
        return scores.squeeze(-1).astype(mx.float32)

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """The head's weights from a Laya or Julia checkpoint, renamed for this module.

        The encoder, the act head and the stored temperature buffer are left out.
        """
        renames = [
            (
                r"^head\.layers\.(\d+)\.self_attn\.in_proj_(weight|bias)$",
                r"layers.\1.self_attn.in_proj.\2",
            ),
            (r"^head\.layers\.", "layers."),
            (r"^scorer\.0\.", "scorer_norm."),
            (r"^scorer\.1\.", "scorer_in."),
            (r"^scorer\.3\.", "scorer_out."),
            (r"^type_emb\.", "type_emb."),
        ]
        out = {}
        for key, value in weights.items():
            for pattern, replacement in renames:
                if re.match(pattern, key):
                    out[re.sub(pattern, replacement, key)] = value
                    break
        return out
