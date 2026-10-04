# Ported from joint_schema_model.py in Cloudflare's Clef release
# (https://huggingface.co/Cloudflare/clef-flash), Apache License 2.0.
# See LICENSES/clef-Apache-2.0.txt.
#
# Modified for mlx-decision: rewritten from PyTorch to MLX, one request at a
# time. Parameter names match joint_head.safetensors.
"""Clef's joint schema head: scores every option of every question together."""

import math

import mlx.core as mx
import mlx.nn as nn

from .encode import EncodedQuestion

MAX_LOGIT_SCALE = math.log(100.0)


class MultiheadAttention(nn.Module):
    """Attention with PyTorch's packed ``in_proj`` parameter layout; inputs are ``[N, E]``."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)

    def __call__(self, queries: mx.array, keys: mx.array, values: mx.array) -> mx.array:
        width = queries.shape[-1]
        weight, bias = self.in_proj_weight, self.in_proj_bias

        def project(x: mx.array, part: int) -> mx.array:
            rows = slice(part * width, (part + 1) * width)
            x = x @ weight[rows].T + bias[rows]
            return x.reshape(x.shape[0], self.heads, width // self.heads).transpose(1, 0, 2)

        q, k, v = project(queries, 0), project(keys, 1), project(values, 2)
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(width // self.heads)
        out = mx.softmax(scores, axis=-1) @ v
        return self.out_proj(out.transpose(1, 0, 2).reshape(-1, width))


def _feedforward(width: int, hidden: int, out: int) -> list:
    # Positions 1 and 2 hold the activation and dropout in the reference, so the
    # two linear layers are numbered 0 and 3 in the weights file.
    return [nn.Linear(width, hidden), nn.GELU(), nn.Identity(), nn.Linear(hidden, out)]


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = MultiheadAttention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = _feedforward(width, feedforward, width)

    def __call__(self, queries: mx.array, memory: mx.array) -> mx.array:
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory)
        x = self.feedforward_norm(queries)
        for layer in self.feedforward:
            x = layer(x)
        return queries + x


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (self-attention, cross-attention, MLP)."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = MultiheadAttention(width, heads)
        self.multihead_attn = MultiheadAttention(width, heads)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def __call__(self, fields: mx.array, memory: mx.array) -> mx.array:
        x = self.norm1(fields)
        fields = fields + self.self_attn(x, x, x)
        fields = fields + self.multihead_attn(self.norm2(fields), memory, memory)
        return fields + self.linear2(nn.gelu(self.linear1(self.norm3(fields))))


def _normalize(x: mx.array, eps: float) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class JointSchemaHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(3, width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [DecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = _feedforward(width * 4, width, 1)
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        questions: tuple[EncodedQuestion, ...],
        output_embeddings: mx.array,
    ) -> list[mx.array]:
        """Logits per question, one per option, for one request.

        ``hidden_states`` is ``[L, hidden_size]``, ``input_ids`` is ``[L]``.
        """
        hidden = self.hidden_norm(hidden_states)
        memory = self.memory_projection(hidden)
        global_vector = hidden[-1]
        question_vectors = mx.stack(
            [hidden[q.question_span[0] : q.question_span[1]].mean(axis=0) for q in questions]
        )

        # Each option starts from its own tokens (in context and as bare
        # embeddings) plus its question, then gathers evidence from the input.
        lexical_options: list[mx.array] = []
        option_queries: list[mx.array] = []
        for index, question in enumerate(questions):
            context = mx.stack([hidden[s:e].mean(axis=0) for s, e in question.option_spans])
            lexical = mx.stack(
                [output_embeddings[input_ids[s:e]].mean(axis=0) for s, e in question.option_spans]
            )
            lexical_options.append(lexical)
            option_queries.append(
                self.option_context_projection(context)
                + self.option_lexical_projection(lexical)
                + self.option_question_projection(question_vectors[index])[None]
            )
        routed = mx.concatenate(option_queries, axis=0)
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        split_options: list[mx.array] = []
        start = 0
        for question in questions:
            split_options.append(routed[start : start + len(question.option_spans)])
            start += len(question.option_spans)

        # One vector per question, refined jointly across questions.
        base_fields = self.question_projection(question_vectors)
        summaries = []
        for field, options in zip(base_fields, split_options, strict=True):
            weights = mx.softmax((options @ field) / math.sqrt(options.shape[-1]), axis=0)
            summaries.append((weights[:, None] * options).sum(axis=0))
        type_ids = mx.array([q.question_type for q in questions])
        fields = (
            base_fields
            + self.option_summary_norm(mx.stack(summaries))
            + self.global_projection(global_vector)[None]
            + self.type_embedding(type_ids)
        )
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields)

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, MAX_LOGIT_SCALE))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, MAX_LOGIT_SCALE))
        gate = mx.sigmoid(self.residual_gate)
        logits: list[mx.array] = []
        for index, (field, lexical, options) in enumerate(
            zip(fields, lexical_options, split_options, strict=True)
        ):
            anchor = _normalize(question_vectors[index] + global_vector, 1e-12)
            prior = prior_scale * (_normalize(lexical, 1e-12) @ anchor)
            options = self.option_norm(options)
            repeated = mx.broadcast_to(field[None], options.shape)
            cosine = (_normalize(repeated, 1e-8) * _normalize(options, 1e-8)).sum(axis=-1)
            features = mx.concatenate(
                [repeated, options, repeated * options, mx.abs(repeated - options)], axis=-1
            )
            for layer in self.residual_scorer:
                features = layer(features)
            joint = joint_scale * cosine + features.squeeze(-1)
            logits.append(prior + gate * joint)
        return logits
