"""The loss of the trained kinds: cross-entropy with label smoothing plus a Brier term.

Clef's supervised recipe (without its reinforcement stage), over rows of
option logits where questions have different option counts.
"""

import mlx.core as mx

SMOOTHING = 0.1
BRIER = 1.0


def decision_loss(
    logits: mx.array,
    valid: mx.array,
    labels: mx.array,
    smoothing: float = SMOOTHING,
    brier: float = BRIER,
) -> mx.array:
    """Mean over rows of cross-entropy with label smoothing plus ``brier`` times the
    Brier score.

    ``logits`` (rows, options), already divided by any temperature; ``valid``
    (rows, options) marks the real options of each row; ``labels`` (rows,) the
    gold option's index. Smoothing spreads ``smoothing`` evenly over a row's
    real options.
    """
    logits = mx.where(valid, logits.astype(mx.float32), -1e9)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    count = valid.sum(-1)
    gold = mx.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]
    uniform = mx.where(valid, logp, 0.0).sum(-1) / count
    cross_entropy = -((1 - smoothing) * gold + smoothing * uniform)
    onehot = mx.arange(logits.shape[-1])[None] == labels[:, None]
    squared = mx.where(valid, (mx.exp(logp) - onehot) ** 2, 0.0).sum(-1)
    return (cross_entropy + brier * squared).mean()
