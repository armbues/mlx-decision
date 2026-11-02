"""The loss of the trained kinds: cross-entropy with label smoothing plus a Brier term.

Clef's supervised recipe (without its reinforcement stage), over rows of
option logits where questions have different option counts. A row's gold
is an option index (a hard label) or a distribution over the options (a
soft one): soft rows train against the distribution itself, without label
smoothing. Rows that are all hard take the hard path alone, so data without
soft labels trains as it always has.
"""

from collections.abc import Sequence

import mlx.core as mx

SMOOTHING = 0.1
BRIER = 1.0

Gold = int | tuple[float, ...]  # an option index, or probabilities in the options' order
Soft = tuple[mx.array, mx.array]  # targets (rows, options), which rows are soft (rows,)


def gold_arrays(golds: Sequence[Gold], width: int) -> tuple[mx.array, Soft | None]:
    """The rows' gold option indexes (a soft row's most likely option, the first
    on a tie) and, when any row is soft, the targets: every row's distribution
    padded to ``width`` options (a hard row's one-hot) and which rows are soft."""
    labels = [g if isinstance(g, int) else max(range(len(g)), key=g.__getitem__) for g in golds]
    indexes = mx.array(labels, dtype=mx.int32)
    if all(isinstance(g, int) for g in golds):
        return indexes, None
    rows = [
        [1.0 if i == g else 0.0 for i in range(width)]
        if isinstance(g, int)
        else [*g, *([0.0] * (width - len(g)))]
        for g in golds
    ]
    soft = mx.array([not isinstance(g, int) for g in golds])
    return indexes, (mx.array(rows, dtype=mx.float32), soft)


def nll_rows(logits: mx.array, labels: mx.array, soft: Soft | None = None) -> mx.array:
    """Each row's negative log-likelihood of its gold: of the gold option, or a
    soft row's cross-entropy against its targets. ``logits`` padded with a large
    negative number where a row has no option."""
    lse = mx.logsumexp(logits, axis=-1)
    hard = lse - mx.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    if soft is None:
        return hard
    targets, rows = soft
    return mx.where(rows, lse - (targets * logits).sum(-1), hard)


def decision_loss(
    logits: mx.array,
    valid: mx.array,
    labels: mx.array,
    smoothing: float = SMOOTHING,
    brier: float = BRIER,
    soft: Soft | None = None,
) -> mx.array:
    """Mean over rows of cross-entropy with label smoothing plus ``brier`` times the
    Brier score.

    ``logits`` (rows, options), already divided by any temperature; ``valid``
    (rows, options) marks the real options of each row; ``labels`` (rows,) the
    gold option's index. Smoothing spreads ``smoothing`` evenly over a row's
    real options. ``soft`` (``gold_arrays``): the soft rows' cross-entropy and
    Brier score are against their targets, without smoothing.
    """
    logits = mx.where(valid, logits.astype(mx.float32), -1e9)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    count = valid.sum(-1)
    gold = mx.take_along_axis(logp, labels[:, None], axis=-1)[:, 0]
    uniform = mx.where(valid, logp, 0.0).sum(-1) / count
    cross_entropy = -((1 - smoothing) * gold + smoothing * uniform)
    onehot = mx.arange(logits.shape[-1])[None] == labels[:, None]
    if soft is None:
        squared = mx.where(valid, (mx.exp(logp) - onehot) ** 2, 0.0).sum(-1)
        return (cross_entropy + brier * squared).mean()
    targets, rows = soft
    targets = targets[:, : logits.shape[-1]]
    against = mx.where(rows[:, None], targets, onehot.astype(mx.float32))
    squared = mx.where(valid, (mx.exp(logp) - against) ** 2, 0.0).sum(-1)
    soft_entropy = -mx.where(valid, targets * logp, 0.0).sum(-1)
    cross_entropy = mx.where(rows, soft_entropy, cross_entropy)
    return (cross_entropy + brier * squared).mean()
