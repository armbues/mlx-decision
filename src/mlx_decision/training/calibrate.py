"""The calibration fit: a temperature per question, and a bias per option.

Answers become ``softmax(log p / temperature + bias)``, the parameters
minimizing the negative log-likelihood of the labels (Adam, full batch, a
fixed number of steps from temperature 1 and no bias, so the same data give
the same result). The temperature is kept between 0.05 and 20: past that a
model worse than chance would be fitted towards uniform answers.
"""

import math

import mlx.core as mx

from ..fitted import Calibration

STEPS = 3000
LEARNING_RATE = 0.05
LOG_T_RANGE = (math.log(0.05), math.log(20.0))
# log(0) is replaced by log of this, so an option a model rules out can still
# be fitted.
FLOOR = 1e-12


def nll(log_t: mx.array, bias: mx.array, logp: mx.array, labels: mx.array) -> mx.array:
    """Mean negative log-likelihood of ``labels`` under the calibrated answers."""
    logits = logp / mx.exp(log_t) + bias
    gold = mx.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    return (mx.logsumexp(logits, axis=-1) - gold).mean()


def fit_calibration(
    probabilities: list[list[float]], labels: list[int], bias: bool = True
) -> Calibration:
    """The calibration of answers ``probabilities`` (rows of the options, in the
    question's order) for the gold option indexes ``labels``; with ``bias``, one
    bias per option as well."""
    logp = mx.log(mx.maximum(mx.array(probabilities, dtype=mx.float32), FLOOR))
    gold = mx.array(labels, dtype=mx.int32)
    mask = mx.array(1.0 if bias else 0.0)
    grad = mx.grad(lambda x: nll(x[0], x[1:] * mask, logp, gold))

    @mx.compile
    def step(x, m, v, t):
        # Adam (betas 0.9 / 0.999), then the temperature kept in its range.
        g = grad(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        update = LEARNING_RATE * (m / (1 - 0.9**t)) / (mx.sqrt(v / (1 - 0.999**t)) + 1e-8)
        x = x - update
        x = mx.concatenate([mx.clip(x[:1], *LOG_T_RANGE), x[1:]])
        return x, m, v

    x = mx.zeros(1 + logp.shape[1])
    m, v = mx.zeros_like(x), mx.zeros_like(x)
    for t in range(1, STEPS + 1):
        x, m, v = step(x, m, v, mx.array(float(t)))
        if t % 100 == 0:
            mx.eval(x, m, v)
    values = x.tolist()
    return Calibration(math.exp(values[0]), [float(b) for b in values[1:]] if bias else None)
