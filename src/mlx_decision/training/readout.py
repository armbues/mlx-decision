"""The readout fit: the option rows of pplx, H2O and d1, trained on last hidden states.

These families read a question's answer from the hidden state at its last
token through one row per scored token (``models/readout.py``). The fit
trains an untied copy of a question's rows, plus one bias per option, on
the hidden states the features pass computed once: cross-entropy with
label smoothing and an L2 pull towards the release's rows, Adam on the full
batch. The learning rate, the L2 weight and the number of steps are those
with the lowest negative log-likelihood on dev, searched over a small grid
and every few steps; the rows at that step are kept (dev is not trained
on, so it can fit the closing calibration). Nothing is random, so the same
data give the same rows.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.optimizers as optim

from ..models.readout import Readout, group_logits
from .events import Cancelled
from .loss import decision_loss

LEARNING_RATES = (1e-5, 1e-4, 1e-3)
L2_WEIGHTS = (0.0, 0.01)
STEPS = 400
# Dev is measured every this many steps.
EVERY = 20
SMOOTHING = 0.1


def search_steps() -> int:
    """Training steps of one question's search (every setting, every step)."""
    return len(LEARNING_RATES) * len(L2_WEIGHTS) * STEPS


@dataclass
class Picked:
    """The setting dev picked and its negative log-likelihood, with the release's
    rows (step 0) for comparison."""

    learning_rate: float | None
    l2: float | None
    steps: int
    dev_nll_release: float
    dev_nll: float

    def as_json(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "steps": self.steps,
            "dev_nll_release": round(self.dev_nll_release, 4),
            "dev_nll": round(self.dev_nll, 4),
        }


def nll(
    params: dict[str, mx.array],
    hidden: mx.array,
    labels: mx.array,
    groups: list[list[int]],
    temperature: float,
) -> mx.array:
    """Mean negative log-likelihood of ``labels`` under ``params``' rows and bias."""
    logits = group_logits(params["rows"], hidden, groups, temperature, params["bias"])
    gold = mx.take_along_axis(logits, labels[:, None], axis=-1)[:, 0]
    return (mx.logsumexp(logits, axis=-1) - gold).mean()


def fit_readout(
    option_ids: list[str],
    rows: mx.array,
    groups: list[list[int]],
    temperature: float,
    fit: tuple[mx.array, list[int]],
    dev: tuple[mx.array, list[int]],
    on_step: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[Readout | None, Picked]:
    """A question's readout fitted on ``fit`` (hidden states ``[n, width]`` and gold
    option indexes), its setting picked on ``dev``; None when the release's rows
    are best on dev. ``on_step(steps)`` reports the search's progress every few
    steps; ``Cancelled`` once ``cancelled()`` is true."""
    rows0 = rows.astype(mx.float32)
    hidden, labels = fit[0], mx.array(fit[1], dtype=mx.int32)
    dev_hidden, dev_labels = dev[0], mx.array(dev[1], dtype=mx.int32)
    valid = mx.ones((len(fit[1]), len(groups)), dtype=mx.bool_)
    start = {"rows": rows0, "bias": mx.zeros(len(groups))}
    release = nll(start, dev_hidden, dev_labels, groups, temperature).item()
    best = (release, None, None, 0, start)
    done = 0
    for learning_rate in LEARNING_RATES:
        for l2 in L2_WEIGHTS:

            def loss(params, l2=l2):
                logits = group_logits(params["rows"], hidden, groups, temperature, params["bias"])
                pull = ((params["rows"] - rows0) ** 2).sum()
                return decision_loss(logits, valid, labels, SMOOTHING, brier=0.0) + l2 * pull

            gradient = mx.grad(loss)
            optimizer = optim.Adam(learning_rate=learning_rate)
            params = dict(start)
            for step in range(1, STEPS + 1):
                # ``update`` replaces the dict's entries, so a kept dict is a snapshot.
                optimizer.update(params, gradient(params))
                mx.eval(params, optimizer.state)
                if step % EVERY:
                    continue
                score = nll(params, dev_hidden, dev_labels, groups, temperature).item()
                if score < best[0]:
                    best = (score, learning_rate, l2, step, dict(params))
                done += EVERY
                if on_step is not None:
                    on_step(done)
                if cancelled is not None and cancelled():
                    raise Cancelled()
    score, learning_rate, l2, steps, params = best
    picked = Picked(learning_rate, l2, steps, release, score)
    if steps == 0:
        return None, picked
    return Readout(list(option_ids), params["rows"], params["bias"]), picked
