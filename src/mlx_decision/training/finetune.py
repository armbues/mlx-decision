"""Fine-tuning Laya and Julia: the encoder and the marker head (or the head alone).

The trained modules compute in bfloat16, while the optimizer keeps float32
master weights (copies of them) and float32 moments, so
updates far smaller than bfloat16's resolution still add up; it updates one
tensor at a time, which keeps the memory at about 20 bytes per trained
parameter. AdamW
(weight decay 0.01), 5% linear warm-up then linear decay, gradients clipped
to norm 1; batches of examples in a seeded random order, up to 4,096 padded
tokens and 32 rows; each encoder layer is checkpointed. After every epoch
the dev part's loss is measured; the epoch with the lowest is kept. The
loss is the decision loss over the served logits (divided by the release's
temperatures, as serving does).
"""

import random
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .checkpoint import checkpointed
from .events import Cancelled
from .loss import decision_loss

LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP = 0.05
BATCH_TOKENS = 4096
BATCH_ROWS = 32
CLIP_NORM = 1.0
TRAIN_DTYPE = mx.bfloat16


@dataclass(frozen=True)
class Item:
    """A labelled question: the family's training item and the gold option's index."""

    item: Any
    label: int


@dataclass
class Trained:
    """The kept epoch's tensors (bfloat16, named ``encoder.*`` / ``head.*``) and
    every epoch's training loss, dev loss and seconds."""

    tensors: dict[str, mx.array]
    epochs: list[dict] = field(default_factory=list)
    best: int = 0


def batches(items: list[Item], rng: random.Random | None = None) -> list[list[Item]]:
    """Items in batches of at most ``BATCH_TOKENS`` padded tokens and ``BATCH_ROWS``
    rows, in a shuffled order (``rng``) or as given."""
    order = list(range(len(items)))
    if rng is not None:
        rng.shuffle(order)
    result, batch, longest = [], [], 0
    for index in order:
        length = items[index].item.length
        if batch and (
            max(longest, length) * (len(batch) + 1) > BATCH_TOKENS or len(batch) >= BATCH_ROWS
        ):
            result.append(batch)
            batch, longest = [], 0
        batch.append(items[index])
        longest = max(longest, length)
    if batch:
        result.append(batch)
    return result


def _loss(backend, batch: list[Item]) -> mx.array:
    logits, valid = backend.scores([entry.item for entry in batch])
    return decision_loss(logits, valid, mx.array([entry.label for entry in batch]))


def dev_loss(backend, items: list[Item]) -> float:
    """Mean loss over ``items`` (by row)."""
    total, count = 0.0, 0
    for batch in batches(items):
        loss = _loss(backend, batch)
        mx.eval(loss)
        total += loss.item() * len(batch)
        count += len(batch)
    return total / count if count else float("nan")


class MixedAdamW:
    """AdamW (as ``mlx.optimizers.AdamW``: no bias correction) over float32 copies
    of the trained weights and float32 moments, one tensor at a time: the
    gradients are cast, clipped by their global norm and applied tensor by
    tensor, each evaluated before the next, so no second copy of all weights,
    moments or gradients is held at once."""

    def __init__(self, weights: dict[str, mx.array], schedule, weight_decay: float):
        self.master, self.first, self.second = {}, {}, {}
        for key, value in weights.items():
            self.master[key] = value.astype(mx.float32)
            self.first[key] = mx.zeros(value.shape, mx.float32)
            self.second[key] = mx.zeros(value.shape, mx.float32)
            mx.eval(self.master[key], self.first[key], self.second[key])
        self.schedule = schedule
        self.weight_decay = weight_decay
        self.step = 0

    def update(self, grads: dict[str, mx.array]) -> dict[str, mx.array]:
        """The new weights in the training dtype, by name, for ``grads`` by name."""
        squared = mx.array(0.0)
        for grad in grads.values():
            squared = _add_squares(squared, grad)
            mx.eval(squared)
        scale = mx.minimum(CLIP_NORM / (mx.sqrt(squared) + 1e-6), 1.0)
        rate = mx.array(self.schedule(self.step), mx.float32)
        decay = mx.array(self.weight_decay, mx.float32)
        self.step += 1
        weights = {}
        for key, grad in grads.items():
            master, first, second, weight = _adamw(
                grad, self.master[key], self.first[key], self.second[key], scale, rate, decay
            )
            mx.eval(master, first, second, weight)
            self.master[key], self.first[key], self.second[key] = master, first, second
            weights[key] = weight
        return weights


# Compiled, so a large tensor (an embedding table) makes no float32 temporaries.
@partial(mx.compile, shapeless=True)
def _add_squares(total: mx.array, grad: mx.array) -> mx.array:
    return total + (grad.astype(mx.float32) ** 2).sum()


@partial(mx.compile, shapeless=True)
def _adamw(grad, master, first, second, scale, rate, decay):
    grad = grad.astype(mx.float32) * scale
    first = 0.9 * first + 0.1 * grad
    second = 0.999 * second + 0.001 * grad * grad
    master = master * (1 - rate * decay) - rate * first / (mx.sqrt(second) + 1e-8)
    return master, first, second, master.astype(TRAIN_DTYPE)


def train(
    backend,
    fit_items: list[Item],
    dev_items: list[Item],
    *,
    epochs: int = 3,
    head_only: bool = False,
    seed: int = 0,
    on_step: Callable[[int, int, float], None] | None = None,
    on_epoch: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Trained:
    """Train ``backend``'s modules in place on ``fit_items``; afterwards they hold the
    kept epoch's weights, in their dtype from before. ``on_step(step, total, loss)``
    after each batch, ``on_epoch(row)`` after each epoch; ``Cancelled`` between
    batches once ``cancelled()`` is true (the modules then hold the last step's
    weights, in their dtype from before: a model to discard)."""
    if epochs < 1:
        raise ValueError(f"epochs must be 1 or more, not {epochs}")
    if not fit_items:
        raise ValueError("no training examples to fine-tune on")
    parts = backend.trainable()
    # Only the trained modules change dtype: a round trip through bfloat16 would
    # change a frozen encoder's float16 weights.
    trained_part = parts.head if head_only else parts
    serving_dtype = tree_flatten(trained_part.parameters())[0][1].dtype
    rng = random.Random(f"{seed}:finetune")
    plans = [batches(fit_items, rng) for _ in range(epochs)]
    total = sum(map(len, plans))
    rate = HEAD_LEARNING_RATE if head_only else LEARNING_RATE
    warmup = max(1, round(total * WARMUP))
    rise = optim.linear_schedule(0.0, rate, warmup)
    decay = optim.linear_schedule(rate, 0.0, max(1, total - warmup))
    schedule = optim.join_schedules([rise, decay], [warmup])

    trained_part.set_dtype(TRAIN_DTYPE)
    if head_only:
        parts.encoder.freeze()
    try:
        optimizer = MixedAdamW(
            dict(tree_flatten(parts.trainable_parameters())), schedule, WEIGHT_DECAY
        )
        loss_and_grad = nn.value_and_grad(parts, lambda batch: _loss(backend, batch))
        trained = Trained({})
        best_loss = float("inf")
        step = 0
        layer_class = type(parts.encoder.layers[0])
        for epoch, plan in enumerate(plans, 1):
            started = time.perf_counter()
            losses = []
            with nullcontext() if head_only else checkpointed(layer_class):
                for batch in plan:
                    if cancelled is not None and cancelled():
                        raise Cancelled()
                    loss, grads = loss_and_grad(batch)
                    mx.eval(loss, grads)
                    weights = optimizer.update(dict(tree_flatten(grads)))
                    del grads
                    parts.update(tree_unflatten(list(weights.items())))
                    losses.append(loss.item())
                    step += 1
                    if on_step is not None:
                        on_step(step, total, losses[-1])
            seconds = time.perf_counter() - started
            loss_on_dev = dev_loss(backend, dev_items)
            row = {
                "epoch": epoch,
                "loss": round(sum(losses) / len(losses), 4),
                "dev_loss": round(loss_on_dev, 4),
                "seconds": round(seconds, 1),
            }
            trained.epochs.append(row)
            if loss_on_dev < best_loss or not trained.tensors:
                best_loss, trained.best = loss_on_dev, epoch
                # References to this epoch's arrays: later updates make new ones.
                trained.tensors = dict(tree_flatten(parts.trainable_parameters()))
            if on_epoch is not None:
                on_epoch(row)
        del optimizer
        parts.update(tree_unflatten(list(trained.tensors.items())))
    finally:
        if head_only:
            parts.encoder.unfreeze()
        trained_part.set_dtype(serving_dtype)
        mx.eval(parts.parameters())
    return trained
