"""The LoRA fit: low-rank adapters on the network of a family that reads out (d3, d1 LFM3).

The adapters (``adapters.LoRALinear``) go on the MLP projections of every
decoder layer; with them train a copy of each fitted question's readout
rows and one bias per option, starting from the release's rows (or a fitted
base's readout). The loss runs through the family's own passes as serving
runs them (``backend.Adaptable``): the prompt, its sets and the readout, so
a fit trains on what the model answers. The network's weights stay frozen,
images go through the vision tower once, before training.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..adapters import adapt, adapter_tensors
from ..backend import TrainingPass
from ..models.readout import group_logits
from .loss import Gold, decision_loss, gold_arrays

RANK = 8
# Alpha over rank is the update's scale: alpha 16 at rank 8 is scale 2, and a
# change of rank keeps the learning rate right.
ALPHA = 16


def scale_for(rank: int) -> float:
    return ALPHA / rank


@dataclass(frozen=True)
class Item:
    """A training pass and the gold of each of its fitted questions: the index of
    the question's readout in the trainee and its gold (an option index, or
    a soft label's probabilities)."""

    passed: TrainingPass
    golds: dict[str, tuple[int, Gold]]

    @property
    def length(self) -> int:
        return self.passed.length

    @property
    def images(self) -> bool:
        return self.passed.images


class Trainee(nn.Module):
    """What a LoRA fit trains: the family's network with adapters, and per fitted
    question its readout rows and option biases (float32)."""

    def __init__(self, network: nn.Module, rows: list[mx.array], biases: list[mx.array]):
        super().__init__()
        self.network = network
        self.rows = [r.astype(mx.float32) for r in rows]
        self.bias = [b.astype(mx.float32) for b in biases]


def add_adapters(backend, rank: int = RANK, scale: float | None = None, seed: int = 0) -> int:
    """Freeze ``backend``'s network and wrap its targets in adapters; the number
    of adapter parameters."""
    backend.network().freeze()
    return adapt(backend.adapter_targets(), rank, scale_for(rank) if scale is None else scale, seed)


def adapters(backend) -> dict[str, mx.array]:
    """The adapters' tensors by name (``layers.N.<mlp>.<proj>.lora_a`` / ``lora_b``)."""
    return adapter_tensors(backend.adapter_targets())


def logits(backend, trainee: Trainee, batch: list[Item]):
    """The fitted questions' logits of a batch as served (rows padded to the
    widest question), the mask of real options and the golds, row by row."""
    hidden = backend.pass_hidden([item.passed for item in batch])
    rows, golds = [], []
    for item, states in zip(batch, hidden, strict=True):
        for slot, (question_id, features) in enumerate(item.passed.questions.items()):
            if question_id not in item.golds:
                continue
            index, gold = item.golds[question_id]
            rows.append(
                group_logits(
                    trainee.rows[index],
                    states[slot],
                    features.groups,
                    features.temperature,
                    trainee.bias[index],
                )
            )
            golds.append(gold)
    width = max(row.shape[0] for row in rows)
    padded = mx.stack([mx.pad(row, (0, width - row.shape[0])) for row in rows])
    counts = mx.array([row.shape[0] for row in rows])
    valid = mx.arange(width)[None] < counts[:, None]
    return padded, valid, golds


def loss(backend, trainee: Trainee, batch: list[Item]) -> mx.array:
    """The decision loss of a batch's fitted questions."""
    padded, valid, golds = logits(backend, trainee, batch)
    labels, soft = gold_arrays(golds, padded.shape[-1])
    return decision_loss(padded, valid, labels, soft=soft)


# Training

LEARNING_RATE = 1e-4
EPOCHS = 2
WEIGHT_DECAY = 0.01
WARMUP = 0.05
# Text passes go in batches of at most this many padded tokens and rows.
BATCH_TOKENS = 4096
BATCH_ROWS = 16
# Passes with images run one at a time; a step sums the gradients of this many.
IMAGE_STEP = 8


@dataclass
class Trained:
    """The kept epoch's adapters (by name) and readouts (rows, bias per fitted
    question), every epoch's training loss, dev loss and seconds, whether the
    run stopped early, and the trainable parameters."""

    adapters: dict[str, mx.array]
    readouts: list[tuple[mx.array, mx.array]]
    epochs: list[dict]
    best: int
    stopped: bool
    parameters: int
    examples: int  # training passes per epoch
    seconds: float  # training, all epochs (without the dev passes)


def _text_batches(items: list[Item]) -> list[list[Item]]:
    result, batch, longest = [], [], 0
    for item in items:
        if batch and (
            max(longest, item.length) * (len(batch) + 1) > BATCH_TOKENS or len(batch) >= BATCH_ROWS
        ):
            result.append(batch)
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, item.length)
    if batch:
        result.append(batch)
    return result


def steps(items: list[Item], rng=None) -> list[list[list[Item]]]:
    """The optimizer steps of an epoch, each a list of batches whose gradients
    it sums: a batch of text passes (by tokens and rows), or ``IMAGE_STEP``
    passes with images one by one; in a shuffled order (``rng``) or as given."""
    order = list(items)
    if rng is not None:
        rng.shuffle(order)
    text = [item for item in order if not item.images]
    images = [item for item in order if item.images]
    result = [[batch] for batch in _text_batches(text)]
    result += [
        [[item] for item in images[start : start + IMAGE_STEP]]
        for start in range(0, len(images), IMAGE_STEP)
    ]
    if rng is not None:
        rng.shuffle(result)
    return result


def learning_rates(rate: float, total: int):
    """The learning rate at each of ``total`` steps: up to ``rate`` over the
    first ``WARMUP`` share (the first step already above zero, so a run of one
    step trains), then linear decay towards zero (the last step above it)."""
    warmup = max(1, round(total * WARMUP))
    rest = max(1, total - warmup)

    def at(step):
        step = mx.array(step, mx.float32)
        rising = rate * (step + 1) / warmup
        falling = rate * (total - step) / rest
        return mx.where(step < warmup, rising, mx.minimum(falling, rate))

    return at


def _rows(batch: list[Item]) -> int:
    return sum(len(item.golds) for item in batch)


def dev_loss(backend, trainee: Trainee, items: list[Item]) -> float:
    """Mean loss over ``items``' fitted questions."""
    total, count = 0.0, 0
    for step in steps(items):
        for batch in step:
            value = loss(backend, trainee, batch)
            mx.eval(value)
            total += value.item() * _rows(batch)
            count += _rows(batch)
    return total / count if count else float("nan")


def train(
    backend,
    trainee: Trainee,
    fit_items: list[Item],
    dev_items: list[Item],
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = 0,
    early_stop: bool = True,
    on_step=None,
    on_epoch=None,
    cancelled=None,
) -> Trained:
    """Train ``trainee`` (adapters already added) on ``fit_items``: AdamW with a
    short warm-up and linear decay, each decoder layer checkpointed, the dev
    loss after every epoch; afterwards it holds the epoch with the lowest dev
    loss (with ``early_stop``, an epoch worse than the best before it ends the
    run). ``on_step(step, total, loss)``, ``on_epoch(row)``; ``Cancelled``
    between steps once ``cancelled()`` is true."""
    import random
    import time

    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    from .checkpoint import checkpointed
    from .events import Cancelled

    if epochs < 1:
        raise ValueError(f"epochs must be 1 or more, not {epochs}")
    if not fit_items:
        raise ValueError("no training examples to fit adapters on")
    rng = random.Random(f"{seed}:lora")
    plans = [steps(fit_items, rng) for _ in range(epochs)]
    total = sum(map(len, plans))
    schedule = learning_rates(learning_rate, total)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(trainee, lambda batch: loss(backend, trainee, batch))
    parameters = sum(value.size for _, value in tree_flatten(trainee.trainable_parameters()))
    history, best, kept, stopped = [], float("inf"), None, False
    step_count, trained_seconds = 0, 0.0
    try:
        for epoch, plan in enumerate(plans, 1):
            started = time.perf_counter()
            losses = []
            trainee.train()
            with checkpointed(backend.decoder_layer):
                for step in plan:
                    if cancelled is not None and cancelled():
                        raise Cancelled()
                    rows = sum(map(_rows, step))
                    summed, step_loss = None, 0.0
                    for batch in step:
                        value, grads = loss_and_grad(batch)
                        weight = _rows(batch) / rows
                        grads = tree_map(lambda g, w=weight: g * w, grads)
                        summed = grads if summed is None else tree_map(mx.add, summed, grads)
                        mx.eval(value, summed)
                        step_loss += value.item() * weight
                    optimizer.update(trainee, summed)
                    mx.eval(trainee.trainable_parameters(), optimizer.state)
                    del summed
                    losses.append(step_loss)
                    step_count += 1
                    if on_step is not None:
                        on_step(step_count, total, step_loss)
            seconds = time.perf_counter() - started
            trained_seconds += seconds
            trainee.eval()
            loss_on_dev = dev_loss(backend, trainee, dev_items)
            row = {
                "epoch": epoch,
                "loss": round(sum(losses) / len(losses), 4),
                "dev_loss": round(loss_on_dev, 4),
                "seconds": round(seconds, 1),
            }
            history.append(row)
            worse = loss_on_dev > best
            if loss_on_dev < best or kept is None:
                # References to this epoch's arrays: later updates make new ones.
                current = dict(tree_flatten(trainee.trainable_parameters()))
                best, kept = loss_on_dev, (epoch, current)
            if early_stop and worse and epoch < len(plans):
                row["stopped"] = stopped = True
            if on_epoch is not None:
                on_epoch(row)
            if stopped:
                break
    finally:
        trainee.eval()
    del optimizer
    epoch, tensors = kept
    trainee.update(tree_unflatten(list(tensors.items())))
    mx.eval(trainee.parameters())
    return Trained(
        adapters(backend),
        list(zip(trainee.rows, trainee.bias, strict=True)),
        history,
        epoch,
        stopped,
        parameters,
        len(fit_items),
        trained_seconds,
    )
