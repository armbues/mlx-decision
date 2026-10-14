"""Clef's head fit: the joint head trained on stored backbone states, the backbone frozen.

The backbone runs once per request; its hidden states go to a folder on
disk (safetensors shards of up to 256 requests or 256 MB), read back one
request at a time when needed, and the folder is removed afterwards. The
head trains in float32 (cast back to its stored dtype after): AdamW (weight
decay 0.01), 5% linear warm-up then linear decay, gradients summed over
batches of 16 requests (one request's graph at a time, so memory does not
grow with the batch) and clipped to norm 1, the requests in a seeded random
order. After every epoch the dev part's loss is measured; the epoch with the
lowest is kept. The loss is the decision loss over the head's logits (Clef
has no temperature).
"""

import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .events import Cancelled
from .finetune import Trained
from .loss import decision_loss

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP = 0.05
BATCH_REQUESTS = 16
CLIP_NORM = 1.0
SHARD_REQUESTS = 256
SHARD_BYTES = 256 * 2**20


class StateStore:
    """Requests' head inputs with their hidden states on disk, under ``folder``
    (created here, removed by ``remove``). ``add`` keeps a request and returns
    its index; ``get`` gives its inputs back, the states read from disk."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        self.inputs: list[Any] = []  # without their hidden states
        self.where: list[tuple[int, str]] = []
        self.shards: list[Path] = []
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.bytes = 0

    def add(self, inputs) -> int:
        index = len(self.inputs)
        key = f"hidden.{index}"
        mx.eval(inputs.hidden)
        self.pending[key] = inputs.hidden
        self.pending_bytes += inputs.hidden.nbytes
        self.inputs.append(replace(inputs, hidden=None))
        self.where.append((len(self.shards), key))
        if len(self.pending) >= SHARD_REQUESTS or self.pending_bytes >= SHARD_BYTES:
            self.flush()
        return index

    def flush(self) -> None:
        if not self.pending:
            return
        file = self.folder / f"states-{len(self.shards):05d}.safetensors"
        mx.save_safetensors(str(file), self.pending)
        self.bytes += file.stat().st_size
        self.shards.append(file)
        self.pending, self.pending_bytes = {}, 0

    def get(self, index: int):
        shard, key = self.where[index]
        if shard == len(self.shards):
            hidden = self.pending[key]
        else:
            # Loaded afresh each time: ``mx.load`` is lazy, so only this tensor is
            # read, and nothing keeps it once the caller is done with it.
            hidden = mx.load(str(self.shards[shard]))[key]
        return replace(self.inputs[index], hidden=hidden)

    def remove(self) -> None:
        self.shards, self.pending = [], {}
        shutil.rmtree(self.folder, ignore_errors=True)


@dataclass(frozen=True)
class Item:
    """A stored request and its labelled questions: question index -> gold option index."""

    state: int
    labels: dict[int, int]


def _loss(backend, store: StateStore, item: Item, dtype) -> mx.array:
    """The decision loss over an item's labelled questions."""
    inputs = store.get(item.state)
    logits = backend.head_logits(replace(inputs, hidden=inputs.hidden.astype(dtype)))
    rows = [logits[q] for q in item.labels]
    width = max(row.shape[0] for row in rows)
    padded = mx.stack([mx.pad(row.astype(mx.float32), (0, width - row.shape[0])) for row in rows])
    valid = mx.arange(width)[None] < mx.array([row.shape[0] for row in rows])[:, None]
    return decision_loss(padded, valid, mx.array(list(item.labels.values())))


def dev_loss(backend, store: StateStore, items: list[Item], dtype=mx.float32) -> float:
    """Mean loss over ``items`` (by request)."""
    if not items:
        return float("nan")
    total = 0.0
    for item in items:
        loss = _loss(backend, store, item, dtype)
        mx.eval(loss)
        total += loss.item()
    return total / len(items)


def train(
    backend,
    store: StateStore,
    fit_items: list[Item],
    dev_items: list[Item],
    *,
    epochs: int = 3,
    seed: int = 0,
    on_step: Callable[[int, int, float], None] | None = None,
    on_epoch: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Trained:
    """Train ``backend.head`` in place on ``fit_items``; afterwards it holds the kept
    epoch's weights in its dtype from before (also ``Trained.tensors``, named
    ``head.*``). ``on_step(step, total, loss)`` after each batch,
    ``on_epoch(row)`` after each epoch; ``Cancelled`` between batches once
    ``cancelled()`` is true (the head then holds the last step's weights: a
    model to discard)."""
    if epochs < 1:
        raise ValueError(f"epochs must be 1 or more, not {epochs}")
    if not fit_items:
        raise ValueError("no training examples to fit the head on")
    head = backend.head
    serving_dtype = tree_flatten(head.parameters())[0][1].dtype
    rng = random.Random(f"{seed}:head")
    plans = []
    for _ in range(epochs):
        order = list(fit_items)
        rng.shuffle(order)
        plans.append([order[i : i + BATCH_REQUESTS] for i in range(0, len(order), BATCH_REQUESTS)])
    total = sum(map(len, plans))
    warmup = max(1, round(total * WARMUP))
    rise = optim.linear_schedule(0.0, LEARNING_RATE, warmup)
    decay = optim.linear_schedule(LEARNING_RATE, 0.0, max(1, total - warmup))
    optimizer = optim.AdamW(
        learning_rate=optim.join_schedules([rise, decay], [warmup]), weight_decay=WEIGHT_DECAY
    )
    head.set_dtype(mx.float32)
    try:
        loss_and_grad = nn.value_and_grad(
            head, lambda item: _loss(backend, store, item, mx.float32)
        )
        trained = Trained({})
        best_loss = float("inf")
        step = 0
        for epoch, plan in enumerate(plans, 1):
            started = time.perf_counter()
            losses = []
            for batch in plan:
                if cancelled is not None and cancelled():
                    raise Cancelled()
                summed, total_loss = None, 0.0
                for item in batch:
                    loss, grads = loss_and_grad(item)
                    summed = grads if summed is None else tree_map(mx.add, summed, grads)
                    mx.eval(loss, summed)
                    total_loss += loss.item()
                count = len(batch)
                grads = tree_map(lambda g, count=count: g / count, summed)
                grads, _ = optim.clip_grad_norm(grads, CLIP_NORM)
                optimizer.update(head, grads)
                mx.eval(head.parameters(), optimizer.state)
                losses.append(total_loss / len(batch))
                step += 1
                if on_step is not None:
                    on_step(step, total, losses[-1])
            seconds = time.perf_counter() - started
            loss_on_dev = dev_loss(backend, store, dev_items)
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
                trained.tensors = dict(tree_flatten(head.parameters()))
            if on_epoch is not None:
                on_epoch(row)
        del optimizer
        head.update(tree_unflatten(list(trained.tensors.items())))
    finally:
        head.set_dtype(serving_dtype)
        mx.eval(head.parameters())
    trained.tensors = {f"head.{k}": v for k, v in tree_flatten(head.parameters())}
    return trained
