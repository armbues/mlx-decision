# Copyright © 2023 Apple Inc.
#
# After mlx-lm's grad_checkpoint (https://github.com/ml-explore/mlx-examples,
# mlx_lm/tuner/trainer.py), under the MIT License. See LICENSES/mlx-lm-MIT.txt.
#
# Modified for mlx-decision: scoped to a `with` block (the class's __call__ is
# restored on exit, so serving in the same process is unaffected), built on
# mlx.nn.utils.checkpoint, and only the first argument is a checkpoint input
# (the others, such as attention masks, are passed through as they are).
"""Gradient checkpointing per layer: activations are recomputed in the backward
pass instead of being kept, which cuts a fine-tune's memory several times
for about a fifth more time."""

from collections.abc import Iterator
from contextlib import contextmanager

import mlx.nn as nn


@contextmanager
def checkpointed(layer_class: type[nn.Module]) -> Iterator[None]:
    """Within the block, every instance of ``layer_class`` checkpoints its call."""
    original = layer_class.__call__

    def call(layer, x, *args, **kwargs):
        def run(inputs):
            return original(layer, inputs, *args, **kwargs)

        return nn.utils.checkpoint(layer, run)(x)

    layer_class.__call__ = call
    try:
        yield
    finally:
        layer_class.__call__ = original
