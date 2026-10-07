# Copyright © 2023 Apple Inc.
#
# Copied from mlx-lm (https://github.com/ml-explore/mlx-lm), file
# mlx_lm/models/cache.py at commit 5cfec4cb39deba54210b3ff4d86f2337c7bc10b5,
# under the MIT License. See LICENSES/mlx-lm-MIT.txt.
#
# Modified for mlx-decision: reduced to the two caches the Qwen3.5 text model
# uses, for one sequence without padding: ConcatenateKVCache for attention
# layers and ArraysCache for Gated DeltaNet layers. Removed: the other cache
# types, batching (padding, lengths, filter, extend, merge), trimming and
# saving. Both caches replace arrays instead of writing into them, so a
# cache's arrays can be shared by several caches (see ``fork``).

from typing import List, Optional

import mlx.core as mx

from .base import create_causal_mask


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class ConcatenateKVCache:
    """ConcatenateKVCache the simplest KV cache implementation.

    Can be used as a mock KV cache or when large blocks are being processed at
    a time in which case KVCache isn't necessarily faster. Consider using the
    KVCache with a larger step size before using this cache.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        self.offset = self.keys.shape[-2]
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    def fork(self) -> "ConcatenateKVCache":
        """A cache over the same arrays; extending it leaves this one as it is."""
        other = ConcatenateKVCache()
        other.keys, other.values, other.offset = self.keys, self.values, self.offset
        return other

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class ArraysCache:
    def __init__(self, size):
        self.cache = [None] * size

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    def empty(self):
        return self.cache[0] is None

    def fork(self) -> "ArraysCache":
        """A cache over the same arrays; updating it leaves this one as it is."""
        other = ArraysCache(len(self.cache))
        other.cache = list(self.cache)
        return other

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


Cache = ConcatenateKVCache | ArraysCache


def fork_cache(cache: List[Cache]) -> List[Cache]:
    """A copy of a model's cache that shares its arrays (they are never written into)."""
    return [c.fork() for c in cache]
