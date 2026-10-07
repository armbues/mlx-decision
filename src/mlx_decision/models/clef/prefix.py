"""Prefix reuse: the computed state after a request's state, kept for later requests.

Clef reads the system prompt, the images and the state before the questions,
and the backbone is causal, so everything it computes up to the state's last
token depends only on that prefix. A request whose prefix matches a kept one
continues from it and computes only its questions. An entry holds the
backbone's cache at that point and the prefix's final hidden states (the head
attends over every hidden state, the state's included).
"""

import hashlib
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ...backbones.qwen3_5.cache import Cache, fork_cache

DEFAULT_PREFIX_CACHE_GB = 2.0


def prefix_key(input_ids: Sequence[int], pixel_values: Sequence[Any] = ()) -> str:
    """A digest of the prefix tokens and the images' preprocessed pixels.

    The tokens are those after any cut, so a state cut differently is a
    different prefix; the pixels are those after resizing, so the image size
    limit is part of the key. ``pixel_values`` are NumPy arrays (NumPy comes
    with the images extra, so it is not imported here).
    """
    digest = hashlib.sha256(array("q", input_ids).tobytes())
    for pixels in pixel_values:
        digest.update(f"{pixels.dtype}{pixels.shape}".encode())
        digest.update(pixels.tobytes())
    return digest.hexdigest()


@dataclass
class PrefixEntry:
    cache: list[Cache]
    hidden: mx.array  # [prefix length, hidden size]

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.cache) + self.hidden.nbytes


class PrefixCache:
    """Entries by key, least recently used dropped first to stay within ``max_bytes``."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, PrefixEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> PrefixEntry | None:
        """The entry for ``key`` with a cache of its own to continue from, or None."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return PrefixEntry(fork_cache(entry.cache), entry.hidden)

    def put(self, key: str, cache: list[Cache], hidden: mx.array) -> PrefixEntry:
        """Keep a computed prefix (if it fits); returns an entry to continue from."""
        entry = PrefixEntry(cache, hidden)
        size = entry.nbytes
        if size <= self.max_bytes:
            self._entries.pop(key, None)
            while self._entries and self.nbytes + size > self.max_bytes:
                self._entries.popitem(last=False)
            self._entries[key] = entry
        return PrefixEntry(fork_cache(cache), hidden)

    def clear(self) -> None:
        self._entries.clear()
