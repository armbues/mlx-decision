"""Multimodal rotary positions for Qwen3.5: one position per axis (time, height, width).

Text tokens have the same position on all three axes, which makes this equal
to ordinary rotary positions. Image and video tokens differ per axis.

The interleaved assignment of frequencies to axes follows the Qwen3.5 rotary
embedding in transformers (Apache License 2.0, see
LICENSES/transformers-Apache-2.0.txt); the code is written for MLX.
"""

import mlx.core as mx
import mlx.nn as nn


class MRoPE(nn.Module):
    def __init__(self, dims: int, base: float, mrope_section: list[int]):
        super().__init__()
        self.dims = dims
        self._inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        # Frequency i reads its position from axis i % 3, within each axis'
        # section; frequencies beyond the height/width sections use time.
        axes = [0] * (dims // 2)
        for axis, offset in ((1, 1), (2, 2)):
            for index in range(offset, min(mrope_section[axis] * 3, dims // 2), 3):
                axes[index] = axis
        self._axes = mx.array(axes)

    def __call__(self, x: mx.array, position_ids: mx.array) -> mx.array:
        """Rotate ``x`` ``[B, heads, L, head_dim]`` by ``position_ids`` ``[3, B, L]``."""
        positions = position_ids.astype(mx.float32)  # [3, B, L]
        # [B, L, F]: for each frequency, the position on its axis
        selected = mx.take(positions, self._axes, axis=0).transpose(1, 2, 0)
        freqs = selected * self._inv_freq
        cos = mx.concatenate([mx.cos(freqs)] * 2, axis=-1)[:, None].astype(x.dtype)
        sin = mx.concatenate([mx.sin(freqs)] * 2, axis=-1)[:, None].astype(x.dtype)

        rotary, rest = x[..., : self.dims], x[..., self.dims :]
        half = self.dims // 2
        rotated = mx.concatenate([-rotary[..., half:], rotary[..., :half]], axis=-1)
        return mx.concatenate([rotary * cos + rotated * sin, rest], axis=-1)
