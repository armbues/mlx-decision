"""Mixed-precision quantization driven by measured sensitivity.

A *unit* is a group of layers that get the same bit width: the layers of one
block (an MLP, an attention) inside a repeated layer, or a single layer
elsewhere (embeddings, lm_head). A unit's sensitivity is how far the
model's answers move, as mean KL divergence over calibration questions,
when only that unit is quantized at the base bit width. Bits are then
allocated greedily under a target average.
"""

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import mlx.nn as nn

ALLOWED_BITS = (2, 3, 4, 5, 6, 8)
SMALL_FRACTION = 0.01  # of the median layer; smaller layers always get 8 bits
EPS = 1e-9


@dataclass(frozen=True)
class Unit:
    name: str
    layers: tuple[str, ...]
    params: int


def group_units(
    layers: Mapping[str, int], small_fraction: float = SMALL_FRACTION
) -> tuple[list[Unit], list[str]]:
    """Group quantizable layers (path -> parameters) into units; also return the small ones.

    Small layers have fewer than ``small_fraction`` times the parameters of
    the median layer (in Qwen3.5: the linear-attention gates ``in_proj_a`` and
    ``in_proj_b``). They cost almost nothing at 8 bits.
    """
    small = small_fraction * statistics.median(layers.values()) if layers else 0
    groups: dict[str, list[str]] = {}
    small_layers = []
    for path, params in layers.items():
        if params < small:
            small_layers.append(path)
            continue
        parts = path.split(".")
        in_block = "layers" in parts[:-2]
        groups.setdefault(".".join(parts[:-1]) if in_block else path, []).append(path)
    units = [
        Unit(name, tuple(paths), sum(layers[path] for path in paths))
        for name, paths in groups.items()
    ]
    return units, small_layers


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(a * math.log((a + EPS) / (b + EPS)) for a, b in zip(p, q, strict=True) if a > 0)


def mean_kl(reference: list[list[float]], other: list[list[float]]) -> float:
    pairs = list(zip(reference, other, strict=True))
    return sum(kl_divergence(p, q) for p, q in pairs) / len(pairs)


def _module(root: nn.Module, path: str) -> nn.Module:
    node = root
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _swap(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = _module(root, parent_path) if parent_path else root
    if isinstance(parent, list):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def measure_sensitivity(
    root: nn.Module,
    units: Sequence[Unit],
    bits: int,
    group_size: int,
    evaluate: Callable[[], list[list[float]]],
    mode: str = "affine",
    progress: Callable[[int, int, Unit, float], None] | None = None,
) -> dict[str, float]:
    """Mean KL of the answers to ``evaluate()`` unquantized, per unit quantized alone.

    ``root`` is left as it was found.
    """
    reference = evaluate()
    sensitivity = {}
    for index, unit in enumerate(units):
        originals = {path: _module(root, path) for path in unit.layers}
        for path, module in originals.items():
            _swap(root, path, module.to_quantized(group_size=group_size, bits=bits, mode=mode))
        try:
            sensitivity[unit.name] = mean_kl(reference, evaluate())
        finally:
            for path, module in originals.items():
                _swap(root, path, module)
        if progress is not None:
            progress(index + 1, len(units), unit, sensitivity[unit.name])
    return sensitivity


def base_bits(target: float) -> int:
    """The largest allowed bit width below ``target``, so there is budget to upgrade."""
    if target < ALLOWED_BITS[0]:
        raise ValueError(f"target bits must be at least {ALLOWED_BITS[0]}")
    candidates = [bits for bits in ALLOWED_BITS if bits < target]
    return candidates[-1] if candidates else ALLOWED_BITS[0]


def allocate(
    units: Sequence[Unit], sensitivity: Mapping[str, float], base: int, target: float
) -> dict[str, int]:
    """Bits per unit with a parameter-weighted average of at most ``target``.

    Expected error of a unit at ``b`` bits is its sensitivity times
    ``4 ** (base - b)`` (each bit halves the rounding error, a quarter of the
    squared error). Upgrades go, one allowed step at a time, to the unit with
    the largest expected reduction per extra bit of storage.
    """
    bits = {unit.name: base for unit in units}
    total = sum(unit.params for unit in units)
    budget = target * total - base * total

    def error(unit: Unit, level: int) -> float:
        return sensitivity[unit.name] * 4.0 ** (base - level)

    def next_level(level: int) -> int | None:
        higher = [b for b in ALLOWED_BITS if b > level]
        return higher[0] if higher else None

    while True:
        # Units whose quantization changes nothing (gain 0) still take
        # leftover budget, after every unit that gains.
        best, best_gain = None, -1.0
        for unit in units:
            level = next_level(bits[unit.name])
            if level is None:
                continue
            cost = unit.params * (level - bits[unit.name])
            if cost > budget:
                continue
            gain = (error(unit, bits[unit.name]) - error(unit, level)) / cost
            if gain > best_gain:
                best, best_gain = unit, gain
        if best is None:
            return bits
        level = next_level(bits[best.name])
        budget -= best.params * (level - bits[best.name])
        bits[best.name] = level


def average_bits(units: Sequence[Unit], bits: Mapping[str, int]) -> float:
    total = sum(unit.params for unit in units)
    if not total:
        return 0.0
    return sum(unit.params * bits[unit.name] for unit in units) / total
