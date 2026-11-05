# Copyright © 2024 Apple Inc.
#
# After mlx-lm's LoRALinear (https://github.com/ml-explore/mlx-lm), file
# mlx_lm/tuner/lora.py at commit 5cfec4cb39deba54210b3ff4d86f2337c7bc10b5,
# under the MIT License. See LICENSES/mlx-lm-MIT.txt.
#
# Modified for mlx-decision: no dropout; the low-rank matrices start from a
# given random key; ``delta`` gives the weight change on its own, and
# ``fuse`` keeps a quantized base quantized only when asked; ``adapt`` wraps
# named projections of modules, ``adapter_tensors`` reads the trained pairs,
# ``load_adapters`` applies a fitted model's.
"""Low-rank adapters (LoRA) on linear layers: ``y = W x + scale * (x A) B``.

``A`` (inputs by rank) starts uniform in +-1/sqrt(inputs), ``B`` (rank by
outputs) at zero, so a new adapter changes nothing until it is trained.
The base layer stays frozen; only ``A`` and ``B`` train.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """A frozen linear layer (plain or quantized) plus a trained low-rank update."""

    @staticmethod
    def from_base(linear: nn.Module, rank: int, scale: float, key: mx.array | None = None):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits
        adapted = LoRALinear(input_dims, output_dims, rank, scale, key)
        adapted.linear = linear
        return adapted

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int,
        scale: float,
        key: mx.array | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=False)
        self.scale = scale
        bound = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(-bound, bound, (input_dims, rank), key=key)
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)

    def delta(self) -> mx.array:
        """The update to the base's weight (outputs by inputs), in float32."""
        a = self.lora_a.astype(mx.float32)
        b = self.lora_b.astype(mx.float32)
        return (self.scale * b.T) @ a.T

    def fuse(self, dequantize: bool = False) -> nn.Module:
        """The base layer with the update added to its weight; a quantized base is
        dequantized, updated and quantized again unless ``dequantize``."""
        linear = self.linear
        quantized = isinstance(linear, nn.QuantizedLinear)
        weight = linear.weight
        if quantized:
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
                mode=linear.mode,
            )
        output_dims, input_dims = weight.shape
        bias = "bias" in linear
        fused = nn.Linear(input_dims, output_dims, bias=bias)
        fused.weight = (weight.astype(mx.float32) + self.delta()).astype(weight.dtype)
        if bias:
            fused.bias = linear.bias
        if quantized and not dequantize:
            fused = nn.QuantizedLinear.from_linear(
                fused, linear.group_size, linear.bits, mode=linear.mode
            )
        return fused


def adapt(
    targets: list[tuple[str, nn.Module, tuple[str, ...]]],
    rank: int,
    scale: float,
    seed: int = 0,
) -> int:
    """Wrap each ``(prefix, module, names)``'s named linear layers in a
    ``LoRALinear``, in order, each ``A`` from its own key of ``seed``. Returns
    the number of trainable parameters added."""
    key = mx.random.key(seed)
    count = 0
    for _, module, names in targets:
        for name in names:
            key, sub = mx.random.split(key)
            adapted = LoRALinear.from_base(getattr(module, name), rank, scale, sub)
            setattr(module, name, adapted)
            count += adapted.lora_a.size + adapted.lora_b.size
    return count


def adapter_tensors(targets: list[tuple[str, nn.Module, tuple[str, ...]]]) -> dict[str, mx.array]:
    """The adapted layers' ``A`` and ``B`` by ``"<prefix>.<name>.lora_a"`` / ``lora_b``."""
    tensors = {}
    for prefix, module, names in targets:
        for name in names:
            layer = getattr(module, name)
            tensors[f"{prefix}.{name}.lora_a"] = layer.lora_a
            tensors[f"{prefix}.{name}.lora_b"] = layer.lora_b
    return tensors


def load_adapters(targets: list[tuple[str, nn.Module, tuple[str, ...]]], path) -> int:
    """Apply a fitted model's adapters (``fit.safetensors``'s ``lora.*`` tensors and
    ``lora.scale``) to ``targets``: fused into the weights of a plain base,
    unfused on a quantized one (whose weights stay as stored). Returns the
    number of adapted layers; ``ValueError`` when the file's adapters do not
    match the targets."""
    found = {k[len("lora.") :]: v for k, v in mx.load(str(path)).items() if k.startswith("lora.")}
    if not found:
        return 0
    if "scale" not in found:
        raise ValueError(f"{path}: its adapters have no scale")
    scale = float(found.pop("scale").item())
    count = 0
    for prefix, module, names in targets:
        for name in names:
            a = found.pop(f"{prefix}.{name}.lora_a", None)
            b = found.pop(f"{prefix}.{name}.lora_b", None)
            if a is None or b is None:
                raise ValueError(f"{path}: no adapter for {prefix}.{name}")
            linear = getattr(module, name)
            output_dims = linear.weight.shape[0]
            adapted = LoRALinear.from_base(linear, a.shape[1], scale)
            if a.shape != adapted.lora_a.shape or b.shape != (a.shape[1], output_dims):
                raise ValueError(f"{path}: the adapter for {prefix}.{name} has other shapes")
            adapted.lora_a, adapted.lora_b = a.astype(mx.float32), b.astype(mx.float32)
            quantized = isinstance(linear, nn.QuantizedLinear)
            setattr(module, name, adapted if quantized else adapted.fuse())
            count += 1
    if found:
        extra = next(iter(found))
        raise ValueError(f"{path}: an adapter for a layer this model does not have ({extra})")
    return count
