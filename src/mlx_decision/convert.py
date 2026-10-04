"""Write an MLX copy of a model, optionally quantized."""

from pathlib import Path

from .hub import resolve_model_path
from .registry import detect_family, resolve


def convert(model: str | Path, output: str | Path, **options) -> Path:
    """Convert ``model`` (folder or Hub repo id) into the folder ``output``.

    ``options`` go to the family's converter: ``bits`` (None for no
    quantization), ``group_size`` and ``quantize_output_embeddings``.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder is not empty: {output}")
    path = resolve_model_path(model)
    family = detect_family(path)
    if family.converter is None:
        raise ValueError(f"{family.name} models cannot be converted")
    resolve(family.converter)(path, output, source=str(model), **options)
    return output
