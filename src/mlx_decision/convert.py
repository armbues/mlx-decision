"""Write an MLX copy of a model, optionally quantized."""

import shutil
from pathlib import Path

from .errors import require_metal
from .hub import is_repo_id, resolve_model_path
from .registry import detect_family, resolve


def default_output(
    model: str | Path, bits: int | None = None, target_bits: float | None = None
) -> Path:
    """``<source name>-q<bits>``, ``-mq<target>`` (mixed) or ``-mlx``, in the current folder."""
    name = Path(str(model).rstrip("/")).name
    if target_bits is not None:
        suffix = f"mq{target_bits:g}"
    elif bits is not None:
        suffix = f"q{bits}"
    else:
        suffix = "mlx"
    return Path(f"{name}-{suffix}")


def convert(model: str | Path, output: str | Path, **options) -> Path:
    """Convert ``model`` (folder or Hub repo id) into the folder ``output``.

    ``options`` go to the family's converter: ``bits`` (None for no
    quantization), ``group_size`` and ``quantize_output_embeddings``.
    """
    require_metal()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder is not empty: {output}")
    path = resolve_model_path(model)
    family = detect_family(path)
    if family.converter is None:
        raise ValueError(f"{family.name} models cannot be converted; they load as they are")
    # A Hub id, or only the folder name: a local path would put the user's
    # directories into a folder that may be shared.
    source = str(model) if is_repo_id(model) else path.resolve().name
    # Written next to the output and renamed when complete, so a failed
    # conversion leaves no half-written model behind.
    partial = output.with_name(f".{output.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        resolve(family.converter)(path, partial, source=source, **options)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if output.exists():
        output.rmdir()  # empty, checked above
    partial.rename(output)
    return output
