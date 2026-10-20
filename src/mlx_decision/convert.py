"""Write an MLX copy of a model, optionally quantized."""

import shutil
from collections.abc import Callable
from pathlib import Path

from .errors import require_metal
from .events import Cancelled, Event, Steps
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


def partial_folder(output: str | Path) -> Path:
    """Where ``convert`` writes ``output`` before renaming it into place."""
    output = Path(output)
    return output.with_name(f".{output.name}.partial")


def convert(
    model: str | Path,
    output: str | Path,
    *,
    on_event: Callable[[Event], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    **options,
) -> Path:
    """Convert ``model`` (folder or Hub repo id) into the folder ``output``.

    ``options`` go to the family's converter: ``bits`` (None for no
    quantization), ``group_size`` and ``quantize_output_embeddings``, or
    ``target_bits`` and ``progress`` for mixed precision. ``on_event``
    receives the stages ``load``, ``convert`` (tensors as steps; the
    sensitivity sweep's units for mixed precision), ``write`` and ``done``;
    ``cancel()`` (a callable, or a ``threading.Event``) stops the conversion
    at the next tensor (or unit) with ``Cancelled``, writing nothing.
    """
    require_metal()
    output = Path(output)
    emit = on_event or (lambda event: None)
    cancelled = getattr(cancel, "is_set", cancel) or (lambda: False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder is not empty: {output}")
    emit(Event("load", message=f"reading {model}"))
    path = resolve_model_path(model)
    family = detect_family(path)
    if family.converter is None:
        raise ValueError(f"{family.name} models cannot be converted; they load as they are")
    # A Hub id, or only the folder name: a local path would put the user's
    # directories into a folder that may be shared.
    source = str(model) if is_repo_id(model) else path.resolve().name
    steps = Steps("convert", None, emit)
    writing = []

    def write() -> None:
        if not writing:
            writing.append(True)
            emit(Event("write", message=f"writing {output}"))

    def on_step(done: int, total: int) -> None:
        steps.at(done, total)
        if done >= total:
            write()  # the tensors are written; the rest is small

    if (progress := options.get("progress")) is not None:

        def checked(done, total, unit, value) -> None:
            progress(done, total, unit, value)
            if cancelled():
                raise Cancelled()

        options["progress"] = checked
    # Written next to the output and renamed when complete, so a failed
    # conversion leaves no half-written model behind.
    partial = partial_folder(output)
    shutil.rmtree(partial, ignore_errors=True)
    try:
        if cancelled():
            raise Cancelled()
        resolve(family.converter)(
            path, partial, source=source, on_step=on_step, cancelled=cancelled, **options
        )
        write()
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if output.exists():
        output.rmdir()  # empty, checked above
    partial.rename(output)
    emit(Event("done", metrics={"path": str(output)}, message=f"wrote {output}"))
    return output
