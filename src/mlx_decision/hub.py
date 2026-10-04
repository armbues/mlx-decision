"""Resolve a model reference to a local folder."""

from pathlib import Path


def resolve_model_path(model: str | Path) -> Path:
    """Return the folder for a local path or a Hugging Face repo id."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if path.exists() or str(model).startswith((".", "/", "~")):
        raise FileNotFoundError(f"model folder not found: {model}")

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model)))
