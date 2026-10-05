"""Resolve a model reference to a local folder."""

from pathlib import Path


class ModelNotFoundError(FileNotFoundError):
    """A model that is neither a local folder nor downloadable from the Hub."""


def is_repo_id(model: str | Path) -> bool:
    """Whether ``model`` names a Hugging Face repo rather than a local folder."""
    path = Path(model).expanduser()
    return not (path.exists() or str(model).startswith((".", "/", "~")))


def resolve_model_path(model: str | Path) -> Path:
    """Return the folder for a local path or a Hugging Face repo id.

    Raises ``ModelNotFoundError`` with a one-line reason when there is none.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if not is_repo_id(model):
        raise ModelNotFoundError(f"model folder not found: {model}")

    import httpx
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        return Path(snapshot_download(str(model)))
    except GatedRepoError:
        reason = "gated on the Hugging Face Hub: accept its terms there and set HF_TOKEN"
    except RepositoryNotFoundError:
        reason = (
            "not a local folder, and not found on the Hugging Face Hub (or private: set HF_TOKEN)"
        )
    except HFValidationError:
        reason = "not a local folder or a valid Hugging Face repo id (org/name)"
    except LocalEntryNotFoundError:
        reason = "not downloaded yet, and the Hugging Face Hub cannot be reached"
    except (HfHubHTTPError, httpx.HTTPError, OSError) as error:
        reason = f"cannot download from the Hugging Face Hub: {str(error).splitlines()[0]}"
    raise ModelNotFoundError(f"{model}: {reason}") from None
