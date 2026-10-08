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
        if (stored := stored_snapshot(str(model))) is not None:
            return stored
        return Path(_snapshot(str(model), snapshot_download))
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


def stored_snapshot(repo_id: str) -> Path | None:
    """The model ``download`` stored quantized in the Hub cache, if there is one.

    It is the snapshot of the cached main revision, recognised by a marker
    written there as a plain file (a downloaded file is a link to a blob).
    Its full-size weight files are not in the cache, so asking the Hub for
    the snapshot would fetch them again. Needs no network.
    """
    from huggingface_hub import try_to_load_from_cache

    from .registry import MARKER_FILE

    found = try_to_load_from_cache(repo_id, MARKER_FILE)
    if not isinstance(found, str) or Path(found).is_symlink():
        return None
    return Path(found).parent


def _snapshot(repo_id: str, snapshot_download) -> str:
    """The repository in the Hub cache: the files its family needs, or all of it.

    The family is recognised from the JSON files, fetched first.
    """
    from .registry import detect_family

    configs = Path(snapshot_download(repo_id, allow_patterns=["*.json"]))
    try:
        files = detect_family(configs).files
    except ValueError:
        files = None
    patterns = list(files) if files is not None else None
    return snapshot_download(repo_id, allow_patterns=patterns)
