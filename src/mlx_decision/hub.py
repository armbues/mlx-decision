"""Resolve a model reference to a local folder."""

import logging
from functools import partial
from pathlib import Path


class ModelNotFoundError(FileNotFoundError):
    """A model that is neither a local folder nor downloadable from the Hub."""


def is_repo_id(model: str | Path) -> bool:
    """Whether ``model`` names a Hugging Face repo rather than a local folder."""
    path = Path(model).expanduser()
    return not (path.exists() or str(model).startswith((".", "/", "~")))


def resolve_model_path(model: str | Path, revision: str | None = None) -> Path:
    """Return the folder for a local path or a Hugging Face repo id.

    ``revision`` (a commit hash) pins a repo id to that snapshot: the cached
    one if there is one, else downloaded. Raises ``ModelNotFoundError`` with a
    one-line reason when there is none.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if not is_repo_id(model):
        raise ModelNotFoundError(f"model folder not found: {model}")

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
    )

    try:
        if revision is not None:
            if (cached := cached_snapshot(str(model), revision)) is not None:
                return cached
        elif (stored := stored_snapshot(str(model))) is not None:
            return stored
        return Path(_snapshot(str(model), snapshot_download, revision))
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
    except (HfHubHTTPError, *http_errors(), OSError) as error:
        reason = f"cannot download from the Hugging Face Hub: {str(error).splitlines()[0]}"
    raise ModelNotFoundError(f"{model}: {reason}") from None


def http_errors() -> tuple[type[Exception], ...]:
    """The HTTP client's errors for a failed connection: huggingface_hub 1.x
    raises httpx's, 2.x those of httpx2 (installed with it)."""
    import httpx

    try:
        import httpx2
    except ImportError:
        return (httpx.HTTPError,)
    return (httpx.HTTPError, httpx2.HTTPError)


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


def cached_snapshot(repo_id: str, revision: str) -> Path | None:
    """The cached snapshot of a repo at a commit, if there is one (no network)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    folder = Path(HF_HUB_CACHE) / repo_folder_name(repo_id=repo_id, repo_type="model")
    snapshot = folder / "snapshots" / revision
    return snapshot if snapshot.is_dir() else None


def snapshot_revision(path: Path) -> tuple[str, str] | None:
    """The repo id and commit of a Hub cache snapshot folder
    (``.../models--org--name/snapshots/<commit>``), or None for any other folder."""
    parts = Path(path).absolute().parts
    if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
        return None
    org, _, name = parts[-3].removeprefix("models--").partition("--")
    return (f"{org}/{name}", parts[-1]) if org and name else None


def _snapshot(repo_id: str, snapshot_download, revision: str | None = None) -> str:
    """The repository in the Hub cache: the files its family needs, or all of it.

    The family is recognised from the JSON files, fetched first.
    """
    from .registry import detect_family

    if revision is not None:
        snapshot_download = partial(snapshot_download, revision=revision)
    configs = Path(snapshot_download(repo_id, allow_patterns=["*.json"]))
    try:
        family = detect_family(configs)
    except ValueError:
        family = None
    files = family.files if family is not None else None
    patterns = list(files) if files is not None else None
    if family is not None and family.licence and not _weights_cached(repo_id, patterns):
        logging.getLogger(__name__).warning("%s: %s", repo_id, family.licence)
    return snapshot_download(repo_id, allow_patterns=patterns)


def _weights_cached(repo_id: str, patterns: list[str] | None) -> bool:
    """Whether the Hub cache already holds the repository's weights (no network)."""
    from huggingface_hub import snapshot_download

    try:
        folder = snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True)
    except Exception:  # not cached (or a stand-in that cannot tell)
        return False
    return any(Path(folder).glob("*.safetensors"))
