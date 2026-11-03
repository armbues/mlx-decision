"""A Hub stand-in for ``download``: one repo, ``org/fake``, a model of the fake family.

``install(sizes, delay)`` replaces the Hub calls of ``mlx_decision.download``.
The repo holds the fake family's marker plus ``sizes`` (file name to bytes);
a download writes each file in chunks of ``CHUNK`` bytes, ``delay`` seconds
apart, counted on a bytes bar as the Hub counts them. Into the cache it
writes the Hub's layout (``models--org--fake/snapshots/SHA``, ``refs/main``).

For a download run as its own process, ``from_env()`` installs it from
``FAKE_HUB_SIZES`` (JSON) and ``FAKE_HUB_DELAY``.
"""

import json
import os
import time
from pathlib import Path

from huggingface_hub.utils import filter_repo_objects

REPO = "org/fake"
SHA = "f" * 40
CHUNK = 1_000_000


def install(sizes: dict[str, int], delay: float = 0.0, patch=setattr) -> None:
    """Replace the Hub in ``mlx_decision.download``; ``patch`` is ``setattr`` or
    a test's ``monkeypatch.setattr``."""
    from types import SimpleNamespace

    from huggingface_hub import constants
    from huggingface_hub.errors import RepositoryNotFoundError

    import mlx_decision.download as download
    from mlx_decision.registry import MARKER_FILE

    contents = {MARKER_FILE: json.dumps({"family": "fake"}).encode()}
    contents.update({name: b"\0" * size for name, size in sizes.items()})

    class Api:
        def model_info(self, repo_id, files_metadata=False, revision=None):
            if repo_id != REPO:
                import httpx

                request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
                raise RepositoryNotFoundError(
                    "missing", response=httpx.Response(404, request=request)
                )
            siblings = [SimpleNamespace(rfilename=n, size=len(b)) for n, b in contents.items()]
            return SimpleNamespace(siblings=siblings, sha=revision or SHA)

    def snapshot_download(
        repo_id,
        allow_patterns=None,
        local_dir=None,
        revision=None,
        ignore_patterns=None,
        tqdm_class=None,
    ):
        names = list(filter_repo_objects(list(contents), allow_patterns=allow_patterns))
        names = list(filter_repo_objects(names, ignore_patterns=ignore_patterns))
        if local_dir is None:
            repo = Path(constants.HF_HUB_CACHE) / "models--org--fake"
            folder = repo / "snapshots" / SHA
            (repo / "refs").mkdir(parents=True, exist_ok=True)
            (repo / "refs" / "main").write_text(SHA)
        else:
            folder = Path(local_dir)
        folder.mkdir(parents=True, exist_ok=True)
        total = sum(len(contents[name]) for name in names)
        bar = tqdm_class(total=total, unit="B", desc="Downloading")
        try:
            for name in names:
                with open(folder / name, "wb") as file:
                    data = contents[name]
                    for start in range(0, max(len(data), 1), CHUNK):
                        chunk = data[start : start + CHUNK]
                        file.write(chunk)
                        bar.update(len(chunk))
                        if delay:
                            time.sleep(delay)
        finally:
            bar.close()
        return str(folder)

    patch(download, "HfApi", Api)
    patch(download, "snapshot_download", snapshot_download)


def from_env() -> None:
    install(json.loads(os.environ["FAKE_HUB_SIZES"]), float(os.environ.get("FAKE_HUB_DELAY", 0)))
