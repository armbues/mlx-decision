"""Checks the Training page shows before a run: whether a fitted model was
fitted on the very file it would be evaluated on (TRS-4)."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..pool import ModelPool


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes, as a fit records its data's."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def trained_on(pool: ModelPool, name: str) -> str | None:
    """The SHA-256 of the data a fitted model was fitted on, as its folder
    records it; None for a base or a fit that records none."""
    spec = pool.specs.get(name)
    if spec is None or spec.fit is None:
        return None
    recorded = (spec.fit.marker.get("fit") or {}).get("data")
    return recorded if isinstance(recorded, str) else None


def training_data_warnings(
    pool: ModelPool, names: Iterable[str | None], data: Path, data_name: str
) -> list[str]:
    """A warning for each of ``names`` fitted on ``data`` itself (the same
    SHA-256): its scores there count examples it learned from."""
    fitted = [(name, trained_on(pool, name)) for name in names if name]
    fitted = [(name, recorded) for name, recorded in fitted if recorded]
    if not fitted:
        return []
    sha = file_sha256(data)
    return [
        f"{data_name} includes {name}'s training data: {name} was fitted on this very "
        "file (the same SHA-256), so its scores here count examples it learned from"
        for name, recorded in fitted
        if recorded == sha
    ]
