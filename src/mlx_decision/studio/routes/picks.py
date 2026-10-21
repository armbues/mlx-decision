"""A dataset and one of its splits, as the fit, evaluation and plan routes
name them: the dataset by its name in ``data/``, the split by its name (by
default the one its role gives the run)."""

from dataclasses import dataclass
from pathlib import Path

from ...data.dataset import Dataset, open_dataset
from ...errors import DecisionError
from ..workspace import NotFound, Workspace, WorkspaceError, check_name
from . import ApiError, field


@dataclass
class Picked:
    """A dataset's name, the split picked and that split's file."""

    name: str
    split: str
    path: Path
    dataset: Dataset


def pick_split(
    files: Workspace,
    body: dict,
    key: str,
    purpose: str,
    split_key: str | None = None,
    ready: bool = True,
) -> Picked:
    """``body[key]``'s dataset and ``body[split_key]``'s split (else the split
    its role gives ``purpose``: train, report or evaluate). 400 for a name
    that is not one, 404 for a dataset or split that is not there, 422 for a
    dataset a run cannot use yet (``ready``: questions and every value
    mapped) or no split for ``purpose``."""
    split_key = split_key or f"{key}_split"
    name = field(body, key, str)
    try:
        check_name(name, "a dataset's name")
        location = files.dataset_location(name)
    except NotFound:
        raise ApiError(404, f"no dataset is named {name}", param=key) from None
    except WorkspaceError as error:
        raise ApiError(400, error.message, param=key) from None
    try:
        dataset, _ = open_dataset(location)
    except (DecisionError, OSError) as error:
        raise ApiError(422, f"{name}: {error}", param=key) from None
    split = field(body, split_key, str, required=False)
    if split is None:
        try:
            split = dataset.pick(purpose)
        except DecisionError:
            listed = ", ".join(dataset.splits)
            raise ApiError(
                422, f"{name} has no {purpose} split; pick one of {listed}", param=split_key
            ) from None
    elif split not in dataset.splits:
        raise ApiError(404, f"{name} has no split {split}", param=split_key)
    if ready:
        missing = dataset.missing()
        if missing:
            raise ApiError(422, f"{name} is not ready for a run: {'; '.join(missing)}", param=key)
    return Picked(name, split, dataset.split_path(split), dataset)
