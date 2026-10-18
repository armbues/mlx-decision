"""The model picker: the models on this Mac, for ``run`` and ``chat`` without a model.

It lists ``models_dir``'s models, then those of ``models``, then the
known families' complete snapshots in the Hugging Face cache; each group
in the families' order, with family and size on disk. Nothing is loaded.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .registry import Family, families


@dataclass(frozen=True)
class Choice:
    reference: str  # what -m takes
    name: str
    family: str
    size: int | None  # bytes on disk; None for a repo id not downloaded
    place: str
    order: int  # the family's position in the families' order


def choices(config: Config) -> list[Choice]:
    """Every model the picker offers, without one listed twice (the first place wins)."""
    from .hub import cached_models, is_repo_id

    order = {family.name: index for index, family in enumerate(families())}
    groups: list[list[Choice]] = [[], [], []]
    seen: set[str] = set()

    def add(group: int, reference: str, name: str, path: Path | None, place: str) -> None:
        key = str(path.resolve()) if path is not None else reference
        if key in seen:
            return
        identified = _family(path) if path is not None else None
        if path is not None and identified is None:
            return  # no family loads it
        seen.add(key)
        family, fitted = identified or (None, False)
        label = "" if family is None else _title(family) + (", fitted" if fitted else "")
        size = folder_size(path) if path is not None else None
        rank = order.get(family.name, len(order)) if family is not None else len(order)
        groups[group].append(Choice(reference, name, label, size, place, rank))

    folder = config.models_folder
    if folder is not None:
        for path in _subfolders(folder):
            add(0, str(path), path.name, path, "models_dir")
    cached = dict(cached_models())
    for reference in config.models:
        path = Path(reference).expanduser()
        if is_repo_id(reference):
            add(1, reference, reference, cached.get(reference), "models")
        elif path.is_dir() and _family(path) is None:  # a folder of models
            for sub in _subfolders(path):
                add(1, str(sub), sub.name, sub, "models")
        elif path.is_dir():
            add(1, reference, path.name, path, "models")
    for repo_id, snapshot in cached.items():
        add(2, repo_id, repo_id, snapshot, "Hugging Face cache")
    return [
        choice
        for group in groups
        for choice in sorted(group, key=lambda c: (c.order, c.name.casefold()))
    ]


def pick(options: list[Choice]) -> str | None:
    """The reference of the model picked in a menu; None when it is left."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.output import create_output

    from .download import format_size
    from .interactive import pick as menu

    rows = [
        (c.name, c.family, format_size(c.size) if c.size is not None else "", c.place)
        for c in options
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    entries = [
        (c.reference, "  ".join(part.ljust(w) for part, w in zip(row, [*widths, 0], strict=True)))
        for c, row in zip(options, rows, strict=True)
    ]
    # The menu goes to the terminal even when stdout is redirected.
    with create_app_session(output=create_output(always_prefer_tty=True)):
        try:
            return menu("Model:", entries)
        except (KeyboardInterrupt, EOFError):
            return None


def _family(path: Path) -> tuple[Family, bool] | None:
    """The family that loads ``path`` (a fitted model's base) and whether it is fitted."""
    from .pool import identify

    try:
        return identify(path)
    except (ValueError, OSError):
        return None


def _subfolders(folder: Path) -> list[Path]:
    try:
        return [
            path for path in folder.iterdir() if path.is_dir() and not path.name.startswith(".")
        ]
    except OSError:
        return []


def _title(family: Family) -> str:
    return family.listing.title if family.listing else family.name


def folder_size(folder: Path) -> int:
    """The bytes of the files in ``folder`` (a cache snapshot's links followed)."""
    total = 0
    for file in folder.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            pass
    return total
