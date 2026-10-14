"""Where examples come from: a Hugging Face dataset, a saved dataset folder, or a file.

``open_source`` gives a ``Table``: the columns with their kinds (a class
label's names, text, numbers, lists, ...) and a sample value each, the row
count, and the rows by index. Hugging Face datasets (``hf:ORG/NAME``, read
through the HF cache), folders written by ``datasets``' ``save_to_disk``
and Parquet files need the ``training`` extra (the ``datasets`` library);
CSV and JSON lines files are read with the standard library.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HUB_PREFIX = "hf:"
INSTALL_HINT = "pip install 'mlx-decision[training]'"
# Column kinds.
TEXT, INT, FLOAT, BOOL, LABEL, LIST, OBJECT, OTHER = (
    "text", "int", "float", "bool", "label", "list", "object", "other"
)  # fmt: skip


class SourceError(ValueError):
    """A source that cannot be read as given: missing, ambiguous, or needing the extra."""


@dataclass(frozen=True)
class Column:
    """A column: its kind, a class label's names, and a sample value."""

    name: str
    kind: str
    names: list[str] | None = None
    sample: Any = None

    def describe(self) -> str:
        """``name (kind, e.g. "sample")`` for messages and menus."""
        kind = f"class label: {', '.join(self.names[:6])}" if self.names else self.kind
        if self.names and len(self.names) > 6:
            kind += ", ..."
        sample = json.dumps(self.sample, ensure_ascii=False, default=str)
        if len(sample) > 40:
            sample = sample[:39] + "…"
        return f"{self.name} ({kind}, e.g. {sample})"


@dataclass
class Table:
    """A source's rows: ``name`` (the dataset's or file's, for default ids),
    ``split`` (Hub and saved datasets), the columns and ``length``."""

    name: str
    split: str | None
    columns: dict[str, Column]
    length: int
    read: Callable[[list[int]], list[dict]] = field(repr=False)
    values: Callable[[str], list] = field(repr=False)
    origin: dict[str, Any] = field(default_factory=dict)

    def rows(self, indices: list[int]) -> list[dict]:
        """The rows at ``indices``, as plain dicts, in that order."""
        return self.read(indices)

    def column(self, name: str) -> list:
        """A whole column's values."""
        return self.values(name)

    def listing(self) -> str:
        """The columns, for messages."""
        return "; ".join(column.describe() for column in self.columns.values())


def _datasets():
    try:
        import datasets
    except ImportError:
        raise SourceError(
            f"reading Hugging Face datasets and Parquet files needs the training extra: "
            f"{INSTALL_HINT}"
        ) from None
    return datasets


def _feature_column(name: str, feature, sample) -> Column:
    datasets = _datasets()
    if isinstance(feature, datasets.ClassLabel):
        return Column(name, LABEL, list(feature.names), sample)
    if isinstance(feature, datasets.Value):
        dtype = feature.dtype
        if dtype in ("string", "large_string"):
            return Column(name, TEXT, None, sample)
        if dtype == "bool":
            return Column(name, BOOL, None, sample)
        if dtype.startswith(("int", "uint")):
            return Column(name, INT, None, sample)
        if dtype.startswith("float"):
            return Column(name, FLOAT, None, sample)
        return Column(name, OTHER, None, sample)
    if isinstance(feature, dict):
        return Column(name, OBJECT, None, sample)
    if isinstance(
        feature, (list, getattr(datasets, "Sequence", ()), getattr(datasets, "List", ()))
    ):
        # A sequence of dicts (``{"text": [...], "label": [...]}``) is an object.
        inner = getattr(feature, "feature", None)
        return Column(name, OBJECT if isinstance(inner, dict) else LIST, None, sample)
    return Column(name, OTHER, None, sample)


def _from_dataset(dataset, name: str, split: str | None, origin: dict) -> Table:
    first = dataset[0] if len(dataset) else {}
    columns = {
        column: _feature_column(column, feature, first.get(column))
        for column, feature in dataset.features.items()
    }
    return Table(
        name,
        split,
        columns,
        len(dataset),
        lambda indices: dataset.select(indices).to_list(),
        lambda column: dataset[column],
        origin,
    )


def _pick(kind: str, given: str | None, available: list[str], flag: str, what: str) -> str:
    """The config or split to read: the one given, or the only one there is."""
    if given is not None:
        if given not in available:
            raise SourceError(f"{what} has no {kind} {given!r}; it has {', '.join(available)}")
        return given
    if len(available) == 1:
        return available[0]
    raise SourceError(f"{what} has several {kind}s; pass {flag} ({', '.join(available)})")


def hub_configs(repo: str, revision: str | None = None) -> list[str]:
    """A Hub dataset's configs; empty when it has only the default one."""
    datasets = _datasets()
    try:
        configs = datasets.get_dataset_config_names(repo, revision=revision)
    except Exception as error:  # the Hub's errors are many and vary by version
        raise SourceError(
            f"{HUB_PREFIX}{repo} cannot be read from the Hugging Face Hub: {error}"
        ) from None
    return [] if list(configs or []) == ["default"] else list(configs or [])


def hub_splits(repo: str, config: str | None = None, revision: str | None = None) -> list[str]:
    """A Hub dataset's splits (of ``config``)."""
    return list(_datasets().get_dataset_split_names(repo, config, revision=revision))


def saved_splits(path: Path) -> list[str]:
    """A saved dataset folder's splits; empty when it holds one dataset."""
    datasets = _datasets()
    loaded = datasets.load_from_disk(str(path))
    return list(loaded) if isinstance(loaded, datasets.DatasetDict) else []


def load_hub(
    repo: str, config: str | None = None, split: str | None = None, revision: str | None = None
) -> Table:
    """A split of a Hugging Face dataset, through the HF cache."""
    datasets = _datasets()
    what = f"{HUB_PREFIX}{repo}"
    configs = hub_configs(repo, revision)
    if configs:
        config = _pick("config", config, configs, "--config", what)
    elif config is not None:
        raise SourceError(f"{what} has no config {config!r}")
    splits = hub_splits(repo, config, revision)
    where = what + (f" ({config})" if config else "")
    split = _pick("split", split, splits, "--split", where)
    dataset = datasets.load_dataset(repo, config, split=split, revision=revision)
    origin = {"repo": repo, "config": config, "split": split, "revision": revision}
    return _from_dataset(dataset, repo.rsplit("/", 1)[-1], split, origin)


def load_saved(path: Path, split: str | None = None) -> Table:
    """A dataset folder written by ``save_to_disk`` (one split, or a split of several)."""
    datasets = _datasets()
    loaded = datasets.load_from_disk(str(path))
    if isinstance(loaded, datasets.DatasetDict):
        split = _pick("split", split, list(loaded), "--split", str(path))
        loaded = loaded[split]
    elif split is not None:
        raise SourceError(f"{path} holds one split; drop --split")
    return _from_dataset(loaded, path.name, split, {"path": str(path), "split": split})


def _kind(values: list) -> str:
    present = [v for v in values if v is not None and v != ""]
    if not present:
        return TEXT
    if all(isinstance(v, bool) for v in present):
        return BOOL
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return INT
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return FLOAT
    if all(isinstance(v, str) for v in present):
        return TEXT
    if all(isinstance(v, list) for v in present):
        return LIST
    if all(isinstance(v, dict) for v in present):
        return OBJECT
    return OTHER


def _csv_value(text: str) -> Any:
    """A CSV cell as a number or a boolean when it is one, else the text."""
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def _from_rows(rows: list[dict], name: str, origin: dict) -> Table:
    names = list(dict.fromkeys(key for row in rows for key in row))
    columns = {}
    for column in names:
        values = [row.get(column) for row in rows]
        columns[column] = Column(column, _kind(values), None, values[0] if values else None)
    return Table(
        name,
        None,
        columns,
        len(rows),
        lambda indices: [rows[i] for i in indices],
        lambda column: [row.get(column) for row in rows],
        origin,
    )


def load_file(path: Path) -> Table:
    """A ``.csv`` (header row), ``.jsonl`` (one object per line) or ``.parquet`` file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataset = _datasets().load_dataset("parquet", data_files=str(path), split="train")
        return _from_dataset(dataset, path.stem, None, {"path": str(path)})
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as file:
            rows = [
                {key: _csv_value(value) for key, value in row.items() if key is not None}
                for row in csv.DictReader(file)
            ]
    elif suffix in (".jsonl", ".json"):
        rows = []
        with open(path, encoding="utf-8") as file:
            for number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceError(f"{path} line {number}: not JSON ({error.msg})") from None
                if not isinstance(row, dict):
                    raise SourceError(f"{path} line {number}: not an object")
                rows.append(row)
    else:
        raise SourceError(f"{path}: not a .csv, .jsonl or .parquet file")
    if not rows:
        raise SourceError(f"{path} holds no rows")
    return _from_rows(rows, path.stem, {"path": str(path)})


def open_source(
    source: str,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
) -> Table:
    """The table of ``hf:ORG/NAME``, a saved dataset folder or a file."""
    if source.startswith(HUB_PREFIX):
        return load_hub(source[len(HUB_PREFIX) :], config, split, revision)
    if config is not None or revision is not None:
        raise SourceError("--config and --revision are for Hugging Face datasets (hf:ORG/NAME)")
    path = Path(source).expanduser()
    if path.is_dir():
        return load_saved(path, split)
    if not path.is_file():
        raise SourceError(
            f"{source}: no such file or folder (a Hugging Face dataset is hf:ORG/NAME)"
        )
    if split is not None:
        raise SourceError("--split is for datasets, not files")
    return load_file(path)
