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
import re
from collections.abc import Callable, Iterable
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


def hub_card(repo: str, revision: str | None = None) -> str | None:
    """A Hub dataset's card (its README), or None when there is none or it
    cannot be read."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # no card, offline, or the Hub's errors
        return None


# A value and its name in a card: "LABEL_3": "Dividend", "3": "Dividend",
# 3: Dividend, - 3 = Dividend, | 3 | Dividend |.
_LABEL_PAIR = re.compile(r"""["']?(?:LABEL_)?([\w.-]+)["']?\s*[:=]\s*["']([^"'\n]{1,80})["']""")
_LINE_PAIR = re.compile(
    r"""^\s*[-*|]?\s*`?["']?([\w.-]+)["']?`?\s*"""
    r"""(?::|=|\||-|–|—)\s*(.{1,80}?)\s*\|?\s*$"""
)


def names_from_card(card: str, values: list[str]) -> dict[str, str] | None:
    """The names a dataset card gives its label's ``values`` (such as a list of
    ``"LABEL_0": "Analyst Update"``), or None when it names fewer than half of
    them. Only keys that are values count, the first name of each."""
    wanted = set(values)
    found: dict[str, str] = {}
    for key, name in _LABEL_PAIR.findall(card):
        if key in wanted and key not in found and name.strip():
            found[key] = name.strip()
    if len(found) * 2 < len(values):
        for line in card.splitlines():
            match = _LINE_PAIR.match(line)
            if match and match.group(1) in wanted and match.group(1) not in found:
                name = match.group(2).strip().strip("\"'`")
                if name and name not in values:
                    found[match.group(1)] = name
    if not values or len(found) * 2 < len(values):
        return None
    return {value: found[value] for value in values if value in found}


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
        config = _pick("subset", config, configs, "--subset", what)
    elif config is not None:
        raise SourceError(f"{what} has no subset {config!r}")
    splits = hub_splits(repo, config, revision)
    where = what + (f" ({config})" if config else "")
    split = _pick("split", split, splits, "--split", where)
    dataset = datasets.load_dataset(repo, config, split=split, revision=revision)
    origin = {"repo": repo, "config": config, "split": split, "revision": revision}
    return _from_dataset(dataset, repo.rsplit("/", 1)[-1], split, origin)


# The Hub's parquet copy of every public dataset, by config and split.
PARQUET_REVISION = "refs/convert/parquet"
# Row counts read from that copy's file footers (a few KB each), at most.
FOOTER_FILES = 16


@dataclass
class SplitSize:
    """A Hub split's size from the Hub's metadata, nothing of it read: ``bytes``
    of its parquet copy, ``rows`` from the card or the copy's footers (None
    when unknown); ``partial`` when the Hub copied only its first part (a
    split of several GB)."""

    rows: int | None = None
    bytes: int | None = None
    partial: bool = False


def hub_split_sizes(repo: str, config: str | None = None) -> dict[str, SplitSize]:
    """Each split's size (of ``config``) as the Hub knows it, best effort:
    the files of its parquet copy, and the rows from the dataset card's
    ``dataset_info``, else from those files' footers. Empty when the Hub
    says nothing (a private or brand-new dataset)."""
    from huggingface_hub import HfApi

    api = HfApi()
    by_config: dict[str, dict[str, list]] = {}
    try:
        tree = api.list_repo_tree(
            repo, path_in_repo=config, recursive=True, repo_type="dataset",
            revision=PARQUET_REVISION,
        )  # fmt: skip
        for entry in tree:
            parts = entry.path.split("/")
            if len(parts) >= 3 and entry.path.endswith(".parquet"):
                by_config.setdefault(parts[0], {}).setdefault(parts[1], []).append(entry)
    except Exception:  # no parquet copy, offline, or the Hub's errors
        by_config = {}
    # Without a config: "default", or the one config there is (named otherwise).
    folder = config or ("default" if len(by_config) != 1 else next(iter(by_config)))
    files = by_config.get(folder, {})
    sizes: dict[str, SplitSize] = {}
    for name, group in files.items():
        partial = name.startswith("partial-")
        size = sizes.setdefault(name.removeprefix("partial-"), SplitSize(partial=partial))
        size.bytes = sum(getattr(entry, "size", 0) or 0 for entry in group)
    for split, rows in _card_rows(api, repo, folder).items():
        sizes.setdefault(split, SplitSize()).rows = rows
    budget = FOOTER_FILES
    for name, group in files.items():
        size = sizes[name.removeprefix("partial-")]
        if size.rows is None and not size.partial and len(group) <= budget:
            budget -= len(group)
            size.rows = _footer_rows(repo, group)
    return sizes


def _card_rows(api, repo: str, folder: str) -> dict[str, int]:
    """The rows per split the dataset card's ``dataset_info`` gives for ``folder``."""
    try:
        card = api.dataset_info(repo).card_data
        info = card.get("dataset_info") if card is not None else None
    except Exception:
        return {}
    entries = info if isinstance(info, list) else [info] if isinstance(info, dict) else []
    entries = [entry for entry in entries if isinstance(entry, dict)]
    entry = next((e for e in entries if e.get("config_name", "default") == folder), None)
    if entry is None and len(entries) == 1 and folder == "default":
        entry = entries[0]
    splits = (entry or {}).get("splits") or []
    return {
        item["name"]: item["num_examples"]
        for item in splits
        if isinstance(item, dict) and isinstance(item.get("num_examples"), int)
    }


def _footer_rows(repo: str, files: list) -> int | None:
    """The rows of a split's parquet files, from their footers alone."""
    from urllib.parse import quote

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        revision = quote(PARQUET_REVISION, safe="")
        rows = 0
        for entry in files:
            with fs.open(f"datasets/{repo}@{revision}/{entry.path}", block_size=2**16) as handle:
                rows += pq.ParquetFile(handle).metadata.num_rows
        return rows
    except Exception:
        return None


def load_hub_head(repo: str, config: str | None, split: str, count: int) -> Table:
    """The first ``count`` rows of a Hub dataset's split, streamed: the split is
    not downloaded, nothing is stored."""
    from itertools import islice

    datasets = _datasets()
    what = f"{HUB_PREFIX}{repo}" + (f" ({config})" if config else "")
    try:
        stream = datasets.load_dataset(repo, config, split=split, streaming=True)
        rows = list(islice(stream, count))
        dataset = datasets.Dataset.from_list(rows, features=stream.features)
    except Exception as error:  # the Hub's and the loaders' errors vary
        raise SourceError(f"{what}: cannot read the first rows of {split}: {error}") from None
    origin = {"repo": repo, "config": config, "split": split, "revision": None}
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


# The longest CSV cell read (the csv module's default is 128 KB).
MAX_CSV_CELL = 2**31 - 1


def _csv_value(text: str | None) -> Any:
    """A CSV cell as a number or a boolean when it is one, else the text; a cell a
    short row leaves out is None (no value)."""
    if text is None:
        return None
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def _csv_rows(handle: Iterable[str], where: str) -> list[dict]:
    """The rows of CSV text with a header row; ``SourceError`` naming the line for
    CSV that does not read (an unclosed quote)."""
    csv.field_size_limit(max(csv.field_size_limit(), MAX_CSV_CELL))
    reader = csv.DictReader(handle, strict=True)
    rows = []
    try:
        for row in reader:
            rows.append({key: _csv_value(value) for key, value in row.items() if key is not None})
    except csv.Error as error:
        raise SourceError(f"{where}line {reader.line_num}: not valid CSV ({error})") from None
    return rows


def _jsonl_rows(lines: Iterable[str], where: str) -> list[dict]:
    """The objects of JSON lines; ``SourceError`` naming the line for one that is
    not a JSON object (or nests too deep, or holds a number too long)."""
    from ..errors import DecisionError
    from ..types import decode_json

    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = decode_json(line, f"{where}line {number}")
        except DecisionError as error:
            raise SourceError(error.message) from None
        if not isinstance(row, dict):
            raise SourceError(f"{where}line {number}: not an object")
        rows.append(row)
    return rows


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


def table_from_text(text: str, kind: str, name: str) -> Table:
    """The rows of an uploaded ``csv`` (header row) or ``jsonl`` (one object per
    line) file's text."""
    import io

    if kind == "csv":
        rows = _csv_rows(io.StringIO(text.lstrip("\ufeff"), newline=""), "")
    elif kind == "jsonl":
        rows = _jsonl_rows(text.splitlines(), "")
    else:
        raise SourceError("the file is a .csv or a .jsonl file")
    if not rows:
        raise SourceError("the file holds no rows")
    return _from_rows(rows, name, {"upload": name})


def load_file(path: Path) -> Table:
    """A ``.csv`` (header row), ``.jsonl`` (one object per line) or ``.parquet`` file."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        dataset = _datasets().load_dataset("parquet", data_files=str(path), split="train")
        return _from_dataset(dataset, path.stem, None, {"path": str(path)})
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as file:
            rows = _csv_rows(file, f"{path} ")
    elif suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as file:
            rows = _jsonl_rows(file, f"{path} ")
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
        raise SourceError("--subset and --revision are for Hugging Face datasets (hf:ORG/NAME)")
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
