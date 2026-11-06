"""Mixes: a dataset folder made of other datasets, read as one dataset.

A mix's ``dataset.json`` holds ``format`` 2 and ``mix``::

    {"format": 2,
     "mix": {"sources": [{"path": "../trec"},
                         {"path": "../sst5", "cap": 3000},
                         {"path": "../openbookqa", "held_out": true}],
             "cap": 2000, "seed": 0}}

- ``sources``: each a dataset folder, one of its split files or a plain
  ``.jsonl`` file, by a path relative to the mix's folder (or absolute).
  ``cap`` is the most train rows read from it (the mix's ``cap`` when left
  out; every row when neither is given), sampled with the mix's ``seed``. A
  ``held_out`` source is scored, never trained on.
- The mix has three splits, each the sources' rows of that role together:
  ``train`` (each source's train split, capped), ``report`` (its report
  split; a source without one gives a fifth of its capped train rows, cut
  off before training, as a fit's own holdout does) and ``evaluate`` (its
  evaluate split, else its report split, as an evaluation of a dataset
  folder reads; a source with neither gives none). A held-out source gives
  report and evaluate rows only.

Every example's id starts with its source's name (``trec/17``) and its
``source`` names it. A mix may not hold another mix, two sources of one
name, or two sources asking one question id differently.
"""

import hashlib
import json
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import DecisionError
from .format import DataError, Example

if TYPE_CHECKING:
    from .dataset import Dataset

# The dataset.json format a mix is written with (plain datasets keep 1).
MIX_FORMAT = 2
SPLITS = ("train", "report", "evaluate")  # each split is named after its role
# A source without a report split reports on this share of its train rows, as
# a fit's own holdout does.
REPORT_SHARE = 0.2


@dataclass(frozen=True)
class MixSource:
    path: str  # as written: relative to the mix's folder, or absolute
    cap: int | None = None
    held_out: bool = False

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {"path": self.path}
        if self.cap is not None:
            body["cap"] = self.cap
        if self.held_out:
            body["held_out"] = True
        return body


@dataclass(frozen=True)
class Mix:
    sources: tuple[MixSource, ...]
    cap: int | None = None
    seed: int = 0

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {"sources": [source.to_json() for source in self.sources]}
        if self.cap is not None:
            body["cap"] = self.cap
        body["seed"] = self.seed
        return body


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_mix(path: Path, body: Any) -> Mix:
    """A ``dataset.json``'s ``mix``; ``DecisionError`` naming what is wrong."""
    shape = '{"sources": [{"path": "../other"}, ...], "cap": N, "seed": N}'
    if not isinstance(body, dict) or not isinstance(body.get("sources"), list):
        raise DecisionError(f"{path}: mix must be an object: {shape}")
    if not body["sources"]:
        raise DecisionError(f"{path}: the mix has no sources")
    for key in ("cap", "seed"):
        value = body.get(key)
        if value is not None and (not _whole(value) or value < (1 if key == "cap" else 0)):
            least = "1" if key == "cap" else "0"
            raise DecisionError(f"{path}: mix.{key} must be a whole number from {least}")
    sources = []
    for index, source in enumerate(body["sources"]):
        where = f"{path}: mix.sources[{index}]"
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise DecisionError(f'{where} must be an object with a path: {{"path": "../other"}}')
        cap = source.get("cap")
        if cap is not None and (not _whole(cap) or cap < 1):
            raise DecisionError(f"{where}.cap must be a whole number from 1")
        held_out = source.get("held_out", False)
        if not isinstance(held_out, bool):
            raise DecisionError(f"{where}.held_out must be true or false")
        sources.append(MixSource(source["path"], cap, held_out))
    if all(source.held_out for source in sources):
        raise DecisionError(f"{path}: every source of the mix is held out; nothing would train")
    return Mix(tuple(sources), body.get("cap"), body.get("seed") or 0)


@dataclass
class _Opened:
    """A source as opened: its dataset, the split file its path names (None
    for a folder or a plain file) and its name in the mix."""

    source: MixSource
    name: str
    dataset: "Dataset"
    named: str | None

    def train_split(self) -> str:
        return self.named or self.dataset.pick("train")

    def report_split(self) -> str | None:
        """Its own report split, or None (a fifth of its train rows reports)."""
        if self.named is not None or not self.dataset.folder:
            return None
        report = self.dataset.with_role("report")
        if report is None:
            return None
        if self.source.held_out:
            return report
        return None if report == self.train_split() else report

    def evaluate_split(self) -> str | None:
        if self.named is not None or not self.dataset.folder:
            return None
        return self.dataset.with_role("evaluate") or self.dataset.with_role("report")


def _resolve(folder: Path, source: MixSource) -> Path:
    path = Path(source.path).expanduser()
    # Lexically: the mix's folder need not exist yet (``write_mix`` checks first).
    return Path(os.path.normpath(path if path.is_absolute() else folder / path))


def open_sources(folder: Path, mix: Mix) -> list[_Opened]:
    """The mix's sources, opened (no rows read); ``DecisionError`` naming the
    source that cannot be."""
    from .dataset import dataset_name, open_dataset

    opened: list[_Opened] = []
    for source in mix.sources:
        path = _resolve(folder, source)
        try:
            dataset, named = open_dataset(path)
        except FileNotFoundError:
            raise DecisionError(f"{folder}: the mix's source {source.path} is missing") from None
        except (DecisionError, OSError) as error:
            raise DecisionError(f"{folder}: the mix's source {source.path}: {error}") from None
        if dataset.mix is not None:
            raise DecisionError(
                f"{folder}: the mix's source {source.path} is a mix itself; list its "
                "sources instead"
            )
        name = dataset_name(path)
        taken = next((o for o in opened if o.name == name), None)
        if taken is not None:
            raise DecisionError(
                f"{folder}: the mix's sources {taken.source.path} and {source.path} have "
                f"one name, {name}; rename one of them"
            )
        opened.append(_Opened(source, name, dataset, named))
    return opened


def missing(folder: Path, mix: Mix) -> list[str]:
    """What keeps the mix from a fit or an evaluation: its sources' gaps."""
    try:
        opened = open_sources(folder, mix)
    except DecisionError as error:
        return [str(error).removeprefix(f"{folder}: ")]
    found = []
    for o in opened:
        found += [f"source {o.name}: {gap}" for gap in o.dataset.missing()]
        if not o.source.held_out or o.report_split() is None:
            try:
                o.train_split()
            except DecisionError as error:
                found.append(f"source {o.name}: {error}")
    return found


def _read(o: _Opened, split: str, lenient: bool) -> list[Example]:
    try:
        return o.dataset.read(split, lenient)
    except DataError as error:
        error.file = str(o.dataset.split_path(split))
        raise


def _sampled(o: _Opened, mix: Mix, lenient: bool) -> tuple[list[Example], random.Random]:
    """The source's train rows, capped; the generator goes on to cut its report."""
    rows = _read(o, o.train_split(), lenient)
    cap = o.source.cap if o.source.cap is not None else mix.cap
    rng = random.Random(f"{mix.seed}:{o.name}")
    if cap is not None and len(rows) > cap:
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), cap))]
    return rows, rng


def _parts(o: _Opened, mix: Mix, lenient: bool) -> tuple[list[Example], list[Example]]:
    """The source's train and report rows."""
    report = o.report_split()
    if report is not None:
        train = [] if o.source.held_out else _sampled(o, mix, lenient)[0]
        return train, _read(o, report, lenient)
    rows, rng = _sampled(o, mix, lenient)
    order = list(range(len(rows)))
    rng.shuffle(order)
    cut = set(order[: round(len(rows) * REPORT_SHARE)])
    kept = [] if o.source.held_out else [row for i, row in enumerate(rows) if i not in cut]
    return kept, [row for i, row in enumerate(rows) if i in cut]


def _check_split(folder: Path, split: str) -> None:
    if split not in SPLITS:
        raise DecisionError(f"{folder} is a mix: its splits are {', '.join(SPLITS)}, not {split}")


def read(folder: Path, mix: Mix, split: str, lenient: bool = False) -> list[Example]:
    """The examples of the mix's ``split``, each tagged with its source."""
    _check_split(folder, split)
    examples: list[Example] = []
    for o in open_sources(folder, mix):
        if split == "evaluate":
            name = o.evaluate_split()
            rows = _read(o, name, lenient) if name is not None else []
        else:
            rows = _parts(o, mix, lenient)[0 if split == "train" else 1]
        examples += [replace(e, id=f"{o.name}/{e.id}", source=o.name) for e in rows]
    if not examples:
        why = "an evaluate or report split" if split == "evaluate" else f"{split} rows"
        raise DecisionError(f"{folder}: no source of the mix has {why}")
    _check_questions(folder, examples)
    return examples


def _check_questions(folder: Path, examples: list[Example]) -> None:
    """Refuse a question id that two sources ask unless both ask it one and the
    same way (a source's lines may each ask their own, as long as no other
    source has the id)."""
    asked: dict[str, dict[str | None, set[str]]] = {}
    for example in examples:
        for question_id, question in (example.request.get("questions") or {}).items():
            text = json.dumps(question, sort_keys=True, ensure_ascii=False)
            asked.setdefault(question_id, {}).setdefault(example.source, set()).add(text)
    for question_id, by_source in asked.items():
        if len(by_source) < 2:
            continue
        ways = set().union(*by_source.values())
        if len(ways) > 1:
            first, second = list(by_source)[:2]
            raise DecisionError(
                f"{folder}: the mix's sources {first} and {second} both ask "
                f"{question_id!r}, differently; give one of them another question id"
            )


def _source_splits(o: _Opened, split: str) -> list[str]:
    """The source's splits the mix's ``split`` reads rows from."""
    if split == "evaluate":
        name = o.evaluate_split()
        return [name] if name is not None else []
    report = o.report_split()
    if split == "train" and o.source.held_out and report is not None:
        return []
    if split == "report" and report is not None:
        return [report]
    return [o.train_split()]


def sha256(folder: Path, mix: Mix, split: str) -> str:
    """The hash a fit records: the mix itself (sources, caps, seed) and the
    hash of every source split the mix's ``split`` reads."""
    _check_split(folder, split)
    digest = hashlib.sha256(
        json.dumps({"mix": mix.to_json(), "split": split}, sort_keys=True).encode()
    )
    for o in open_sources(folder, mix):
        for name in _source_splits(o, split):
            digest.update(f"{o.name}:{name}:{o.dataset.sha256(name)}\n".encode())
    return digest.hexdigest()


def files(folder: Path, mix: Mix, split: str) -> list[Path]:
    """The files the mix's ``split`` reads (its ``dataset.json``, then each
    source's and its split files), to tell when it changed on disk."""
    from .dataset import DATASET_FILE

    _check_split(folder, split)
    found = [folder / DATASET_FILE]
    for o in open_sources(folder, mix):
        if o.dataset.folder:
            found.append(o.dataset.path / DATASET_FILE)
        found += [o.dataset.split_path(name) for name in _source_splits(o, split)]
    return found


def roots(folder: Path, mix: Mix) -> dict[str, Path]:
    """Each source's folder its lines' image paths are inside, by its name."""
    return {o.name: o.dataset.root for o in open_sources(folder, mix)}


def describe(folder: Path, mix: Mix) -> list[dict[str, Any]]:
    """Each source as a page shows it: ``name``, ``path`` (as written),
    ``location`` (resolved), ``cap`` (the one that applies, None for every
    row) and ``held_out``."""
    return [
        {
            "name": o.name,
            "path": o.source.path,
            "location": str(_resolve(folder, o.source)),
            "cap": o.source.cap if o.source.cap is not None else mix.cap,
            "held_out": o.source.held_out,
        }
        for o in open_sources(folder, mix)
    ]


def write_mix(folder: str | Path, mix: Mix, created: str | None = None) -> Path:
    """Write a mix's folder (only its ``dataset.json``); its sources are
    checked first. An existing folder is not replaced."""
    from .dataset import DATASET_FILE

    folder = Path(folder)
    if folder.exists():
        raise DecisionError(f"{folder} exists; pick another name for the mix")
    open_sources(folder, mix)
    folder.mkdir(parents=True)
    body: dict[str, Any] = {"format": MIX_FORMAT, "mix": mix.to_json()}
    if created is not None:
        body["created"] = created
    (folder / DATASET_FILE).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    return folder
