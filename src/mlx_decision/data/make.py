"""``make``: training examples from a source, a shape and a wording, or from a preset.

What the ``data`` command runs: open the source, pick the rows (all, the
first ``limit`` or a seeded ``sample``), guess or take the shape and
columns, build one labelled question per row, and check the result as
``training fit`` will read it. Presets take their source, split commit and
wording from ``presets``.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from . import presets as preset_data
from .format import Example, parse_examples
from .shapes import Mapping, Wording, build, default_id, guess
from .sources import HUB_PREFIX, SourceError, Table, open_source


@dataclass
class Made:
    """The examples made, the table they came from, the shape used (None for a
    preset) and the rows left out, counted by reason."""

    examples: list[Example]
    table: Table
    mapping: Mapping | None
    skipped: Counter = field(default_factory=Counter)

    def label_counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, Counter] = {}
        for example in self.examples:
            for question_id, label in example.labels.items():
                counts.setdefault(question_id, Counter())[label] += 1
        return {qid: dict(sorted(c.items(), key=lambda kv: -kv[1])) for qid, c in counts.items()}


def rows_to_read(length: int, limit: int | None, sample: int | None, seed: int) -> list[int]:
    """All rows, the first ``limit``, or ``sample`` of them drawn with ``seed`` (in
    the source's order)."""
    if limit is not None and sample is not None:
        raise SourceError("--limit and --sample do not go together")
    if limit is not None:
        return list(range(min(limit, length)))
    if sample is not None:
        return sorted(random.Random(seed).sample(range(length), min(sample, length)))
    return list(range(length))


def checked(examples: list[Example]) -> list[Example]:
    """``examples`` as ``training fit`` reads them back (the format's checks)."""
    if not examples:
        raise SourceError("no row gave an example")
    lines = [json.dumps(example.to_json(), ensure_ascii=False) for example in examples]
    return parse_examples(lines)


def make(
    source: str | None,
    *,
    preset: str | None = None,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    shape: str | None = None,
    columns: dict[str, str | None] | None = None,
    wording: dict[str, Any] | None = None,
) -> Made:
    """Examples from ``source`` (``hf:ORG/NAME``, a saved dataset folder, a file)
    or ``preset``. ``columns`` names columns by role (``text``, ``label``, ...),
    ``wording`` holds ``Wording``'s fields (``id`` defaults to the source's name).
    Raises ``SourceError`` (a ``ValueError``) naming the flag that would settle
    what is open or wrong, and ``DataError`` when a made line fails the format."""
    if preset is not None:
        if source is not None:
            raise SourceError("a preset names its own dataset; drop the source")
        if shape is not None or any((columns or {}).values()) or any((wording or {}).values()):
            raise SourceError("a preset fixes the shape, columns and wording; drop those flags")
        return _preset(preset, split, limit, sample, seed)
    if source is None:
        raise SourceError(
            "name a source (hf:ORG/NAME, a .csv, .jsonl or .parquet file) or --preset"
        )
    table = open_source(source, config, split, revision)
    mapping = guess(table, shape, columns)
    fields = {key: value for key, value in (wording or {}).items() if value is not None}
    fields.setdefault("id", default_id(table))
    built = build(
        table, mapping, Wording(**fields), rows_to_read(table.length, limit, sample, seed)
    )
    return Made(checked(built.examples), table, mapping, built.skipped)


def _preset(name: str, split: str | None, limit, sample, seed) -> Made:
    found = preset_data.PRESETS.get(name)
    if found is None:
        listed = ", ".join(preset_data.PRESETS)
        raise SourceError(f"--preset: no preset {name!r}; the presets are {listed}")
    pinned = found.source
    table = open_source(HUB_PREFIX + pinned.repo, pinned.config, split, pinned.revision)
    indices = rows_to_read(table.length, limit, sample, seed)
    rows = table.rows(indices)
    made = []
    for item in preset_data.examples(found, rows, table.split, indices):
        [question_id] = item["questions"]
        request = {"state": item["state"], "questions": item["questions"]}
        made.append(Example(item["id"], request, {question_id: item["gold"]}))
    return Made(checked(made), table, None)


# The flag of each column role, in the order commands list them.
COLUMN_FLAGS = {
    "text": "--text",
    "label": "--label",
    "premise": "--premise",
    "hypothesis": "--hypothesis",
    "question": "--question-column",
    "choices": "--choices",
    "answer": "--answer",
    "context": "--context",
}


@dataclass
class DataPlan:
    """What a ``data`` run does, as its flags say it (and as the menus fill it
    in): ``arguments()`` gives the flags back, so the menus can print the
    command they stand for."""

    source: str | None = None
    output: str | None = None
    preset: str | None = None
    config: str | None = None
    split: str | None = None
    revision: str | None = None
    limit: int | None = None
    sample: int | None = None
    seed: int = 0
    show: int | None = None
    shape: str | None = None
    columns: dict[str, str | None] = field(default_factory=dict)
    question_id: str | None = None
    instructions: str | None = None
    options: dict[str, str] | None = None
    label_map: dict[str, str] | None = None
    noul: bool = False
    levels: list[str] | None = None
    questions_file: str | None = None
    question: dict[str, Any] | None = None  # read from ``questions_file``

    def wording(self) -> dict[str, Any]:
        return {
            "id": self.question_id,
            "instructions": self.instructions,
            "options": self.options,
            "label_map": self.label_map,
            "noul": self.noul or None,
            "levels": self.levels,
            "question": self.question,
        }

    def has_wording(self) -> bool:
        """Whether the question's wording is given (a preset brings its own)."""
        return bool(self.preset or self.instructions or self.question or self.shape == "yesno")

    def make(self) -> Made:
        return make(
            self.source,
            preset=self.preset,
            config=self.config,
            split=self.split,
            revision=self.revision,
            limit=self.limit,
            sample=self.sample,
            seed=self.seed,
            shape=self.shape,
            columns=self.columns,
            wording=self.wording(),
        )

    def arguments(self) -> list[str]:
        """The flags, in the order ``data --help`` lists them."""
        from ..flags import join_list, join_pairs

        words = [self.source] if self.source else []
        named = [
            ("--preset", self.preset),
            ("--config", self.config),
            ("--split", self.split),
            ("--revision", self.revision),
            ("--limit", self.limit),
            ("--sample", self.sample),
            ("--seed", self.seed if self.sample is not None and self.seed else None),
            ("--shape", self.shape),
            *((flag, self.columns.get(role)) for role, flag in COLUMN_FLAGS.items()),
            ("--questions", self.questions_file),
            ("--id", self.question_id),
            ("--question", self.instructions),
            ("--options", join_pairs(self.options) if self.options else None),
            ("--label-map", join_pairs(self.label_map) if self.label_map else None),
            ("--levels", join_list(self.levels) if self.levels else None),
            ("--show", self.show),
            ("-o", self.output),
        ]
        for flag, value in named:
            if value is not None:
                words += [flag, str(value)]
        if self.noul:
            words.append("--noul")
        return words
