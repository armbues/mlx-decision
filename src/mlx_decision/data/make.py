"""``make``: training examples from a source, a shape and a wording, or from a preset.

What the ``data`` command runs: open the source, pick the rows (all, the
first ``limit`` or a seeded ``sample``), guess or take the shape and
columns, build one labelled question per row, and check the result as
``training fit`` will read it. Presets take their source, split commit and
wording from ``presets``. Into a dataset folder (``-o`` without ``.jsonl``)
the rows keep their raw label values with a map to the answers, and the
wording may be left out (``raw``: rows without questions).
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import presets as preset_data
from .dataset import ROLES, Dataset, SplitOut, partial_folder, raw_value, write_dataset
from .format import Example, _label_to_json, parse_examples
from .format import split as split_examples
from .shapes import (
    Mapping,
    Wording,
    build,
    class_names,
    class_of,
    default_id,
    guess,
    rating_values,
)
from .sources import HUB_PREFIX, SourceError, Table, open_source


@dataclass
class Made:
    """The examples made, the table they came from, the shape used (None for a
    preset) and the rows left out, counted by reason; ``value_column`` and
    ``classes`` (its values in order) when the examples keep raw values."""

    examples: list[Example]
    table: Table
    mapping: Mapping | None
    skipped: Counter = field(default_factory=Counter)
    value_column: str | None = None
    classes: list[str] = field(default_factory=list)

    def answers(self) -> dict[str, dict]:
        """The map from the value column's values to each question's answers,
        as the examples have them; values without an answer are left out."""
        if self.value_column is None:
            return {}
        maps: dict[str, dict[str, Any]] = {}
        for example in self.examples:
            value = (example.values or {}).get(self.value_column)
            if value is None:
                continue
            questions = example.request.get("questions") or {}
            for question_id, label in example.labels.items():
                answer = _label_to_json(questions[question_id], label)
                maps.setdefault(question_id, {}).setdefault(value, answer)
        return {
            question_id: {
                "column": self.value_column,
                "map": mapped,
                "left_out": [value for value in self.classes if value not in mapped],
            }
            for question_id, mapped in maps.items()
        }

    def columns(self) -> dict[str, dict]:
        """The value column's values in order with their rows here."""
        if self.value_column is None:
            return {}
        counts = Counter((e.values or {}).get(self.value_column) for e in self.examples)
        order = list(self.classes) + sorted(v for v in counts if v and v not in self.classes)
        return {self.value_column: {"values": {v: {"rows": counts.get(v, 0)} for v in order}}}

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
    """``examples`` as ``training fit`` reads them back (the format's checks),
    keeping their raw values; rows without questions as they are."""
    if not examples:
        raise SourceError("no row gave an example")
    if "questions" not in examples[0].request:
        return examples
    lines = [json.dumps(example.to_json(), ensure_ascii=False) for example in examples]
    read = parse_examples(lines)
    return [replace(new, values=old.values) for new, old in zip(read, examples, strict=True)]


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
    raw: bool = False,
) -> Made:
    """Examples from ``source`` (``hf:ORG/NAME``, a saved dataset folder, a file)
    or ``preset``. ``columns`` names columns by role (``text``, ``label``, ...),
    ``wording`` holds ``Wording``'s fields (``id`` defaults to the source's name).
    Raises ``SourceError`` (a ``ValueError``) naming the flag that would settle
    what is open or wrong, and ``DataError`` when a made line fails the format.
    ``raw``: no question, the rows' states and label values only."""
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
    indices = rows_to_read(table.length, limit, sample, seed)
    built = build(table, mapping, Wording(**fields), indices, raw=raw)
    made = Made(checked(built.examples), table, mapping, built.skipped)
    kept = mapping.shape in ("text", "pair", "rating")
    value_column = mapping.columns.get("label") if kept else None
    if value_column is not None:
        column = table.columns[value_column]
        made.value_column = value_column
        made.classes = (
            rating_values(table, column) if mapping.shape == "rating" and not raw
            else class_names(column, table, None)
        )  # fmt: skip
    return made


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
    column = table.columns.get(found.value_column) if found.value_column else None
    items = preset_data.examples(found, rows, table.split, indices)
    for row, item in zip(rows, items, strict=True):
        [question_id] = item["questions"]
        request = {"state": item["state"], "questions": item["questions"]}
        values = None
        if column is not None:
            values = {column.name: raw_value(class_of(column, row[column.name]))}
        made.append(Example(item["id"], request, {question_id: item["gold"]}, values))
    result = Made(checked(made), table, None)
    if column is not None:
        result.value_column = column.name
        result.classes = class_names(column, table, None)
    return result


def split_plans(
    plan: "DataPlan", splits: list[str | None], samples: list[int] | None
) -> list["DataPlan"]:
    """One plan per split: ``samples`` once for every split, or once per split
    (0: the whole split). Several splits go into a dataset folder.
    ``SourceError`` for what does not go together."""
    if len(set(splits)) != len(splits):
        raise SourceError("--split: each split once")
    many = len(splits) > 1
    if samples is None:
        samples = [None]
    if len(samples) not in (1, len(splits)):
        raise SourceError("--sample: once for every split, or once per --split")
    if len(samples) == 1:
        samples = samples * len(splits)
    if many and plan.show is not None:
        raise SourceError("--show takes one --split")
    if many and plan.output is not None and not is_folder_output(plan.output):
        raise SourceError(
            "several --split: -o names a dataset folder (-o data/NAME, one file per split)"
        )
    return [
        replace(plan, split=split, sample=sample or None)
        for split, sample in zip(splits, samples, strict=True)
    ]


def partial_path(path: str | Path) -> Path:
    """Where a file is written before it is complete: ``.NAME.partial`` beside it."""
    path = Path(path)
    return path.with_name(f".{path.name}.partial")


def leftovers(plans: list["DataPlan"]) -> list[Path]:
    """What a run of ``plans`` writes on the way (removed after a stop)."""
    if not plans or plans[0].output is None:
        return []
    if plans[0].folder:
        return [partial_folder(plans[0].output)]
    return [partial_path(plans[0].output)]


def write_file(plan: "DataPlan", made: Made, before_rename=None) -> Path:
    """Write the examples to a partial file, then rename it into place.
    ``before_rename`` (when given) runs first and may raise to stop."""
    from .format import write_examples

    output = Path(plan.output)
    partial = partial_path(output)
    try:
        write_examples(partial, made.examples)
        if before_rename is not None:
            before_rename()
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output


# Dataset folders

# The file a part is written to, by its role (another split's name taken:
# SPLIT-ROLE).
PART_NAMES = {"train": "train", "report": "validation", "evaluate": "test"}
# A split's role by its name, when --roles leaves it open.
REPORT_NAMES = ("validation", "valid", "dev", "val")


def is_folder_output(output: str | None) -> bool:
    """Whether ``-o`` names a dataset folder (no ``.jsonl``)."""
    return output is not None and not output.endswith(".jsonl")


RoleSpec = list[tuple[str, float | None]]  # one split's role, or its parts with shares


def parse_roles(text: str, count: int) -> list[RoleSpec]:
    """``--roles``: per split (comma-separated, in ``--split``'s order) a role,
    or parts joined by ``+``, each a role and its share: ``train,report:0.5+
    evaluate:0.5``. A role at most once; a split's shares add up to 1 at most."""
    entries = [entry.strip() for entry in text.split(",")]
    if len(entries) != count:
        raise SourceError("--roles: one entry per --split")
    specs: list[RoleSpec] = []
    seen: set[str] = set()
    for entry in entries:
        spec: RoleSpec = []
        for item in entry.split("+"):
            role, _, share_text = item.strip().partition(":")
            if role not in ROLES:
                raise SourceError(f"--roles: {role!r} is none of {', '.join(ROLES)}")
            if role in seen:
                raise SourceError(f"--roles: {role} at most once")
            seen.add(role)
            share = None
            if share_text:
                try:
                    share = float(share_text)
                except ValueError:
                    raise SourceError(f"--roles: {share_text!r} is no share") from None
                if not 0 < share <= 1:
                    raise SourceError("--roles: a share is above 0 and at most 1")
            elif "+" in entry:
                raise SourceError(f"--roles: each part takes a share ({role}:0.1)")
            spec.append((role, share))
        if sum(share or 0 for _, share in spec) > 1 + 1e-9:
            raise SourceError(f"--roles: the shares of {entry!r} add up to more than 1")
        specs.append(spec)
    return specs


def default_roles(splits: list[str]) -> list[RoleSpec]:
    """Roles by the splits' names: train for train, a validation split for the
    report, test for evaluate when another split reports, else for the report."""
    names = [name.lower() for name in splits]
    roles: list[str | None] = [None] * len(splits)
    if "train" in names:
        roles[names.index("train")] = "train"
    report = next((i for i, n in enumerate(names) if n in REPORT_NAMES), None)
    if report is not None:
        roles[report] = "report"
    if "test" in names:
        roles[names.index("test")] = "evaluate" if report is not None else "report"
    return [[(role, None)] if role else [] for role in roles]


def cut(examples: list[Example], shares: list[float], seed: int | str) -> list[list[Example]]:
    """The examples cut into parts of ``shares`` (in order), each stratified by
    label (or raw value), seeded; shares adding up to 1 leave nothing out."""
    total = len(examples)
    sizes = [round(share * total) for share in shares]
    whole = abs(sum(shares) - 1) < 1e-9
    rest = list(examples)
    parts: list[list[Example]] = []
    for i, size in enumerate(sizes):
        if whole and i == len(sizes) - 1:
            parts.append(rest)
            break
        if size <= 0 or not rest:
            parts.append([])
            continue
        if size >= len(rest):
            parts.append(rest)
            rest = []
            continue
        rest, part = split_examples(rest, size / len(rest), f"{seed}:part{i}")
        parts.append(part)
    return parts


def folder_splits(
    plans: list["DataPlan"], made: list[Made], specs: list[RoleSpec]
) -> list[SplitOut]:
    """The split files of a folder: each split whole with its role, or cut into
    its parts. ``SourceError`` for a part left without examples."""
    whole_names = {
        made_one.table.split or plan.split or "train"
        for plan, made_one, spec in zip(plans, made, specs, strict=True)
        if not any(share is not None for _, share in spec)
    }
    taken = set(whole_names)
    outs = []
    for plan, made_one, spec in zip(plans, made, specs, strict=True):
        source_split = made_one.table.split or plan.split or "train"
        meta = {
            "split": made_one.table.split or plan.split,  # None for a file
            "rows": made_one.table.length,
            "sample": plan.sample,
            "seed": plan.seed if plan.sample is not None else None,
        }
        if not any(share is not None for _, share in spec):
            role = spec[0][0] if spec else None
            entry = {"role": role, **meta, "examples": len(made_one.examples)}
            outs.append(SplitOut(source_split, made_one.examples, entry))
            continue
        seed = f"{plan.seed}:{source_split}"
        parts = cut(made_one.examples, [share or 0 for _, share in spec], seed)
        for (role, share), part in zip(spec, parts, strict=True):
            if not part:
                raise SourceError(f"--roles: the {role} part of {source_split} has no examples")
            name = PART_NAMES[role]
            if name in taken:
                name = f"{source_split}-{role}"
            taken.add(name)
            entry = {"role": role, **meta, "part": share, "examples": len(part)}
            outs.append(SplitOut(name, part, entry))
    return outs


def merged_columns(made: list[Made]) -> dict[str, dict]:
    """The value columns of every split, their rows added up."""
    merged: dict[str, dict] = {}
    for made_one in made:
        for column, body in made_one.columns().items():
            values = merged.setdefault(column, {"values": {}})["values"]
            for value, info in body["values"].items():
                values.setdefault(value, {"rows": 0})["rows"] += info["rows"]
    return merged


def merged_answers(made: list[Made]) -> dict[str, dict]:
    """The answers of every split as one map per question."""
    merged: dict[str, dict] = {}
    for made_one in made:
        for question_id, answer in made_one.answers().items():
            into = merged.setdefault(question_id, {"column": answer["column"], "map": {}})
            for value, mapped in answer["map"].items():
                into["map"].setdefault(value, mapped)
    classes = list(dict.fromkeys(value for made_one in made for value in made_one.classes))
    for answer in merged.values():
        answer["left_out"] = [value for value in classes if value not in answer["map"]]
    return merged


def source_of(plan: "DataPlan", made: Made) -> dict[str, Any]:
    """Where a folder's rows came from (no empty fields)."""
    origin = made.table.origin
    preset = preset_data.PRESETS.get(plan.preset) if plan.preset else None
    body = {
        "source": HUB_PREFIX + preset.source.repo if preset else plan.source,
        "preset": plan.preset,
        "config": origin.get("config", plan.config),
        "revision": origin.get("revision") or plan.revision,
        "shape": made.mapping.shape if made.mapping else None,
        "columns": {k: v for k, v in made.mapping.columns.items() if v} if made.mapping else None,
        "label_map": plan.label_map,
        "limit": plan.limit,
    }
    return {key: value for key, value in body.items() if value is not None}


def write_folder(
    plans: list["DataPlan"], made: list[Made], specs: list[RoleSpec] | None, before_rename=None
) -> list[SplitOut]:
    """Write the made splits as the dataset folder ``-o`` names (``specs``: the
    roles, else by the splits' names); returns the split files written."""
    if specs is None:
        names = [m.table.split or p.split or "train" for p, m in zip(plans, made, strict=True)]
        specs = default_roles(names)
    outs = folder_splits(plans, made, specs)
    columns = merged_columns(made)
    names = plans[0].value_names or {}
    for body in columns.values():
        for value, info in body["values"].items():
            if names.get(value):
                info["name"] = names[value]
    dataset = Dataset(
        Path(plans[0].output),
        answers=merged_answers(made),
        columns=columns,
        source=source_of(plans[0], made[0]),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    write_dataset(plans[0].output, dataset, outs, before_rename)
    return outs


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
    value_names: dict[str, str] | None = None  # names of the label's values (a folder)

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

    @property
    def folder(self) -> bool:
        """Whether the output is a dataset folder."""
        return is_folder_output(self.output)

    @property
    def raw(self) -> bool:
        """Rows without a question: into a folder with no wording given."""
        return self.folder and not self.has_wording()

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
            raw=self.raw,
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
            ("--value-names", join_pairs(self.value_names) if self.value_names else None),
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
