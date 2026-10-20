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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
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


# The record beside a data file NAME.jsonl: NAME.source.json.
RECORD_SUFFIX = ".source.json"
# What a file of a Get is for: training, the fit's report, Evaluate.
ROLES = ("train", "report", "evaluate")
SPLIT_FIELD = "{split}"  # in -o, each split's name


def record_path(output: str | Path) -> Path:
    """Where the record of the data file ``output`` goes."""
    output = Path(output)
    stem = output.name.removesuffix(".jsonl") if output.name.endswith(".jsonl") else output.stem
    return output.with_name(stem + RECORD_SUFFIX)


def file_name(output: str | Path) -> str:
    """A data file's name as the Studio lists it: without ``.jsonl``."""
    name = Path(output).name
    return name.removesuffix(".jsonl")


def get_name(pattern: str) -> str:
    """The name of a Get's files without their split: ``trec`` for
    ``data/trec-{split}.jsonl``."""
    name = file_name(pattern).replace(SPLIT_FIELD, "")
    return name.strip("-_.") or name


def record(
    plan: "DataPlan", made: Made, *, role: str | None, files: list[dict[str, Any]], name: str
) -> dict[str, Any]:
    """Where a data file came from: the source (a preset's dataset on the Hub,
    a Hub dataset, a file), config, revision, split, the split's rows, the
    sample or limit and seed, the examples written, the question and class map
    when given, the file's role, and the files of the same Get (``name``
    without their split) with their splits and roles."""
    origin = made.table.origin
    preset = preset_data.PRESETS.get(plan.preset) if plan.preset else None
    source = HUB_PREFIX + preset.source.repo if preset else plan.source
    question = None
    if plan.question is not None and plan.question_id:
        question = {plan.question_id: plan.question}
    return {
        "source": source,
        "preset": plan.preset,
        "config": origin.get("config", plan.config),
        "revision": origin.get("revision") or plan.revision,
        "split": made.table.split or plan.split,
        "rows": made.table.length,
        "sample": plan.sample,
        "limit": plan.limit,
        "seed": plan.seed if plan.sample is not None else None,
        "examples": len(made.examples),
        "labels": made.label_counts(),
        "question": question,
        "label_map": plan.label_map,
        "role": role,
        "get": {"name": name, "files": files},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def read_record(path: str | Path) -> dict[str, Any] | None:
    """The record beside the data file ``path``, or None (none, or unreadable)."""
    try:
        loaded = json.loads(record_path(path).read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def split_plans(
    plan: "DataPlan", splits: list[str | None], samples: list[int] | None
) -> list["DataPlan"]:
    """One plan per split: ``samples`` once for every split, or once per split
    (0: the whole split); ``{split}`` in the output replaced by each split's
    name. ``SourceError`` for what does not go together."""
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
    if plan.output is not None:
        if many and SPLIT_FIELD not in plan.output:
            raise SourceError(
                f"several --split: put {SPLIT_FIELD} in -o (-o data/NAME-{SPLIT_FIELD}.jsonl)"
            )
        if SPLIT_FIELD in plan.output and splits == [None]:
            raise SourceError(f"-o has {SPLIT_FIELD}: name the split with --split")
    result = []
    for split, sample in zip(splits, samples, strict=True):
        output = plan.output
        if output is not None and split is not None:
            output = output.replace(SPLIT_FIELD, split)
        result.append(
            replace(
                plan, split=split, sample=sample or None, output=output,
                output_pattern=plan.output,
            )
        )  # fmt: skip
    return result


def partial_path(path: str | Path) -> Path:
    """Where a file is written before it is complete: ``.NAME.partial`` beside it."""
    path = Path(path)
    return path.with_name(f".{path.name}.partial")


def leftovers(plans: list["DataPlan"], with_records: bool) -> list[Path]:
    """The partial files a run of ``plans`` writes (removed after a stop)."""
    found = []
    for plan in plans:
        if plan.output is not None:
            found.append(partial_path(plan.output))
            if with_records:
                found.append(partial_path(record_path(plan.output)))
    return found


def write_made(
    plans: list["DataPlan"],
    made: list[Made],
    *,
    with_records: bool = False,
    roles: list[str] | None = None,
    before_rename=None,
) -> list[Path]:
    """Write each plan's examples (and, ``with_records``, its record) to a
    partial file, then rename them all: the files appear together or not at
    all. ``before_rename`` (when given) runs before the renames and may raise
    to stop. An ``OSError`` removes the partials and passes on."""
    from .format import write_examples

    outputs = [Path(plan.output) for plan in plans]
    written: list[tuple[Path, Path]] = []
    try:
        files = [
            {"name": file_name(output), "split": plan.split, "role": roles[i] if roles else None}
            for i, (plan, output) in enumerate(zip(plans, outputs, strict=True))
        ]
        name = get_name(plans[0].output_pattern or plans[0].output)
        for i, (plan, made_one, output) in enumerate(zip(plans, made, outputs, strict=True)):
            partial = partial_path(output)
            write_examples(partial, made_one.examples)
            written.append((partial, output))
            if with_records:
                target = record_path(output)
                body = record(plan, made_one, role=files[i]["role"], files=files, name=name)
                partial = partial_path(target)
                partial.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
                written.append((partial, target))
        if before_rename is not None:
            before_rename()
        for partial, target in written:
            partial.replace(target)
    except BaseException:
        for partial, _ in written:
            partial.unlink(missing_ok=True)
        raise
    return outputs


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
    output_pattern: str | None = None  # -o with {split}, when split into plans

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
