"""Dataset shapes: how a table's columns become one labelled question per row.

- ``text``: a text column is the state, a class column the answer of a
  ``choice`` (or of a ``noul`` for two classes).
- ``pair``: premise and hypothesis together are the state (an object), the
  class column the answer of a ``choice`` (NLI).
- ``choice``: multiple choice; the question (with a context column, if
  any) is the state, the row's choices are the options, the answer column
  names the right one.
- ``yesno``: a passage is the state, the row's question a ``noul``, a
  yes / no column the answer.
- ``rating``: a text column is the state, a numeric (or ordered class)
  column the level of a ``score``.
- ``object``: several columns of any kind together are the state, an object
  keyed by their names (a class label's cells by name), a class column the
  answer as in ``text``. Its columns are named (``--columns A,B``), never
  guessed; naming them picks the shape.

``guess`` finds the shape and the columns from the kinds and common names,
taking what the caller gives; a guess that is not clear stops with the
columns listed. ``Wording`` holds the question's id, instructions, options
and label map; ``build`` turns rows into examples. The text, pair, rating and
object shapes keep each row's raw label in the example's ``values`` (for a
dataset folder); with ``raw`` they ask no question and keep only that.
"""

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..flags import join_list, split_list
from .format import Example
from .sources import BOOL, FLOAT, INT, LABEL, LIST, OBJECT, OTHER, TEXT, Column, SourceError, Table

SHAPES = ("text", "pair", "choice", "yesno", "rating", "object")
# The shapes whose rows keep their label's raw value (and may ask nothing).
VALUE_SHAPES = ("text", "pair", "rating", "object")
# The columns each shape reads, and the flag that names each.
ROLES = {
    "text": ("text", "label"),
    "pair": ("premise", "hypothesis", "label"),
    "choice": ("question", "choices", "answer", "context"),
    "yesno": ("text", "question", "label"),
    "rating": ("text", "label"),
    "object": ("columns", "label"),
}
OPTIONAL = {"context"}
FLAGS = {
    "text": "--text",
    "columns": "--columns",
    "label": "--label",
    "premise": "--premise",
    "hypothesis": "--hypothesis",
    "question": "--question-column",
    "choices": "--choices",
    "answer": "--answer",
    "context": "--context",
}
# Common column names per role, most likely first.
NAMES = {
    "text": ("text", "sentence", "content", "review", "tweet", "document", "comment", "body",
             "utterance", "passage", "paragraph"),
    "label": ("label", "labels", "coarse_label", "class", "category", "target", "answer",
              "sentiment", "stars", "rating", "score"),
    "premise": ("premise", "sentence1", "text_a"),
    "hypothesis": ("hypothesis", "sentence2", "text_b"),
    "question": ("question", "question_stem", "query", "goal", "stem"),
    "choices": ("choices", "options", "endings"),
    "answer": ("answerkey", "answer_key", "answer", "label", "correct"),
    "context": ("context", "fact1", "passage", "article", "support"),
    "columns": (),
}  # fmt: skip
# The kinds a role's column may have.
KINDS = {
    "text": (TEXT,),
    "label": (LABEL, INT, TEXT, BOOL, FLOAT),
    "premise": (TEXT,),
    "hypothesis": (TEXT,),
    "question": (TEXT,),
    "choices": (LIST, OBJECT),
    "answer": (TEXT, INT, LABEL, OTHER),
    "context": (TEXT,),
    "columns": (TEXT, INT, FLOAT, BOOL, LABEL, LIST, OBJECT, OTHER),
}
# A rating's label column, when its name says so.
RATING_NAMES = ("stars", "rating", "score")
# The most levels a score question takes (the Jev API's limit).
MAX_LEVELS = 10


@dataclass(frozen=True)
class Mapping:
    """A shape and the column of each of its roles (None: an optional one left out)."""

    shape: str
    columns: dict[str, str | None]


def candidates(table: Table, role: str, used: set[str]) -> list[str]:
    fitting = [c for c in table.columns.values() if c.kind in KINDS[role] and c.name not in used]
    named = [c.name for c in fitting if c.name.lower() in NAMES[role]]
    if named:
        return named
    if role == "label":
        return [c.name for c in fitting if c.kind == LABEL]
    if role == "text":
        return [c.name for c in fitting]
    return []


def state_columns(value: str) -> list[str]:
    """The object shape's columns from ``--columns`` (``A,B``; ``\\,`` for a comma)."""
    return [name for name in split_list(value) if name]


def _resolve(table: Table, shape: str, given: dict[str, str]) -> Mapping:
    columns: dict[str, str | None] = {}
    used: set[str] = set()
    if shape == "object":
        names = state_columns(given.get("columns") or "")
        if not names:
            raise SourceError(
                f"the object shape needs the state's columns: pass --columns A,B "
                f"(columns: {table.listing()})"
            )
        for name in names:
            if name not in table.columns:
                raise SourceError(
                    f"--columns: no column {name!r}; the columns are {table.listing()}"
                )
        if len(set(names)) < len(names):
            raise SourceError("--columns: a column is named twice")
        columns["columns"] = join_list(names)
        used.update(names)
        if given.get("label") in used:
            raise SourceError("--label: the label is not one of the state's columns")
    for role in ROLES[shape]:
        if role in columns:
            continue
        name = given.get(role)
        if name is not None:
            if name not in table.columns:
                raise SourceError(
                    f"{FLAGS[role]}: no column {name!r}; the columns are {table.listing()}"
                )
            columns[role] = name
            used.add(name)
    for role in ROLES[shape]:
        if role in columns:
            continue
        found = candidates(table, role, used)
        if len(found) == 1:
            columns[role] = found[0]
            used.add(found[0])
        elif not found and role in OPTIONAL:
            columns[role] = None
        else:
            which = "is not clear" if found else "is not found"
            listed = ", ".join(found) if found else table.listing()
            raise SourceError(
                f"the {role} column of the {shape} shape {which}; pass {FLAGS[role]} "
                f"({'one of ' + listed if found else 'columns: ' + listed})"
            )
    return Mapping(shape, {role: columns[role] for role in ROLES[shape]})


def guess(table: Table, shape: str | None = None, given: dict[str, str] | None = None) -> Mapping:
    """The shape (``shape`` if given) and its columns (``given`` by role)."""
    given = {role: name for role, name in (given or {}).items() if name is not None}
    if shape is None and given.get("columns"):
        shape = "object"  # its columns are named, never guessed
    if shape is not None:
        if shape not in SHAPES:
            raise SourceError(f"--shape: no shape {shape!r}; the shapes are {', '.join(SHAPES)}")
        return _resolve(table, shape, given)
    for role in given:
        if not any(role in ROLES[s] for s in SHAPES):
            raise SourceError(f"no shape reads a {role} column")
    names = {name.lower(): column for name, column in table.columns.items()}
    tried = []
    if any(n in names for n in NAMES["premise"]) and any(n in names for n in NAMES["hypothesis"]):
        tried.append("pair")
    if any(c.kind in KINDS["choices"] and c.name.lower() in NAMES["choices"]
           for c in table.columns.values()):  # fmt: skip
        tried.append("choice")
    if any(n in names for n in NAMES["question"]) and any(
        c.kind == BOOL for c in table.columns.values()
    ):
        tried.append("yesno")
    if not tried:
        label = next((names[n] for n in NAMES["label"] if n in names), None)
        if label is not None and label.kind in (INT, FLOAT) and label.name.lower() in RATING_NAMES:
            tried.append("rating")
        else:
            tried.append("text")
    if len(tried) > 1:
        raise SourceError(
            f"the shape is not clear ({' or '.join(tried)}); pass --shape; "
            f"the columns are {table.listing()}"
        )
    return _resolve(table, tried[0], given)


@dataclass
class Wording:
    """The question every row asks: its id, instructions, option ids with
    descriptions (in order), raw labels mapped to option ids, whether two
    classes become a ``noul``, a rating's level descriptions, or a whole
    question (``-q``: type, instructions, criteria)."""

    id: str
    instructions: Any = None
    options: dict[str, Any] | None = None
    label_map: dict[str, str] | None = None
    noul: bool = False
    levels: list[str] | None = None
    question: dict[str, Any] | None = None


@dataclass
class Built:
    """Examples built from rows, and the rows left out with why (counted)."""

    examples: list[Example] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)


def _raw(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def class_names(column: Column, table: Table, label_map: dict[str, str] | None) -> list[str]:
    """The raw classes of a label column, in order: a class label's names, else
    the label map's keys, else the distinct values sorted."""
    if column.names:
        return list(column.names)
    if label_map:
        return list(label_map)
    values = {v for v in table.column(column.name) if v is not None and v != ""}
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    return [_raw(v) for v in sorted(values, key=float if numeric else str)]


def class_of(column: Column, value: Any) -> str | None:
    """A row's raw class: a class label's name (None for -1, no label), else the value."""
    if value is None or value == "":
        return None
    if column.names:
        if isinstance(value, int) and 0 <= value < len(column.names):
            return column.names[value]
        return None
    return _raw(value)


def _lookup(label_map: dict[str, str] | None, column: Column, value: Any, name: str | None):
    """The option id of a raw label: by the map (its index or its name), else the name."""
    if label_map:
        for key in (_raw(value), name):
            if key is not None and key in label_map:
                return label_map[key]
        return None
    return name


def _classes(table: Table, mapping: Mapping, wording: Wording):
    """The options of a class question: ``(option ids, descriptions)`` in order."""
    column = table.columns[mapping.columns["label"]]
    names = class_names(column, table, wording.label_map)
    if wording.options:
        return list(wording.options), dict(wording.options)
    if wording.label_map:
        ids = list(dict.fromkeys(wording.label_map.values()))
        return ids, {option: option for option in ids}
    return names, {name: name for name in names}


def _class_question(table: Table, mapping: Mapping, wording: Wording) -> tuple[dict, list[str]]:
    """The question of a text, pair or object shape and its option ids."""
    if wording.question is not None:
        question = dict(wording.question)
        kind = question.get("type")
        if kind == "noul":
            return question, ["true", "false"]
        if kind == "choice":
            return question, list(question.get("criteria") or {})
        raise SourceError(f"-q: a {mapping.shape} question is a choice or a noul, not {kind!r}")
    ids, descriptions = _classes(table, mapping, wording)
    if wording.noul:
        if len(ids) != 2:
            raise SourceError(
                f"--noul needs two classes; the label has {len(ids)}: {', '.join(ids)}"
            )
        true, false = ("true", "false") if set(ids) == {"true", "false"} else (ids[1], ids[0])
        question = {"type": "noul", "instructions": wording.instructions}
        question["criteria"] = {"true": descriptions[true], "false": descriptions[false]}
        return question, [true, false]
    question = {"type": "choice", "instructions": wording.instructions, "criteria": descriptions}
    return question, ids


def _require_instructions(wording: Wording, shape: str) -> None:
    if wording.question is None and wording.instructions is None:
        raise SourceError(
            f"the question's wording is open: pass --question TEXT (or -q FILE) for the "
            f"{shape} shape"
        )


def _choice_options(value: Any) -> tuple[list[str], list[Any]] | None:
    """A row's choices as option ids and texts: ``{"label": [...], "text": [...]}``
    keeps its labels, a plain list gets A, B, C, ..."""
    if isinstance(value, dict) and "text" in value:
        texts = list(value["text"])
        labels = value.get("label") or list(string.ascii_uppercase[: len(texts)])
        return [str(label) for label in labels], texts
    if isinstance(value, list):
        return list(string.ascii_uppercase[: len(value)]), list(value)
    return None


def _choice_answer(answer: Any, ids: list[str], texts: list[Any]) -> str | None:
    """The answer as an option id: an id, a position, or a choice's own text."""
    if isinstance(answer, bool) or answer is None:
        return None
    if isinstance(answer, int) or (
        isinstance(answer, str) and answer.isdigit() and answer not in ids
    ):
        index = int(answer)
        return ids[index] if 0 <= index < len(ids) else None
    if str(answer) in ids:
        return str(answer)
    if answer in texts:
        return ids[texts.index(answer)]
    return None


def _yes_no(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _raw(value).strip().lower()
    if text in ("true", "yes", "1"):
        return "true"
    if text in ("false", "no", "0"):
        return "false"
    return None


def rating_values(table: Table, column: Column) -> list[str]:
    """A rating column's values in order, one level each: a class label's names,
    whole numbers from the lowest to the highest, or the distinct values."""
    raws = class_names(column, table, None)
    if column.kind == INT and raws:
        # Whole-number ratings keep every level between the lowest and highest.
        low, high = int(raws[0]), int(raws[-1])
        if high - low + 1 > MAX_LEVELS:
            raise SourceError(
                f"{column.name} runs from {low} to {high}: more than {MAX_LEVELS} levels"
            )
        raws = [str(value) for value in range(low, high + 1)]
    return raws


def _levels(table: Table, mapping: Mapping, wording: Wording) -> tuple[list[str], list[Any]]:
    """A rating's raw values in order and the level descriptions."""
    column = table.columns[mapping.columns["label"]]
    if column.kind not in (INT, FLOAT, LABEL):
        raise SourceError(
            f"the rating shape needs a numeric or class label column; {column.name} is "
            f"{column.kind}"
        )
    if wording.label_map or wording.options:
        raise SourceError(
            "a rating's levels are its values in order; use --levels for their descriptions"
        )
    raws = rating_values(table, column)
    if wording.levels is not None:
        if len(wording.levels) != len(raws):
            raise SourceError(
                f"--levels: {len(wording.levels)} given for {len(raws)} values ({', '.join(raws)})"
            )
        return raws, list(wording.levels)
    return raws, list(raws)


def _example(
    table: Table,
    index: int,
    state,
    question_id: str,
    question: dict | None,
    label: str | None,
    values: dict[str, str] | None = None,
):
    """The example of row ``index``: id from the table's name and split, one
    question (none for a raw row: its ``values`` only)."""
    name = f"{table.name}:{table.split}/{index}" if table.split else f"{table.name}/{index}"
    if question is None:
        return Example(name, {"state": state}, {}, values)
    return Example(
        name, {"state": state, "questions": {question_id: question}}, {question_id: label}, values
    )


def _state(table: Table, shape: str, columns: dict, row: dict) -> Any:
    if shape == "pair":
        return {"premise": row[columns["premise"]], "hypothesis": row[columns["hypothesis"]]}
    if shape == "object":
        state = {}
        for name in state_columns(columns["columns"]):
            column, value = table.columns[name], row.get(name)
            state[name] = class_of(column, value) if column.names else value
        return state
    return row[columns["text"]]


def _raw_rows(table: Table, mapping: Mapping, indices: list[int], rows: list[dict]) -> Built:
    """Rows as examples without a question: the state and the label's raw value."""
    built = Built()
    column_name = mapping.columns["label"]
    label_column = table.columns[column_name]
    for index, row in zip(indices, rows, strict=True):
        name = class_of(label_column, row.get(column_name))
        if name is None:
            built.skipped["no label"] += 1
            continue
        state = _state(table, mapping.shape, mapping.columns, row)
        built.examples.append(_example(table, index, state, "", None, None, {column_name: name}))
    return built


def build(
    table: Table, mapping: Mapping, wording: Wording, indices: list[int], raw: bool = False
) -> Built:
    """The examples of the rows at ``indices``; rows without a usable label are
    left out and counted by reason. ``raw``: no question (text, pair, rating
    and object shapes), each row's state and label value only."""
    shape, columns = mapping.shape, mapping.columns
    rows = table.rows(indices)
    if raw:
        if shape not in VALUE_SHAPES:
            raise SourceError(
                f"the {shape} shape needs its question's wording: pass --question TEXT (or -q FILE)"
            )
        return _raw_rows(table, mapping, indices, rows)
    built = Built()
    if shape in ("text", "pair", "object"):
        _require_instructions(wording, shape)
        question, ids = _class_question(table, mapping, wording)
        label_column = table.columns[columns["label"]]
        for index, row in zip(indices, rows, strict=True):
            value = row.get(columns["label"])
            name = class_of(label_column, value)
            option = _lookup(wording.label_map, label_column, value, name) if name else None
            if option is None:
                built.skipped["no label" if name is None else f"label {name!r} unmapped"] += 1
                continue
            if option not in ids:
                raise SourceError(
                    f"row {index}: label {name!r} is no option of the question "
                    f"({', '.join(ids)}); map it with --label-map"
                )
            if question["type"] == "noul":
                option = "true" if option == ids[0] else "false"
            state = _state(table, shape, columns, row)
            built.examples.append(
                _example(
                    table, index, state, wording.id, question, option, {columns["label"]: name}
                )
            )
        return built
    if shape == "choice":
        _require_instructions(wording, shape)
        instructions = (wording.question or {}).get("instructions", wording.instructions)
        for index, row in zip(indices, rows, strict=True):
            options = _choice_options(row.get(columns["choices"]))
            if options is None:
                built.skipped["no choices"] += 1
                continue
            ids, texts = options
            option = _choice_answer(row.get(columns["answer"]), ids, texts)
            if option is None:
                built.skipped["no answer"] += 1
                continue
            stem = row[columns["question"]]
            context = row.get(columns["context"]) if columns.get("context") else None
            state = {"context": context, "question": stem} if context else stem
            question = {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(zip(ids, texts, strict=True)),
            }
            built.examples.append(_example(table, index, state, wording.id, question, option))
        return built
    if shape == "yesno":
        if wording.instructions is not None or wording.question is not None:
            raise SourceError("the yesno shape asks each row's question; drop --question and -q")
        for index, row in zip(indices, rows, strict=True):
            option = _yes_no(row.get(columns["label"]))
            if option is None:
                built.skipped["no yes / no label"] += 1
                continue
            question = {"type": "noul", "instructions": row[columns["question"]]}
            built.examples.append(
                _example(table, index, row[columns["text"]], wording.id, question, option)
            )
        return built
    # rating
    _require_instructions(wording, shape)
    raws, descriptions = _levels(table, mapping, wording)
    question = {"type": "score", "instructions": wording.instructions, "criteria": descriptions}
    if wording.question is not None:
        question = dict(wording.question)
        if question.get("type") != "score" or len(question.get("criteria") or []) != len(raws):
            raise SourceError(f"-q: a rating question is a score with {len(raws)} levels")
    label_column = table.columns[columns["label"]]
    for index, row in zip(indices, rows, strict=True):
        value = class_of(label_column, row.get(columns["label"]))
        if value is None or value not in raws:
            built.skipped["no label"] += 1
            continue
        label = str(raws.index(value))
        values = {columns["label"]: value}
        built.examples.append(
            _example(table, index, row[columns["text"]], wording.id, question, label, values)
        )
    return built


def safe_id(text: str, fallback: str) -> str:
    """``text`` as an id: lower case, each run of characters other than letters,
    digits, '.' and '-' an underscore; ``fallback`` when nothing is left."""
    return re.sub(r"[^\w.\-]+", "_", text).strip("_").lower() or fallback


def safe_ids(names: list[str], fallback: str = "option") -> list[str]:
    """Each name as an id (``safe_id``), distinct: a repeated one gets ``_2``,
    ``_3``, ... as the Playground numbers a copy."""
    ids: list[str] = []
    for name in names:
        base = safe_id(name, fallback)
        candidate, n = base, 2
        while candidate in ids:
            candidate, n = f"{base}_{n}", n + 1
        ids.append(candidate)
    return ids


def default_id(table: Table) -> str:
    """A question id from the dataset's or file's name."""
    return safe_id(table.name, "question")
