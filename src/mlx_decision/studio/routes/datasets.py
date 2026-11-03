"""Public datasets for Download on the Datasets page (TRS-6): the ``data`` command's
presets, and any Hugging Face dataset previewed.

``GET /studio/api/datasets`` lists the presets: each with its title and
what it asks, the dataset on the Hub (repo, config, the pinned commit), its
question as the training data will have it (built by the preset's own
code) and the splits it can be read from with their rows, the coverage
suite's evaluation split marked.

``POST /studio/api/datasets/preview`` ``{source, config, split}`` reads any
``hf:ORG/NAME``: first its configs, splits and each split's rows and size
(the Hub's metadata, nothing read), then, with both settled, the split
itself (in a thread): a small one whole (into the Hugging Face cache), a
large one (or one of unknown size) by its first rows, streamed, nothing
stored. The answer has the columns with their kinds, a few rows, the text
columns that can be the state and the label columns with each value and
its count (in the rows read; and the values' names when the dataset card
lists them, as ``"LABEL_0": "Analyst Update"``). Downloading either is a
``data`` job (``POST /studio/api/jobs``) that writes a dataset folder.
"""

import asyncio
import json
from collections import Counter
from typing import Any

from fastapi import APIRouter, Request

from ...data import sources
from ...data.presets import PRESETS, SPLITS, describe
from ...data.shapes import class_names, class_of, guess
from ...data.sources import HUB_PREFIX, LABEL, TEXT, SourceError, Table
from . import ApiError, field, json_body
from .jobs import MAX_CLASSES, REPO_ID

ROWS_SHOWN = 5
CELL_CHARS = 160
# A split whose parquet copy on the Hub is at most this is read whole for the
# preview; a larger one, or one of unknown size, by its first HEAD_ROWS rows.
WHOLE_BYTES = 50 * 10**6
HEAD_ROWS = 5000
# A label column has at most MAX_CLASSES classes, and repeats them (more, or
# about one value a row, is free text or ids, not labels).

# How the sheet names each preset, and its task in a few words (the question
# and its options, shown beside it, say the rest).
TITLES = {
    "trec": ("TREC", "Question classification"),
    "offensive": ("TweetEval Offensive", "Offensive tweets"),
    "sst5": ("SST-5", "Movie review sentiment"),
    "openbookqa": ("OpenBookQA", "Elementary science questions"),
    "anli-r3": ("ANLI round 3", "Natural language inference"),
}
# The split a training file is read from by default.
TRAIN_SPLITS = {"anli-r3": "train_r3"}


def datasets_router(max_body_bytes: int = 2**20) -> APIRouter:
    router = APIRouter(prefix="/studio/api/datasets")

    @router.get("")
    async def presets():
        """``{presets}``: ``name``, ``title``, ``summary``, ``repo``, ``config``,
        ``revision``, ``question`` (``id``, ``type``, ``instructions``,
        ``options`` as ``[id, description]``), ``splits`` (``name``, ``rows``,
        ``evaluation``) and ``default_split``."""
        listed = []
        for name in ("trec", "offensive", "sst5", "openbookqa", "anli-r3"):
            preset = PRESETS[name]
            question = describe(preset)
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                options = [[option, text] for option, text in criteria.items()]
            else:  # a score's levels, lowest first
                options = [[str(level), text] for level, text in enumerate(criteria or [])]
            title, summary = TITLES[name]
            listed.append(
                {
                    "name": name,
                    "title": title,
                    "summary": summary,
                    "repo": preset.source.repo,
                    "config": preset.source.config,
                    "revision": preset.source.revision,
                    "question": {
                        "id": question["id"],
                        "type": question["type"],
                        "instructions": question.get("instructions"),
                        "options": options,
                    },
                    "splits": [
                        {"name": split, "rows": rows, "evaluation": split == preset.source.split}
                        for split, rows in SPLITS[name].items()
                    ],
                    "default_split": TRAIN_SPLITS.get(name, "train"),
                }
            )
        return {"presets": listed}

    @router.post("/preview")
    async def preview(request: Request):
        """``{source, config, split}`` -> ``{source, configs, config, splits,
        sizes, split}`` (``sizes``: per split its ``rows``, ``bytes`` and
        ``partial`` as the Hub's metadata gives them, and ``whole``: whether
        the preview reads it whole; a split the Hub says nothing of left
        out) and, once config and split are settled, ``rows`` (the split's
        length; None when only its first rows were read and the Hub does not
        say), ``sampled`` (the rows read when not all; else None),
        ``columns`` (``name``, ``kind``, ``names``), ``sample`` (the first
        rows, cells cut short), ``text_columns`` and ``label_columns``
        (``name``, ``classes`` as ``[class, count]`` over the rows read),
        with the guessed ``text`` and ``label``. 400 for a source that is no ``hf:ORG/NAME``,
        404 for one the Hub does not show, 422 for one that cannot be read."""
        body = await json_body(request, max_body_bytes)
        source = field(body, "source", str).strip()
        repo = source.removeprefix(HUB_PREFIX)
        if not source.startswith(HUB_PREFIX) or not REPO_ID.fullmatch(repo) or ".." in repo:
            raise ApiError(400, "give a Hugging Face dataset as hf:ORG/NAME", param="source")
        config = field(body, "config", str, required=False)
        split = field(body, "split", str, required=False)
        try:
            return await asyncio.to_thread(read_preview, repo, config, split)
        except SourceError as error:
            message = str(error)
            status = 404 if "doesn't exist" in message or "not found" in message.lower() else 422
            raise ApiError(status, message, param="source") from None

    return router


def read_preview(repo: str, config: str | None, split: str | None) -> dict[str, Any]:
    """What the preview route answers (blocks: the Hub, the dataset's split)."""
    source = f"{HUB_PREFIX}{repo}"
    configs = sources.hub_configs(repo)
    answer: dict[str, Any] = {"source": source, "configs": configs, "config": None}
    if configs:
        if config is None and len(configs) == 1:
            config = configs[0]
        if config is None:
            return answer | {"splits": None, "split": None}
        if config not in configs:
            raise SourceError(f"{source} has no subset {config!r}; it has {', '.join(configs)}")
    answer["config"] = config
    try:
        splits = sources.hub_splits(repo, config)
    except Exception as error:  # the Hub's errors vary by version
        raise SourceError(f"{source}: cannot list its splits: {error}") from None
    answer["splits"] = splits
    sizes = sources.hub_split_sizes(repo, config)
    answer["sizes"] = {
        name: {"rows": size.rows, "bytes": size.bytes, "partial": size.partial,
               "whole": read_whole(size)}
        for name in splits
        if (size := sizes.get(name)) is not None
    }  # fmt: skip
    if split is None and len(splits) == 1:
        split = splits[0]
    if split is None:
        return answer | {"split": None}
    if split not in splits:
        raise SourceError(f"{source} has no split {split!r}; it has {', '.join(splits)}")
    size = sizes.get(split)
    sampled = None
    if read_whole(size):
        table = sources.load_hub(repo, config, split)
        rows = table.length
    else:
        table = sources.load_hub_head(repo, config, split, HEAD_ROWS)
        if table.length < HEAD_ROWS:
            rows = table.length  # that was all of it
        else:
            rows, sampled = (size.rows if size else None), table.length
    described = describe_table(table) | {"rows": rows, "sampled": sampled}
    card = None
    for column in described["label_columns"]:
        if column["kind"] == LABEL:
            continue  # a class label names its classes itself
        card = card if card is not None else (sources.hub_card(repo) or "")
        values = [value for value, _ in column["classes"]]
        column["names"] = sources.names_from_card(card, values) if card else None
    return answer | {"split": split, **described}


def read_whole(size: sources.SplitSize | None) -> bool:
    """Whether the preview reads a split whole: its size is known and small."""
    return (
        size is not None
        and not size.partial
        and size.bytes is not None
        and (size.bytes <= WHOLE_BYTES)
    )


def describe_table(table: Table) -> dict[str, Any]:
    """The columns, a few rows and the columns that can be the state or the label."""
    columns = [
        {"name": column.name, "kind": column.kind, "names": column.names}
        for column in table.columns.values()
    ]
    rows = table.rows(list(range(min(ROWS_SHOWN, table.length))))
    # A class label's cells by name, as the classes are mapped below.
    sample = [
        {
            name: cell(class_of(column, row.get(name)) if column.names else row.get(name))
            for name, column in table.columns.items()
        }
        for row in rows
    ]
    text_columns = [column.name for column in table.columns.values() if column.kind == TEXT]
    label_columns = []
    for column in table.columns.values():
        if column.kind not in (LABEL, TEXT, "int", "bool"):
            continue
        classes = class_names(column, table, None)
        if not classes or len(classes) > MAX_CLASSES:
            continue
        if column.kind != LABEL and len(classes) > max(2, table.length // 2):
            continue  # nearly every row its own value: free text or ids, not classes
        counts = Counter(class_of(column, value) for value in table.column(column.name))
        counts.pop(None, None)
        label_columns.append(
            {
                "name": column.name,
                "kind": column.kind,
                "classes": [[name, counts.get(name, 0)] for name in classes],
                "names": None,  # the values' names, from the dataset card
            }
        )
    labels = [column["name"] for column in label_columns]
    text = next((name for name in text_columns if name not in labels), None)
    text = next((name for name in ("text", "sentence", "content") if name in text_columns), text)
    if text is None and text_columns:
        # Every text column repeats its values (a small file): the one with the
        # most is the state.
        values = {column["name"]: len(column["classes"]) for column in label_columns}
        text = max(text_columns, key=lambda name: values.get(name, MAX_CLASSES + 1))
    others = [name for name in labels if name != text]
    label = next(
        (name for name in ("label", "labels", "class", "category") if name in others),
        others[0] if others else None,
    )
    return {
        "rows": table.length,
        "columns": columns,
        "sample": sample,
        "text_columns": text_columns,
        "label_columns": label_columns,
        "text": text,
        "label": label,
        "guess": guessed(table, labels),
    }


def guessed(table: Table, labels: list[str]) -> dict[str, Any] | None:
    """The sheet's shape as the ``data`` command guesses it: ``{shape, ...}``
    with "text" (``text``, ``label``), "object" (``columns``, ``label``; a
    pair's or a yes / no passage's columns) or "choice" (``question``,
    ``choices``, ``answer``, ``context``); None when the guess is not clear."""
    try:
        mapping = guess(table)
    except SourceError:
        return None
    columns = mapping.columns
    label = columns.get("label")
    if label is not None and label not in labels:
        label = None  # not a column of a few values: the sheet asks for one
    if mapping.shape in ("text", "rating"):
        return {"shape": "text", "text": columns["text"], "label": label}
    if mapping.shape == "pair":
        state = [columns["premise"], columns["hypothesis"]]
        return {"shape": "object", "columns": state, "label": label}
    if mapping.shape == "yesno":
        return {
            "shape": "object",
            "columns": [columns["text"], columns["question"]],
            "label": label,
        }
    if mapping.shape == "choice":
        return {
            "shape": "choice",
            **{role: columns.get(role) for role in ("question", "choices", "answer", "context")},
        }
    return None


def cell(value: Any) -> str:
    """A cell as the preview shows it, cut short."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= CELL_CHARS else text[: CELL_CHARS - 1] + "…"
