"""Presets: five public classification datasets turned into decision requests.

TREC-6 (answer type, ``choice``), TweetEval Offensive (``noul``), ANLI round 3
(``choice``), OpenBookQA (``choice``) and SST-5 (sentiment, ``score``), each
pinned to a Hub commit, with the question wording, option descriptions and
label maps fixed here. The same code builds the coverage suite of
``scripts/accuracy_benchmark.py`` (the evaluation splits) and the ``data``
command's presets (any split), so the two cannot drift apart.

An example is a dict: ``id`` (stable: the preset's key and the split's own id
or row index), ``state``, ``questions`` (one question, in the wire format) and
``gold`` (the expected option id; for ``noul`` "true" or "false", for
``score`` the level as a string). Rows are plain dicts as the ``datasets``
library yields them; this module does not import it.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

# TREC's six coarse answer types, in its label order (ABBR, ENTY, DESC, HUM, LOC, NUM).
TREC_TYPES = {
    "abbreviation": "An abbreviation, or what an abbreviation stands for",
    "entity": "A thing: an object, animal, plant, color, event, food, product, "
    "substance, language, term or other entity",
    "description": "A description, definition, manner or reason",
    "human": "A person or a group of people, or a description of one",
    "location": "A place: a city, country, state, mountain or other location",
    "numeric": "A number: a count, date, distance, amount of money, percentage, "
    "period, speed, temperature, size or weight",
}
# TweetEval's offensive class follows OffensEval 2019's definition.
OFFENSIVE_CRITERIA = {
    "true": "It contains profanity or swear words, or a targeted offense such as an "
    "insult or a threat, veiled or direct",
    "false": "It contains no profanity and no targeted offense",
}
# In ANLI's label order (entailment, neutral, contradiction).
NLI = {
    "entailment": "The premise shows that the hypothesis is true",
    "neutral": "The premise neither proves nor contradicts the hypothesis",
    "contradiction": "The premise shows that the hypothesis is false",
}
# SST-5's labels 0 to 4, in order: the score level is the label.
SENTIMENT = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


@dataclass(frozen=True)
class Source:
    """A dataset split pinned to a Hub commit, with what a copy of it must hold."""

    repo: str
    config: str | None
    split: str
    revision: str
    rows: int
    digest: str


def digest(rows: list[dict]) -> str:
    """First 16 hex digits of the SHA-256 of the rows as sorted JSON lines."""
    sha = hashlib.sha256()
    for row in rows:
        sha.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode())
    return sha.hexdigest()[:16]


@dataclass(frozen=True)
class Preset:
    """One dataset: where it is, which split the coverage suite answers, and how a
    row becomes an example."""

    name: str  # as the data command takes it ("anli-r3")
    key: str  # the suite's benchmark name and the start of every id ("anli_r3")
    source: Source  # the evaluation split (the coverage suite's), at the pinned commit
    row_id: Callable[[dict, int], str]  # the split's own id of a row, or its index
    make: Callable[[dict], dict]  # row -> state, questions, gold


def _label(names: list[str], row: dict, column: str) -> str:
    value = row[column]
    if not isinstance(value, int) or not 0 <= value < len(names):
        raise ValueError(f"{column} {value!r} is none of the labels 0 to {len(names) - 1}")
    return names[value]


def _trec(row: dict) -> dict:
    return {
        "state": row["text"],
        "questions": {
            "answer_type": {
                "type": "choice",
                "instructions": "What kind of answer does this question ask for?",
                "criteria": TREC_TYPES,
            }
        },
        "gold": _label(list(TREC_TYPES), row, "coarse_label"),
    }


def _offensive(row: dict) -> dict:
    return {
        "state": row["text"],
        "questions": {
            "offensive": {
                "type": "noul",
                "instructions": "Is this tweet offensive?",
                "criteria": OFFENSIVE_CRITERIA,
            }
        },
        "gold": _label(["false", "true"], row, "label"),
    }


def _anli(row: dict) -> dict:
    return {
        "state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
        "questions": {
            "relation": {
                "type": "choice",
                "instructions": "How does the premise relate to the hypothesis?",
                "criteria": NLI,
            }
        },
        "gold": _label(list(NLI), row, "label"),
    }


def _openbookqa(row: dict) -> dict:
    return {
        "state": row["question_stem"],
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": "Which option answers or completes this science "
                "question correctly?",
                "criteria": dict(zip(row["choices"]["label"], row["choices"]["text"], strict=True)),
            }
        },
        "gold": row["answerKey"],
    }


def _sst5(row: dict) -> dict:
    return {
        "state": row["text"],
        "questions": {
            "sentiment": {
                "type": "score",
                "instructions": "How positive is the sentiment of this movie review?",
                "criteria": SENTIMENT,
            }
        },
        "gold": _label([str(level) for level in range(len(SENTIMENT))], row, "label"),
    }


def _index(row: dict, index: int) -> str:
    return str(index)


# The coverage suite's splits, at the Hub commits they were saved from (2026-10-08).
PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            "trec",
            "trec",
            Source(
                "CogComp/trec",
                None,
                "test",
                "65752bf53af25bc935a0dce92fb5b6c930728450",
                500,
                "da2ece276b070172",
            ),
            _index,
            _trec,
        ),
        Preset(
            "offensive",
            "offensive",
            Source(
                "cardiffnlp/tweet_eval",
                "offensive",
                "test",
                "b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
                860,
                "bda9a442fbcd6f91",
            ),
            _index,
            _offensive,
        ),
        Preset(
            "anli-r3",
            "anli_r3",
            Source(
                "facebook/anli",
                None,
                "test_r3",
                "8e4813d81f46d313dac7892e1c28076917cfcdf9",
                1200,
                "a9a449ce009d9136",
            ),
            lambda row, index: row["uid"],
            _anli,
        ),
        Preset(
            "openbookqa",
            "openbookqa",
            Source(
                "allenai/openbookqa",
                "main",
                "test",
                "388097ea7776314e93a529163e0fea805b8a6454",
                500,
                "1e448ea67d38e7c1",
            ),
            lambda row, index: row["id"],
            _openbookqa,
        ),
        Preset(
            "sst5",
            "sst5",
            Source(
                "SetFit/sst5",
                None,
                "test",
                "e51bdcd8cd3a30da231967c1a249ba59361279a3",
                2210,
                "2f7e332b4d7193ca",
            ),
            _index,
            _sst5,
        ),
    )
}


def examples(
    preset: Preset, rows: list[dict], split: str | None = None, indices: list[int] | None = None
) -> list[dict]:
    """The examples of a split's rows, in order (``split`` None: the evaluation
    split). Ids of the evaluation split are ``<key>:<row id>``, of other splits
    ``<key>:<split>/<row id>``; a row's id is its own, or its index in the split
    (``indices``, when ``rows`` are some of them; else their position)."""
    split = split or preset.source.split
    prefix = f"{preset.key}:" if split == preset.source.split else f"{preset.key}:{split}/"
    result = []
    for index, row in zip(indices or range(len(rows)), rows, strict=True):
        try:
            example = preset.make(row)
        except ValueError as error:
            raise ValueError(f"{preset.name} {split} row {index}: {error}") from None
        result.append({"id": prefix + preset.row_id(row, index), **example})
    return result
