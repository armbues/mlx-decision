# Copyright Convai Innovations (the laya authors) and Supersonic Labs.
#
# Ported from laya (https://pypi.org/project/laya/) 0.3.28, files
# laya/common.py (build_head, build_sequence, render_options,
# render_criterion, serialize_state) and laya/agent.py (question checks,
# state truncation), and from Julia 1 (https://huggingface.co/SupersonicLabs/Julia-1),
# files julia/data.py (sequence, strict encoding) and julia/typed.py (named
# questions to option lists), all under the Apache License 2.0. See
# LICENSES/laya-Apache-2.0.txt.
#
# Modified for mlx-decision: one sequence builder for both families;
# tokenizes with the `tokenizers` library; requests arrive in the wire
# format and errors name the offending field.
"""Render a request into the marker sequences Laya and Julia were trained on.

One sequence per question:
``[CLS] <type> question: <instructions> [SEP] [MASK] option ... [SEP] state [SEP]``.
Each option starts with a ``[MASK]`` marker whose hidden state the head scores.
"""

import json
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from ...errors import DecisionError
from ...types import Choice, Noul, Request, Score
from .head import QUESTION_TYPES

OPTION_TOKENS = 48  # tokens kept per option text
MIN_HEAD_ROOM = 16  # below this, options are cut evenly to leave room for the question
MIN_OPTION_TOKENS = 4
MIN_QUESTION_TOKENS = 8
JULIA_OPTIONS = (2, 20)


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    input_ids: list[int]
    markers: list[int]
    question_type: int
    option_ids: list[str]
    truncated: bool


def serialize(value: Any) -> str:
    """State and instructions as text: strings as they are, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def laya_options(question_id: str, question) -> tuple[list[str], list[str]]:
    """Option ids and texts as Laya renders them."""
    where = f"questions.{question_id}.criteria"
    if isinstance(question, Choice):
        ids = list(question.criteria)
        texts = [
            key if value is None or value == "" else f"{key}: {_criterion(value)}"
            for key, value in question.criteria.items()
        ]
        return ids, texts
    if isinstance(question, Score):
        if any(level is None for level in question.criteria):
            raise DecisionError("score levels must not be null", param=where)
        texts = [f"level {i}: {_criterion(level)}" for i, level in enumerate(question.criteria)]
        return [str(i) for i in range(len(texts))], texts
    criteria = question.criteria or {}
    texts = []
    for key, default in (
        ("false", "no, the statement does not hold"),
        ("true", "yes, the statement holds"),
    ):
        value = criteria.get(key)
        texts.append(f"{key}: " + (_criterion(value) if value not in (None, "") else default))
    return ["false", "true"], texts


def julia_options(question_id: str, question) -> tuple[list[str], list[str]]:
    """Option ids and texts as Julia's named-question API passes them: descriptions only."""
    where = f"questions.{question_id}.criteria"
    if isinstance(question, Noul):
        if question.criteria is None:
            return ["false", "true"], ["false", "true"]
        if set(question.criteria) != {"false", "true"}:
            raise DecisionError(
                f"{_name(question)} criteria must describe both 'true' and 'false'", param=where
            )
        texts = [question.criteria["false"], question.criteria["true"]]
        ids = ["false", "true"]
    elif isinstance(question, Choice):
        ids, texts = list(question.criteria), list(question.criteria.values())
    else:
        ids, texts = [str(i) for i in range(len(question.criteria))], list(question.criteria)
    if not all(isinstance(text, str) and text for text in texts):
        raise DecisionError(
            f"{_name(question)} descriptions must be non-empty strings for this model",
            param=where,
        )
    low, high = JULIA_OPTIONS
    if not low <= len(texts) <= high:
        raise DecisionError(
            f"this model answers questions with {low} to {high} options", param=where
        )
    return ids, texts


def _name(question) -> str:
    return question.type


def _missing(value: Any) -> bool:
    return value is None or value == ""


def fill_defaults(request: Request, family: str) -> Request:
    """Fill in what Laya and Julia need but a request may leave out.

    Both families were trained with instructions; Julia also with a description
    for every option. Short forms (``run --choice``, questions built in
    ``chat``) leave them out, and the families' own code refuses such
    requests. Here a missing instruction becomes the question id, and for
    Julia a missing choice description the option id (as Laya writes an
    option without one) and a missing noul side its literal ``false`` /
    ``true`` (Julia's default when no side is described).
    """
    questions = {}
    for question_id, question in request.questions.items():
        update = {}
        if _missing(question.instructions):
            update["instructions"] = question_id
        if (
            family == "julia"
            and isinstance(question, Choice)
            and any(_missing(v) for v in question.criteria.values())
        ):
            update["criteria"] = {
                key: key if _missing(value) else value for key, value in question.criteria.items()
            }
        if family == "julia" and isinstance(question, Noul) and question.criteria is not None:
            update["criteria"] = {
                key: key if _missing(question.criteria.get(key)) else question.criteria[key]
                for key in ("false", "true")
            }
        questions[question_id] = question.model_copy(update=update) if update else question
    return request.model_copy(update={"questions": questions})


def encode_request(tokenizer: Tokenizer, special, settings, request: Request):
    """One ``EncodedQuestion`` per question, in request order."""
    julia = settings.family == "julia"
    request = fill_defaults(request, settings.family)
    strict = julia and settings.strict

    def tokens(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    def clean(text: str, param: str) -> str:
        if special.mask_text in text:
            if strict:
                raise DecisionError(
                    f"contains the model's marker token {special.mask_text}", param=param
                )
            return text.replace(special.mask_text, " ")
        return text

    if julia and not isinstance(request.state, (str, dict, list)):
        raise DecisionError("state must be text, an object or an array", param="state")
    if request.state is None:
        raise DecisionError("state must not be null", param="state")
    state_ids = tokens(clean(serialize(request.state), "state"))
    # Laya keeps the end of a conversation (a list state) when it has to cut.
    keep_end = not julia and isinstance(request.state, list)

    encoded = []
    for question_id, question in request.questions.items():
        where = f"questions.{question_id}"
        if julia:
            if not isinstance(question.instructions, str):
                raise DecisionError(
                    "instructions must be text for this model", param=f"{where}.instructions"
                )
            option_ids, texts = julia_options(question_id, question)
        else:
            option_ids, texts = laya_options(question_id, question)
        instructions = clean(serialize(question.instructions), f"{where}.instructions")
        question_ids = tokens(f"{question.type} question: {instructions}")
        options = []
        for text in texts:
            ids = tokens(" " + clean(text, f"{where}.criteria"))
            if strict and len(ids) > OPTION_TOKENS:
                raise DecisionError(
                    f"an option is longer than {OPTION_TOKENS} tokens", param=f"{where}.criteria"
                )
            options.append([special.mask] + ids[:OPTION_TOKENS])
        budget = settings.head_length - sum(map(len, options))
        if budget < MIN_HEAD_ROOM:
            per_option = max(
                MIN_OPTION_TOKENS, (settings.head_length - MIN_HEAD_ROOM) // len(options)
            )
            if strict and any(len(option) > per_option for option in options):
                raise DecisionError(
                    f"the options need more than the {settings.head_length} tokens "
                    "this model gives a question",
                    param=f"{where}.criteria",
                )
            options = [option[:per_option] for option in options]
            budget = settings.head_length - sum(map(len, options))
        if strict and len(question_ids) > budget:
            raise DecisionError(
                f"the question and its options need more than the {settings.head_length} "
                "tokens this model gives a question",
                param=f"{where}.instructions",
            )
        ids = [special.cls] + question_ids[: max(MIN_QUESTION_TOKENS, budget)] + [special.sep]
        markers = []
        for option in options:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(special.sep)

        room = settings.max_length - len(ids) - 1
        if julia and room < 1:
            raise DecisionError(
                f"the question and its options need {len(ids) + 1} tokens; "
                f"the limit is {settings.max_length}",
                param=f"{where}.criteria",
            )
        room = max(0, room)
        if strict and len(state_ids) > room:
            raise DecisionError(
                f"the state needs {len(state_ids)} tokens; {room} are left after the "
                f"question {question_id!r} (limit {settings.max_length})",
                param="state",
            )
        state = state_ids[max(0, len(state_ids) - room) :] if keep_end else state_ids[:room]
        ids = (ids + state + [special.sep])[: settings.max_length]
        if any(marker >= settings.max_length for marker in markers):
            raise DecisionError(
                f"the options do not fit in {settings.max_length} tokens",
                param=f"{where}.criteria",
            )
        encoded.append(
            EncodedQuestion(
                question_id,
                ids,
                markers,
                QUESTION_TYPES[question.type],
                option_ids,
                truncated=len(state) < len(state_ids),
            )
        )
    return encoded
