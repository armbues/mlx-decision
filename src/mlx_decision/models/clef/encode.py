# Ported from joint_schema_model.py in Cloudflare's Clef release
# (https://huggingface.co/Cloudflare/clef-flash), Apache License 2.0.
# See LICENSES/clef-Apache-2.0.txt.
#
# Modified for mlx-decision: text only, tokenizes with the `tokenizers`
# library, reports truncation of the state.
"""Render a request into the token sequence Clef was trained on."""

import json
from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer

from ...errors import DecisionError
from ...types import Choice, Noul, Request, Score

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
DEFAULT_MAX_LENGTH = 16384


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRequest:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    truncated: bool


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: Noul | Choice | Score) -> list[tuple[str, Any]]:
    """The options of a question as (id, description), in the order the model reads them."""
    if isinstance(question, Noul):
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.criteria or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if isinstance(question, Choice):
        return sorted((str(key), value) for key, value in question.criteria.items())
    return [(str(index), value) for index, value in enumerate(question.criteria)]


def encode_request(
    tokenizer: Tokenizer,
    request: Request,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> EncodedRequest:
    def tokens(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    schema_ids = tokens("\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for index, (question_id, question) in enumerate(request.questions.items()):
        schema_ids.extend(
            tokens(
                f"\nFIELD {index + 1}\nID: {question_id}\nTYPE: {question.type}\nINSTRUCTION: "
            )
        )
        question_start = len(schema_ids)
        instructions = question.instructions
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(tokens(render(instructions)))
        question_end = len(schema_ids)
        schema_ids.extend(tokens("\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(question_options(question)):
            schema_ids.extend(tokens(f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(tokens(render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(tokens("\n"))
        schema_ids.extend(tokens("END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[question.type],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = tokens(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    )
    suffix_ids = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise DecisionError(
            f"the questions alone need {fixed_length} tokens; the limit is {max_length}",
            param="questions",
        )
    state_ids = tokens(render(request.state))
    room = max_length - fixed_length
    truncated = len(state_ids) > room
    state_ids = state_ids[:room]

    offset = len(prefix_ids) + len(state_ids)
    shifted = tuple(
        EncodedQuestion(
            question_id=q.question_id,
            question_type=q.question_type,
            question_span=(q.question_span[0] + offset, q.question_span[1] + offset),
            option_spans=tuple((start + offset, end + offset) for start, end in q.option_spans),
            option_ids=q.option_ids,
        )
        for q in questions
    )
    return EncodedRequest(
        input_ids=tuple(prefix_ids + state_ids + schema_ids + suffix_ids),
        questions=shifted,
        truncated=truncated,
    )
