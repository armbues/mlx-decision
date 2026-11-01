# Written from SPEC.md and the observed reference data in fixtures/; no
# code from the d1-2B release was read or used.
"""Prompt texts, packed sets and picture sizes of d1-2B."""

import json
import math
import re
from string import ascii_lowercase, ascii_uppercase

START = "<|startoftext|><|im_start|>user\n"
QUESTION_HEADER = "QUESTION:\n"
STATE_END = "\n\n\n" + QUESTION_HEADER
TURN_END = "<|im_end|>\n<|im_start|>assistant\n<think></think>"
PICTURE_START, PICTURE_TOKEN, PICTURE_END = "<|vision_start|>", "<|image|>", "<|vision_end|>"
MAX_PIXELS = 1_638_400
SET_SIZE = 16
TYPES = ("noul", "choice", "score")
ZERO_WIDTH_SPACE = "\u200b"


# Validation


def check_question(question: dict) -> None:
    """Raise ValueError for a question the model does not take."""
    kind = question.get("type")
    if kind not in TYPES:
        raise ValueError(f"unknown question type {kind!r}; expected one of {', '.join(TYPES)}")
    instructions = question.get("instructions")
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        raise ValueError('"instructions" is required and must not be empty')
    criteria = question.get("criteria")
    if kind == "score" and not (
        isinstance(criteria, list) and 2 <= len(criteria) <= 10 and all(isinstance(c, str) for c in criteria)
    ):
        raise ValueError('a score\'s "criteria" must be a list of 2 to 10 strings, lowest level first')
    if kind == "choice" and not (isinstance(criteria, dict) and criteria):
        raise ValueError('a choice\'s "criteria" must be a non-empty object of option -> description')


# Text


def as_text(value) -> str:
    return value if isinstance(value, str) else str(value)


# The tokenizer's special tokens (added tokens marked special); other added
# tokens such as <think> stay as they are in caller text.
SPECIAL_TOKENS = (
    *(
        f"<|{name}|>"
        for name in [
            "pad",
            "startoftext",
            "endoftext",
            "fim_pre",
            "fim_mid",
            "fim_suf",
            "im_start",
            "im_end",
            "tool_list_start",
            "tool_list_end",
            "img_thumbnail",
            "vision_start",
            "vision_end",
            "image_split",
            "audio_start",
            "text_start",
            "text_end",
            "mixed_start",
            "mixed_end",
            "video",
            "image",
        ]  # fmt: skip
    ),
    "<image>",
    *(f"<|img_row_{row}_col_{col}|>" for row in range(1, 11) for col in range(1, 11)),
)
_SPECIAL = re.compile("|".join(re.escape(t) for t in sorted(SPECIAL_TOKENS, key=len, reverse=True)))


def escape(text: str) -> str:
    """Break up every special token in caller text with a zero-width space after its first character."""
    return _SPECIAL.sub(lambda m: m.group(0)[0] + ZERO_WIDTH_SPACE + m.group(0)[1:], text)


def state_text(state) -> str | None:
    """The state as the prompt shows it, before escaping."""
    if state is None:
        return None
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def prefix(state, image_tokens=()) -> str:
    """Everything before the first question: the user turn's start, the pictures, the state."""
    parts = [START]
    for count in image_tokens:
        parts.append(PICTURE_START + PICTURE_TOKEN * count + PICTURE_END)
    text = state_text(state)
    if text is not None:
        parts.append(escape(text) + STATE_END)
    return "".join(parts)


def suffix(question: dict, tokenizer) -> str:
    """A question's block and the turn's end."""
    check_question(question)

    def text(value) -> str:
        return escape(as_text(value))

    kind = question["type"]
    criteria = question.get("criteria")
    body = text(question["instructions"])
    if kind == "noul":
        if isinstance(criteria, dict) and criteria:
            body += f"\nYes: {text(criteria.get('true'))}\nNo: {text(criteria.get('false'))}"
        body += "\n\nReply with yes or no only."
    elif kind == "choice":
        lines = []
        for code, (key, description) in zip(option_codes(question, tokenizer), criteria.items(), strict=True):
            label = text(description) if description else text(key.replace("_", " "))
            lines.append(f"{code} {label}")
        body += "\n\nOptions:\n" + "\n".join(lines) + "\n\nReply with the option code only."
    else:
        lines = [f"{i} {text(level)}" for i, level in enumerate(criteria)]
        body += "\n\n" + "\n".join(lines) + f"\n\nReply with a single digit 0-{len(criteria) - 1} only."
    return body + TURN_END


# Option codes


def _two_letter_codes():
    for first in ascii_uppercase:
        for second in ascii_uppercase:
            yield first + second


FALLBACK_CODES = (
    list(ascii_uppercase)
    + [f"{i:02d}" for i in range(100)]
    + list(ascii_lowercase)
    + [f"#{i}" for i in range(200)]
    + list(_two_letter_codes())
)


def single_token(text: str, tokenizer) -> int | None:
    """The id of `text` if it encodes to exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None


def option_codes(question: dict, tokenizer) -> list[str]:
    """The code shown for each option of a choice, in option order."""
    labels = [str(key).strip() for key in question["criteria"]]
    if all(len(label) == 1 and label.isalpha() for label in labels):
        wanted = labels
    elif len(labels) <= 26:
        wanted = list(ascii_uppercase[: len(labels)])
    else:
        generated = list(ascii_uppercase) + list(_two_letter_codes())
        wanted = generated[: len(labels)]

    codes, taken = [], set()
    for code in wanted:
        token = single_token(code, tokenizer)
        if token is None or token in taken:
            code, token = next(
                (c, t) for c in FALLBACK_CODES if (t := single_token(c, tokenizer)) is not None and t not in taken
            )
        codes.append(code)
        taken.add(token)
    return codes


# Sets


def packed_sets(state, questions: list[dict], tokenizer, set_size: int = SET_SIZE, image_tokens=()) -> list[dict]:
    """The token sequences the model reads: up to `set_size` questions each, every one with the whole prefix."""
    for question in questions:
        check_question(question)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    head = prefix(state, image_tokens)
    state_length = len(encode(head))
    header = "" if state is None else QUESTION_HEADER
    sets = []
    for first in range(0, len(questions), set_size):
        group = questions[first : first + set_size]
        ids = encode(head + suffix(group[0], tokenizer))
        spans = [[state_length, len(ids)]]
        for question in group[1:]:
            start = len(ids)
            ids += encode(header + suffix(question, tokenizer))
            spans.append([start, len(ids)])
        sets.append({"ids": ids, "state_length": state_length, "spans": spans, "ends": [end - 1 for _, end in spans]})
    return sets


# Pictures


def picture_size(width: int, height: int, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """A picture's size after the cap: scaled down to at most `max_pixels`, sides truncated."""
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    return int(width * scale), int(height * scale)
