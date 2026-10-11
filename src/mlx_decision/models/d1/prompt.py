# Written from a behaviour specification and the observed outputs of the
# d1-3B release (LiquidAI/d1-3B); no code from the release was read or used.
"""Prompt text for d1-3B: a prefix per state, a suffix per question."""

import json
import string

USER = "<|startoftext|><|im_start|>user\n"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"


def as_text(value) -> str:
    """Strings verbatim, anything else as Python's str()."""
    return value if isinstance(value, str) else str(value)


def prefix(state) -> str:
    """Everything before the question; shared by all questions on a state."""
    if state is None:
        return USER
    block = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
    return f"{USER}{block}\n\n\nQUESTION:\n"


def instructions(question: dict) -> str:
    value = question.get("instructions")
    if value is None:
        raise ValueError("question without instructions")
    return as_text(value)


def single_token(text: str, tokenizer) -> int | None:
    """The id of `text` if it encodes to exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None


def _fallback_codes():
    yield from string.ascii_uppercase
    yield from (f"{i:02d}" for i in range(100))
    yield from string.ascii_lowercase
    yield from (f"#{i}" for i in range(200))
    yield from (a + b for a in string.ascii_uppercase for b in string.ascii_uppercase)


def _wanted_codes(keys: list[str]) -> list[str]:
    labels = [key.strip() for key in keys]
    if all(len(label) == 1 and label.isalpha() for label in labels):
        return labels
    if len(labels) <= 26:
        return list(string.ascii_uppercase[: len(labels)])
    return [f"{i:02d}" for i in range(len(labels))]


def option_codes(question: dict, tokenizer) -> list[tuple[str, int]]:
    """(code, token id) per option of a choice question, in option order."""
    codes, taken = [], set()
    for wanted in _wanted_codes(list(question["criteria"])):
        token = single_token(wanted, tokenizer)
        if token is None or token in taken:
            for wanted in _fallback_codes():
                token = single_token(wanted, tokenizer)
                if token is not None and token not in taken:
                    break
            else:
                raise ValueError("no single-token option code left")
        codes.append((wanted, token))
        taken.add(token)
    return codes


def _noul(question: dict) -> str:
    text = instructions(question)
    criteria = question.get("criteria")
    if isinstance(criteria, dict) and criteria:
        text += f"\nYes: {as_text(criteria.get('true'))}\nNo: {as_text(criteria.get('false'))}"
    return text + "\n\nReply with yes or no only."


def _choice(question: dict, tokenizer) -> str:
    text = instructions(question)
    criteria = question["criteria"]
    lines = []
    for (code, _), (key, description) in zip(
        option_codes(question, tokenizer), criteria.items(), strict=True
    ):
        label = as_text(description) if description else key.replace("_", " ")
        lines.append(f"{code} {label}")
    return f"{text}\n\nOptions:\n" + "\n".join(lines) + "\n\nReply with the option code only."


def _score(question: dict) -> str:
    text = instructions(question)
    levels = question["criteria"]
    lines = "\n".join(f"{i} {as_text(level)}" for i, level in enumerate(levels))
    return f"{text}\n\n{lines}\n\nReply with a single digit 0-{len(levels) - 1} only."


def suffix(question: dict, tokenizer) -> str:
    """The question block and the assistant header."""
    kind = question.get("type")
    if kind == "noul":
        block = _noul(question)
    elif kind == "choice":
        block = _choice(question, tokenizer)
    elif kind == "score":
        block = _score(question)
    else:
        raise ValueError(f"unknown question type: {kind!r}")
    return block + ASSISTANT
