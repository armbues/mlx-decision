# Copyright the OneJev authors.
#
# Ported from OneJev (https://github.com/OmniJev/OneJev, commit
# 81ce62f1597c91e46767d4d02d6ac2e18534fe94, with Base Labs' serve.patch from
# baselabs/elephant-1 revision b3f8a93462fb1051a9de031e94a7ff380e254072),
# file qev/prompt.py (the answer codes, render_entry, render_state,
# render_question's default "tagged" layout, state_block, build_messages and
# the prompt strings) and qev/engine.py (the shared prefix and its check in
# _prefix_and_suffix_ids and _split), with the text the releases' chat
# template (chat_template.jinja, Qwen's) renders for those messages, all
# under the Apache License 2.0. See LICENSES/onejev-Apache-2.0.txt.
#
# Modified for mlx-decision: questions arrive as mlx-decision's request
# types; only the production template (no other layouts, no option order
# shifts); the chat template's output is written out instead of rendered
# with Jinja (thinking off, a generation prompt); tokenized with
# ``tokenizers``.
"""The prompt Base Labs' models were trained on (OneJev's ``qev-labels-v2``).

One prompt per question: a fixed system message, then the user message:
the state in a ``<state>`` block, the question, its options under answer
codes and the answer line; then an empty thinking block. Every question of
a request starts alike up to the end of the state block: that shared
prefix runs once, each question continues from it. An option's answer code
is ``A`` to ``Z``, then two-letter codes, 256 in all; each is one token of
the models' tokenizer.
"""

import bisect
import json
from dataclasses import dataclass

from ...types import Choice, Noul, Score

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Two-letter codes the tokenizer splits, left out of the codes.
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ"}
_NOT_ONE_TOKEN |= {"GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
CODES: tuple[str, ...] = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)]
)

SYSTEM = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
QUESTION_HEADER = "Question:"
SCORE_HEADER = "Rate the state:"
ANSWER_LINE = "Answer with one letter: {codes}."
ANSWER_LINE_MANY = "Answer with one label: {codes}."
NOUL_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_TRUE = "the statement is true / the answer is yes"
NOUL_FALSE = "the statement is false / the answer is no"
CHOICE_INSTRUCTIONS = "Which option applies to the state?"
SCORE_INSTRUCTIONS = "Which level describes the state?"
# The chat template's turns around the user message.
BEFORE_USER = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
AFTER_USER = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_entry(entry: object) -> str:
    """An instructions or criteria entry as text: strings stripped, None empty,
    anything else as JSON indented by 2."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: object) -> str:
    """The state as text: strings as they are, None empty, anything else as JSON
    indented by 2."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def state_block(state: object) -> str:
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def options(question: Noul | Choice | Score) -> tuple[list[str], list[str], list[object]]:
    """Option ids (as in the answer), the names the prompt shows and their descriptions."""
    if isinstance(question, Noul):
        criteria = question.criteria or {}
        true, false = criteria.get("true"), criteria.get("false")
        # ``is not None``, as the release: an empty description stays empty.
        return (
            ["true", "false"],
            ["yes", "no"],
            [NOUL_TRUE if true is None else true, NOUL_FALSE if false is None else false],
        )
    if isinstance(question, Choice):
        ids = list(question.criteria)
        return ids, ids, list(question.criteria.values())
    count = len(question.criteria)
    return [str(i) for i in range(count)], [f"level {i}" for i in range(count)], question.criteria


def render_question(question: Noul | Choice | Score) -> str:
    """A question's part of the prompt: after the state block, to the answer line."""
    if isinstance(question, Noul):
        header, default = QUESTION_HEADER, NOUL_INSTRUCTIONS
    elif isinstance(question, Choice):
        header, default = QUESTION_HEADER, CHOICE_INSTRUCTIONS
    else:
        header, default = SCORE_HEADER, SCORE_INSTRUCTIONS
    instructions = render_entry(question.instructions) or default
    _, names, descriptions = options(question)
    codes = CODES[: len(names)]
    lines = []
    for code, name, description in zip(codes, names, descriptions, strict=True):
        text = render_entry(description)
        lines.append(f"{code}. {name}: {text}" if text else f"{code}. {name}")
    answer = ANSWER_LINE if len(codes) <= len(LETTERS) else ANSWER_LINE_MANY
    return (
        f"{header} {instructions}\n\nOptions:\n"
        + "\n".join(lines)
        + "\n\n"
        + answer.format(codes=", ".join(codes))
    )


def render(state: object, suffix: str) -> str:
    """The whole prompt of a question, ready to tokenize (no special tokens added)."""
    return BEFORE_USER + state_block(state) + suffix + AFTER_USER


def prefix_text(state: object) -> str:
    """What every question's prompt starts with: up to the end of the state block."""
    return BEFORE_USER + state_block(state)


@dataclass
class Layout:
    """A request's prompts as tokens: the shared prefix and each question's rest."""

    prefix: list[int]
    suffixes: list[list[int]]


def layout(tokenizer, state: object, suffixes: list[str]) -> Layout:
    """The prompts of ``suffixes`` (``render_question``'s) tokenized, split into the
    longest common token prefix that ends at or before the state block's end and
    each question's rest. Raises ``ValueError`` when a question's tokens do not
    break at that point (the prefix's attention never sees a question, so its
    cache is only shared there) or a question has no tokens of its own."""
    full = []
    for suffix in suffixes:
        encoding = tokenizer.encode(render(state, suffix), add_special_tokens=False)
        full.append(encoding)
    prefix = tokenizer.encode(prefix_text(state), add_special_tokens=False).ids
    count = len(prefix)
    while count > 0 and not all(e.ids[:count] == prefix[:count] for e in full):
        count -= 1
    boundary = len(prefix_text(state))
    for encoding in full:
        ends = [end for _, end in encoding.offsets]
        # The tokens that end at or before the state block's end.
        if bisect.bisect_right(ends, boundary) != count:
            raise ValueError(
                "the prompt's tokens do not break at the end of the state block "
                f"(token {bisect.bisect_right(ends, boundary)}, shared prefix {count})"
            )
    suffix_ids = [e.ids[count:] for e in full]
    if any(not ids for ids in suffix_ids):
        raise ValueError("a question has no tokens after the state block")
    return Layout(prefix[:count], suffix_ids)
