# Copyright H2O.ai (the H2O-Lightning authors).
#
# Ported from H2O-Lightning-4B (https://huggingface.co/h2oai/h2o-lightning-4b,
# revision acaf0d4ea251e54de928c75ef4352670d33192d3, tag v1.2.3), file
# h2o_lightning_shim.py (render_state, desc_text, render_instr, options_of,
# option_lines, user_turn, render_chat and the state cut of truncate_state),
# under the Apache License 2.0. See LICENSES/h2o-lightning-Apache-2.0.txt.
#
# Modified for mlx-decision: questions arrive as mlx-decision's request
# types; only the plain layout with the chat template; settings passed in
# rather than read from the environment; errors name the offending field.
"""Render one question into the prompt H2O-Lightning was fine-tuned on.

One causal sequence per question: the system prompt, then the user message
(the images, ``record:``, ``question:`` and the options, one ``A) name:
description`` line each), then an empty thinking block and the answer
prefill. The answer is read at the last token, over the labels' tokens.
"""

import json
from dataclasses import dataclass

from ...types import Choice, Noul, Score

CHAT = {
    "system": "<|im_start|>system\n{content}<|im_end|>\n",
    "user": "<|im_start|>user\n{content}<|im_end|>\n",
    "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
}
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
CUT_MARK = "\n...\n"


@dataclass(frozen=True)
class PromptSettings:
    """The ``prompt`` block of the release's ``serve_config.json``."""

    system: str | None
    prefill: str
    state_compact: bool
    max_state_tokens: int
    default_instructions: str

    @classmethod
    def from_config(cls, prompt: dict) -> "PromptSettings":
        if not prompt.get("chat_template", True) or prompt.get("enable_thinking", False):
            raise ValueError("only the chat template with thinking off is supported")
        if prompt.get("layout", "plain") != "plain":
            raise ValueError(f"prompt layout {prompt['layout']!r} is not supported")
        return cls(
            system=prompt.get("system") or None,
            prefill=prompt.get("prefill", "Answer:"),
            state_compact=bool(prompt.get("state_compact", True)),
            max_state_tokens=int(prompt.get("max_state_tokens", 32000)),
            default_instructions=prompt.get("default_instructions", "Answer the question below."),
        )


def render_state(state: object, compact: bool) -> str:
    """The record: a string verbatim, anything else as JSON."""
    if isinstance(state, str):
        return state
    if compact:
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(state, indent=1)


def cut_state(text: str, max_tokens: int, encode, decode) -> tuple[str, bool]:
    """A state over ``max_tokens`` keeps its head and tail halves around a "..." line.

    Returns the text and whether it was cut. A byte-level BPE never has more
    tokens than the text has UTF-8 bytes, so a short text is not tokenized.
    """
    if not max_tokens or len(text.encode("utf-8", "ignore")) <= max_tokens:
        return text, False
    ids = encode(text)
    if len(ids) <= max_tokens:
        return text, False
    half = max_tokens // 2
    return decode(ids[:half]) + CUT_MARK + decode(ids[-half:]), True


def describe(value: object) -> str:
    """An option description or score level: a string verbatim, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def instructions_text(question: Noul | Choice | Score) -> str:
    """The instructions as the release reads them from the request's JSON.

    A string verbatim, anything else as JSON; left out, they are empty (and
    the default takes over), while an explicit null is written as JSON.
    """
    value = question.instructions
    if value is None and "instructions" not in question.model_fields_set:
        return ""
    return value if isinstance(value, str) else json.dumps(value, indent=1)


def options(question: Noul | Choice | Score) -> tuple[list[str], list[str]]:
    """Option ids (as in the answer, in the prompt's order) and their descriptions."""
    if isinstance(question, Noul):
        criteria = question.criteria or {}
        true, false = criteria.get("true"), criteria.get("false")
        return ["true", "false"], [
            describe(true) if true not in (None, "") else "the statement holds",
            describe(false) if false not in (None, "") else "it does not",
        ]
    if isinstance(question, Score):
        return [str(i) for i in range(len(question.criteria))], [
            describe(value) for value in question.criteria
        ]
    criteria = question.criteria
    return list(criteria), [
        describe(value) if value is not None else name for name, value in criteria.items()
    ]


def render(
    state_text: str,
    question: Noul | Choice | Score,
    labels: list[str],
    settings: PromptSettings,
    images: int = 0,
) -> tuple[str, list[str]]:
    """The prompt text for one question (no special tokens added) and its option ids.

    ``state_text`` is the rendered (and cut) state; ``labels`` has at least
    as many labels as the question has options.
    """
    names, descriptions = options(question)
    instructions = instructions_text(question).lstrip() or settings.default_instructions
    body = "\n".join(
        f"{label}) {name}: {description}"
        for label, name, description in zip(labels, names, descriptions, strict=False)
    )
    user = f"record: {state_text}\nquestion: {instructions}\noptions:\n{body}"
    # The chat template trims both messages.
    text = "" if settings.system is None else CHAT["system"].format(content=settings.system.strip())
    text += CHAT["user"].format(content=IMAGE * images + user.strip())
    return text + CHAT["generation"] + settings.prefill, names
