# Copyright H2O.ai (the H2O-Lightning authors).
#
# Ported from H2O-Lightning-4B (https://huggingface.co/h2oai/h2o-lightning-4b,
# revision ee4e0e624a0c62a803df878536bd1b59a40cc9f7, tag v1.4.2), file
# h2o_lightning_shim.py (Shim.fit_states, select_record, fit_question,
# cut_text and image_tokens_bound), under the Apache License 2.0. See
# LICENSES/h2o-lightning-Apache-2.0.txt.
#
# Modified for mlx-decision: questions arrive as mlx-decision's request
# types; the tokenizer, the prompt settings and the context are passed in;
# no state cap from the environment; the notes become one "cut" flag.
"""Fit each question's prompt to the context, as H2O-Lightning from v1.4 does.

A record is read whole while a question's prompt fits the context. When it
does not, that question gets the record selected to the room its own prompt
leaves (``selector.select_state``: BM25 over the record's parts, with their
headings), or its head and tail where nothing in it matches the question.
A question too long even without the record is read without it, its
instructions, then its option descriptions, then its option names cut until
it fits. Each question is fitted on its own; no request is refused for its
length.
"""

from collections.abc import Callable
from dataclasses import dataclass

from . import selector
from .prompt import CUT_MARK, PromptSettings, chat, instructions_text, options, render, user_text

# Context tokens kept free: the answer token, seams.
CONTEXT_SLACK = 16
# Under this many tokens of room the record is left out of the prompt.
MIN_STATE_BUDGET = 64
# Qwen3.5's vision tower: 16-pixel patches merged 2x2 into one token.
PIXELS_PER_IMAGE_TOKEN = 32 * 32
# The image processor's own largest image, when no bound is set.
LARGEST_IMAGE_PIXELS = 16777216


@dataclass(frozen=True)
class Fit:
    """One question's record; ``user`` replaces its user message when the
    question itself was cut (its record is then empty)."""

    record: str
    user: str | None = None


def cut_text(text: str, max_tokens: int, encode: Callable, decode: Callable) -> str:
    """``text`` cut to ``max_tokens`` by its head and tail around a "..." line."""
    # A byte-level BPE never has more tokens than the text has bytes.
    if not max_tokens or len(text.encode("utf-8", "ignore")) <= max_tokens:
        return text
    ids = encode(text)
    if len(ids) <= max_tokens:
        return text
    half = max(1, max_tokens // 2)
    return decode(ids[:half]) + CUT_MARK + decode(ids[-half:])


def image_tokens_bound(max_pixels: int) -> int:
    """At most one image's tokens: its pixel bound over the pixels per token,
    plus the vision start and end tokens."""
    return (max_pixels or LARGEST_IMAGE_PIXELS) // PIXELS_PER_IMAGE_TOKEN + 2


def fit_records(
    raw_state: object,
    full_text: str,
    questions: dict,
    labels: list[str],
    settings: PromptSettings,
    encode: Callable,
    decode: Callable,
    context: int,
    images: int = 0,
    max_pixels: int = 0,
) -> list[Fit]:
    """Each question's record (in the order of ``questions``) for a request
    whose record, rendered, is ``full_text``; ``context`` is the prompt limit,
    ``images`` the request's image count and ``max_pixels`` their bound."""
    img = images * image_tokens_bound(max_pixels)
    limit = context - CONTEXT_SLACK

    def prompt(record: str, question) -> str:
        # Measured as the text prompt: the images count by their bound.
        return render(record, question, labels, settings)[0]

    fits, state_tokens = [], None
    for question in questions.values():
        if len(prompt(full_text, question).encode("utf-8", "ignore")) + img <= limit:
            fits.append(Fit(full_text))  # fits: no tokenizer call
            continue
        if state_tokens is None:
            state_tokens = len(encode(full_text))
        room = limit - (len(encode(prompt("", question))) + img)
        budget = room
        if state_tokens <= budget:
            fits.append(Fit(full_text))
            continue
        if room < MIN_STATE_BUDGET:
            user = fit_question(question, labels, settings, encode, decode, limit, img)
            fits.append(Fit("", user))
            continue
        record = select_record(raw_state, full_text, question, budget, encode, decode)
        for _ in range(3):  # the exact count: the record's seams can add a token
            tokens = len(encode(prompt(record, question))) + img
            if tokens <= limit:
                break
            budget -= (tokens - limit) + 8
            if budget < MIN_STATE_BUDGET:
                record = ""
                break
            record = select_record(raw_state, full_text, question, budget, encode, decode)
        fits.append(Fit(record))
    return fits


def select_record(
    raw_state, full_text: str, question, budget: int, encode: Callable, decode: Callable
) -> str:
    """The record cut to ``budget`` tokens for ``question``: the selector's
    BM25 on the raw state, else (nothing matches, or a record that cannot be
    split to fit) its head and tail."""
    try:
        return selector.select_state(
            raw_state, instructions_text(question), question.criteria, budget, encode
        )
    except selector.EvidenceSelectionUnsupported:
        return cut_text(full_text, budget, encode, decode)


def fit_question(
    question, labels, settings: PromptSettings, encode, decode, limit: int, img: int
) -> str | None:
    """The user message of a question too long for the context without its
    record: the instructions, then each option description, then each option
    name (in the prompt only; the answer keeps the request's names) cut by
    head and tail until the prompt fits ``limit``. None when the question
    fits with an empty record."""
    names, descriptions = options(question)
    labels = labels[: len(names)]
    instructions = instructions_text(question).lstrip() or settings.default_instructions
    parts = [instructions, list(descriptions), list(names)]

    def user() -> str:
        return user_text("", parts[0], parts[2], parts[1], labels)

    def over() -> int:
        return len(encode(chat(user(), settings))) + img - limit

    if over() <= 0:
        return None
    for level in (0, 1, 2):
        excess = over()
        if excess <= 0:
            break
        if level == 0:
            tokens = len(encode(parts[0]))
            parts[0] = cut_text(parts[0], max(16, tokens - excess - 8), encode, decode)
        else:
            total = sum(len(encode(x)) for x in parts[level])
            per = max(4, (total - excess - 8) // max(1, len(parts[level])))
            parts[level] = [cut_text(x, per, encode, decode) for x in parts[level]]
    return user()
