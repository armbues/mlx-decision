"""Latency benchmark: load time, and median and p95 per request over a grid.

The grid crosses state lengths (in tokens) with numbers of questions. Each
cell runs one warm-up request, then ``repeats`` timed ones, end to end
through ``DecisionModel.decide_request``. It works for any family: states
are filler text, sized from the token counts the model reports.

For a model that keeps computed prefixes (Clef), each cell also times
requests on a state it has seen: the "warm" times, which cover only the
questions. The timed cold requests drop kept prefixes first.
"""

import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import mlx.core as mx

from .machine import describe, machine_info, software_info
from .model import DecisionModel, load

# Version of the JSON written by ``to_dict``: format 2 added everything but
# the numbers (date, machine, software, model info, options), format 3 the
# warm times and the prefix cache option.
FORMAT = 3

FILLER = (
    "The customer wrote again about the delayed delivery and asked whether the "
    "order could still arrive before the weekend, adding that the tracking page "
    "had not changed since Tuesday and that support had not answered yet. "
)
DEFAULT_LENGTHS = (250, 1000, 4000, 15000)
DEFAULT_QUESTIONS = (1, 5, 20)


@dataclass
class Cell:
    state_tokens: int
    questions: int
    input_tokens: int
    median_s: float
    p95_s: float
    tokens_per_s: float
    warm_median_s: float | None = None
    warm_p95_s: float | None = None


@dataclass
class Report:
    model: str
    load_s: float
    peak_memory_gb: float
    repeats: int
    cells: list[Cell]
    date: str = ""
    machine: dict = field(default_factory=dict)
    software: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    lengths: tuple[int, ...] = ()
    question_counts: tuple[int, ...] = ()
    prefix_cache_gb: float | None = None


def make_questions(count: int) -> dict:
    """``count`` questions, cycling noul, choice (four options), score (five levels)."""
    kinds = [
        {"type": "noul", "instructions": "The customer asks for a refund."},
        {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "shipping": "Delivery and tracking",
                "billing": "Payments",
                "technical": "Bugs",
                "sales": "Plans and pricing",
            },
        },
        {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["Calm", "Slightly annoyed", "Annoyed", "Angry", "Furious"],
        },
    ]
    return {f"q{index + 1}": kinds[index % len(kinds)] for index in range(count)}


def make_state(words: int) -> str:
    filler = FILLER.split()
    return " ".join(filler[i % len(filler)] for i in range(words))


def default_lengths(limit: int | None) -> tuple[int, ...]:
    """The default state lengths that fit the model's input limit.

    When those all stay below half the limit (Laya's 512 or 1,024 tokens), a
    length near the limit is added, so that the longest inputs are measured.
    """
    if limit is None:
        return DEFAULT_LENGTHS
    lengths = [length for length in DEFAULT_LENGTHS if length < limit]
    if not lengths or lengths[-1] < limit / 2:
        lengths.append(int(limit * 0.9) // 50 * 50)
    return tuple(lengths)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def tokens_per_word(model: DecisionModel) -> float:
    questions = make_questions(1)
    base = model.decide(make_state(0), questions).usage.input_tokens
    probe = model.decide(make_state(500), questions).usage.input_tokens
    return max((probe - base) / 500, 1e-3)


def prefix_cache(model: DecisionModel):
    """The backend's cache of computed prefixes, or None (off, or a family without one)."""
    return getattr(model.backend, "prefix_cache", None)


def forget_prefixes(model: DecisionModel) -> None:
    """Drop kept prefixes, so a repeated request is timed as a new one."""
    cache = prefix_cache(model)
    if cache is not None:
        cache.clear()


def time_warm(model: DecisionModel, request: dict, repeats: int) -> list[float] | None:
    """Times of ``request`` with its state's prefix kept, or None if it was not.

    The request repeats unchanged: only the prefix is looked up, so its
    questions are computed each time, as new questions would be. A prefix
    larger than the cache limit is not kept, and then there is nothing to time.
    """
    cache = prefix_cache(model)
    if cache is None:
        return None
    hits = cache.hits
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.decide_request(request)
        times.append(time.perf_counter() - start)
    return times if cache.hits - hits == repeats else None


def run(
    model: DecisionModel,
    lengths: tuple[int, ...] | None = None,
    question_counts: tuple[int, ...] = DEFAULT_QUESTIONS,
    repeats: int = 5,
    load_s: float = 0.0,
) -> Report:
    """Time every cell; ``lengths`` default to ``default_lengths`` for the model's limit."""
    if lengths is None:
        lengths = default_lengths(model.backend.capabilities.max_input_tokens)
    rate = tokens_per_word(model)
    mx.reset_peak_memory()
    cells = []
    for length in lengths:
        state = make_state(round(length / rate))
        for count in question_counts:
            request = {"state": state, "questions": make_questions(count)}
            result = model.decide_request(request)  # warm-up
            times = []
            for _ in range(repeats):
                forget_prefixes(model)
                start = time.perf_counter()
                model.decide_request(request)
                times.append(time.perf_counter() - start)
            warm = time_warm(model, request, repeats)
            median = statistics.median(times)
            cells.append(
                Cell(
                    state_tokens=length,
                    questions=count,
                    input_tokens=result.usage.input_tokens,
                    median_s=median,
                    p95_s=percentile(times, 0.95),
                    tokens_per_s=result.usage.input_tokens / median,
                    warm_median_s=statistics.median(warm) if warm else None,
                    warm_p95_s=percentile(warm, 0.95) if warm else None,
                )
            )
    return Report(
        model=model.name,
        load_s=load_s,
        peak_memory_gb=mx.get_peak_memory() / 2**30,
        repeats=repeats,
        cells=cells,
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        machine=machine_info(),
        software=software_info(),
        model_info=model.info(),
        lengths=tuple(lengths),
        question_counts=tuple(question_counts),
        prefix_cache_gb=cache.max_bytes / 1e9 if (cache := prefix_cache(model)) else None,
    )


def benchmark(reference: str, check_memory: bool = True, **options) -> Report:
    """Load ``reference`` (timed) and run the grid."""
    start = time.perf_counter()
    model = load(reference, check_memory=check_memory)
    load_s = time.perf_counter() - start
    return run(model, load_s=load_s, **options)


def format_report(report: Report) -> str:
    """A Markdown table; warm columns only when some cell has warm times."""
    warm = any(cell.warm_median_s is not None for cell in report.cells)
    header = "| State tokens | Questions | Input tokens | Median s | p95 s | Tokens/s |"
    lines = [
        describe(report.machine),
        f"{report.model}: load {report.load_s:.1f} s, peak memory "
        f"{report.peak_memory_gb:.1f} GB, {report.repeats} timed runs per row after a warm-up",
        "",
        header + (" Warm median s | Warm p95 s |" if warm else ""),
        "|---" * (8 if warm else 6) + "|",
    ]
    for cell in report.cells:
        line = (
            f"| {cell.state_tokens} | {cell.questions} | {cell.input_tokens} "
            f"| {cell.median_s:.3f} | {cell.p95_s:.3f} | {cell.tokens_per_s:.0f} |"
        )
        if warm:
            line += f" {_seconds(cell.warm_median_s)} | {_seconds(cell.warm_p95_s)} |"
        lines.append(line)
    if warm:
        lines += [
            "",
            "Warm: the same request again with its state's computed prefix kept "
            "(only the questions are computed); - where the prefix was too large to keep.",
        ]
    return "\n".join(lines)


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def to_dict(report: Report) -> dict:
    """The report as one JSON object that names where its numbers come from."""
    return {
        "format": FORMAT,
        "date": report.date,
        "machine": report.machine,
        "software": report.software,
        "model": report.model,
        "model_info": report.model_info,
        "options": {
            "lengths": list(report.lengths),
            "question_counts": list(report.question_counts),
            "repeats": report.repeats,
            "prefix_cache_gb": report.prefix_cache_gb,
        },
        "load_s": report.load_s,
        "peak_memory_gb": report.peak_memory_gb,
        "repeats": report.repeats,
        "cells": [asdict(cell) for cell in report.cells],
    }
