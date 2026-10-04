"""Turn per-option probabilities into answers. Shared by every backend."""

from collections.abc import Mapping, Sequence

from .types import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer


def choice_confidence(probabilities: Sequence[float]) -> float:
    """How far the top probability sits above an even split, from 0 to 1."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, (max(probabilities) - 1 / n) / (1 - 1 / n))


def score_confidence(probabilities: Sequence[float]) -> float:
    """One minus the spread around the most likely level, relative to an even spread."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    peak = max(range(n), key=probabilities.__getitem__)
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - spread / even_spread)


def build_answer(question: Noul | Choice | Score, probabilities: Mapping[str, float]) -> Answer:
    """Build the answer for one question from its option probabilities.

    ``probabilities`` is keyed by option id: ``"true"``/``"false"`` for a noul,
    the criteria keys for a choice, ``"0"``, ``"1"``, ... for a score.
    """
    if isinstance(question, Noul):
        return NoulAnswer(noul=probabilities["true"])
    if isinstance(question, Choice):
        options = list(question.criteria)
        ordered = {option: probabilities[option] for option in options}
        return ChoiceAnswer(
            choice=max(options, key=ordered.__getitem__),
            confidence=choice_confidence(list(ordered.values())),
            probabilities=ordered,
        )
    levels = [str(index) for index in range(len(question.criteria))]
    values = [probabilities[level] for level in levels]
    return ScoreAnswer(
        score=sum(index * p for index, p in enumerate(values)),
        confidence=score_confidence(values),
        legend=dict(zip(levels, question.criteria)),
        probabilities=dict(zip(levels, values)),
    )
