"""What a model family implements to take part in the framework."""

from dataclasses import dataclass
from typing import Protocol

from .types import Request


@dataclass(frozen=True)
class Capabilities:
    question_types: frozenset[str] = frozenset({"noul", "choice", "score"})
    max_input_tokens: int | None = None
    max_choice_options: int | None = None
    max_score_levels: int | None = None
    requires_instructions: bool = False
    supports_images: bool = False
    # False when inputs over max_input_tokens are refused instead of cut.
    truncates_input: bool = True


@dataclass
class BackendOutput:
    """Probabilities per question id, each keyed by option id.

    Option ids are ``"true"``/``"false"`` for a noul, the criteria keys for a
    choice and ``"0"``, ``"1"``, ... for a score.
    """

    probabilities: dict[str, dict[str, float]]
    input_tokens: int
    truncated: bool = False


class Backend(Protocol):
    """A loaded model of one family.

    ``load`` also sets ``family`` (the registry's name for it). A backend may
    set ``precision``, a short description of how its weights are stored
    (``"bfloat16"``, ``"8-bit"``, ...), shown by ``DecisionModel.info``.
    """

    name: str
    capabilities: Capabilities

    def score(self, request: Request) -> BackendOutput: ...
