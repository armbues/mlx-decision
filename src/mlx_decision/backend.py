"""What a model family implements to take part in the framework."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .types import Question, Request

if TYPE_CHECKING:
    from .models.readout import Features, Readout


@dataclass(frozen=True)
class Capabilities:
    question_types: frozenset[str] = frozenset({"noul", "choice", "score"})
    max_input_tokens: int | None = None
    max_choice_options: int | None = None
    max_score_levels: int | None = None
    # Tokens one question and its options share; options are cut to fit.
    question_tokens: int | None = None
    requires_instructions: bool = False
    supports_images: bool = False
    # False when inputs over max_input_tokens are refused instead of cut.
    truncates_input: bool = True
    # Tokens of state a prompt shows at most, when less than the input limit
    # (longer states are cut).
    max_state_tokens: int | None = None


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


@runtime_checkable
class Readable(Protocol):
    """A family that reads its answers from the last hidden state (H2O, d1, pplx),
    so a readout can be fitted on its features and applied per call.

    ``score(request, readouts, finish)`` uses a ``Readout`` instead of the
    release's rows for each question (by id) whose options it matches;
    ``finish=False`` leaves out ``finish``, the steps after the readout (H2O's
    yes/no floor; nothing for the others), so a calibration can come first.
    """

    def features(self, request: Request) -> dict[str, "Features"]: ...

    def score(
        self,
        request: Request,
        readouts: dict[str, "Readout"] | None = None,
        finish: bool = True,
    ) -> BackendOutput: ...

    def finish(self, question: Question, probabilities: dict[str, float]) -> dict[str, float]: ...
