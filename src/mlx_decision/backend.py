"""What a model family implements to take part in the framework."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    # For a model that reads images: larger ones are scaled down to this many
    # pixels (keeping their shape); None when it does not read images.
    max_image_pixels: int | None = None
    # Images one request takes at most (None: no limit of the model's own).
    max_images: int | None = None
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
    """A family that reads its answers from the last hidden state (pplx, d3, Base
    Labs, H2O, d1 LFM3, d1), so a readout can be fitted on its features and applied per call.

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


@runtime_checkable
class HeadTrainable(Protocol):
    """A family whose decision head trains on stored backbone states (Clef).

    ``head_inputs(request)`` is the backbone's part of answering (hidden
    states, token ids, spans); ``head_logits(inputs)`` the head's logits per
    question, differentiable in ``head``'s parameters; ``read_head(inputs)``
    the probabilities ``score`` answers with. ``head_size(request)`` is the
    request's hidden states' size in bytes, computed without the model. A
    loader option ``fit_weights`` loads a trained head over the release's.
    """

    head: Any

    def head_inputs(self, request: Request) -> Any: ...

    def head_logits(self, inputs: Any) -> list[Any]: ...

    def read_head(self, inputs: Any) -> dict[str, dict[str, float]]: ...

    def head_size(self, request: Request) -> int: ...


@runtime_checkable
class Trainable(Protocol):
    """A family whose weights a fine-tune trains (Laya, Julia).

    ``trainable()`` is the module trained in place (its ``encoder`` and
    ``head``); ``items(request)`` the request's questions as training items
    by question id (each with ``option_ids`` and a ``length`` in tokens);
    ``scores(items)`` differentiable logits (rows, options) as served,
    divided by any temperature, and a mask of the real options. A loader
    option ``fit_weights`` loads a fine-tune's tensors over the release's.
    """

    def trainable(self) -> Any: ...

    def items(self, request: Request) -> dict[str, Any]: ...

    def scores(self, items: list[Any]) -> tuple[Any, Any]: ...


@dataclass
class TrainingPass:
    """One sequence of a request as a LoRA fit runs it: its token ids, each of
    its questions' readout (by id, in the order of their answer slots; the
    ``Features`` without ``hidden``), the family's frozen inputs (such as
    the vision tower's rows, computed once) and whether it holds images."""

    tokens: list[int]
    questions: dict[str, "Features"]
    frozen: Any = None
    images: bool = False
    truncated: bool = False

    @property
    def length(self) -> int:
        return len(self.tokens)


@runtime_checkable
class Adaptable(Protocol):
    """A family whose network a LoRA fit adapts (d3, d1 LFM3).

    ``network()`` is the text model the adapters go into; ``adapter_targets()``
    the projections they wrap, as ``(prefix, module, names)`` with prefixes
    named after the weights; ``decoder_layer`` the class each checkpointed
    layer is. ``training_passes(request)`` encodes a checked request into the
    sequences it runs (any images through the vision tower now, frozen);
    ``pass_hidden(passes)`` runs passes as one batch (text passes only, or a
    single pass with images) and gives each pass's final hidden states at its
    questions' answer slots (float32, ``[questions, width]``), differentiable
    and not evaluated, as serving computes them.
    """

    decoder_layer: type

    def network(self) -> Any: ...

    def adapter_targets(self) -> list[tuple[str, Any, tuple[str, ...]]]: ...

    def training_passes(self, request: Request) -> list[TrainingPass]: ...

    def pass_hidden(self, passes: list[TrainingPass]) -> list[Any]: ...
