"""The public entry points: ``load`` a model and ask it to ``decide``."""

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .answers import build_answer
from .backend import Backend
from .errors import DecisionError, require_metal
from .hub import is_repo_id, resolve_model_path
from .memory import check_fits
from .registry import load_backend
from .state_images import lift_images
from .types import Choice, Request, Result, Score, Usage, parse_request


class DecisionModel:
    """A loaded decision model; ``load()`` returns one.

    ``decide`` and ``decide_request`` answer questions; ``check`` validates a
    request without answering it. Not thread-safe: answer from one thread.
    """

    def __init__(self, backend: Backend):
        self.backend = backend

    @property
    def name(self) -> str:
        return self.backend.name

    def info(self) -> dict[str, Any]:
        """What the model is and what requests it takes, as plain JSON values."""
        caps = self.backend.capabilities
        return {
            "name": self.name,
            "family": getattr(self.backend, "family", None),
            "precision": getattr(self.backend, "precision", None),
            "max_input_tokens": caps.max_input_tokens,
            "truncates_input": caps.truncates_input,
            "question_types": sorted(caps.question_types),
            "max_choice_options": caps.max_choice_options,
            "max_score_levels": caps.max_score_levels,
            "question_tokens": caps.question_tokens,
            "requires_instructions": caps.requires_instructions,
            "supports_images": caps.supports_images,
        }

    def decide(
        self, state: Any, questions: Mapping[str, Any], images: Sequence[Any] | None = None
    ) -> Result:
        """Answer ``questions`` (dicts or ``Choice``/``Score``/``Noul``) about ``state``.

        ``images`` may hold file paths, image bytes, PIL images or data URLs
        (``data:image/png;base64,...``); they need the ``images`` extra.
        """
        request = {"state": state, "questions": dict(questions)}
        if images:
            request["images"] = list(images)
        return self.decide_request(request)

    def decide_request(self, request: Request | Mapping[str, Any]) -> Result:
        """Answer a whole request body in the wire format."""
        request, lifted = self._checked(request)
        try:
            output = self.backend.score(request)
        except DecisionError as error:
            raise _in_state(error, lifted) from None
        for question_id, probabilities in output.probabilities.items():
            if not all(math.isfinite(p) for p in probabilities.values()):
                raise RuntimeError(f"{self.name} gave non-finite probabilities for {question_id!r}")
        answers = {
            question_id: build_answer(question, output.probabilities[question_id])
            for question_id, question in request.questions.items()
        }
        return Result(
            model=self.backend.name,
            answers=answers,
            usage=Usage(input_tokens=output.input_tokens),
            truncated=output.truncated,
        )

    def check(self, request: Request | Mapping[str, Any]) -> Request:
        """Validate a request for this model without answering it.

        Raises ``DecisionError`` for an invalid body and for what the model
        does not support (question types, limits, media).
        """
        return self._checked(request)[0]

    def _checked(self, request: Request | Mapping[str, Any]) -> tuple[Request, dict[int, str]]:
        """The request validated, with the images in its state lifted out (for a
        model that reads images), and where each lifted image was by its index."""
        request = parse_request(request)
        lifted = {}
        if self.backend.capabilities.supports_images:
            request, lifted = _lift_state_images(request)
        self._check_supported(request)
        return request, lifted

    def _check_supported(self, request: Request) -> None:
        caps = self.backend.capabilities
        if request.videos:
            raise DecisionError(
                "videos are not supported yet", param="videos", code="unsupported_media"
            )
        if request.images and not caps.supports_images:
            raise DecisionError(
                f"{self.backend.name} does not support images",
                param="images",
                code="unsupported_media",
            )
        for question_id, question in request.questions.items():
            where = f"questions.{question_id}"
            if question.type not in caps.question_types:
                raise DecisionError(
                    f"{self.backend.name} does not support {question.type} questions",
                    param=f"{where}.type",
                )
            if caps.requires_instructions and question.instructions in (None, ""):
                raise DecisionError("instructions are required", param=f"{where}.instructions")
            if (
                isinstance(question, Choice)
                and caps.max_choice_options is not None
                and len(question.criteria) > caps.max_choice_options
            ):
                raise DecisionError(
                    f"a choice can have at most {caps.max_choice_options} options",
                    param=f"{where}.criteria",
                )
            if (
                isinstance(question, Score)
                and caps.max_score_levels is not None
                and len(question.criteria) > caps.max_score_levels
            ):
                raise DecisionError(
                    f"a score can have at most {caps.max_score_levels} levels",
                    param=f"{where}.criteria",
                )


def load(model: str | Path, check_memory: bool = True, **options) -> DecisionModel:
    """Load a decision model from a local folder or a Hugging Face repo id.

    ``options`` go to the model family's loader. For Clef: ``max_input_tokens``
    (default 16,384; longer states are truncated), ``vision=True`` to load the
    vision tower now rather than with the first image request, and
    ``max_image_pixels`` (default 2**21; larger images are shrunk, None keeps
    only the processor's own maximum). For pplx: ``max_input_tokens`` (default
    and maximum 8,192 per question; longer input is refused). For
    H2O-Lightning: ``max_input_tokens`` (default and maximum 40,960 per
    question; states over 32,000 tokens, or too long for a lowered limit, are
    cut) and
    ``noul_floor`` (default the release's 0.801: less sure yes/no answers are
    reported at it; 0 turns it off). For d1: ``max_input_tokens`` (default
    and maximum 32,768 per question; states too long for it are cut). For
    Clef, H2O-Lightning and d1: ``prefix_cache_gb`` (memory for recently
    computed states, default 2; 0 turns it off). For Laya and
    Julia: ``max_input_tokens`` (default: the release's own limit, at most
    8,192) and ``dtype`` (``"float32"``, ``"float16"`` or ``"bfloat16"``;
    default float16); for Julia also ``strict_encoding`` (default True, as its release: refuse
    requests that would have to be cut).

    Before reading the weights, ``load`` checks that the model fits in the
    GPU's recommended working set (its weights plus 10%) and raises
    ``ModelTooLargeError`` if not; ``check_memory=False`` skips the check.
    """
    require_metal()
    path = resolve_model_path(model)
    if check_memory:
        check_fits(path, options, name=str(model))
    backend = load_backend(path, **options)
    if is_repo_id(model):
        # Named after the repo, not the cache's snapshot folder.
        backend.name = str(model).rstrip("/").rsplit("/", 1)[-1]
    return DecisionModel(backend)


def _lift_state_images(request: Request) -> tuple[Request, dict[int, str]]:
    """Images in the state replaced by ``[image N]`` and added after ``images``."""
    first = len(request.images or [])
    state, found = lift_images(request.state, first)
    if not found:
        return request, {}
    images = [*(request.images or []), *(image for image, _ in found)]
    lifted = {first + index: where for index, (_, where) in enumerate(found)}
    return request.model_copy(update={"state": state, "images": images}), lifted


def _in_state(error: DecisionError, lifted: dict[int, str]) -> DecisionError:
    """An error about a lifted image, naming where it was in the state."""
    match = re.fullmatch(r"images\.(\d+)", error.param or "")
    if match is None or int(match[1]) not in lifted:
        return error
    return DecisionError(error.message, param=lifted[int(match[1])], code=error.code)
