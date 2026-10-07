"""The public entry points: ``load`` a model and ask it to ``decide``."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .answers import build_answer
from .backend import Backend
from .errors import DecisionError, require_metal
from .hub import is_repo_id, resolve_model_path
from .memory import check_fits
from .registry import load_backend
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
        request = self.check(request)
        output = self.backend.score(request)
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
        request = parse_request(request)
        self._check_supported(request)
        return request

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
    only the processor's own maximum). For Laya and Julia: ``max_input_tokens``
    (default: the release's own limit, at most 8,192) and ``dtype``
    (``"float32"``, ``"float16"`` or ``"bfloat16"``; default float16); for
    Julia also ``strict_encoding`` (default True, as its release: refuse
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
