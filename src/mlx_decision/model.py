"""The public entry points: ``load`` a model and ask it to ``decide``."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .answers import build_answer
from .backend import Backend
from .errors import DecisionError, require_metal
from .hub import is_repo_id, resolve_model_path
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


def load(model: str | Path, **options) -> DecisionModel:
    """Load a decision model from a local folder or a Hugging Face repo id.

    ``options`` go to the model family's loader. For Clef: ``max_input_tokens``
    (default 16,384; longer states are truncated), ``vision=True`` to load the
    vision tower now rather than with the first image request, and
    ``max_image_pixels`` (default 2**21; larger images are shrunk, None keeps
    only the processor's own maximum). For Laya and Julia: ``max_input_tokens``
    (default: the release's own limit, at most 8,192) and ``dtype``
    (``"float32"``, ``"float16"`` or ``"bfloat16"``; default: as stored); for
    Julia also ``strict_encoding`` (default True, as its release: refuse
    requests that would have to be cut).
    """
    require_metal()
    backend = load_backend(resolve_model_path(model), **options)
    if is_repo_id(model):
        # Named after the repo, not the cache's snapshot folder.
        backend.name = str(model).rstrip("/").rsplit("/", 1)[-1]
    return DecisionModel(backend)
