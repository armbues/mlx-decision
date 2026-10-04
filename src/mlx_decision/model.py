"""The public entry points: ``load`` a model and ask it to ``decide``."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .answers import build_answer
from .backend import Backend
from .errors import DecisionError
from .hub import resolve_model_path
from .registry import load_backend
from .types import Choice, Request, Result, Score, Usage, parse_request


class DecisionModel:
    def __init__(self, backend: Backend):
        self.backend = backend

    @property
    def name(self) -> str:
        return self.backend.name

    def decide(self, state: Any, questions: Mapping[str, Any]) -> Result:
        """Answer ``questions`` (dicts or ``Choice``/``Score``/``Noul``) about ``state``."""
        return self.decide_request({"state": state, "questions": dict(questions)})

    def decide_request(self, request: Request | Mapping[str, Any]) -> Result:
        """Answer a whole request body in the wire format."""
        request = parse_request(request)
        self._check_supported(request)
        output = self.backend.score(request)
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

    def _check_supported(self, request: Request) -> None:
        caps = self.backend.capabilities
        if (request.images or request.videos) and not caps.supports_media:
            raise DecisionError(
                "images and videos are not supported yet",
                param="images" if request.images else "videos",
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
            if isinstance(question, Choice) and caps.max_choice_options is not None:
                if len(question.criteria) > caps.max_choice_options:
                    raise DecisionError(
                        f"a choice can have at most {caps.max_choice_options} options",
                        param=f"{where}.criteria",
                    )
            if isinstance(question, Score) and caps.max_score_levels is not None:
                if len(question.criteria) > caps.max_score_levels:
                    raise DecisionError(
                        f"a score can have at most {caps.max_score_levels} levels",
                        param=f"{where}.criteria",
                    )


def load(model: str | Path, **options) -> DecisionModel:
    """Load a decision model from a local folder or a Hugging Face repo id."""
    return DecisionModel(load_backend(resolve_model_path(model), **options))
