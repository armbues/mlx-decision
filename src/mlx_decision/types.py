"""The wire format: requests, questions, answers and responses.

These models mirror the JSON bodies of ``POST /v1/systemone``.
"""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import DecisionError

WIRE_DECIMALS = 4


class Noul(BaseModel):
    """A yes/no question. The answer is the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: Any = None
    criteria: dict[Literal["true", "false"], Any] | None = None


class Choice(BaseModel):
    """Pick one option. ``criteria`` maps option id to a description (or None)."""

    type: Literal["choice"] = "choice"
    instructions: Any = None
    criteria: dict[str, Any] = Field(min_length=1)


class Score(BaseModel):
    """Rate on ordered levels. ``criteria`` lists the level descriptions."""

    type: Literal["score"] = "score"
    instructions: Any = None
    criteria: list[Any] = Field(min_length=2)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class Request(BaseModel):
    """A request body: a ``state`` (any JSON value), ``questions`` by id, optional ``images``.

    ``model`` is accepted for compatibility with API clients and ignored;
    unknown fields are ignored too.
    """

    model_config = ConfigDict(extra="ignore")

    state: Any
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None
    images: list[Any] | None = None
    videos: list[Any] | None = None


class NoulAnswer(BaseModel):
    """``noul``: the probability that the answer is yes."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    """The most likely option, its confidence, and every option's probability."""

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    """``score``: the expected level (0-based); ``legend`` maps levels to their descriptions."""

    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    """Tokens the model read; decision models write none."""

    input_tokens: int
    output_tokens: int = 0


class Response(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class Result(Response):
    """A response plus what the wire format has no place for."""

    truncated: bool = Field(default=False, exclude=True)

    def to_wire(self) -> dict[str, Any]:
        """The response body as plain JSON data, with floats rounded."""
        return _round_floats(self.model_dump())


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, WIRE_DECIMALS)
    if isinstance(value, dict):
        return {k: _round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v) for v in value]
    return value


def parse_request(data: Any) -> Request:
    """Validate a request body, raising ``DecisionError`` naming the bad field."""
    if isinstance(data, Request):
        return data
    try:
        request = Request.model_validate(data)
    except ValidationError as error:
        message, loc = _describe(error.errors()[0])
        raise DecisionError(message, param=".".join(loc) or None) from None
    _check_text(request.state, "state")
    for question_id, question in request.questions.items():
        if not question_id.strip():
            raise DecisionError("question ids must not be empty", param="questions")
        where = f"questions.{question_id}"
        _check_text(question.instructions, f"{where}.instructions")
        _check_text(question.criteria, f"{where}.criteria")
    return request


def _check_text(value: Any, param: str) -> None:
    """Fail unless ``value`` can be written as JSON text (models read it as text)."""
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        # TypeError: not JSON data (a date, a set, mixed key types);
        # UnicodeEncodeError (a ValueError): a lone surrogate in a string.
        field = param.rsplit(".", 1)[-1]
        if isinstance(error, UnicodeError):
            message = f"{field} is not valid text: {error.reason} ({error.object[error.start]!r})"
        elif "not supported between instances" in str(error):
            message = f"{field} mixes key types (keys must be strings)"
        else:
            message = f"{field} is not JSON data: {error}"
        raise DecisionError(message, param=param) from None


QUESTION_TYPES = "noul, choice or score"


def _describe(error: dict[str, Any]) -> tuple[str, list[str]]:
    """A readable message and the field path for one pydantic error."""
    loc = [str(part) for part in error["loc"]]
    # Drop the union tag pydantic inserts after a question id, and the marker
    # it appends when a mapping key is at fault.
    if len(loc) > 2 and loc[0] == "questions" and loc[2] in ("noul", "choice", "score"):
        question_type = loc.pop(2)
    else:
        question_type = None
    if loc and loc[-1] == "[key]":
        loc.pop()
    kind = error["type"]
    field = loc[-1] if loc else "request"

    if kind == "union_tag_invalid":
        found = error["ctx"]["tag"]
        return f"unknown question type {found!r}; expected {QUESTION_TYPES}", [*loc, "type"]
    if kind == "union_tag_not_found":
        return f"type is required: {QUESTION_TYPES}", [*loc, "type"]
    if kind == "missing":
        return f"{field} is required", loc
    if kind == "too_short":
        if loc == ["questions"]:
            return "at least one question is required", loc
        if question_type == "choice":
            return "a choice needs at least one option", loc
        if question_type == "score":
            return "a score needs at least two levels", loc
    if kind == "literal_error" and question_type == "noul":
        return "noul criteria can only describe 'true' and 'false'", loc
    if kind in ("model_type", "model_attributes_type", "dict_type") and not loc:
        return "the request body must be a JSON object", loc
    if kind == "dict_type" and loc == ["questions"]:
        return "questions must be an object mapping question ids to questions", loc
    if kind == "list_type" and loc in (["images"], ["videos"]):
        return f"{field} must be a list", loc
    if kind in ("model_attributes_type", "dict_type") and len(loc) == 2 and loc[0] == "questions":
        return "a question must be an object with a type", loc
    if kind == "dict_type" and question_type == "choice":
        return "choice criteria must map option ids to descriptions", loc
    if kind == "dict_type" and question_type == "noul":
        return "noul criteria must be an object describing 'true' and/or 'false'", loc
    if kind == "list_type" and question_type == "score":
        return "score criteria must be a list of level descriptions", loc
    return error["msg"], loc
