"""The wire format: requests, questions, answers and responses.

These models mirror the JSON bodies of ``POST /v1/systemone``.
"""

import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import DecisionError

WIRE_DECIMALS = 4
# Deepest nesting of a state, instructions or criteria. Families render JSON
# with Python's own encoder, which recurses once per level.
MAX_DEPTH = 100

# Validating an instance checks it again (it may have been changed since it
# was built) and keeps which fields were given: H2O tells a left-out field
# from an explicit null.
REVALIDATE = ConfigDict(revalidate_instances="always")


class Noul(BaseModel):
    """A yes/no question. The answer is the probability of yes."""

    model_config = REVALIDATE

    type: Literal["noul"] = "noul"
    instructions: Any = None
    criteria: dict[Literal["true", "false"], Any] | None = None


class Choice(BaseModel):
    """Pick one option. ``criteria`` maps option id to a description (or None)."""

    model_config = REVALIDATE

    type: Literal["choice"] = "choice"
    instructions: Any = None
    criteria: dict[str, Any] = Field(min_length=1)


class Score(BaseModel):
    """Rate on ordered levels. ``criteria`` lists the level descriptions."""

    model_config = REVALIDATE

    type: Literal["score"] = "score"
    instructions: Any = None
    criteria: list[Any] = Field(min_length=2)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class Request(BaseModel):
    """A request body: a ``state`` (any JSON value), ``questions`` by id, optional ``images``.

    ``model`` is accepted for compatibility with API clients and ignored;
    unknown fields are ignored too.
    """

    model_config = ConfigDict(extra="ignore", revalidate_instances="always")

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


def decode_json(text: str | bytes, subject: str | None = None, param: str | None = None) -> Any:
    """``json.loads``, with each way it refuses input as a ``DecisionError``.

    ``subject`` names the input in the message ("the body is not valid JSON").
    """
    try:
        return json.loads(text)
    except RecursionError:
        problem = "is nested too deeply"
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        problem = f"is not valid JSON: {error}"
    except ValueError as error:  # an integer longer than Python converts
        problem = f"holds a number too long to read: {str(error).split(';')[0]}"
    message = f"{subject} {problem}" if subject else problem.removeprefix("is ")
    raise DecisionError(message, param=param)


def parse_request(data: Any) -> Request:
    """Validate a request body, raising ``DecisionError`` naming the bad field.

    A ``Request`` is checked like a body (it may hold anything in ``state``,
    or have been changed since it was built), and comes back as a new one.
    """
    # Before pydantic, whose error paths would show such an id garbled.
    if isinstance(data, Request):
        questions = data.questions
    else:
        questions = data.get("questions") if isinstance(data, Mapping) else None
    for question_id in questions if isinstance(questions, Mapping) else ():
        if isinstance(question_id, str) and not _is_text(question_id):
            raise DecisionError(
                f"question id {ascii(question_id)} is not valid text", param="questions"
            )
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


def _is_text(text: str) -> bool:
    """Whether ``text`` can be written as UTF-8 (a lone surrogate cannot)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _depth(value: Any) -> int:
    """How deeply ``value`` nests lists and objects, counted without recursion
    and only up to one level past ``MAX_DEPTH``."""
    deepest, stack = 0, [(value, 0)]
    while stack and deepest <= MAX_DEPTH:
        item, depth = stack.pop()
        if isinstance(item, dict):
            item = item.values()
        elif not isinstance(item, (list, tuple)):
            continue
        deepest = max(deepest, depth + 1)
        stack.extend((child, depth + 1) for child in item)
    return deepest


def _check_text(value: Any, param: str) -> None:
    """Fail unless ``value`` can be written as JSON text (models read it as text)."""
    if _depth(value) > MAX_DEPTH:
        field = param.rsplit(".", 1)[-1]
        raise DecisionError(f"{field} is nested deeper than {MAX_DEPTH} levels", param=param)
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        # TypeError: not JSON data (a date, a set, mixed key types);
        # UnicodeEncodeError (a ValueError): a lone surrogate in a string;
        # ValueError: NaN or infinity (which a JSON body can spell, e.g. 1e400).
        field = param.rsplit(".", 1)[-1]
        if isinstance(error, UnicodeError):
            message = f"{field} is not valid text: {error.reason} ({error.object[error.start]!r})"
        elif str(error).startswith("Out of range float"):
            message = f"{field} holds a number that is not finite (NaN or infinity)"
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
