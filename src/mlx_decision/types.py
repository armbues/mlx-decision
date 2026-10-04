"""The wire format: requests, questions, answers and responses.

These models mirror the JSON bodies of ``POST /v1/systemone``.
"""

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
    model_config = ConfigDict(extra="ignore")

    state: Any
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None
    images: list[Any] | None = None
    videos: list[Any] | None = None


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
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
        return Request.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        # Drop the union tag pydantic inserts after a question id.
        loc = [str(part) for part in first["loc"]]
        if len(loc) > 2 and loc[0] == "questions" and loc[2] in ("noul", "choice", "score"):
            del loc[2]
        raise DecisionError(first["msg"], param=".".join(loc) or None) from None
