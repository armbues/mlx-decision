"""Invalid requests are rejected, naming the field and the problem, before any model work."""

import pytest

from fake_backend import FakeBackend
from mlx_decision import DecisionError, DecisionModel
from mlx_decision.types import Choice, Noul, Request, Score, parse_request


def body(**questions):
    return {"state": "text", "questions": questions}


@pytest.mark.parametrize(
    ("data", "param", "message"),
    [
        (
            body(q={"type": "rank", "criteria": {"a": None}}),
            "questions.q.type",
            "unknown question type 'rank'",
        ),
        (body(q={"criteria": {"a": None}}), "questions.q.type", "type is required"),
        (body(q={"type": "choice"}), "questions.q.criteria", "criteria is required"),
        (
            body(q={"type": "choice", "criteria": {}}),
            "questions.q.criteria",
            "at least one option",
        ),
        (
            body(q={"type": "choice", "criteria": ["a", "b"]}),
            "questions.q.criteria",
            "must map option ids",
        ),
        (body(q={"type": "score"}), "questions.q.criteria", "criteria is required"),
        (
            body(q={"type": "score", "criteria": ["only"]}),
            "questions.q.criteria",
            "at least two levels",
        ),
        (
            body(q={"type": "score", "criteria": {"a": "b"}}),
            "questions.q.criteria",
            "must be a list",
        ),
        (
            body(q={"type": "noul", "criteria": {"yes": "y"}}),
            "questions.q.criteria.yes",
            "'true' and 'false'",
        ),
        (body(), "questions", "at least one question"),
        ({"state": "x", "questions": []}, "questions", "an object mapping question ids"),
        (body(**{"": {"type": "noul"}}), "questions", "question ids must not be empty"),
        (body(q={"type": "noul", "criteria": "x"}), "questions.q.criteria", "must be an object"),
        ({**body(q={"type": "noul"}), "images": "data:x"}, "images", "images must be a list"),
        ({**body(q={"type": "noul"}), "state": "a\ud800"}, "state", "not valid text"),
        ({**body(q={"type": "noul"}), "state": {1: "a", "b": 2}}, "state", "mixes key types"),
        (
            body(q={"type": "choice", "criteria": {"a": {1, 2}}}),
            "questions.q.criteria",
            "not JSON data",
        ),
        (
            body(q={"type": "noul", "instructions": b"bytes"}),
            "questions.q.instructions",
            "not JSON data",
        ),
        ({"state": "text"}, "questions", "questions is required"),
        ({"questions": {"q": {"type": "noul"}}}, "state", "state is required"),
        (body(q="Is it urgent?"), "questions.q", "must be an object"),
        (["text"], None, "must be a JSON object"),
        # The question id in the path is the one at fault, not the first.
        (
            body(ok={"type": "noul"}, bad={"type": "score", "criteria": []}),
            "questions.bad.criteria",
            "at least two levels",
        ),
    ],
)
def test_structure_errors(data, param, message):
    with pytest.raises(DecisionError) as caught:
        parse_request(data)
    assert caught.value.param == param
    assert message in caught.value.message
    if param:
        assert str(caught.value).startswith(f"{param}: ")


def test_valid_request_parses_into_typed_questions():
    request = parse_request(
        body(
            a={"type": "noul"},
            b={"type": "choice", "criteria": {"x": None}},
            c={"type": "score", "criteria": [1, 2]},
        )
    )
    assert isinstance(request, Request)
    assert [type(q) for q in request.questions.values()] == [Noul, Choice, Score]
    assert parse_request(request) is request


def test_unknown_top_level_fields_are_ignored():
    request = parse_request({**body(a={"type": "noul"}), "model": "jev-latest", "seed": 1})
    assert request.model == "jev-latest"


class CountingBackend(FakeBackend):
    calls = 0

    def score(self, request):
        self.calls += 1
        return super().score(request)


def decide_with(data, **capabilities):
    backend = CountingBackend(**capabilities)
    model = DecisionModel(backend)
    with pytest.raises(DecisionError) as caught:
        model.decide_request(data)
    assert backend.calls == 0
    return caught.value


def test_invalid_structure_never_reaches_the_backend():
    error = decide_with(body(q={"type": "rank"}))
    assert error.param == "questions.q.type"


def test_unsupported_question_type():
    error = decide_with(
        body(a={"type": "noul"}, b={"type": "score", "criteria": [1, 2]}),
        question_types=frozenset({"noul", "choice"}),
    )
    assert error.param == "questions.b.type"
    assert "score" in error.message


def test_missing_instructions_when_required():
    error = decide_with(body(q={"type": "noul", "instructions": ""}), requires_instructions=True)
    assert error.param == "questions.q.instructions"


def test_missing_instructions_are_fine_when_not_required():
    model = DecisionModel(FakeBackend())
    assert model.decide_request(body(q={"type": "noul"})).answers["q"].noul == 0.75


def test_choice_option_limit():
    criteria = {str(i): None for i in range(4)}
    error = decide_with(body(q={"type": "choice", "criteria": criteria}), max_choice_options=3)
    assert error.param == "questions.q.criteria"
    assert "at most 3 options" in error.message
    DecisionModel(FakeBackend(max_choice_options=4)).decide_request(
        body(q={"type": "choice", "criteria": criteria})
    )


def test_score_level_limit():
    error = decide_with(body(q={"type": "score", "criteria": list(range(11))}), max_score_levels=10)
    assert error.param == "questions.q.criteria"
    assert "at most 10 levels" in error.message


@pytest.mark.parametrize(
    ("media", "capabilities", "message"),
    [
        ("images", {}, "fake does not support images"),
        ("videos", {}, "videos are not supported yet"),
        ("videos", {"supports_images": True}, "videos are not supported yet"),
    ],
)
def test_media_is_rejected_when_not_supported(media, capabilities, message):
    error = decide_with({**body(q={"type": "noul"}), media: ["data:..."]}, **capabilities)
    assert error.param == media
    assert error.code == "unsupported_media"
    assert error.message == message


def test_empty_media_lists_are_fine():
    model = DecisionModel(FakeBackend())
    model.decide_request({**body(q={"type": "noul"}), "images": [], "videos": None})


def test_non_finite_probabilities_never_become_answers():
    model = DecisionModel(FakeBackend(fixed={"q": {"true": float("nan"), "false": 0.5}}))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.decide("text", {"q": {"type": "noul"}})
