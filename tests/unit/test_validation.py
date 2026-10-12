"""Invalid requests are rejected, naming the field and the problem, before any model work."""

import pytest

from fake_backend import FakeBackend
from mlx_decision import DecisionError, DecisionModel
from mlx_decision.types import MAX_DEPTH, Choice, Noul, Request, Score, decode_json, parse_request


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
        (body(**{"a\ud800": {"type": "noul"}}), "questions", "question id 'a\\ud800'"),
        # The id is the error, not what pydantic finds in its question.
        (body(**{"a\ud800": {"type": "rank"}}), "questions", "is not valid text"),
        ({**body(q={"type": "noul"}), "state": float("inf")}, "state", "not finite"),
        ({**body(q={"type": "noul"}), "state": {"a": [1, float("nan")]}}, "state", "not finite"),
        (
            body(q={"type": "noul", "instructions": {"limit": -float("inf")}}),
            "questions.q.instructions",
            "not finite",
        ),
        (
            body(q={"type": "score", "criteria": [0, float("inf")]}),
            "questions.q.criteria",
            "not finite",
        ),
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


def nested(depth: int) -> list:
    """Lists nested ``depth`` levels deep (``[]`` is one level)."""
    value: list = []
    for _ in range(depth - 1):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("data", "param"),
    [
        ({**body(q={"type": "noul"}), "state": nested(MAX_DEPTH + 1)}, "state"),
        # Deeper than Python's own encoder can write; no RecursionError.
        ({**body(q={"type": "noul"}), "state": {"a": nested(5000)}}, "state"),
        (
            body(q={"type": "noul", "instructions": nested(MAX_DEPTH + 1)}),
            "questions.q.instructions",
        ),
        (body(q={"type": "score", "criteria": [0, nested(MAX_DEPTH)]}), "questions.q.criteria"),
    ],
)
def test_nesting_deeper_than_the_limit_is_refused(data, param):
    with pytest.raises(DecisionError) as caught:
        parse_request(data)
    assert caught.value.param == param
    assert f"nested deeper than {MAX_DEPTH} levels" in caught.value.message


def test_nesting_up_to_the_limit_is_fine():
    parse_request({**body(q={"type": "noul"}), "state": {"a": nested(MAX_DEPTH - 1)}})
    parse_request(Request(state=nested(MAX_DEPTH), questions={"q": Noul()}))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{broken", "the body is not valid JSON: "),
        (b'"caf\xe9"', "the body is not valid JSON: "),
        ("[" * 100_000 + "]" * 100_000, "the body is nested too deeply"),
        ("1" * 5000, "the body holds a number too long to read: Exceeds the limit"),
    ],
    ids=["broken", "not-utf8", "deep", "long-number"],
)
def test_decode_json_turns_every_refusal_into_a_decision_error(text, message):
    with pytest.raises(DecisionError) as caught:
        decode_json(text, "the body", param="body")
    assert caught.value.message.startswith(message)
    assert caught.value.param == "body"


def test_decode_json_without_a_subject():
    with pytest.raises(DecisionError) as caught:
        decode_json("{")
    assert caught.value.message.startswith("not valid JSON: ")


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
    assert parse_request(request) == request


def test_typed_requests_are_checked_like_bodies():
    with pytest.raises(DecisionError) as caught:
        parse_request(Request(state={1, 2}, questions={"q": Noul()}))
    assert (caught.value.param, "not JSON data" in caught.value.message) == ("state", True)
    with pytest.raises(DecisionError) as caught:
        parse_request(Request(state="text", questions={"": Noul()}))
    assert caught.value.message == "question ids must not be empty"
    with pytest.raises(DecisionError) as caught:
        parse_request(Request(state=float("nan"), questions={"q": Noul()}))
    assert caught.value.param == "state"
    # Changed after it was built: pydantic does not check assignments.
    changed = Request(state="text", questions={"q": Choice(criteria={"a": None})})
    changed.questions["q"].criteria = {}
    with pytest.raises(DecisionError) as caught:
        parse_request(changed)
    assert caught.value.param == "questions.q.criteria"


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


def test_a_checked_typed_request_keeps_which_fields_were_given():
    # H2O writes left-out instructions as empty, an explicit null as JSON.
    request = Request(state="x", questions={"a": Noul(), "b": Noul(instructions=None)})
    checked = parse_request(request)
    assert checked.questions["a"].model_fields_set == set()
    assert checked.questions["b"].model_fields_set == {"instructions"}


def test_invalid_typed_requests_never_reach_the_backend():
    error = decide_with(Request(state={1, 2}, questions={"q": Noul()}))
    assert error.param == "state"


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
