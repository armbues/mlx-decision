"""Training data: reading, checking and writing JSON lines, and the seeded splits."""

import base64
import json
from collections import Counter
from pathlib import Path

import pytest

from mlx_decision.data.format import (
    DataError,
    Example,
    parse_examples,
    read_examples,
    split,
    training_splits,
    write_examples,
)
from mlx_decision.errors import DecisionError

QUESTIONS = {
    "team": {"type": "choice", "criteria": {"billing": "Money", "tech": "Bugs"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "mood": {"type": "score", "criteria": ["Calm", "Annoyed", "Angry"]},
}


def line(labels, state="My invoice is wrong", **extra) -> str:
    return json.dumps({**extra, "state": state, "questions": QUESTIONS, "labels": labels})


def error_of(*lines) -> DataError:
    with pytest.raises(DataError) as caught:
        parse_examples(lines)
    return caught.value


def test_labels_become_option_ids_and_ids_default_to_the_line_number():
    lines = [
        line({"team": "billing", "urgent": True, "mood": 2}),
        "",
        line({"urgent": "false", "mood": "0"}, id="ticket-7"),
    ]
    first, second = parse_examples(lines)
    assert (first.id, first.labels) == ("1", {"team": "billing", "urgent": "true", "mood": "2"})
    assert (second.id, second.labels) == ("ticket-7", {"urgent": "false", "mood": "0"})
    assert first.request == {"state": "My invoice is wrong", "questions": QUESTIONS}


def test_written_examples_read_back_the_same(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(line({"team": "tech", "urgent": False, "mood": 1}) + "\n", encoding="utf-8")
    examples = read_examples(path)
    out = tmp_path / "out.jsonl"
    assert write_examples(out, examples) == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["id"] == "1" and data["labels"] == {"team": "tech", "urgent": False, "mood": 1}
    assert read_examples(out) == examples


def test_unlabelled_questions_are_allowed():
    (example,) = parse_examples([line({"team": "tech"})])
    assert example.labels == {"team": "tech"}
    first, _ = parse_examples([line({}), line({"team": "tech"})])
    assert first.labels == {}


@pytest.mark.parametrize(
    ("labels", "param", "message"),
    [
        ({"team": "sales"}, "labels.team", '"sales" is none of the options (billing, tech)'),
        ({"team": 1}, "labels.team", "is none of the options"),
        ({"urgent": "yes"}, "labels.urgent", "a noul label is true or false"),
        ({"mood": 3}, "labels.mood", "a score label is a level from 0 to 2"),
        ({"mood": True}, "labels.mood", "a score label"),
        ({"mood": "-1"}, "labels.mood", "a score label"),
        ({"other": "x"}, "labels.other", "the request has no question with this id"),
    ],
)
def test_a_label_that_is_no_answer_names_line_and_question(labels, param, message):
    error = error_of(line({"team": "tech"}), line(labels))
    assert (error.line, error.param) == (2, param)
    assert message in error.message
    assert str(error).startswith(f"line 2: {param}: ")


def test_requests_get_the_checks_every_model_makes():
    bad = json.dumps({"state": "x", "questions": {"q": {"type": "choice"}}, "labels": {}})
    error = error_of(bad)
    assert (error.line, error.param) == (1, "questions.q.criteria")


@pytest.mark.parametrize(
    ("text", "param", "message"),
    [
        ("{not json", None, "not valid JSON"),
        ("[1, 2]", None, "a line must be an object"),
        (json.dumps({"state": "x", "questions": QUESTIONS}), None, "no labels"),
        (line(["tech"]), "labels", "labels must be an object"),
        (line({}, id=7), "id", "an id is a non-empty string"),
        (line({}, id=" "), "id", "an id is a non-empty string"),
    ],
)
def test_malformed_lines(text, param, message):
    error = error_of(text)
    assert (error.line, error.param) == (1, param)
    assert message in error.message


def test_invalid_utf8_names_the_line():
    error = error_of(line({}).encode(), b'{"state": "\xff"}')
    assert error.line == 2 and "UTF-8" in error.message


def test_ids_are_unique_including_line_numbers():
    error = error_of(line({}), line({}, id="1"))
    assert (error.line, error.param, error.message) == (2, "id", "the same id as line 1")


def test_images_are_refused():
    error = error_of(line({}, images=["data:image/png;base64,AAAA"]))
    assert (error.param, error.message) == ("images", "training data cannot hold images")
    png = base64.b64encode(bytes.fromhex("89504e470d0a1a0a0000000d49484452") + bytes(64)).decode()
    error = error_of(line({}, state={"text": "hi", "photo": f"data:image/png;base64,{png}"}))
    assert (error.param, error.message) == ("state.photo", "training data cannot hold images")


def test_a_file_without_examples_or_labels_is_refused():
    with pytest.raises(DecisionError, match="holds no examples"):
        parse_examples(["", "  "])
    with pytest.raises(DecisionError, match="no example has a label"):
        parse_examples([line({})])


def labelled(count: int, labels=("a", "b", "c"), weights=(6, 3, 1)) -> list[Example]:
    cycle = [label for label, weight in zip(labels, weights, strict=True) for _ in range(weight)]
    return [
        Example(f"e{i}", {"state": i, "questions": {}}, {"q": cycle[i % len(cycle)]})
        for i in range(count)
    ]


def test_split_is_seeded_stratified_and_keeps_the_order():
    examples = labelled(1000)
    rest, part = split(examples, 0.2, seed=0)
    assert (len(rest), len(part)) == (800, 200)
    assert {e.id for e in rest} | {e.id for e in part} == {e.id for e in examples}
    assert Counter(e.labels["q"] for e in part) == {"a": 120, "b": 60, "c": 20}
    assert [int(e.id[1:]) for e in part] == sorted(int(e.id[1:]) for e in part)
    assert split(examples, 0.2, seed=0) == (rest, part)
    assert split(examples, 0.2, seed=1)[1] != part


def test_split_sizes_add_up_with_small_strata():
    examples = labelled(23, labels=("a", "b", "c", "d"), weights=(5, 3, 2, 1))
    rest, part = split(examples, 0.3, seed=5)
    assert len(part) == round(0.3 * 23) and len(rest) + len(part) == 23
    assert len(split(examples, 0.01, seed=0, minimum=4)[1]) == 4
    assert len(split(examples[:1], 0.5)[1]) == 0  # never all of them


def test_training_splits():
    splits = training_splits(labelled(1000), seed=3)
    assert (len(splits.holdout), len(splits.fit), len(splits.dev)) == (200, 720, 80)
    assert training_splits(labelled(1000), seed=3) == splits
    # dev: at least 50 ...
    splits = training_splits(labelled(300))
    assert (len(splits.holdout), len(splits.fit), len(splits.dev)) == (60, 190, 50)
    # ... but at most half of the rest
    splits = training_splits(labelled(40))
    assert (len(splits.holdout), len(splits.fit), len(splits.dev)) == (8, 16, 16)
    ids = [e.id for part in (splits.holdout, splits.fit, splits.dev) for e in part]
    assert sorted(ids) == sorted(e.id for e in labelled(40))


def test_a_validation_set_replaces_the_holdout():
    validation = labelled(30)
    splits = training_splits(labelled(600), validation=validation, holdout=5)
    assert splits.holdout == validation
    assert (len(splits.fit), len(splits.dev)) == (540, 60)
    with pytest.raises(ValueError, match="between 0 and 1"):
        training_splits(labelled(600), holdout=1.0)
