"""Presets: rows of the five datasets turned into examples, without the datasets."""

import pytest

from mlx_decision.data.presets import PRESETS, digest, examples

ROWS = {
    "trec": {"text": "Who wrote Hamlet?", "coarse_label": 3, "fine_label": 29},
    "offensive": {"text": "@user what a lovely day", "label": 0},
    "anli-r3": {
        "uid": "abc-1",
        "premise": "A cat sleeps.",
        "hypothesis": "An animal rests.",
        "label": 0,
        "reason": "",
    },
    "openbookqa": {
        "id": "7-980",
        "question_stem": "The sun is responsible for",
        "choices": {"text": ["puppies", "plants", "rain", "night"], "label": list("ABCD")},
        "answerKey": "B",
    },
    "sst5": {"text": "A gorgeous film.", "label": 4, "label_text": "very positive"},
}


def test_every_preset_builds_one_question_with_its_gold_among_the_options():
    golds = {"trec": "human", "offensive": "false", "anli-r3": "entailment"}
    golds |= {"openbookqa": "B", "sst5": "4"}
    for name, preset in PRESETS.items():
        (example,) = examples(preset, [ROWS[name]])
        assert list(example) == ["id", "state", "questions", "gold"]
        (question,) = example["questions"].values()
        assert example["gold"] == golds[name]
        if question["type"] == "score":
            assert int(example["gold"]) < len(question["criteria"])
        else:
            assert example["gold"] in question["criteria"]


def test_ids_are_the_suites_on_the_evaluation_split_and_carry_other_splits():
    assert examples(PRESETS["trec"], [ROWS["trec"]] * 2)[1]["id"] == "trec:1"
    assert examples(PRESETS["trec"], [ROWS["trec"]], "train")[0]["id"] == "trec:train/0"
    assert examples(PRESETS["anli-r3"], [ROWS["anli-r3"]])[0]["id"] == "anli_r3:abc-1"
    assert examples(PRESETS["anli-r3"], [ROWS["anli-r3"]], "train_r3")[0]["id"] == (
        "anli_r3:train_r3/abc-1"
    )
    assert examples(PRESETS["openbookqa"], [ROWS["openbookqa"]])[0]["id"] == "openbookqa:7-980"


def test_anli_state_holds_premise_and_hypothesis():
    (example,) = examples(PRESETS["anli-r3"], [ROWS["anli-r3"]])
    assert example["state"] == {"premise": "A cat sleeps.", "hypothesis": "An animal rests."}


def test_a_label_outside_the_map_names_the_row():
    rows = [ROWS["anli-r3"], {**ROWS["anli-r3"], "label": -1}]
    with pytest.raises(ValueError, match="anli-r3 dev_r3 row 1: label -1 is none of the labels"):
        examples(PRESETS["anli-r3"], rows, "dev_r3")


def test_digest_is_stable():
    assert digest([{"b": 1, "a": "é"}]) == digest([{"a": "é", "b": 1}])
    # The saved copies of the coverage suite are checked against digests made this way.
    assert digest([{"a": 1}]) == "e8c628edc9968ef0"
