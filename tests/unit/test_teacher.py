"""Teacher labels: answers as soft or hard labels the training format reads
back as given, what a teacher's folder says about its calibration, and when
a student would inherit poor or unknown calibration."""

import json

import pytest

from fake_backend import TEAM, write_fitted, write_model
from mlx_decision.answers import build_answer
from mlx_decision.data.dataset import read_for
from mlx_decision.registry import MARKER_FILE
from mlx_decision.training import fit
from mlx_decision.training.teacher import (
    Teacher,
    confidence,
    describe_teacher,
    hard_label,
    label,
    soft_label,
    teacher_note,
)
from mlx_decision.types import parse_request

QUESTIONS = {
    "team": TEAM,
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    "mood": {"type": "score", "criteria": ["calm", "cross", "furious"]},
}
GIVEN = {
    "team": {"billing": 0.2, "tech": 0.7, "sales": 0.1},
    "urgent": {"true": 0.3, "false": 0.7},
    "mood": {"0": 0.1, "1": 0.3, "2": 0.6},
}


def answers() -> dict[str, dict]:
    questions = parse_request({"state": "s", "questions": QUESTIONS}).questions
    return {q: build_answer(questions[q], GIVEN[q]).model_dump() for q in QUESTIONS}


def test_soft_and_hard_labels_per_question_type():
    made = answers()
    assert soft_label(made["team"]) == pytest.approx(GIVEN["team"])
    assert soft_label(made["urgent"]) == pytest.approx(0.3)
    assert soft_label(made["mood"]) == pytest.approx(GIVEN["mood"])
    assert [hard_label(made[q]) for q in QUESTIONS] == ["tech", False, 2]
    assert confidence(made["team"]) == pytest.approx((0.7 - 1 / 3) / (2 / 3))  # above an even split
    assert confidence(made["urgent"]) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="soft or hard"):
        label(made["team"], "both")


def test_the_labels_read_back_as_the_answers(tmp_path):
    made = answers()
    path = tmp_path / "taught.jsonl"
    soft = {
        "state": "s",
        "questions": QUESTIONS,
        "labels": {q: soft_label(a) for q, a in made.items()},
    }
    hard = {
        "state": "t",
        "questions": QUESTIONS,
        "labels": {q: hard_label(a) for q, a in made.items()},
    }
    path.write_text(json.dumps(soft) + "\n" + json.dumps(hard) + "\n")
    first, second = read_for(path).examples
    assert first.targets["team"] == pytest.approx(GIVEN["team"])
    assert first.targets["urgent"] == pytest.approx(GIVEN["urgent"])
    assert first.targets["mood"] == pytest.approx(GIVEN["mood"])
    assert first.labels == {"team": "tech", "urgent": "false", "mood": "2"}
    assert second.targets is None and second.labels == first.labels


def test_a_release_has_no_closing_calibration(tmp_path):
    teacher = describe_teacher(write_model(tmp_path / "flash"))
    assert teacher == Teacher("flash")
    assert teacher.as_json() == {"name": "flash"}
    note = teacher_note(teacher, ["team"])
    assert note.startswith("flash has no closing calibration, so how far its probabilities")
    assert "fit flash on labelled rows first" in note


def test_a_fitted_teacher_names_its_base_calibration_and_report(tmp_path):
    base = write_model(tmp_path / "base")
    teams = ["billing", "tech", "sales"]
    lines = [{"state": f"t {i}", "questions": {"team": TEAM}, "labels": {"team": teams[i % 3]}}
             for i in range(90)]  # fmt: skip
    data = tmp_path / "tickets.jsonl"
    data.write_text("".join(json.dumps(line) + "\n" for line in lines))
    fit(base, data, output=tmp_path / "taught")
    teacher = describe_teacher(tmp_path / "taught")
    assert (teacher.name, teacher.base, teacher.kinds) == ("taught", "base", ("calibrate",))
    assert teacher.calibration == "auto" and teacher.calibrated == {"team"}
    assert set(teacher.ece) == {"team"}
    assert teacher.as_json()["calibrated"] == ["team"]
    # Another question than its fit's: unknown for that one.
    assert "taught has no closing calibration for urgent" in teacher_note(
        teacher, ["team", "urgent"]
    )


def test_a_fit_with_calibration_none_counts_as_uncalibrated(tmp_path):
    base = write_model(tmp_path / "base")
    folder = write_fitted(tmp_path / "taught", base)
    teacher = describe_teacher(folder)
    assert teacher.base == "base" and teacher.kinds == ("calibrate",)
    marker = json.loads((folder / MARKER_FILE).read_text())
    marker["fit"]["calibration"] = {"choice": "none", "questions": {}}
    (folder / MARKER_FILE).write_text(json.dumps(marker))
    none = describe_teacher(folder)
    assert none.calibration is None and none.calibrated == frozenset()
    assert teacher_note(none, ["team"]).startswith("taught has no closing calibration,")


@pytest.mark.parametrize(
    ("teacher", "measured", "expected"),
    [
        (Teacher("t", "b", ("calibrate",), "auto", frozenset({"q"}), {"q": 0.02}), None, None),
        (Teacher("t", "b", ("calibrate",), "auto", frozenset({"q"}), {"q": 0.09}), None,
         "t's fit left an ECE of q 9.0% on its report"),
        (Teacher("t"), {"q": 0.03}, None),  # measured on labelled rows: that counts
        (Teacher("t"), {"q": 0.135},
         "t's probabilities are off on the labelled rows (ECE q 13.5%)"),
    ],
)  # fmt: skip
def test_the_note_follows_what_is_known(teacher, measured, expected):
    note = teacher_note(teacher, ["q"], measured)
    if expected is None:
        assert note is None
    else:
        assert note.startswith(expected) and "inherits that" in note
