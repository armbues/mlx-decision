"""The ``chat`` session, typed through prompt_toolkit's pipe input."""

import json

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.validation import ValidationError

from fake_backend import write_model
from mlx_decision import load
from mlx_decision.interactive import OptionsValidator, Session

ENTER = "\r"
DOWN = "\x1b[B"
CLEAR_LINE = "\x15"  # Ctrl-U
BACKSPACE = "\x7f"
CTRL_C = "\x03"
CTRL_D = "\x04"

# Menu entries are picked by their number, then Enter; Enter alone keeps the default.
CHOICE, SCORE, NOUL = "1", "2", "3"
START, ADD = "1", "2"

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": {"about": "payments"}, "technical": None},
    },
    "anger": {"type": "score", "criteria": ["calm", "angry"]},
}


def lines(*parts: str) -> str:
    """Each part typed and followed by Enter."""
    return "".join(part + ENTER for part in parts)


def drive(model, typed: str, body=None, state_json=False):
    with create_pipe_input() as pipe:
        pipe.send_text(typed + CTRL_D)
        with create_app_session(input=pipe, output=DummyOutput()):
            session = Session(model, {} if body is None else body, state_json, as_json=True)
            session.run()
    return session


def bodies(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines()]


def test_builder_opens_without_questions(fake_model, capsys):
    typed = lines(
        CHOICE, "team", "Which team?", "billing: payments and refunds", "technical", "",
        ADD,
        NOUL, "", "Is it urgent?",
        START,
        "Stripe is down", "",
    )  # fmt: skip
    session = drive(fake_model, typed)
    assert session.questions == {
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "payments and refunds", "technical": None},
        },
        "noul_1": {"type": "noul", "instructions": "Is it urgent?"},
    }
    out, err = capsys.readouterr()
    assert "added:\n  team (choice): Which team?\n    billing: payments and refunds" in err
    assert "added:\n  noul_1 (noul): Is it urgent?" in err
    assert [list(b["answers"]) for b in bodies(out)] == [["team", "noul_1"]]


def test_julia_hints_at_undescribed_options(fake_model, capsys):
    fake_model.backend.family = "julia"
    typed = lines(
        CHOICE, "team", "", "billing: Billing and payment disputes", "sales", "",
        ADD,
        CHOICE, "kind", "", "bug: A bug or an error", "question: A question", "",
        START,
    )  # fmt: skip
    drive(fake_model, typed)
    err = capsys.readouterr().err
    assert "hint: team: Julia answers much worse without option descriptions" in err
    assert "/edit ID" in err
    assert "hint: kind" not in err


def test_the_type_menu_works_with_arrow_keys(fake_model):
    session = drive(fake_model, DOWN + ENTER + lines("anger", "", "calm", "angry", "", ""))
    assert session.questions == {"anger": {"type": "score", "criteria": ["calm", "angry"]}}


def test_cancelling_the_first_question_ends_the_session(fake_model, capsys):
    session = drive(fake_model, lines(CHOICE) + CTRL_C)
    assert session.questions == {}
    assert "cancelled" in capsys.readouterr().err


def test_an_invalid_question_is_asked_again_with_what_was_typed(fake_model, capsys):
    typed = lines(SCORE, "anger", "", "calm", "") + lines("", "", "", "angry", "")
    session = drive(fake_model, typed + lines(START))
    assert session.questions == {"anger": {"type": "score", "criteria": ["calm", "angry"]}}
    assert "questions.anger.criteria: a score needs at least two levels" in capsys.readouterr().err


def test_the_builder_checks_the_models_limits(tmp_path, capsys):
    model = load(write_model(tmp_path / "small", max_choice_options=2))
    typed = lines(CHOICE, "team", "", "a", "b", "c", "")
    typed += lines("", "", "", BACKSPACE * 2, START)  # the third option deleted
    session = drive(model, typed)
    assert "a choice can have at most 2 options" in capsys.readouterr().err
    assert list(session.questions["team"]["criteria"]) == ["a", "b"]


def test_a_noul_needs_its_text(fake_model):
    # The empty instructions are refused in place; the text typed next is taken.
    session = drive(fake_model, lines(NOUL, "q", "", "Is it?", START))
    assert session.questions == {"q": {"type": "noul", "instructions": "Is it?"}}


def test_states_are_answered_and_kept_in_the_history(fake_model, capsys):
    typed = lines("one two", "three", "", "", "four", "")
    session = drive(fake_model, typed, body={"questions": dict(QUESTIONS)})
    out, err = capsys.readouterr()
    assert [b["usage"]["input_tokens"] for b in bodies(out)] == [3, 1]
    assert "/help lists the commands" in err
    # Typed-ahead keys run before the history loads, so recall is not driven here.
    assert session.states.history.get_strings() == ["one two\nthree", "four"]


def test_the_toolbar_counts_the_questions(fake_model):
    session = drive(fake_model, "", body={"questions": dict(QUESTIONS)})
    assert session.toolbar().startswith(" 2 questions · blank line: answer")


def test_a_bad_state_is_reported_and_the_session_goes_on(fake_model, capsys):
    typed = lines("{oops", "", "[1, 2]", "")
    drive(fake_model, typed, body={"questions": dict(QUESTIONS)}, state_json=True)
    out, err = capsys.readouterr()
    assert "error: state: the state is not valid JSON" in err
    assert len(bodies(out)) == 1


def test_list_help_and_unknown_commands(fake_model, capsys):
    drive(fake_model, lines("/list", "/help", "/nope"), body={"questions": dict(QUESTIONS)})
    err = capsys.readouterr().err
    assert 'team (choice): Which team?\n  billing: {"about": "payments"}\n  technical' in err
    assert "anger (score)\n  0  calm\n  1  angry" in err
    assert "/save" in err and "//" in err
    assert "unknown command /nope" in err


def test_edit_renames_in_place_and_keeps_untouched_values(fake_model):
    # The type menu starts on the current type, so Enter keeps it.
    typed = lines("/edit team", "", CLEAR_LINE + "department", "", "sales", "")
    session = drive(fake_model, typed, body={"questions": dict(QUESTIONS)})
    assert list(session.questions) == ["department", "anger"]
    assert session.questions["department"] == {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": {"about": "payments"}, "technical": None, "sales": None},
    }


def test_edit_without_an_id_offers_a_menu(fake_model):
    typed = lines("/edit", "2", NOUL, "", "Is the customer angry?")
    session = drive(fake_model, typed, body={"questions": dict(QUESTIONS)})
    assert session.questions["anger"] == {"type": "noul", "instructions": "Is the customer angry?"}
    assert session.questions["team"] == QUESTIONS["team"]


def test_remove_by_id_and_from_the_menu(fake_model, capsys):
    typed = lines("/remove team", "/remove", "", "/remove", "/remove nope", "a state", "")
    session = drive(fake_model, typed, body={"questions": dict(QUESTIONS)})
    assert session.questions == {}
    out, err = capsys.readouterr()
    assert "removed team" in err and "removed anger" in err
    assert err.count("no questions; /add one") == 3
    assert out == ""


def test_an_unknown_id_names_the_known_ones(fake_model, capsys):
    drive(fake_model, lines("/edit nope"), body={"questions": dict(QUESTIONS)})
    assert "no question 'nope' (team, anger)" in capsys.readouterr().err


def test_save_writes_a_file_that_q_reads(fake_model, tmp_path, capsys):
    from mlx_decision.cli import build_questions

    path = tmp_path / "questions.json"
    path.write_text("{}")
    typed = lines("/add", NOUL, "urgent", "Urgent?", f"/save {path}") + "y" + lines("/save")
    session = drive(fake_model, typed, body={"questions": dict(QUESTIONS)})
    assert "/save needs a file name" in capsys.readouterr().err
    assert build_questions(path, {})["questions"] == session.questions
    assert list(session.questions) == ["team", "anger", "urgent"]


def test_save_keeps_an_existing_file_unless_confirmed(fake_model, tmp_path):
    path = tmp_path / "questions.json"
    path.write_text("{}")
    drive(fake_model, lines(f"/save {path}") + "n", body={"questions": dict(QUESTIONS)})
    assert path.read_text() == "{}"


def test_a_state_starting_with_a_slash(fake_model, capsys):
    drive(fake_model, lines("//usr/bin is full", "", "/quit", "ignored", ""), body={
        "questions": {"q": {"type": "noul", "instructions": "Disk?"}}
    })  # fmt: skip
    assert [b["usage"]["input_tokens"] for b in bodies(capsys.readouterr().out)] == [3]


def test_options_given_twice_are_refused():
    with pytest.raises(ValidationError, match="given twice"):
        OptionsValidator().validate(Document("a: first\nb\na\n"))


def test_ctrl_d_leaves_a_menu(fake_model, capsys):
    session = drive(fake_model, lines("/edit") + CTRL_D + lines("/list"), body={
        "questions": dict(QUESTIONS)
    })  # fmt: skip
    assert session.questions == QUESTIONS
    assert "team (choice)" in capsys.readouterr().err


def test_ctrl_c_while_answering_keeps_the_session(fake_model, capsys, monkeypatch):
    calls = []

    def decide_request(body):
        calls.append(body["state"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return load_decide(body)

    load_decide = fake_model.decide_request
    monkeypatch.setattr(fake_model, "decide_request", decide_request)
    drive(fake_model, lines("first", "", "second", ""), body={"questions": QUESTIONS})
    out, err = capsys.readouterr()
    assert calls == ["first", "second"]
    assert "cancelled" in err
    assert len(bodies(out)) == 1  # the second state was still answered
