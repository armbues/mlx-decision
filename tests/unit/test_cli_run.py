"""``mlx-decision run``, through the fake backend."""

import json

import pytest
from typer.testing import CliRunner

from fake_backend import write_model
from mlx_decision import __version__
from mlx_decision.cli import app

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs", "billing": "Payments", "sales": "Pricing"},
    },
    "frustration": {"type": "score", "criteria": ["calm", "upset", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
}


@pytest.fixture
def cli(fake_model_path):
    runner = CliRunner()

    def invoke(*args, input=None):
        return runner.invoke(app, ["run", "-m", str(fake_model_path), *args], input=input)

    return invoke


@pytest.fixture
def questions_file(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS))
    return path


def test_json_output_equals_the_python_api(cli, questions_file, fake_model):
    result = cli("-q", str(questions_file), "-s", "Stripe is down", "--json")
    assert result.exit_code == 0, result.output
    expected = fake_model.decide("Stripe is down", QUESTIONS).to_wire()
    assert json.loads(result.stdout) == expected


def test_a_whole_request_body_as_questions_file(cli, tmp_path, fake_model):
    path = tmp_path / "body.json"
    path.write_text(json.dumps({"state": {"ticket": 1}, "questions": QUESTIONS, "model": "x"}))
    result = cli("-q", str(path), "--json")
    assert json.loads(result.stdout) == fake_model.decide({"ticket": 1}, QUESTIONS).to_wire()


def test_state_from_stdin(cli, questions_file):
    result = cli("-q", str(questions_file), "--json", input="one two three")
    assert json.loads(result.stdout)["usage"]["input_tokens"] == 3


def test_state_from_a_file_and_as_json(cli, questions_file, tmp_path, fake_model):
    path = tmp_path / "state.json"
    path.write_text('{"a": [1, 2]}')
    result = cli("-q", str(questions_file), "--state-file", str(path), "--state-json", "--json")
    assert json.loads(result.stdout) == fake_model.decide({"a": [1, 2]}, QUESTIONS).to_wire()


def test_shorthand_questions(cli, fake_model_path):
    result = cli(
        "-s",
        "text",
        "--noul",
        "Is this urgent?",
        "--noul",
        "spam=Is it spam? a=b",
        "--choice",
        "team=billing, technical,sales",
        "--score",
        "anger=calm,angry",
        "--json",
    )
    assert result.exit_code == 0, result.output
    answers = json.loads(result.stdout)["answers"]
    assert list(answers) == ["noul_1", "spam", "team", "anger"]
    assert list(answers["team"]["probabilities"]) == ["billing", "technical", "sales"]
    assert answers["anger"]["legend"] == {"0": "calm", "1": "angry"}


def test_shorthand_noul_text_is_the_instruction(fake_model_path):
    from mlx_decision.cli import shorthand_questions

    questions = shorthand_questions(["Is x=y?", "q=Is it?"], [], [])
    assert questions == {
        "noul_1": {"type": "noul", "instructions": "Is x=y?"},
        "q": {"type": "noul", "instructions": "Is it?"},
    }


def test_shorthand_and_file_together(cli, questions_file):
    result = cli("-q", str(questions_file), "--noul", "extra=Extra?", "-s", "x", "--json")
    assert list(json.loads(result.stdout)["answers"]) == [*QUESTIONS, "extra"]


def test_readable_output(cli, questions_file):
    result = cli("-q", str(questions_file), "-s", "Stripe is down")
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "department (choice): technical, confidence" in out
    assert "frustration (score):" in out and "on 0-2" in out
    assert "is_urgent (noul): yes, p(yes) 0.750" in out
    assert "fake · 3 input tokens" in out


def test_truncation_is_noted(fake_model_path, questions_file):
    write_model(fake_model_path, max_input_tokens=2)
    runner = CliRunner()
    args = ["run", "-m", str(fake_model_path), "-q", str(questions_file), "-s", "a b c"]
    readable = runner.invoke(app, args)
    assert "state truncated" in readable.stdout
    as_json = runner.invoke(app, [*args, "--json"])
    assert "truncated" in as_json.stderr
    assert "truncated" not in as_json.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-s", "x"], "no questions"),
        (["-s", "x", "--choice", "team=a,b", "--choice", "team=c,d"], "given twice"),
        (["-s", "x", "--noul", "a=one", "--noul", "a=two"], "'a' given twice"),
        (["-s", "x", "--noul", "first", "--noul", "noul_1=second"], "'noul_1' given twice"),
        (["-s", "x", "--choice", "team=a,a"], "option 'a' given twice"),
        (["-s", "x", "--score", "anger=calm"], "questions.anger.criteria: a score needs"),
        (["-s", "x", "--noul", "q=?", "--state-file", "missing.txt"], "either --state"),
        (["-s", "{", "--noul", "q=?", "--state-json"], "not valid JSON"),
    ],
)
def test_errors_exit_non_zero(cli, args, message):
    result = cli(*args)
    assert result.exit_code != 0
    assert message in result.output


def test_invalid_request_shows_the_validation_message(cli, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"q": {"type": "rank"}}))
    result = cli("-q", str(path), "-s", "x")
    assert result.exit_code == 1
    assert "error: questions.q.type: unknown question type 'rank'" in result.stderr


def test_choice_shorthand_needs_an_id(cli):
    result = cli("-s", "x", "--choice", "a,b")
    assert result.exit_code == 2
    assert "ID=OPTION" in result.output


def test_unknown_model_folder(tmp_path):
    result = CliRunner().invoke(app, ["run", "-m", str(tmp_path), "-s", "x", "--noul", "q?"])
    assert result.exit_code == 1
    assert "not a supported decision model" in result.stderr


def test_chat_needs_a_terminal(fake_model_path):
    result = CliRunner().invoke(app, ["chat", "-m", str(fake_model_path)], input="x\n\n")
    assert result.exit_code == 1
    assert "chat needs a terminal" in result.stderr
    assert "run --states" in result.stderr


def test_run_has_no_interactive_flag(cli):
    result = cli("--noul", "q=Is it?", "-i")
    assert result.exit_code == 2


def lines_of(*requests) -> str:
    return "".join(json.dumps(request) + "\n" for request in requests)


def test_states_one_body_per_line_in_order(cli, questions_file, tmp_path, fake_model):
    path = tmp_path / "requests.jsonl"
    states = ["Stripe is down", {"ticket": 1}, "a b c d"]
    path.write_text(lines_of(*({"state": s} for s in states)) + "\n")
    result = cli("-q", str(questions_file), "--states", str(path))
    assert result.exit_code == 0, result.output
    expected = [fake_model.decide(s, QUESTIONS).to_wire() for s in states]
    assert [json.loads(line) for line in result.stdout.splitlines()] == expected


def test_states_lines_may_bring_their_own_questions(cli, fake_model):
    own = {"spam": {"type": "noul", "instructions": "Spam?"}}
    typed = lines_of({"state": "x", "questions": own}, {"state": "y"})
    result = cli("--noul", "q=Is it?", "--states", "-", input=typed)
    first, second = (json.loads(line) for line in result.stdout.splitlines())
    assert list(first["answers"]) == ["spam"]
    assert list(second["answers"]) == ["q"]


def test_states_without_question_flags(cli):
    own = {"q": {"type": "noul", "instructions": "Is it?"}}
    typed = lines_of({"state": "x", "questions": own}, {"state": "y"})
    result = cli("--states", "-", input=typed)
    assert result.exit_code == 0, result.output
    answered, missing = (json.loads(line) for line in result.stdout.splitlines())
    assert list(answered["answers"]) == ["q"]
    assert missing["error"]["param"] == "questions"


def test_states_error_lines(cli):
    typed = lines_of({"state": "one two"}) + '{broken\n\n"three"\n' + lines_of({"text": 1})
    typed += lines_of({"state": "three"})
    result = cli("--noul", "q=Is it?", "--states", "-", input=typed)
    assert result.exit_code == 0, result.output
    first, broken, bare, no_state, last = (json.loads(line) for line in result.stdout.splitlines())
    assert first["usage"]["input_tokens"] == 2
    assert broken["line"] == 2
    assert broken["error"]["type"] == "invalid_request_error"
    assert broken["error"]["message"].startswith("not valid JSON")
    assert bare["line"] == 4 and "a line must be a request body" in bare["error"]["message"]
    assert no_state["error"]["param"] == "state"
    assert no_state["error"]["message"] == "state is required"
    assert last["usage"]["input_tokens"] == 1


def test_states_note_truncation_on_stderr(fake_model_path, questions_file):
    write_model(fake_model_path, max_input_tokens=2)
    args = ["run", "-m", str(fake_model_path), "-q", str(questions_file), "--states", "-"]
    typed = lines_of({"state": "a b"}, {"state": "a b c"})
    result = CliRunner().invoke(app, args, input=typed)
    assert "line 2: the state was truncated" in result.stderr
    assert "line 1" not in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--states", "missing.jsonl"], "cannot read the states"),
        (["--states", "-", "-s", "x"], "drop --state"),
        (["--states", "-", "--state-json"], "drop --state"),
        (["--states", "-", "--score", "anger=calm"], "questions.anger.criteria"),
    ],
)
def test_states_errors(cli, args, message):
    result = cli("--noul", "q=Is it?", *args, input="")
    assert result.exit_code == 1
    assert message in result.stderr


def test_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout == f"mlx-decision {__version__}\n"


def test_files_that_are_not_utf8_are_reported(cli, tmp_path):
    path = tmp_path / "latin1.json"
    path.write_bytes('{"q": {"type": "noul", "instructions": "caf\xe9"}}'.encode("latin-1"))
    result = cli("-q", str(path), "-s", "x")
    assert result.exit_code == 1
    assert "cannot read questions from" in result.stderr
    assert "Traceback" not in result.output


def test_server_checks_the_model_before_starting(tmp_path):
    result = CliRunner().invoke(app, ["server", "-m", str(tmp_path)])
    assert result.exit_code == 1
    assert "not a supported decision model" in result.stderr


def test_help_shows_the_images_extra():
    result = CliRunner().invoke(app, ["run", "--help"], terminal_width=200)
    assert "mlx-decision[images]" in result.output


def test_run_checks_questions_before_loading(tmp_path, monkeypatch):
    import mlx_decision.model

    def no_load(*args, **kwargs):
        raise AssertionError("loaded the model for an invalid request")

    monkeypatch.setattr(mlx_decision.model, "load", no_load)
    result = CliRunner().invoke(app, ["run", "-m", str(tmp_path), "-s", "x", "--score", "a=only"])
    assert result.exit_code == 1
    assert "a score needs at least two levels" in result.stderr
