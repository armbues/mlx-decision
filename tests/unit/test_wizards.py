"""The ``data`` and ``training fit`` menus, typed through prompt_toolkit's pipe input.

Menu entries are picked by their number, then Enter; Enter alone keeps the
default. Each run is checked for the plan it ends with, and for the command
line it prints: run without menus, that command makes the same examples.
"""

import json
import shlex

import datasets
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from test_cli_training import examples as ticket_examples
from test_data_command import hub, labelled
from typer.testing import CliRunner

from fake_backend import write_model
from mlx_decision.cli import app, default_output
from mlx_decision.data.format import write_examples
from mlx_decision.data.make import DataPlan
from mlx_decision.data.wizard import Cancelled, ask_data, same_without_menus
from mlx_decision.flags import join_list, join_pairs, split_list, split_pairs
from mlx_decision.training import wizard as fit_menus
from mlx_decision.training.wizard import FitPlan, ask_eval, ask_fit

ENTER = "\r"
CTRL_C = "\x03"
CLEAR_LINE = "\x15"
BACKSPACE = "\x7f"
# The options block a two-class file starts with.
TICKET_OPTIONS = "billing: billing\ntech: tech\n"


def block(old: str, *new: str) -> str:
    """A prefilled block ``old`` replaced by the lines ``new`` (Enter ends it)."""
    return BACKSPACE * len(old) + "".join(line + ENTER for line in new)


def typed(*parts: str) -> str:
    """Each part followed by Enter (a part ending in "!" is sent as is: y / n)."""
    return "".join(part[:-1] if part.endswith("!") else part + ENTER for part in parts)


def drive(function, keys: str, *args):
    """``function(*args)`` with ``keys`` typed; Ctrl-C after them, so a run that
    asks more than the test types is cancelled instead of waiting."""
    with create_pipe_input() as pipe:
        pipe.send_text(keys + CTRL_C)
        with create_app_session(input=pipe, output=DummyOutput()):
            return function(*args)


def rerun(plan: DataPlan, tmp_path) -> list[dict]:
    """The printed command run without menus; the lines it writes."""
    words = shlex.split(same_without_menus(plan).splitlines()[1])
    assert words[:2] == ["mlx-decision", "data"]
    out = tmp_path / "again.jsonl"
    args = [w for w in words[2:]]
    args[args.index("-o") + 1] = str(out)
    result = CliRunner().invoke(app, ["data", *args])
    assert result.exit_code == 0, result.output
    return [json.loads(line) for line in out.read_text().splitlines()]


@pytest.fixture
def tickets(tmp_path):
    path = tmp_path / "tickets.csv"
    path.write_text("text,label\nrefund please,billing\napp crashes,tech\nrefund,billing\n")
    return path


def test_a_file_through_every_step(tmp_path, tickets, monkeypatch):
    monkeypatch.chdir(tmp_path)
    keys = typed(
        "",  # shape: text, guessed
        "",  # text column: text
        "",  # label column: label
        "",  # type: choice
        "",  # id: tickets
        "Which team handles it?",
        block(TICKET_OPTIONS, "money: Payments, refunds", "tech: Bugs"),  # renamed
        "",  # write to: tickets.jsonl
        "y!",
    )
    plan, made = drive(ask_data, keys, DataPlan(source=str(tickets)))
    assert plan.shape == "text" and plan.columns == {"text": "text", "label": "label"}
    assert plan.output == "tickets.jsonl"
    assert plan.label_map == {"billing": "money", "tech": "tech"}
    assert plan.options == {"money": "Payments, refunds", "tech": "Bugs"}
    assert "money=Payments\\, refunds,tech=Bugs" in plan.arguments()
    first = made.examples[0]
    assert first.request["questions"]["tickets"]["criteria"] == plan.options
    assert first.labels == {"tickets": "money"}
    again = rerun(plan, tmp_path)
    assert again == [example.to_json() for example in made.examples]


def test_two_classes_can_become_a_yes_no_question(tmp_path, tickets):
    keys = typed("", "", "", "2", "", "Is it about money?", CLEAR_LINE + "out.jsonl", "y!")
    plan, made = drive(ask_data, keys, DataPlan(source=str(tickets)))
    assert plan.noul and "--noul" in plan.arguments()
    question = made.examples[0].request["questions"]["tickets"]
    assert question["type"] == "noul"
    assert [e.labels["tickets"] for e in made.examples] == ["false", "true", "false"]  # tech: yes
    assert rerun(plan, tmp_path) == [example.to_json() for example in made.examples]


def test_the_hub_asks_config_and_split_with_train_preselected(tmp_path, monkeypatch):
    train = labelled(["happy", "sad", "glad"], [0, 1, 0], ["joy", "sadness"])
    hub(monkeypatch, {"org/emotion": {"split": {"test": train, "train": train}, "raw": {}}})
    keys = typed(
        "6",  # a Hugging Face dataset
        "org/emotion",
        "1",  # config: split
        "",  # split: train, preselected
        "",  # shape
        "",  # text column
        "",  # label column
        "",  # type: choice
        "",  # id
        "Which emotion?",
        "",  # options as they are
        CLEAR_LINE + str(tmp_path / "e.jsonl"),
        "y!",
    )
    plan, made = drive(ask_data, keys, DataPlan())
    assert (plan.source, plan.config, plan.split) == ("hf:org/emotion", "split", "train")
    assert plan.arguments()[:5] == ["hf:org/emotion", "--config", "split", "--split", "train"]
    assert len(made.examples) == 3


def test_a_preset_needs_only_its_split_and_the_output(tmp_path, monkeypatch):
    from mlx_decision.data.presets import PRESETS

    preset = PRESETS["sst5"]
    features = datasets.Features(
        {"text": datasets.Value("string"), "label": datasets.Value("int64")}
    )
    rows = datasets.Dataset.from_dict({"text": ["bad", "good"], "label": [0, 4]}, features=features)
    hub(monkeypatch, {preset.source.repo: {"default": {"train": rows, "test": rows}}})
    keys = typed(
        "5", "", CLEAR_LINE + str(tmp_path / "s.jsonl"), "y!"
    )  # sst5, then train (preselected)
    plan, made = drive(ask_data, keys, DataPlan())
    assert (plan.preset, plan.split) == ("sst5", "train")
    assert plan.arguments() == [
        "--preset",
        "sst5",
        "--split",
        "train",
        "-o",
        str(tmp_path / "s.jsonl"),
    ]
    assert [e.labels["sentiment"] for e in made.examples] == ["0", "4"]


def test_ratings_edit_their_levels(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("review,stars\ngreat,3\nbad,1\n")
    levels = block("1\n2\n3\n", "bad", "ok", "good")
    keys = typed("", "", "", "", "How good?", levels, CLEAR_LINE + "r.jsonl", "y!")
    plan, made = drive(ask_data, keys, DataPlan(source=str(path)))
    assert plan.shape == "rating" and plan.levels == ["bad", "ok", "good"]
    assert made.examples[0].request["questions"]["reviews"]["criteria"] == ["bad", "ok", "good"]
    assert rerun(plan, tmp_path) == [example.to_json() for example in made.examples]


def test_a_wrong_number_of_options_is_asked_again(tmp_path, tickets, capsys):
    keys = typed(
        "", "", "", "", "", "Team?", block(TICKET_OPTIONS, "only"),
        "", "", "", block("only\n", "a", "b"), CLEAR_LINE + "x.jsonl", "y!",
    )  # fmt: skip
    plan, _ = drive(ask_data, keys, DataPlan(source=str(tickets)))
    assert "give one entry per class or value: 2, not 1" in capsys.readouterr().err
    assert plan.label_map == {"billing": "a", "tech": "b"}


def test_leaving_the_menus_cancels(tmp_path, tickets):
    with pytest.raises(Cancelled):
        drive(ask_data, CTRL_C, DataPlan(source=str(tickets)))
    with pytest.raises(Cancelled):
        drive(ask_data, typed("", "", "", "", "", "Team?", "", CLEAR_LINE + "x.jsonl", "n!"),
              DataPlan(source=str(tickets)))  # fmt: skip


def test_flags_given_are_not_asked_again(tmp_path, tickets):
    plan = DataPlan(source=str(tickets), shape="text", columns={"text": "text", "label": "label"},
                    instructions="Team?")  # fmt: skip
    plan, _ = drive(ask_data, typed(CLEAR_LINE + "o.jsonl", "y!"), plan)  # only the output is open
    assert plan.output == "o.jsonl"


# training fit


@pytest.fixture
def fit_data(tmp_path):
    base = write_model(tmp_path / "base")
    data = tmp_path / "tickets.jsonl"
    write_examples(data, ticket_examples())
    return base, data


def test_fit_asks_the_kind_holdout_and_output(tmp_path, fit_data, capsys):
    base, data = fit_data
    keys = typed(
        "", "2", CLEAR_LINE + "0.3", "", "y!"
    )  # calibrate, another share, 0.3, default output
    plan = drive(ask_fit, keys, FitPlan(model=str(base), data=str(data)), default_output)
    assert (plan.method, plan.holdout, plan.output) == ("calibrate", 0.3, "base-tickets")
    err = capsys.readouterr().err
    assert "calibrate" in err and "about" in err
    assert "data: 100 examples: 30 holdout" in err
    assert f"mlx-decision training fit -m {base} --data {data} -o base-tickets --method " \
           "calibrate --holdout 0.3" in err  # fmt: skip


def test_fit_takes_a_validation_file_and_replaces_an_output_when_told(tmp_path, fit_data):
    base, data = fit_data
    (tmp_path / "out").mkdir()
    keys = typed("", "3", str(data), CLEAR_LINE + str(tmp_path / "out"), "y!", "y!")
    plan = drive(ask_fit, keys, FitPlan(model=str(base), data=str(data)), default_output)
    assert plan.validation == str(data) and plan.overwrite
    assert "--overwrite" in plan.arguments() and "--validation" in plan.arguments()
    with pytest.raises(fit_menus.Cancelled):
        drive(ask_fit, typed("", "", "", "n!"), FitPlan(model=str(base), data=str(data)),
              default_output)  # fmt: skip


def test_epochs_are_asked_for_the_kinds_that_train_in_epochs(tmp_path, fit_data, monkeypatch):
    base, data = fit_data
    monkeypatch.setitem(fit_menus.kinds_plan.MAIN_KIND, "fake", "finetune")
    plan = FitPlan(model=str(base), data=str(data), method="finetune", holdout=0.2)
    asked = drive(
        ask_fit,
        typed(CLEAR_LINE + "5", CLEAR_LINE + str(tmp_path / "f"), "y!"),
        plan,
        default_output,
    )
    assert asked.epochs == 5 and "--epochs" in asked.arguments()


def test_fit_and_eval_ask_for_a_missing_model_and_data(tmp_path, fit_data):
    base, data = fit_data
    assert drive(ask_eval, typed(str(base), str(data)), None, None) == (str(base), str(data))
    plan = drive(ask_fit, typed(str(base), str(data), "", "", "", "y!"), FitPlan(),
                 default_output)  # fmt: skip
    assert (plan.model, plan.data) == (str(base), str(data))


def test_without_a_terminal_the_defaults_apply(tmp_path, fit_data, monkeypatch):
    base, data = fit_data
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["training", "fit", "-m", str(base), "--data", str(data)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "base-tickets").is_dir()
    missing = CliRunner().invoke(app, ["training", "fit", "--data", str(data)])
    assert missing.exit_code == 1 and "pass -m MODEL" in missing.stderr
    plan = FitPlan(model="m", data="d.jsonl", early_stop=False).with_defaults(default_output)
    assert plan.arguments()[-1] == "--no-early-stop"
    assert (plan.method, plan.holdout, plan.epochs, plan.output) == ("auto", 0.2, 3, "m-d")


# flag values


@pytest.mark.parametrize("values", [["a", "b"], ["x, y", "back\\slash"], ["", "z"]])
def test_lists_with_commas_survive_a_round_trip(values):
    assert split_list(join_list(values)) == values


def test_pairs_split_at_the_first_equals_sign():
    assert split_pairs("a=x=1,b=y\\,z") == {"a": "x=1", "b": "y,z"}
    assert split_pairs(join_pairs({"k": "v, w"})) == {"k": "v, w"}
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        split_pairs("a=1,b")
