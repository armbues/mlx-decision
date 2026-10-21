"""Dataset folders: ``data -o FOLDER`` writes ``dataset.json`` and one file per
split, the rows keeping their raw label values; questions and the answers map
live in ``dataset.json``; fits and evaluations read a folder's splits by role."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mlx_decision.data.make as make
from fake_backend import TEAM, write_model
from mlx_decision.cli import app
from mlx_decision.data.dataset import (
    DATASET_FILE,
    Dataset,
    SplitOut,
    open_dataset,
    partial_folder,
    read_for,
    report_split,
    write_dataset,
)
from mlx_decision.data.format import DataError, Example
from mlx_decision.data.make import cut, default_roles, parse_roles
from mlx_decision.data.sources import SourceError, open_source
from mlx_decision.errors import DecisionError
from mlx_decision.training import evaluate, fit

datasets = pytest.importorskip("datasets")


@pytest.fixture
def saved(tmp_path) -> Path:
    """Tickets with a train (30 rows) and a test (12 rows) split: a text and a
    team as plain text, every third one billing."""

    def split(count, offset):
        texts = [f"ticket number {offset + i}" for i in range(count)]
        names = ["billing" if i % 3 == 0 else "tech" for i in range(count)]
        return datasets.Dataset.from_dict({"text": texts, "label": names})

    folder = tmp_path / "tickets"
    datasets.DatasetDict({"train": split(30, 0), "test": split(12, 100)}).save_to_disk(folder)
    return folder


def run(*args: str):
    return CliRunner().invoke(app, list(args))


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def described(folder: Path) -> dict:
    return json.loads((folder / DATASET_FILE).read_text())


def write_questions(folder: Path, questions: dict, answers: dict) -> None:
    body = described(folder)
    body["questions"] = questions
    body["answers"] = answers
    (folder / DATASET_FILE).write_text(json.dumps(body))


# Writing


def test_a_dataset_without_a_question_keeps_its_values(saved, tmp_path):
    out = tmp_path / "data" / "tickets"
    result = run("data", str(saved), "--text", "text", "--label", "label",
                 "--split", "train", "--split", "test", "-o", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "no question yet" in result.stderr
    assert sorted(p.name for p in out.iterdir()) == [DATASET_FILE, "test.jsonl", "train.jsonl"]
    body = described(out)
    assert "questions" not in body and "answers" not in body
    # No validation split: test reports.
    assert body["splits"]["train"]["role"] == "train"
    assert body["splits"]["test"]["role"] == "report"
    assert body["splits"]["train"]["examples"] == 30
    assert body["columns"]["label"]["values"] == {"billing": {"rows": 14}, "tech": {"rows": 28}}
    assert body["source"]["columns"] == {"text": "text", "label": "label"}
    first = lines(out / "train.jsonl")[0]
    assert first == {"id": "tickets:train/0", "state": "ticket number 0",
                     "values": {"label": "billing"}}  # fmt: skip
    assert not partial_folder(out).exists()


def test_fit_refuses_a_dataset_without_questions_then_fits_it(saved, tmp_path):
    out = tmp_path / "tickets-data"
    assert run("data", str(saved), "--split", "train", "-o", str(out)).exit_code == 0
    base = write_model(tmp_path / "base")
    refused = run("training", "fit", "-m", str(base), "--data", str(out), "-o",
                  str(tmp_path / "f"))  # fmt: skip
    assert refused.exit_code != 0
    assert "has no questions yet" in refused.output
    with pytest.raises(DecisionError, match="no questions yet"):
        read_for(out)
    write_questions(
        out,
        {"team": TEAM},
        {"team": {"column": "label", "map": {"billing": "billing", "tech": "tech"}}},
    )
    result = fit(base, out, output=tmp_path / "fitted")
    assert result.report["data"]["examples"] == 30
    assert result.report["data"]["file"] == f"{out} (train split)"


def test_unmapped_values_are_named(saved, tmp_path):
    out = tmp_path / "t"
    run("data", str(saved), "--split", "train", "-o", str(out))
    write_questions(out, {"team": TEAM}, {"team": {"column": "label", "map": {"tech": "tech"}}})
    dataset, _ = open_dataset(out)
    assert dataset.missing() == ["team: label values not mapped to an answer or left out: billing"]
    with pytest.raises(DecisionError, match="billing"):
        read_for(out)


def test_a_map_merges_leaves_out_and_makes_yes_no(saved, tmp_path):
    out = tmp_path / "t"
    run("data", str(saved), "--split", "train", "-o", str(out))
    urgent = {"type": "noul", "instructions": "Is it about money?"}
    write_questions(
        out,
        {"team": TEAM, "money": urgent},
        {
            # Both values one option.
            "team": {"column": "label", "map": {"billing": "sales", "tech": "sales"}},
            # billing is yes, tech left out: its rows are not labelled for money.
            "money": {"column": "label", "map": {"billing": True}, "left_out": ["tech"]},
        },
    )
    examples = read_for(out).examples
    assert {e.labels["team"] for e in examples} == {"sales"}
    money = [e for e in examples if "money" in e.labels]
    assert len(money) == 10 and {e.labels["money"] for e in money} == {"true"}
    assert examples[0].request["questions"] == {"team": TEAM, "money": urgent}
    assert examples[0].values == {"label": "billing"}


def test_with_a_question_the_folder_holds_it_once_with_its_map(saved, tmp_path):
    out = tmp_path / "t"
    result = run("data", str(saved), "--question", "Which team?", "--id", "team",
                 "--split", "train", "--split", "test", "--label-map",
                 "billing=money,tech=bugs", "-o", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    body = described(out)
    assert body["questions"]["team"]["type"] == "choice"
    assert body["answers"] == {
        "team": {"column": "label", "map": {"billing": "money", "tech": "bugs"}, "left_out": []}
    }
    line = lines(out / "train.jsonl")[0]
    assert "questions" not in line and "labels" not in line
    examples = read_for(out).examples
    assert examples[0].labels == {"team": "money"}
    assert len(read_for(out, "evaluate").examples) == 12


def test_a_lines_own_questions_replace_the_datasets(tmp_path):
    folder = tmp_path / "own"
    own = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
    dataset = Dataset(folder, questions={"team": TEAM})
    examples = [
        Example("1", {"state": "x"}, {}, {"label": "tech"}),
        Example("2", {"state": "y", "questions": {"pick": own}}, {"pick": "b"}),
    ]
    dataset.answers = {"team": {"column": "label", "map": {"tech": "tech"}}}
    # The questions differ per line, so each line keeps its own.
    examples[0].request["questions"] = {"team": TEAM}
    examples[0].labels["team"] = "tech"
    write_dataset(folder, dataset, [SplitOut("all", examples, {"role": "train"})])
    read = read_for(folder).examples
    assert read[0].request["questions"] == {"team": TEAM}
    assert read[1].request["questions"] == {"pick": own} and read[1].labels == {"pick": "b"}


def test_errors_name_the_split_file_and_line(saved, tmp_path):
    out = tmp_path / "t"
    run("data", str(saved), "--split", "train", "-o", str(out))
    write_questions(out, {"team": TEAM}, {"team": {"column": "label", "map": {
        "billing": "billing", "tech": "tech"}}})  # fmt: skip
    path = out / "train.jsonl"
    rows = path.read_text().splitlines()
    rows[1] = json.dumps({"id": "x", "state": "s", "values": {"label": "tech"}, "images": ["a"]})
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(DataError) as error:
        read_for(out)
    assert str(error.value).startswith("train.jsonl line 2: ")
    assert "images are not used for training yet" in str(error.value)


def test_a_stop_before_the_rename_leaves_nothing(tmp_path):
    folder = tmp_path / "gone"

    def stop():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        write_dataset(folder, Dataset(folder), [SplitOut("train", [], {})], before_rename=stop)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_folder_is_not_replaced(saved, tmp_path):
    out = tmp_path / "t"
    out.mkdir()
    result = run("data", str(saved), "--split", "train", "-o", str(out))
    assert result.exit_code != 0 and "exists" in result.output


def test_events_end_with_the_files_of_the_folder(saved, tmp_path):
    out = tmp_path / "t"
    result = run("data", str(saved), "--split", "train", "--split", "test",
                 "--roles", "train,evaluate", "-o", str(out), "--events")  # fmt: skip
    assert result.exit_code == 0, result.output
    done = json.loads(result.stdout.splitlines()[-1])
    assert done["stage"] == "done"
    assert done["metrics"]["path"] == str(out)
    assert [(f["split"], f["role"], f["examples"]) for f in done["metrics"]["files"]] == [
        ("train", "train", 30),
        ("test", "evaluate", 12),
    ]


# Parts


def test_parts_cut_one_split_by_role(saved, tmp_path):
    out = tmp_path / "t"
    result = run("data", str(saved), "--split", "train", "--roles",
                 "train:0.6+report:0.2+evaluate:0.2", "-o", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    splits = described(out)["splits"]
    assert {name: (s["role"], s["examples"], s["part"]) for name, s in splits.items()} == {
        "train": ("train", 18, 0.6),
        "validation": ("report", 6, 0.2),
        "test": ("evaluate", 6, 0.2),
    }
    # Stratified: a third of each part is billing.
    for name in splits:
        values = [line["values"]["label"] for line in lines(out / f"{name}.jsonl")]
        assert values.count("billing") * 3 == len(values)
    ids = [line["id"] for name in splits for line in lines(out / f"{name}.jsonl")]
    assert len(set(ids)) == 30
    again = tmp_path / "again"
    run("data", str(saved), "--split", "train", "--roles",
        "train:0.6+report:0.2+evaluate:0.2", "-o", str(again))  # fmt: skip
    assert (again / "test.jsonl").read_text() == (out / "test.jsonl").read_text()


def test_a_part_named_as_another_split_takes_its_splits_name(saved, tmp_path):
    out = tmp_path / "t"
    result = run("data", str(saved), "--split", "train", "--split", "test", "--roles",
                 "train:0.8+evaluate:0.2,report", "-o", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    roles = {name: s["role"] for name, s in described(out)["splits"].items()}
    assert roles == {"train": "train", "train-evaluate": "evaluate", "test": "report"}


@pytest.mark.parametrize(
    ("roles", "message"),
    [
        ("train,train", "train at most once"),
        ("train:0.8+report:0.4", "add up to more than 1"),
        ("fit", "none of train, report, evaluate"),
        ("train+report", "each part takes a share"),
        ("train:x", "no share"),
    ],
)
def test_roles_that_do_not_go(roles, message):
    with pytest.raises(SourceError, match=message):
        parse_roles(roles, roles.count(",") + 1)


def test_default_roles_follow_the_split_names():
    assert default_roles(["train", "validation", "test"]) == [
        [("train", None)], [("report", None)], [("evaluate", None)]]  # fmt: skip
    assert default_roles(["train", "test"]) == [[("train", None)], [("report", None)]]
    assert default_roles(["extra"]) == [[]]


def test_cut_leaves_out_what_the_shares_leave():
    examples = [Example(str(i), {"state": "s"}, {}, {"v": str(i % 2)}) for i in range(10)]
    parts = cut(examples, [0.4, 0.2], seed=1)
    assert [len(part) for part in parts] == [4, 2]
    assert [e.values["v"] for e in parts[0]].count("0") == 2


# Fit and eval on folders


@pytest.fixture
def ready(saved, tmp_path) -> Path:
    out = tmp_path / "data" / "tickets"
    result = run("data", str(saved), "--split", "train", "--split", "test",
                 "--roles", "train,report:0.5+evaluate:0.5", "--question", "Which team?",
                 "--id", "team", "-o", str(out))  # fmt: skip
    assert result.exit_code == 0, result.output
    return out


def test_fit_on_a_folder_reports_on_its_report_split(ready, tmp_path, monkeypatch):
    base = write_model(tmp_path / "base")
    assert report_split(ready) == ready / "validation.jsonl"
    assert report_split(ready / "train.jsonl") == ready / "validation.jsonl"
    monkeypatch.chdir(tmp_path)
    result = run("training", "fit", "-m", str(base), "--data", str(ready))
    assert result.exit_code == 0, result.output
    written = json.loads((tmp_path / "base-tickets" / "training.json").read_text())
    assert written["data"]["file"] == f"{ready} (train split)"
    assert written["options"]["validation"] == f"{ready} (validation split)"
    assert written["data"]["holdout"] == 6
    # --holdout keeps the holdout of the train split.
    held = run("training", "fit", "-m", str(base), "--data", str(ready), "--holdout", "0.2",
               "-o", str(tmp_path / "held"))  # fmt: skip
    assert held.exit_code == 0, held.output
    held_report = json.loads((tmp_path / "held" / "training.json").read_text())
    assert held_report["options"]["validation"] is None


def test_eval_on_a_folder_reads_its_evaluate_split(ready, tmp_path):
    base = write_model(tmp_path / "base")
    result = evaluate(base, ready)
    assert result.data["file"] == f"{ready} (test split)"
    assert result.data["examples"] == 6
    by_file = evaluate(base, ready / "train.jsonl")
    assert by_file.data["examples"] == 30


def test_the_hash_covers_the_questions(ready):
    dataset, _ = open_dataset(ready)
    before = dataset.sha256("train")
    body = described(ready)
    body["questions"]["team"]["instructions"] = "Which team handles it?"
    (ready / DATASET_FILE).write_text(json.dumps(body))
    dataset, _ = open_dataset(ready)
    assert dataset.sha256("train") != before


# Presets


@pytest.fixture
def trec_like(tmp_path, monkeypatch):
    """TREC's shape (a class label of six coarse types) saved locally, opened
    in place of the Hub."""
    names = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
    features = datasets.Features(
        {"text": datasets.Value("string"), "coarse_label": datasets.ClassLabel(names=names)}
    )
    rows = {
        "text": [f"question {i}" for i in range(12)],
        "coarse_label": [i % 6 for i in range(12)],
    }
    folder = tmp_path / "trec"
    datasets.DatasetDict({"train": datasets.Dataset.from_dict(rows, features=features)}
                         ).save_to_disk(folder)  # fmt: skip
    opened = open_source(str(folder), None, "train")
    monkeypatch.setattr(make, "open_source", lambda *a, **k: opened)
    return folder


def test_a_preset_writes_its_answers_map(trec_like, tmp_path):
    out = tmp_path / "trec-data"
    result = run("data", "--preset", "trec", "--split", "train", "-o", str(out))
    assert result.exit_code == 0, result.output
    body = described(out)
    assert list(body["questions"]) == ["answer_type"]
    assert body["answers"]["answer_type"]["column"] == "coarse_label"
    assert body["answers"]["answer_type"]["map"]["ABBR"] == "abbreviation"
    assert body["answers"]["answer_type"]["map"]["NUM"] == "numeric"
    assert lines(out / "train.jsonl")[0]["values"] == {"coarse_label": "ABBR"}
    assert read_for(out).examples[3].labels == {"answer_type": "human"}


def test_openbookqa_keeps_each_lines_question(tmp_path, monkeypatch):
    rows = {
        "id": ["q1", "q2"],
        "question_stem": ["The sun is", "Water is"],
        "choices": [
            {"label": ["A", "B"], "text": ["a star", "a moon"]},
            {"label": ["A", "B"], "text": ["dry", "wet"]},
        ],
        "answerKey": ["A", "B"],
    }
    folder = tmp_path / "obqa"
    datasets.DatasetDict({"test": datasets.Dataset.from_dict(rows)}).save_to_disk(folder)
    monkeypatch.setattr(make, "open_source", lambda *a, **k: open_source(str(folder), None, "test"))
    out = tmp_path / "obqa-data"
    result = run("data", "--preset", "openbookqa", "-o", str(out))
    assert result.exit_code == 0, result.output
    body = described(out)
    assert "questions" not in body and "answers" not in body
    first, second = lines(out / "test.jsonl")
    assert first["questions"]["answer"]["criteria"] == {"A": "a star", "B": "a moon"}
    assert second["labels"] == {"answer": "B"}
    assert open_dataset(out)[0].missing() == []
    assert [e.labels for e in read_for(out, "evaluate").examples] == [
        {"answer": "A"}, {"answer": "B"}]  # fmt: skip


def test_a_plain_file_is_a_dataset_of_one_split(tmp_path):
    path = tmp_path / "plain.jsonl"
    line = {"state": "s", "questions": {"team": TEAM}, "labels": {"team": "tech"}}
    path.write_text(json.dumps(line) + "\n")
    dataset, split = open_dataset(path)
    assert not dataset.folder and split is None and list(dataset.splits) == ["plain"]
    read = read_for(path)
    assert read.description == str(path) and read.examples[0].labels == {"team": "tech"}
