"""Mixes: a dataset folder listing other datasets, read as one dataset by fits
and evaluations: its train, report and evaluate splits are the sources'
together, capped and sampled with the mix's seed, held-out sources scored only."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fake_backend import TEAM, write_model
from mlx_decision.cli import app
from mlx_decision.data.dataset import DATASET_FILE, open_dataset, read_for, report_split
from mlx_decision.data.format import DataError
from mlx_decision.data.mix import Mix, MixSource, write_mix
from mlx_decision.errors import DecisionError
from mlx_decision.training import evaluate, fit
from mlx_decision.training.console import format_eval, format_fit

URGENT = {"type": "noul", "instructions": "The message is urgent"}
MOOD = {"type": "choice", "criteria": {"calm": "Calm", "angry": "Angry"}}


def rows(count: int, question: str, label: str, prefix: str) -> list[dict]:
    return [
        {"id": f"{i}", "state": f"{prefix} {i}", "labels": {question: label}} for i in range(count)
    ]


def write_folder(folder: Path, questions: dict, splits: dict[str, tuple[str, list[dict]]]) -> Path:
    folder.mkdir(parents=True)
    for name, (_, lines) in splits.items():
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    roles = {name: {"role": role} for name, (role, _) in splits.items()}
    body = {"questions": questions, "splits": roles}
    (folder / DATASET_FILE).write_text(json.dumps(body))
    return folder


@pytest.fixture
def sources(tmp_path) -> Path:
    """tickets (a folder: train 30, validation 10, test 8), urgent (a plain
    file of 20 rows: no report split) and moods (a folder: train 12, test 6)."""
    data = tmp_path / "datasets"
    write_folder(data / "tickets", {"team": TEAM}, {
        "train": ("train", rows(30, "team", "billing", "ticket")),
        "validation": ("report", rows(10, "team", "tech", "check")),
        "test": ("evaluate", rows(8, "team", "sales", "exam")),
    })  # fmt: skip
    (data / "urgent.jsonl").write_text(
        "".join(
            json.dumps({**r, "questions": {"urgent": URGENT}}) + "\n"
            for r in rows(20, "urgent", "true", "alarm")
        )
    )
    write_folder(data / "moods", {"mood": MOOD}, {
        "train": ("train", rows(12, "mood", "calm", "mood")),
        "test": ("evaluate", rows(6, "mood", "angry", "exam")),
    })  # fmt: skip
    return data


def mix_folder(data: Path, body: dict, name: str = "mix") -> Path:
    folder = data / name
    folder.mkdir()
    (folder / DATASET_FILE).write_text(json.dumps({"format": 2, "mix": body}))
    return folder


def by_source(examples) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.source] = counts.get(example.source, 0) + 1
    return counts


THREE = {"sources": [{"path": "../tickets"}, {"path": "../urgent.jsonl"},
                     {"path": "../moods", "held_out": True}]}  # fmt: skip


def test_the_splits_are_the_sources_splits_together(sources):
    folder = mix_folder(sources, THREE)
    dataset, split = open_dataset(folder)
    assert split is None and list(dataset.splits) == ["train", "report", "evaluate"]
    train, report, test = (dataset.read(name) for name in ("train", "report", "evaluate"))
    # urgent has no report split: a fifth of its rows reports; moods is held out.
    assert by_source(train) == {"tickets": 30, "urgent": 16}
    assert by_source(report) == {"tickets": 10, "urgent": 4, "moods": 2}
    assert by_source(test) == {"tickets": 8, "moods": 6}
    assert {e.id for e in train if e.source == "urgent"}.isdisjoint(
        e.id for e in report if e.source == "urgent"
    )
    assert train[0].id == "tickets/0" and train[0].labels == {"team": "billing"}
    assert all(e.id.startswith(f"{e.source}/") for e in train + report + test)
    assert dataset.missing() == []


def test_caps_sample_with_the_seed(sources):
    body = {"sources": [{"path": "../tickets", "cap": 5}, {"path": "../urgent.jsonl"}],
            "cap": 10, "seed": 3}  # fmt: skip
    dataset, _ = open_dataset(mix_folder(sources, body))
    train = dataset.read("train")
    assert by_source(train) == {"tickets": 5, "urgent": 8}  # 10 capped, a fifth reports
    assert [e.id for e in train] == [e.id for e in dataset.read("train")]
    other, _ = open_dataset(mix_folder(sources, {**body, "seed": 4}, "other"))
    assert [e.id for e in other.read("train")] != [e.id for e in train]
    # A source's report split is read whole, its cap is on train rows.
    assert by_source(dataset.read("report")) == {"tickets": 10, "urgent": 2}


def test_the_hash_covers_sources_caps_and_seed(sources):
    folder = mix_folder(sources, THREE)
    dataset, _ = open_dataset(folder)
    first = dataset.sha256("train")
    assert dataset.sha256("train") == first != dataset.sha256("report")
    capped, _ = open_dataset(mix_folder(sources, {**THREE, "cap": 5}, "capped"))
    assert capped.sha256("train") != first
    lines = (sources / "tickets" / "train.jsonl").read_text()
    (sources / "tickets" / "train.jsonl").write_text(lines.replace("ticket 3", "ticket three"))
    assert dataset.sha256("train") != first


def test_a_fit_and_an_evaluation_read_the_mix(sources, tmp_path, monkeypatch):
    folder = mix_folder(sources, THREE)
    assert report_split(folder) == folder / "report.jsonl"
    read = read_for(folder)
    assert read.split == "train" and read.description.endswith("(a mix of 3 datasets, train split)")
    base = write_model(tmp_path / "base")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["training", "fit", "-m", str(base), "--data", str(folder)])
    assert result.exit_code == 0, result.output
    written = json.loads((tmp_path / "base-mix" / "training.json").read_text())
    assert written["data"]["file"] == f"{folder} (a mix of 3 datasets, train split)"
    assert written["options"]["validation"] == f"{folder} (a mix of 3 datasets, report split)"
    evaluated = evaluate(base, folder)
    assert evaluated.data["examples"] == 14
    command = CliRunner().invoke(app, ["training", "eval", "-m", str(base), "--data", str(folder)])
    assert command.exit_code == 0, command.output
    assert "(a mix of 3 datasets, evaluate split)" in command.output


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"sources": [{"path": "../gone"}]}, "source ../gone is missing"),
        ({"sources": [{"path": "../tickets", "held_out": True}]}, "every source .* is held out"),
        ({"sources": []}, "has no sources"),
        ({"sources": [{"path": "../tickets", "cap": 0}]}, r"cap must be a whole number from 1"),
        ({"sources": ["../tickets"]}, "must be an object with a path"),
        ({"sources": [{"path": "../tickets"}], "seed": "1"}, "seed must be a whole number"),
        ({"sources": [{"path": "../tickets"}, {"path": "../tickets/train.jsonl"}]},
         "one name, tickets"),
    ],
)  # fmt: skip
def test_a_mix_that_cannot_be_read_says_why(sources, body, message):
    folder = mix_folder(sources, body)
    with pytest.raises(DecisionError, match=message):
        dataset, _ = open_dataset(folder)
        dataset.require_ready()


def test_a_mix_inside_a_mix_is_refused(sources):
    mix_folder(sources, THREE, "inner")
    outer = mix_folder(sources, {"sources": [{"path": "../inner"}]}, "outer")
    dataset, _ = open_dataset(outer)
    with pytest.raises(DecisionError, match="is a mix itself"):
        dataset.require_ready()


def test_one_question_id_asked_two_ways_is_refused(sources):
    def six(question: dict, label: str) -> dict:
        return {"team": question}, {"train": ("train", rows(6, "team", label, "row"))}

    write_folder(sources / "teams", *six(MOOD, "calm"))
    dataset, _ = open_dataset(
        mix_folder(sources, {"sources": [{"path": "../tickets"}, {"path": "../teams"}]})
    )
    with pytest.raises(DecisionError, match="tickets and teams both ask 'team', differently"):
        dataset.read("train")
    # Asked the same way, one question id may come from two sources.
    write_folder(sources / "more", *six(TEAM, "tech"))
    same, _ = open_dataset(
        mix_folder(sources, {"sources": [{"path": "../tickets"}, {"path": "../more"}]}, "same")
    )
    assert by_source(same.read("train")) == {"tickets": 30, "more": 5}


def test_a_line_error_names_the_sources_file(sources):
    (sources / "tickets" / "train.jsonl").write_text('{"state": "s", "labels": {"team": "nope"}}\n')
    dataset, _ = open_dataset(mix_folder(sources, THREE))
    with pytest.raises(DataError) as raised:
        dataset.read("train")
    assert str(raised.value).startswith(f"{sources / 'tickets' / 'train.jsonl'} line 1:")


def test_a_mix_has_three_splits_only(sources):
    folder = mix_folder(sources, THREE)
    with pytest.raises(DecisionError, match="its splits are train, report, evaluate"):
        read_for(folder / "test.jsonl")
    (folder / DATASET_FILE).write_text(
        json.dumps({"format": 2, "mix": THREE, "questions": {"team": TEAM}})
    )
    with pytest.raises(DecisionError, match="takes its questions from its sources"):
        open_dataset(folder)


def test_write_mix_checks_its_sources_and_round_trips(sources):
    mix = Mix((MixSource("../tickets", cap=5), MixSource("../moods", held_out=True)), cap=9, seed=2)
    folder = write_mix(sources / "made", mix, created="2026-11-06")
    body = json.loads((folder / DATASET_FILE).read_text())
    assert list(body) == ["format", "mix", "created"] and body["format"] == 2
    assert body["mix"] == {"sources": [{"path": "../tickets", "cap": 5},
                                       {"path": "../moods", "held_out": True}],
                           "cap": 9, "seed": 2}  # fmt: skip
    dataset, _ = open_dataset(folder)
    assert dataset.mix == mix and dataset.to_json() == body
    with pytest.raises(DecisionError, match="exists"):
        write_mix(folder, mix)
    with pytest.raises(DecisionError, match="source ../gone is missing"):
        write_mix(sources / "broken", Mix((MixSource("../gone"),)))
    assert not (sources / "broken").exists()


# Reports per source


def test_a_fit_reports_each_source_with_held_out_ones_marked(sources, tmp_path):
    folder = mix_folder(sources, THREE)
    base = write_model(tmp_path / "base")
    result = fit(base, folder, output=tmp_path / "fitted")
    assert {name: info["held_out"] for name, info in result.sources.items()} == {
        "tickets": False, "urgent": False, "moods": True
    }  # fmt: skip
    counts = {
        (name, question_id): entry["before"]["n"]
        for name, info in result.sources.items()
        for question_id, entry in info["questions"].items()
    }
    assert counts == {("tickets", "team"): 10, ("urgent", "urgent"): 4, ("moods", "mood"): 2}
    assert set(result.sources["urgent"]["questions"]["urgent"]["after"]) >= {"accuracy", "ece"}
    written = json.loads((tmp_path / "fitted" / "training.json").read_text())
    assert written["sources"] == result.sources
    text = format_fit(result.report, str(result.path))
    assert "per source (accuracy / ECE on the report examples)" in text
    assert "moods *" in text and "held out: scored, not trained on" in text


def test_an_evaluation_reports_each_source(sources, tmp_path):
    folder = mix_folder(sources, THREE)
    base = write_model(tmp_path / "base")
    result = evaluate(base, folder, compare=base)
    for model in result.models:
        assert {s: list(q) for s, q in model.sources.items()} == {
            "tickets": ["team"], "moods": ["mood"]
        }  # fmt: skip
        assert model.sources["tickets"]["team"]["n"] == 8
        assert model.as_json()["sources"] == model.sources
    text = format_eval(result)
    assert "per source (accuracy / ECE)" in text


def test_data_without_sources_reports_none(sources, tmp_path):
    base = write_model(tmp_path / "base")
    result = fit(base, sources / "tickets", output=tmp_path / "fitted")
    assert result.sources == {} and "sources" not in result.report
    assert "per source" not in format_fit(result.report, str(result.path))
    evaluated = evaluate(base, sources / "tickets")
    assert evaluated.models[0].sources == {} and "sources" not in evaluated.models[0].as_json()
