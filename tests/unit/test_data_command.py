"""``mlx-decision data``: sources, shapes, wording and presets; the Hub mocked."""

import json

import datasets
import pytest
from typer.testing import CliRunner

from mlx_decision.cli import app
from mlx_decision.data.format import read_examples
from mlx_decision.data.make import make, rows_to_read
from mlx_decision.data.presets import PRESETS
from mlx_decision.data.presets import examples as preset_examples
from mlx_decision.data.shapes import guess
from mlx_decision.data.sources import SourceError, open_source

ASK = {"instructions": "Which fits?"}


def hub(monkeypatch, repos: dict[str, dict[str, dict[str, datasets.Dataset]]]):
    """Mock the Hub: ``repos[repo][config][split]`` (config "default" for none)."""
    calls = []

    def configs(repo, revision=None):
        return list(repos[repo])

    def splits(repo, config=None, revision=None):
        return list(repos[repo][config or "default"])

    def load(repo, config=None, split=None, revision=None):
        calls.append((repo, config, split, revision))
        return repos[repo][config or "default"][split]

    monkeypatch.setattr(datasets, "get_dataset_config_names", configs)
    monkeypatch.setattr(datasets, "get_dataset_split_names", splits)
    monkeypatch.setattr(datasets, "load_dataset", load)
    return calls


def labelled(texts, labels, names) -> datasets.Dataset:
    features = datasets.Features(
        {"text": datasets.Value("string"), "label": datasets.ClassLabel(names=names)}
    )
    return datasets.Dataset.from_dict({"text": texts, "label": labels}, features=features)


@pytest.fixture
def emotion(monkeypatch):
    train = labelled(["so happy", "so sad", "happy again", "sad again"], [0, 1, 0, 1],
                     ["joy", "sadness"])  # fmt: skip
    test = labelled(["glad"], [0], ["joy", "sadness"])
    return hub(monkeypatch, {"org/emotion": {"default": {"train": train, "test": test}}})


def test_a_class_label_dataset_becomes_choices_named_by_its_labels(emotion):
    made = make("hf:org/emotion", split="train", wording=ASK)
    first = made.examples[0]
    assert first.id == "emotion:train/0"
    assert first.request == {
        "state": "so happy",
        "questions": {
            "emotion": {
                "type": "choice",
                "instructions": "Which fits?",
                "criteria": {"joy": "joy", "sadness": "sadness"},
            }
        },
    }
    assert [e.labels["emotion"] for e in made.examples] == ["joy", "sadness", "joy", "sadness"]
    assert made.mapping.shape == "text" and made.label_counts() == {
        "emotion": {"joy": 2, "sadness": 2}
    }
    assert emotion == [("org/emotion", None, "train", None)]


def test_open_choices_are_errors_naming_the_flag(emotion, monkeypatch):
    with pytest.raises(SourceError, match=r"several splits; pass --split \(train, test\)"):
        make("hf:org/emotion", wording=ASK)
    with pytest.raises(SourceError, match="wording is open: pass --question TEXT"):
        make("hf:org/emotion", split="train")
    with pytest.raises(SourceError, match="no split 'dev'; it has train, test"):
        make("hf:org/emotion", split="dev", wording=ASK)
    hub(monkeypatch, {"org/multi": {"a": {"train": labelled(["x"], [0], ["y"])}, "b": {}}})
    with pytest.raises(SourceError, match=r"several configs; pass --config \(a, b\)"):
        make("hf:org/multi", wording=ASK)


def test_two_class_columns_are_ambiguous_until_one_is_named(monkeypatch):
    features = datasets.Features({
        "body": datasets.Value("string"),
        "topic": datasets.ClassLabel(names=["a", "b"]),
        "tone": datasets.ClassLabel(names=["calm", "angry"]),
    })  # fmt: skip
    dataset = datasets.Dataset.from_dict(
        {"body": ["x", "y"], "topic": [0, 1], "tone": [1, 0]}, features=features
    )
    hub(monkeypatch, {"org/mail": {"default": {"train": dataset}}})
    with pytest.raises(SourceError, match=r"label column of the text shape is not clear; pass "
                       r"--label \(one of topic, tone\)"):  # fmt: skip
        make("hf:org/mail", wording=ASK)
    made = make("hf:org/mail", columns={"label": "tone"}, wording={**ASK, "noul": True})
    question = made.examples[0].request["questions"]["mail"]
    assert question["type"] == "noul"
    assert question["criteria"] == {"true": "angry", "false": "calm"}  # the second class is yes
    assert [e.labels["mail"] for e in made.examples] == ["true", "false"]


def test_pairs_put_premise_and_hypothesis_in_the_state(monkeypatch):
    features = datasets.Features({
        "premise": datasets.Value("string"),
        "hypothesis": datasets.Value("string"),
        "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
    })  # fmt: skip
    dataset = datasets.Dataset.from_dict(
        {"premise": ["p1", "p2"], "hypothesis": ["h1", "h2"], "label": [2, -1]}, features=features
    )
    hub(monkeypatch, {"org/nli": {"default": {"validation": dataset}}})
    made = make("hf:org/nli", wording={**ASK, "id": "relation"})
    [example] = made.examples
    assert example.request["state"] == {"premise": "p1", "hypothesis": "h1"}
    assert example.labels == {"relation": "contradiction"}
    assert made.skipped == {"no label": 1}  # -1: the row has no gold label


def test_multiple_choice_rows_bring_their_own_options(tmp_path):
    rows = [
        {"question": "2 + 2?", "choices": {"text": ["3", "4"], "label": ["A", "B"]},
         "answerKey": "B"},
        {"question": "Sky?", "choices": ["green", "blue"], "answer": 1},
        {"question": "Sea?", "choices": ["wet", "dry"], "answer": "wet"},
    ]  # fmt: skip
    path = tmp_path / "quiz.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows[:1]))
    made = make(str(path), wording=ASK)
    assert made.mapping.shape == "choice"
    first = made.examples[0]
    assert first.request["questions"]["quiz"]["criteria"] == {"A": "3", "B": "4"}
    assert first.labels == {"quiz": "B"}
    plain = tmp_path / "plain.jsonl"
    plain.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
    made = make(str(plain), wording=ASK)
    assert [e.labels["plain"] for e in made.examples] == ["B", "A"]  # a position, a text
    assert made.examples[0].request["questions"]["plain"]["criteria"] == {"A": "green", "B": "blue"}


def test_yes_no_rows_ask_their_own_question(tmp_path):
    rows = [{"passage": "Water is wet.", "question": "is water wet", "answer": True},
            {"passage": "Fire is cold.", "question": "is fire hot", "answer": False}]  # fmt: skip
    path = tmp_path / "boolq.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    made = make(str(path))
    assert made.mapping.shape == "yesno"
    assert made.examples[1].request == {
        "state": "Fire is cold.",
        "questions": {"boolq": {"type": "noul", "instructions": "is fire hot"}},
    }
    assert made.examples[1].labels == {"boolq": "false"}
    with pytest.raises(SourceError, match="asks each row's question; drop --question"):
        make(str(path), wording=ASK)


def test_ratings_keep_every_level_in_their_range(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("review,stars\ngreat,5\nbad,1\nok,3\n")
    made = make(str(path), wording=ASK)
    assert made.mapping.shape == "rating"  # the column's name says so
    question = made.examples[0].request["questions"]["reviews"]
    assert question["criteria"] == ["1", "2", "3", "4", "5"]
    assert [e.labels["reviews"] for e in made.examples] == ["4", "0", "2"]
    levels = ["awful", "bad", "ok", "good", "great"]
    made = make(str(path), wording={**ASK, "levels": levels})
    assert made.examples[0].request["questions"]["reviews"]["criteria"] == levels
    with pytest.raises(SourceError, match="--levels: 2 given for 5 values"):
        make(str(path), wording={**ASK, "levels": ["low", "high"]})


def test_label_maps_and_options_name_the_answers(tmp_path):
    path = tmp_path / "tickets.csv"
    path.write_text("message,team\nrefund,0\ncrash,1\nprice,2\n")
    team = {"label": "team"}  # not a name the guess knows
    with pytest.raises(SourceError, match="label column of the text shape is not found"):
        make(str(path), wording=ASK)
    codes = {"0": "billing", "1": "tech", "2": "sales"}
    mapped = make(str(path), columns=team, wording={**ASK, "id": "team", "label_map": codes})
    assert [e.labels["team"] for e in mapped.examples] == ["billing", "tech", "sales"]
    options = {"billing": "Payments and refunds", "tech": "Bugs"}
    merged = {"0": "billing", "1": "tech", "2": "billing"}
    described = make(str(path), columns=team, wording={**ASK, "label_map": merged,
                                                       "options": options})  # fmt: skip
    question = described.examples[0].request["questions"]["tickets"]
    assert question["criteria"] == options
    assert [e.labels["tickets"] for e in described.examples] == ["billing", "tech", "billing"]
    with pytest.raises(SourceError, match="row 2: label '2' is no option"):
        make(str(path), columns=team, wording={**ASK, "label_map": {**codes, "2": "x"},
                                               "options": options})  # fmt: skip


def test_rows_are_read_whole_first_or_sampled_with_a_seed():
    assert rows_to_read(5, None, None, 0) == [0, 1, 2, 3, 4]
    assert rows_to_read(5, 2, None, 0) == [0, 1]
    sampled = rows_to_read(100, None, 10, 3)
    assert sampled == sorted(sampled) and sampled == rows_to_read(100, None, 10, 3)
    assert sampled != rows_to_read(100, None, 10, 4)
    with pytest.raises(SourceError, match="--limit and --sample do not go together"):
        rows_to_read(5, 1, 1, 0)


def test_presets_read_their_pinned_commit_and_name_rows_by_index(monkeypatch):
    preset = PRESETS["trec"]
    features = datasets.Features({
        "text": datasets.Value("string"),
        "coarse_label": datasets.ClassLabel(names=["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]),
        "fine_label": datasets.Value("int32"),
    })  # fmt: skip
    rows = {"text": [f"q{i}?" for i in range(6)], "coarse_label": list(range(6)),
            "fine_label": [0] * 6}  # fmt: skip
    train = datasets.Dataset.from_dict(rows, features=features)
    calls = hub(monkeypatch, {preset.source.repo: {"default": {"train": train, "test": train}}})
    made = make(None, preset="trec", split="train", sample=2, seed=1)
    assert calls == [(preset.source.repo, None, "train", preset.source.revision)]
    indices = rows_to_read(6, None, 2, 1)
    expected = preset_examples(preset, train.select(indices).to_list(), "train", indices)
    assert [e.id for e in made.examples] == [item["id"] for item in expected]
    assert made.examples[0].id == f"trec:train/{indices[0]}"
    with pytest.raises(SourceError, match="a preset fixes the shape, columns and wording"):
        make(None, preset="trec", split="train", wording=ASK)
    with pytest.raises(SourceError, match="no preset 'imdb'; the presets are trec, "):
        make(None, preset="imdb")


def test_saved_folders_and_parquet_files_are_sources(tmp_path):
    dataset = labelled(["good", "bad"], [0, 1], ["pos", "neg"])
    dataset.save_to_disk(str(tmp_path / "saved"))
    dataset.to_parquet(str(tmp_path / "reviews.parquet"))
    saved = open_source(str(tmp_path / "saved"))
    assert saved.columns["label"].names == ["pos", "neg"] and saved.length == 2
    parquet = open_source(str(tmp_path / "reviews.parquet"))
    assert guess(parquet).shape == "text"
    with pytest.raises(SourceError, match="--split is for datasets, not files"):
        open_source(str(tmp_path / "reviews.parquet"), split="train")
    with pytest.raises(SourceError, match="no such file or folder"):
        open_source(str(tmp_path / "missing.csv"))


def test_without_the_extra_the_hub_says_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_datasets(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_datasets)
    with pytest.raises(SourceError, match=r"needs the training extra: pip install "
                       r"'mlx-decision\[training\]'"):  # fmt: skip
        open_source("hf:org/x")


def test_the_command_previews_writes_and_repeats_itself(tmp_path, emotion):
    runner = CliRunner()
    args = ["data", "hf:org/emotion", "--split", "train", "--question", "Which emotion?"]
    shown = runner.invoke(app, [*args, "--show", "1"])
    assert shown.exit_code == 0, shown.output
    lines = shown.stdout.splitlines()
    assert json.loads(lines[0])["labels"] == {"emotion": "joy"}
    assert "4 examples (text shape: text=text, label=label)" in shown.stdout
    assert "labels: emotion: joy 2, sadness 2" in shown.stdout
    for name in ("a.jsonl", "b.jsonl"):
        written = runner.invoke(app, [*args, "-o", str(tmp_path / name)])
        assert written.exit_code == 0, written.output
        assert f"wrote {tmp_path / name}" in written.stderr
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert len(read_examples(tmp_path / "a.jsonl")) == 4
    missing = runner.invoke(app, args)
    assert missing.exit_code == 1 and "pass -o FILE.jsonl (or --show N" in missing.stderr
    question = tmp_path / "q.json"
    question.write_text(json.dumps({"mood": {"type": "noul", "instructions": "Happy?"}}))
    with_q = runner.invoke(app, ["data", "hf:org/emotion", "--split", "train", "-q",
                                 str(question), "--label-map", "joy=true,sadness=false",
                                 "--show", "1"])  # fmt: skip
    assert with_q.exit_code == 0, with_q.output
    first = json.loads(with_q.stdout.splitlines()[0])
    assert first["questions"] == {"mood": {"type": "noul", "instructions": "Happy?"}}
    assert first["labels"] == {"mood": True}
