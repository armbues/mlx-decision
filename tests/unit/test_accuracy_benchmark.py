"""The accuracy benchmark script: records, metrics, sampling and resuming, without weights."""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import accuracy_benchmark as ab  # noqa: E402


def choice_example(i: int, state: str = "short state", options: int = 3) -> dict:
    return {
        "id": f"bench:{i}",
        "state": state,
        "questions": {
            "q": {"type": "choice", "criteria": {f"o{k}": f"Option {k}" for k in range(options)}}
        },
        "gold": "o0" if i % 2 == 0 else "o1",
    }


def test_record_choice():
    answer = {
        "type": "choice",
        "choice": "b",
        "confidence": 0.5,
        "probabilities": {"a": 0.3, "b": 0.7},
    }
    r = ab.record(answer, "b")
    assert (r["predicted"], r["p_max"], r["confidence"]) == ("b", 0.7, 0.5)
    assert "score" not in r


def test_record_noul_predicts_yes_from_one_half():
    r = ab.record({"type": "noul", "noul": 0.5}, "true")
    assert (r["predicted"], r["p_max"], r["noul"]) == ("true", 0.5, 0.5)
    r = ab.record({"type": "noul", "noul": 0.2}, "true")
    assert (r["predicted"], r["p_max"]) == ("false", 0.8)


def test_record_score_keeps_the_most_probable_level_and_the_expected_one():
    answer = {
        "type": "score",
        "score": 1.4,
        "confidence": 0.6,
        "legend": {},
        "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5},
    }
    r = ab.record(answer, "1")
    assert (r["predicted"], r["p_max"], r["score"]) == ("2", 0.5, 1.4)


def test_metrics():
    records = [
        {"gold": "a", "predicted": "a", "p_max": 0.9},
        {"gold": "a", "predicted": "b", "p_max": 0.9, "truncated": True},
        {"gold": "b", "predicted": "b", "p_max": 0.7},
        {"gold": "b", "predicted": None, "refused": "too long"},
    ]
    m = ab.metrics(records)
    assert (m["n"], m["refused"], m["truncated"]) == (4, 1, 1)
    assert m["accuracy"] == 0.5
    # a: tp 1, fp 0, fn 1 -> 2/3; b: tp 1, fp 1, fn 1 -> 1/2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    # bin 0.9: 2 answered, accuracy 0.5 -> 0.4; bin 0.7: accuracy 1 -> 0.3
    assert m["ece"] == pytest.approx(2 / 3 * 0.4 + 1 / 3 * 0.3)
    assert "mae" not in m


def test_metrics_score_mae_leaves_out_refusals():
    records = [
        {"gold": "1", "predicted": "1", "p_max": 0.6, "score": 1.5},
        {"gold": "3", "predicted": "2", "p_max": 0.5, "score": 2.0},
        {"gold": "0", "predicted": None, "refused": "x"},
    ]
    m = ab.metrics(records)
    assert m["mae"] == pytest.approx(0.75)
    assert "MAE 0.75" in ab.cell(m) and "(3, 1 refused)" in ab.cell(m)


def test_select_picks_what_sampling_the_rows_picked():
    rows = [{"x": i} for i in range(50)]
    assert ab.select(rows, 7) == random.Random(ab.SEED).sample(rows, 7)
    assert ab.select(rows, None) == rows
    assert ab.select(rows, 80) == rows


def test_load_source_checks_rows(tmp_path: Path):
    datasets = pytest.importorskip("datasets")
    rows = [{"text": "a", "label": 0}, {"text": "b", "label": 1}]
    datasets.Dataset.from_list(rows).save_to_disk(str(tmp_path / "org" / "name"))
    source = ab.Source("org/name", None, "test", "0" * 40, 2, ab.digest(rows))
    _, loaded = ab.load_source(tmp_path, source)
    assert loaded == rows
    with pytest.raises(SystemExit, match="not org/name test"):
        ab.load_source(tmp_path, ab.Source("org/name", None, "test", "0" * 40, 2, "f" * 16))


def test_answer_all_records_and_resumes(tmp_path: Path):
    import mlx_decision
    from fake_backend import write_model

    model = mlx_decision.load(write_model(tmp_path / "m", max_input_tokens=3))
    examples = [choice_example(0), choice_example(1, state="a state of five words")]
    out = tmp_path / "results.json"
    results = {"benchmarks": {}}
    ab.answer_all(model, "bench", examples, results, out)
    records = results["benchmarks"]["bench"]
    assert [r["id"] for r in records] == ["bench:0", "bench:1"]
    assert [r["truncated"] for r in records] == [False, True]
    assert records[0]["predicted"] == "o0" and records[0]["gold"] == "o0"
    assert json.loads(out.read_text())["benchmarks"]["bench"] == records

    more = [*examples, choice_example(2)]
    ab.answer_all(model, "bench", more, results, out)
    assert [r["id"] for r in results["benchmarks"]["bench"]] == ["bench:0", "bench:1", "bench:2"]


def test_answer_all_keeps_refusals(tmp_path: Path):
    import mlx_decision
    from fake_backend import write_model

    model = mlx_decision.load(write_model(tmp_path / "m", max_choice_options=2))
    results = {"benchmarks": {}}
    ab.answer_all(model, "bench", [choice_example(0, options=3)], results, tmp_path / "r.json")
    (r,) = results["benchmarks"]["bench"]
    assert r["predicted"] is None and r["refused"]


def test_start_results_refuses_other_results(tmp_path: Path):
    import mlx_decision
    from fake_backend import write_model

    model = mlx_decision.load(write_model(tmp_path / "m"))
    out = tmp_path / "results.json"
    args = SimpleNamespace(suite="coverage", samples=None, out=out)
    results = ab.start_results(args, model, ["trec"])
    assert results["datasets"]["trec"]["revision"] == ab.TREC.revision
    ab.write(out, results)
    assert ab.start_results(args, model, ["trec"])["model"] == "fake"
    with pytest.raises(SystemExit, match="holds other results"):
        ab.start_results(SimpleNamespace(suite="coverage", samples=50, out=out), model, ["trec"])
