"""The accuracy benchmark script: suites, sampling, resuming and the report, without weights."""

import json
import math
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


def test_cell_shows_mae_and_refusals():
    m = {"n": 3, "refused": 1, "truncated": 0, "accuracy": 0.5, "macro_f1": 0.5, "ece": 0.1}
    assert ab.cell(m) == "50.0 / 50.0 / 10.0 (3, 1 refused)"
    assert "MAE 0.75" in ab.cell({**m, "mae": 0.75})


def test_coverage_suite_is_the_presets():
    from mlx_decision.data.presets import PRESETS

    coverage = ab.SUITES["coverage"]
    presets = [preset for preset in PRESETS.values() if preset.coverage]
    assert list(coverage) == [preset.key for preset in presets]
    assert [b.source for b in coverage.values()] == [p.source for p in presets]
    # typed-decisions is a suite of its own, from the preset's split.
    assert (
        ab.SUITES["typed-decisions"]["typed_decisions"].source == PRESETS["typed-decisions"].source
    )


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
    assert ab.load_source(tmp_path, source) == rows
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
    assert results["datasets"]["trec"]["revision"] == ab.PRESETS["trec"].source.revision
    ab.write(out, results)
    assert ab.start_results(args, model, ["trec"])["model"] == "fake"
    with pytest.raises(SystemExit, match="holds other results"):
        ab.start_results(SimpleNamespace(suite="coverage", samples=50, out=out), model, ["trec"])


# typed-decisions: whole cases, scored as the benchmark's board scores them.

ROOT = Path(__file__).resolve().parents[2]


def typed_case(i: int, workflow: str = "invoices") -> dict:
    return {
        "id": f"case:{i}",
        "workflow": workflow,
        "state": f"invoice {i} for 40 EUR",
        "questions": {
            "pay": {"type": "noul", "instructions": "Should it be paid?"},
            "route": {"type": "choice", "criteria": {"approve": "Fine", "hold": "Wait"}},
            "risk": {"type": "score", "criteria": ["Low", "Medium", "High"]},
        },
        "gold": {
            "pay": {"type": "noul", "probabilities": {"false": 0.3, "true": 0.7}},
            "route": {"type": "choice", "probabilities": {"approve": 0.6, "hold": 0.4}},
            "risk": {"type": "score", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
        },
    }


def test_score_decision_follows_the_boards_scorer():
    gold = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5, "c": 0.0}}
    # Ties go to the first option in the gold's order, on both sides.
    ok, kl, brier, top = ab.score_decision({"a": 1, "b": 1, "c": 1}, gold)
    assert ok and top == pytest.approx(1 / 3)
    assert kl == pytest.approx(math.log(1.5))  # 2 x 0.5 log(0.5 / (1/3))
    assert brier == pytest.approx(2 * (1 / 6) ** 2 + (1 / 3) ** 2)
    # Renormalised; a zero prediction floored at 1e-9 in KL; missing is uniform.
    assert ab.score_decision({"a": 2, "b": 6}, gold)[:2] == (
        False,
        pytest.approx(0.5 * math.log(2) + 0.5 * math.log(0.5 / 0.75)),
    )
    assert ab.score_decision({"c": 1}, gold)[1] == pytest.approx(math.log(0.5 / 1e-9))
    assert ab.score_decision({}, gold) == ab.score_decision({"a": 1, "b": 1, "c": 1}, gold)


def test_pooled_ece_weighs_each_bin_by_its_size():
    scored = [(True, 0, 0, 0.95), (False, 0, 0, 0.95), (True, 0, 0, 0.15), (True, 0, 0, 1.0)]
    # Bin 9: three decisions, 2 right, mean confidence 0.9667; bin 1: one, right, 0.15.
    expected = 3 / 4 * abs(2 / 3 - (0.95 + 0.95 + 1.0) / 3) + 1 / 4 * abs(1 - 0.15)
    assert ab.pooled_ece(scored) == pytest.approx(expected)


def test_answer_cases_records_distributions_and_resumes(tmp_path: Path):
    import mlx_decision
    from fake_backend import write_model

    model = mlx_decision.load(write_model(tmp_path / "m", max_choice_options=2))
    out = tmp_path / "results.json"
    results = {"benchmarks": {}}
    wide = typed_case(1)
    wide["questions"]["route"]["criteria"]["reject"] = "No"  # three options: refused
    ab.answer_cases(model, "typed", [typed_case(0), wide], results, out)
    answered, refused = results["benchmarks"]["typed"]
    assert answered["gold"] == typed_case(0)["gold"] and answered["workflow"] == "invoices"
    predicted = answered["predicted"]
    assert set(predicted) == {"pay", "route", "risk"}
    assert set(predicted["pay"]) == {"true", "false"}
    assert sum(predicted["pay"].values()) == pytest.approx(1)
    assert set(predicted["risk"]) == {"0", "1", "2"}  # levels, as the gold names them
    assert refused["predicted"] is None and refused["refused"]
    assert json.loads(out.read_text())["benchmarks"]["typed"] == [answered, refused]
    ab.answer_cases(model, "typed", [typed_case(0), wide, typed_case(2)], results, out)
    assert [r["id"] for r in results["benchmarks"]["typed"]] == ["case:0", "case:1", "case:2"]


def test_case_metrics_by_type_and_workflow():
    exact = {
        **typed_case(0),
        "predicted": {q: g["probabilities"] for q, g in typed_case(0)["gold"].items()},
    }
    other = {**typed_case(1, "security"), "predicted": None, "refused": "too long"}
    m = ab.case_metrics([exact, other])
    assert (m["n"], m["cases"], m["refused"], m["truncated"]) == (6, 2, 1, 0)
    assert m["by_workflow"]["invoices"]["accuracy"] == 1
    assert m["by_workflow"]["invoices"]["kl"] == pytest.approx(0, abs=1e-12)
    # The refused case answers uniform: right only where the gold's first option wins.
    assert m["by_workflow"]["security"]["accuracy"] == pytest.approx(1 / 3)
    assert m["by_type"]["noul"]["accuracy"] == 0.5 and m["by_type"]["noul"]["n"] == 2
    assert m["accuracy"] == pytest.approx(4 / 6)


def test_report_has_a_table_for_cases(tmp_path: Path, capsys):
    record = {
        **typed_case(0),
        "predicted": {q: g["probabilities"] for q, g in typed_case(0)["gold"].items()},
    }
    typed = tmp_path / "typed.json"
    typed.write_text(
        json.dumps(
            {"suite": "typed-decisions", "model": "m", "benchmarks": {"typed_decisions": [record]}}
        )
    )
    classic = tmp_path / "classic.json"
    classic.write_text(json.dumps({"suite": "coverage", "model": "c", "benchmarks": {}}))
    ab.report(SimpleNamespace(results=[str(classic), str(typed)], json=False))
    text = capsys.readouterr().out
    assert "| m | 1.000 / 0.000 / 0.000 / " in text
    assert "| Model | choice | noul | score |" not in text  # types sit in the first table
    assert "| choice | noul | score | Cases |" in text and "| invoices |" in text
    ab.report(SimpleNamespace(results=[str(typed)], json=True))
    (row,) = json.loads(capsys.readouterr().out)
    assert row["metrics"]["by_type"]["score"]["accuracy"] == 1


@pytest.mark.skipif(
    not (ROOT / "_datasets" / "LocalLLaMA" / "typed-decisions").is_dir(),
    reason="typed-decisions is not saved in _datasets",
)
def test_the_boards_reference_predictions():
    """Uniform and a prior of train label frequencies score as the board's
    published scorer scores them (its KL and Brier also the card's Uniform row)."""
    import collections

    from datasets import load_from_disk

    cases = ab.typed_decisions(ROOT / "_datasets")
    assert len(cases) == 400 and sum(len(c["gold"]) for c in cases) == 2000
    train = load_from_disk(str(ROOT / "_datasets" / "LocalLLaMA" / "typed-decisions"))["train"]
    counts = collections.defaultdict(collections.Counter)
    for row in train:
        for qid, gold in json.loads(row["gold"]).items():
            counts[(row["workflow"], qid)][gold["label"]] += 1

    def scored(predict):
        records = [
            {**c, "predicted": {q: predict(c, q, g) for q, g in c["gold"].items()}} for c in cases
        ]
        m = ab.case_metrics(records)
        return [round(m[key], 4) for key in ("accuracy", "kl", "brier", "ece")]

    uniform = scored(lambda c, q, g: {o: 1.0 for o in g["probabilities"]})
    assert uniform == [0.2765, 0.4445, 0.2382, 0.045]
    prior = scored(lambda c, q, g: {o: counts[(c["workflow"], q)][o] for o in g["probabilities"]})
    assert prior == [0.4815, 0.4174, 0.1973, 0.0367]
