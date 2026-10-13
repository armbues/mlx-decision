"""The accuracy benchmark script: suites, sampling, resuming and the report, without weights."""

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


def test_cell_shows_mae_and_refusals():
    m = {"n": 3, "refused": 1, "truncated": 0, "accuracy": 0.5, "macro_f1": 0.5, "ece": 0.1}
    assert ab.cell(m) == "50.0 / 50.0 / 10.0 (3, 1 refused)"
    assert "MAE 0.75" in ab.cell({**m, "mae": 0.75})


def test_coverage_suite_is_the_presets():
    from mlx_decision.data.presets import PRESETS

    coverage = ab.SUITES["coverage"]
    assert list(coverage) == [preset.key for preset in PRESETS.values()]
    assert [b.source for b in coverage.values()] == [p.source for p in PRESETS.values()]


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
