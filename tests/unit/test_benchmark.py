"""The latency benchmark, through the fake backend."""

import json

from typer.testing import CliRunner

from mlx_decision.benchmark import make_questions, percentile, run
from mlx_decision.cli import app
from mlx_decision.types import parse_request


def test_questions_cycle_through_the_types():
    questions = make_questions(5)
    assert [q["type"] for q in questions.values()] == ["noul", "choice", "score", "noul", "choice"]
    parse_request({"state": "x", "questions": questions})


def test_percentile_is_nearest_rank():
    values = [5.0, 1.0, 4.0, 2.0, 3.0]
    assert percentile(values, 0.95) == 5.0
    assert percentile(values, 0.5) == 3.0
    assert percentile([7.0], 0.95) == 7.0


def test_grid_and_sizes(fake_model):
    report = run(fake_model, lengths=(10, 100), question_counts=(1, 3), repeats=2)
    assert report.model == "fake"
    assert report.repeats == 2
    assert [(c.state_tokens, c.questions) for c in report.cells] == [
        (10, 1),
        (10, 3),
        (100, 1),
        (100, 3),
    ]
    # The fake backend counts words, so the states hit their sizes exactly.
    assert [c.input_tokens for c in report.cells] == [10, 10, 100, 100]
    assert all(c.p95_s >= c.median_s > 0 for c in report.cells)


def test_cli(fake_model_path):
    runner = CliRunner()
    args = ["benchmark", "-m", str(fake_model_path), "--lengths", "20", "--questions", "2"]
    table = runner.invoke(app, [*args, "--repeats", "2"])
    assert table.exit_code == 0, table.output
    assert "| 20 | 2 | 20 |" in table.stdout
    assert "fake: load" in table.stdout
    as_json = json.loads(runner.invoke(app, [*args, "--json"]).stdout)
    assert as_json["cells"][0]["input_tokens"] == 20
    assert as_json["repeats"] == 5


def test_cli_rejects_bad_lists(fake_model_path):
    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "-m", str(fake_model_path), "--lengths", "a,b"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["benchmark", "-m", str(fake_model_path), "--questions", "0"])
    assert result.exit_code == 2
