"""The latency benchmark, through the fake backend."""

import json

from typer.testing import CliRunner

from mlx_decision.benchmark import format_report, make_questions, percentile, run
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
    # The fake backend keeps no prefixes: no warm times, no warm columns.
    assert all(c.warm_median_s is None for c in report.cells)
    assert "Warm" not in format_report(report)


class KeptStates:
    """A stand-in prefix cache: keeps states up to ``max_words`` words, counts hits."""

    def __init__(self, max_words: int):
        self.max_bytes = 2e9
        self.max_words = max_words
        self.kept: set[str] = set()
        self.hits = 0
        self.cleared = 0

    def clear(self):
        self.kept.clear()
        self.cleared += 1

    def wrap(self, score):
        def scored(request):
            if request.state in self.kept:
                self.hits += 1
            elif len(request.state.split()) <= self.max_words:
                self.kept.add(request.state)
            return score(request)

        return scored


def test_warm_times_with_a_prefix_cache(fake_model):
    cache = KeptStates(max_words=50)
    fake_model.backend.prefix_cache = cache
    fake_model.backend.score = cache.wrap(fake_model.backend.score)
    report = run(fake_model, lengths=(10, 100), question_counts=(1,), repeats=3)
    small, large = report.cells
    assert small.warm_p95_s >= small.warm_median_s > 0
    # Too large to keep: every warm request missed, so there is nothing to show.
    assert large.warm_median_s is None and large.warm_p95_s is None
    assert cache.hits == 3
    # Each cold request dropped the kept prefixes first.
    assert cache.cleared == 2 * 3
    assert report.prefix_cache_gb == 2.0
    table = format_report(report)
    assert "| Warm median s | Warm p95 s |" in table
    assert "| 100 | 1 | 100 |" in table and table.count(" - | - |") == 1


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


def test_results_file_names_machine_software_model_and_options(
    fake_model_path, tmp_path, monkeypatch
):
    import mlx_decision.benchmark as benchmark

    monkeypatch.setattr(benchmark, "machine_info", lambda: dict(MACHINE))
    out = tmp_path / "result.json"
    args = ["benchmark", "-m", str(fake_model_path), "--lengths", "20", "--questions", "2"]
    result = CliRunner().invoke(app, [*args, "--repeats", "2", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("Apple M1 (MacBookPro17,1), 4 Performance + 4 Efficiency")
    assert f"results written to {out}" in result.stderr
    data = json.loads(out.read_text())
    assert data["format"] == 3
    assert data["date"].endswith("+00:00")
    assert data["machine"] == MACHINE
    assert set(data["software"]) == {"mlx_decision", "mlx", "python"}
    assert data["model"] == "fake"
    assert data["model_info"]["family"] == "fake"
    assert data["options"] == {
        "lengths": [20],
        "question_counts": [2],
        "repeats": 2,
        "prefix_cache_gb": None,
    }
    assert data["cells"][0]["warm_median_s"] is None
    assert data["cells"][0]["input_tokens"] == 20


MACHINE = {
    "model": "MacBookPro17,1",
    "chip": "Apple M1",
    "cpu_cores": [{"name": "Performance", "count": 4}, {"name": "Efficiency", "count": 4}],
    "gpu_cores": 8,
    "memory_bytes": 16 * 2**30,
    "gpu_working_set_bytes": 11_453_251_584,
    "macos": "15.6",
}


def test_machine_description():
    from mlx_decision.machine import describe

    assert describe(MACHINE) == (
        "Apple M1 (MacBookPro17,1), 4 Performance + 4 Efficiency CPU cores, 8 GPU cores, "
        "16 GB memory, 10.7 GB GPU working set, macOS 15.6"
    )
    assert describe({}) == "unknown chip"


def test_unreadable_machine_values_are_none(monkeypatch):
    import mlx_decision.machine as machine

    monkeypatch.setattr(machine, "_command", lambda *args: None)
    monkeypatch.setattr(machine, "working_set_bytes", lambda: None)
    monkeypatch.setattr(machine.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert machine.machine_info() == dict.fromkeys(MACHINE)


def test_probes_parse_sysctl_and_system_profiler(monkeypatch):
    import mlx_decision.machine as machine

    answers = {
        ("sysctl", "-n", "hw.nperflevels"): "2",
        ("sysctl", "-n", "hw.perflevel0.physicalcpu"): "6",
        ("sysctl", "-n", "hw.perflevel0.name"): "Super",
        ("sysctl", "-n", "hw.perflevel1.physicalcpu"): "12",
        ("sysctl", "-n", "hw.perflevel1.name"): "Performance",
        ("system_profiler", "SPDisplaysDataType", "-json"): json.dumps(
            {"SPDisplaysDataType": [{"sppci_model": "Apple M5 Pro", "sppci_cores": "20"}]}
        ),
    }
    monkeypatch.setattr(machine, "_command", lambda *args: answers.get(args))
    assert machine.cpu_cores() == [
        {"name": "Super", "count": 6},
        {"name": "Performance", "count": 12},
    ]
    assert machine.gpu_cores() == 20
    answers[("system_profiler", "SPDisplaysDataType", "-json")] = "not json"
    assert machine.gpu_cores() is None


def test_cli_rejects_bad_lists(fake_model_path):
    runner = CliRunner()
    result = runner.invoke(app, ["benchmark", "-m", str(fake_model_path), "--lengths", "a,b"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["benchmark", "-m", str(fake_model_path), "--questions", "0"])
    assert result.exit_code == 2


def test_default_lengths_fit_the_input_limit():
    from mlx_decision.benchmark import DEFAULT_LENGTHS, default_lengths

    assert default_lengths(16384) == DEFAULT_LENGTHS
    assert default_lengths(None) == DEFAULT_LENGTHS
    assert default_lengths(8192) == (250, 1000, 4000, 7350)
    assert default_lengths(1024) == (250, 1000)
    assert default_lengths(512) == (250, 450)
    assert default_lengths(128) == (100,)
