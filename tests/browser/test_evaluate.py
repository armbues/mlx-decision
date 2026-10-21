"""Evaluate in the Training page: the form from the Datasets page's
"Evaluate on It…" (a fitted model and its base by default), the
warning when the model was fitted on this very file, the job's progress and
its report of both models with the differences, and the History listing
fits and evaluations. The job is ``tests/fake_job.py`` acting as
``training eval``."""

import json
import re

import pytest
from drive import open_studio, shot
from fake_studio import fake_jobs

from mlx_decision.fitted import MARKER_FILE
from mlx_decision.studio.checks import file_sha256

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


@pytest.fixture
def studio(make_studio):
    studio = make_studio(sizes=True)  # fake-fit: fitted over fake
    marker = studio.home / "fake-models" / "fake-fit" / MARKER_FILE
    recorded = json.loads(marker.read_text())
    recorded["fit"]["data"] = file_sha256(studio.workspace / "data" / "tickets.jsonl")
    marker.write_text(json.dumps(recorded))  # as if fitted on tickets
    # The Studio read the marker when it started: the same in its list.
    studio.app.state.gpu.pool.specs["fake-fit"].fit.marker["fit"]["data"] = recorded["fit"]["data"]
    (studio.workspace / "data" / "other.jsonl").write_text(
        (studio.workspace / "data" / "tickets.jsonl").read_text().replace("Stripe", "PayPal")
    )
    fake_jobs(studio.app, steps=10, every=0.05, fits=True)
    return studio


def test_evaluate_a_fitted_model_against_its_base(studio, page):
    open_studio(page, studio, "datasets/tickets")
    page.get_by_role("button", name="Evaluate on It…").click()
    expect(page).to_have_url(f"{studio.base}/#/training/evaluate")
    form = page.locator(".eval-form")
    expect(form.locator("#eval-data")).to_have_value("tickets")
    expect(form.locator("#eval-model")).to_have_value("fake-fit")  # the newest fitted model
    expect(form.locator("#eval-compare")).to_have_value("fake")  # its base
    expect(form).to_contain_text("its base, to see what the fit changed")
    expect(form.locator(".q-chip")).to_have_text(["department choice", "is_urgent noul"])
    warning = form.locator(".warn-text")
    expect(warning).to_have_text(
        "tickets includes fake-fit's training data: fake-fit was fitted on this very file "
        "(the same SHA-256), so its scores here count examples it learned from"
    )
    expect(page.locator(".sidebar .new-list .sb-item[aria-current='true']")).to_have_text(
        "New Evaluation"
    )
    shot(page, "evaluate-form")
    # Another file: no warning.
    form.locator("#eval-data").select_option("other")
    expect(warning).to_have_count(0)
    form.locator("#eval-data").select_option("tickets")
    expect(warning).to_have_count(1)

    page.locator(".run-btn").click()  # Start Evaluation
    expect(page).to_have_url(re.compile(r"#/training/\d{8}-\d{6}-eval-[0-9a-f]{6}$"))
    head = page.locator(".centre .pane-head")
    expect(head.locator("h1")).to_have_text("Evaluate fake-fit against fake")
    expect(head.locator("p")).to_have_text("fake-fit against fake · on tickets")
    expect(page.locator(".stepper li span:not(.dot)")).to_have_text(
        ["Load fake-fit", "Score fake-fit", "Load fake", "Score fake"]
    )
    expect(page.locator(".run-warning")).to_contain_text("fake-fit was fitted on this very file")

    # The report: both models, the difference.
    table = page.locator(".eval-table").first
    expect(table).to_be_visible(timeout=15_000)
    expect(table.locator("thead th")).to_have_text(["", "fake-fit", "fake", "Difference"])
    accuracy = table.locator("tr", has_text="Accuracy")
    expect(accuracy.locator("td")).to_have_text(["90.0%", "70.0%", re.compile(r"\+20\.0 pts")])
    expect(accuracy.locator(".change")).to_have_class(re.compile("better"))
    expect(table.locator("tr", has_text="Calibration error").locator(".change")).to_have_class(
        re.compile("better")
    )
    page.locator(".confusions summary").first.click()
    expect(page.locator(".confusion h4")).to_have_text(["fake-fit", "fake"])
    expect(page.locator(".confusion").first.locator("td.hit").first).to_have_text("5")
    expect(page.locator(".stepper li.done")).to_have_count(4)
    command = page.locator(".kv", has_text="Command").locator("code")
    expect(command).to_contain_text("mlx-decision training eval -m")
    expect(command).to_contain_text("--compare")
    shot(page, "evaluate-report")

    # The History: the evaluation, tagged, opening its report.
    row = page.locator(".sidebar .fit-row").first
    expect(row.locator(".label")).to_have_text("fake-fit against fake")
    expect(row.locator(".run-kind")).to_have_text("eval")
    page.goto(f"{studio.base}/#/training")
    row.locator(".sb-item").click()
    expect(page.locator(".eval-table").first).to_be_visible()


def test_no_comparison(studio, page):
    open_studio(page, studio, "training/evaluate")
    form = page.locator(".eval-form")
    form.locator("#eval-model").select_option("fake")
    expect(form.locator("#eval-compare")).to_have_value("")  # a base: nothing to compare
    form.locator("#eval-data").select_option("other")
    page.locator(".run-btn").click()
    table = page.locator(".eval-table").first
    expect(table).to_be_visible(timeout=15_000)
    expect(table.locator("thead th")).to_have_text(["", "fake"])
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert "--compare" not in job.argv and job.details["warnings"] == []


def test_evaluate_on_a_folders_evaluate_split(studio, page):
    folder = studio.workspace / "data" / "help"
    folder.mkdir()
    for split, count in (("train", 4), ("validation", 3), ("test", 2)):
        rows = [{"id": f"{split}{i}", "state": f"{split} {i}", "values": {"label": "yes"}}
                for i in range(count)]  # fmt: skip
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    body = {
        "questions": {"ok": {"type": "noul", "instructions": "Is it fine?"}},
        "answers": {"ok": {"column": "label", "map": {"yes": True}}},
        "columns": {"label": {"values": {"yes": {"rows": 9}}}},
        "splits": {"train": {"role": "train"}, "validation": {"role": "report"},
                   "test": {"role": "evaluate"}},
    }  # fmt: skip
    (folder / "dataset.json").write_text(json.dumps(body))
    open_studio(page, studio, "datasets/help")
    page.get_by_role("button", name="Evaluate on It…").click()
    form = page.locator(".eval-form")
    expect(form.locator("#eval-data")).to_have_value("help")
    split = form.locator("#eval-split")
    expect(split).to_have_value("test")  # its evaluate split
    expect(split.locator("option")).to_have_text(
        ["train (train)", "validation (report)", "test (evaluate)"]
    )
    expect(form).to_contain_text("2 examples")
    split.select_option("validation")
    expect(form).to_contain_text("3 examples")
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".centre .pane-head p")).to_contain_text("on help (validation)")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["split"] == "validation"
    assert job.argv[job.argv.index("--data") + 1].endswith("help/validation.jsonl")
