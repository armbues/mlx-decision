"""The fit form: its dataset, split and labels, the base, the method cards with
their estimates, the result's name and options, and Start Fit (⌘↩) as a job."""

import json
import re

import pytest
from drive import open_studio, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip


@pytest.fixture
def studio(make_studio):
    studio = make_studio(convertible=True)  # the tiny Clef: head, then calibrate
    labels = ["tech"] * 30 + ["billing"] * 12 + ["sales"] * 18
    lines = [
        json.dumps({"state": f"ticket {i}", "questions": {"team": TEAM}, "labels": {"team": label}})
        for i, label in enumerate(labels)
    ]
    (studio.workspace / "data" / "support.jsonl").write_text("\n".join(lines) + "\n")
    return studio


def test_the_form_and_start_fit(studio, page):
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    data = form.locator("#fit-data")
    expect(data.locator("option")).to_have_text(["support", "tickets"])
    expect(data).to_have_value("support")  # the first, until one is picked
    expect(page.locator(".sidebar .new-list .sb-item")).to_have_text(
        ["New Fit…", "New Evaluation…"]
    )
    expect(page.locator(".sidebar .sb-empty", has_text="fits")).to_have_text(
        "No fits or evaluations yet"
    )
    expect(page.locator(".app")).to_have_class(re.compile(r"\bno-inspector\b"))  # none here
    form.locator("#fit-base").select_option("tiny-clef")

    cards = form.locator(".method")
    expect(cards.locator(".mt b")).to_have_text(["Head", "Calibrate"])
    head = form.locator(".method[data-method='head']")
    expect(head).to_have_attribute("aria-checked", "true")  # auto's pick
    expect(head.locator(".pill-tag")).to_have_text("auto")
    expect(head).to_contain_text(re.compile(r"Memory at the peak\s*about [\d.]+ (MB|GB)"))
    expect(head).to_contain_text("On disk while it runs")  # Clef's stored states
    expect(head.locator(".fit")).to_have_text("Fits this Mac")
    expect(head).to_contain_text("Estimated from the model and the data")
    legend = form.locator(".split-legend")
    expect(legend).to_contain_text("12 kept aside for the report")
    # The labels in the question's own order (billing, tech, sales), not by name.
    expect(form.locator(".fit-labels .line span b")).to_have_text(["billing", "tech", "sales"])
    name = form.locator("#fit-name")
    expect(name).to_have_value("tiny-clef-support")

    form.locator(".adv summary").click()
    expect(form.locator("#fit-epochs")).to_be_visible()  # the head trains in epochs
    form.locator("#fit-holdout").fill("25")
    form.locator("#fit-holdout").press("Tab")
    expect(legend).to_contain_text("15 kept aside for the report")
    form.locator(".method[data-method='calibrate']").click()
    expect(form.locator("#fit-epochs")).to_have_count(0)  # calibrate has no epochs

    # A name in the list: Start Fit is off, with the reason.
    name.fill("fake")
    run = page.locator(".run-btn")
    expect(run).to_be_disabled()
    expect(run).to_have_attribute("title", "A model named fake exists already")
    name.fill("support-calibrated")
    expect(run).to_be_enabled()
    page.keyboard.press("Meta+Enter")
    # Started: the page shows the fit's progress.
    expect(page.locator(".centre h1")).to_have_text("Fit support-calibrated")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["method"] == "calibrate" and job.details["holdout"] == 0.25
    assert job.argv[job.argv.index("--method") + 1] == "calibrate"
    # One GPU job at a time.
    page.locator(".run-btn", has_text="New Fit").click()
    expect(run).to_have_attribute("title", re.compile("Fit support-calibrated has the GPU"))
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()
    expect(page.locator(".sidebar .job small")).to_have_text("Stopped")


def test_a_family_without_a_main_kind_offers_calibrate(studio, page):
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    form.locator("#fit-data").select_option("support")
    form.locator("#fit-base").select_option("fake")
    expect(form.locator(".method .mt b")).to_have_text(["Calibrate"])
    expect(form.locator(".method").first).to_have_attribute("aria-checked", "true")
    expect(form.locator("#fit-name")).to_have_value("fake-support")


def write_folder(studio, name, splits):
    """A dataset folder: ``splits`` maps a split to (role, rows), each row a label."""
    folder = studio.workspace / "data" / name
    folder.mkdir()
    for split, (_, labels) in splits.items():
        lines = [
            json.dumps({"state": f"{split} {i}", "values": {"label": label}})
            for i, label in enumerate(labels)
        ]
        (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    body = {
        "questions": {"team": TEAM},
        "answers": {"team": {"column": "label", "map": {v: v for v in TEAM["criteria"]}}},
        "columns": {"label": {"values": {v: {"rows": 1} for v in TEAM["criteria"]}}},
        "splits": {
            split: {"role": role, "examples": len(rows)} for split, (role, rows) in splits.items()
        },
        "source": {"source": "hf:org/help"},
    }
    (folder / "dataset.json").write_text(json.dumps(body))


def test_report_on_the_datasets_report_split(studio, page):
    labels = ["tech", "billing", "sales"]
    write_folder(studio, "help", {
        "train": ("train", labels * 20),
        "test": ("report", labels * 4),
        "eval": ("evaluate", labels * 2),
    })  # fmt: skip
    open_studio(page, studio, "datasets/help")
    page.get_by_role("button", name="Fit on It…").click()
    form = page.locator(".fit-form")
    report = form.locator("#fit-report")
    # The dataset's report split, preselected; its evaluate split is not offered.
    expect(report).to_have_value("split:test")
    options = report.locator("option")
    expect(options).to_have_text(["test · 12 · Report", "A share of its train split", "support",
                                  "tickets"])  # fmt: skip
    expect(form).to_contain_text("its report split")
    expect(form.locator("#fit-holdout")).to_have_count(0)
    legend = form.locator(".split-legend")
    expect(legend).to_contain_text("12 in test for the report")
    expect(form.locator(".split")).to_have_class(re.compile("other-file"))
    expect(form).to_contain_text("its train split: 60 rows")
    shot(page, "fit-report-on")
    expect(form.locator(".fit-labels .line span").first).to_contain_text("20")  # all 60 train
    # Another dataset with other questions: refused, with the reason.
    report.select_option("data:support/support")  # the same question: fine
    expect(legend).to_contain_text("in support for the report")
    report.select_option("data:tickets/tickets")
    expect(form.locator(".warn-text").last).to_contain_text("tickets labels other questions")
    expect(page.locator(".run-btn")).to_be_disabled()
    # Back to the report split, then a share of the train split.
    report.select_option("split:test")
    expect(page.locator(".run-btn")).to_be_enabled()
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".centre h1")).to_have_text("Fit fake-help")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert (job.details["report"], job.details["report_split"]) == ("help", "test")
    assert job.details["holdout"] is None
    assert job.argv[job.argv.index("--data") + 1].endswith("help/train.jsonl")
    assert job.argv[job.argv.index("--validation") + 1].endswith("help/test.jsonl")
    expect(page.locator(".centre .pane-head p")).to_contain_text("reported on help (test)")
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()
    page.locator(".run-btn", has_text="New Fit").click()
    report.select_option("")
    expect(form.locator("#fit-holdout")).to_have_value("20")
    expect(legend).to_contain_text("12 kept aside for the report")


def test_a_dataset_without_questions_is_not_fitted(studio, page):
    folder = studio.workspace / "data" / "raw"
    folder.mkdir()
    (folder / "train.jsonl").write_text(json.dumps({"state": "s", "values": {"label": "1"}}) + "\n")
    body = {
        "columns": {"label": {"values": {"1": {"rows": 1}}}},
        "splits": {"train": {"role": "train"}},
    }
    (folder / "dataset.json").write_text(json.dumps(body))
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    expect(form.locator("#fit-data option", has_text="raw")).to_have_text("raw (not ready)")
    form.locator("#fit-data").select_option("raw")
    warning = form.locator(".warn-text").first
    expect(warning).to_contain_text("raw is not ready for a fit: it has no questions yet")
    expect(warning.get_by_role("button", name="Write Questions")).to_be_visible()
    expect(page.locator(".run-btn")).to_be_disabled()
    expect(form.locator(".method")).to_have_count(0)


def test_report_overlap_is_said(studio, page):
    data = studio.workspace / "data"
    lines = (data / "support.jsonl").read_text().splitlines()
    (data / "copy.jsonl").write_text("\n".join(lines[:5]) + "\n")  # five of the same states
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    form.locator("#fit-data").select_option("support")
    form.locator("#fit-report").select_option("data:copy/copy")
    expect(form.locator(".warn-text", has_text="also in")).to_have_text(
        "5 of 5 examples in copy have a state that is also in support: the report would "
        "score examples the fit learns from"
    )
