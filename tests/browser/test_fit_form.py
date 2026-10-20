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
    expect(page.locator(".sidebar .fit-list .sb-item")).to_have_text("New Fit")
    expect(page.locator(".sidebar .sb-empty", has_text="fits")).to_have_text("No fits yet")
    expect(page.locator(".inspector .data-inspector h2")).to_have_text("support")
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
    expect(run).to_have_attribute("title", re.compile("Fit support-calibrated is running"))
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


def write_get(studio, files):
    """Data files written as one Get with records: ``files`` maps a name to
    (role, rows), each row a label."""
    data = studio.workspace / "data"
    listed = [{"name": name, "split": name.split("-")[-1], "role": role}
              for name, (role, _) in files.items()]  # fmt: skip
    for name, (role, labels) in files.items():
        lines = [
            json.dumps(
                {"state": f"{name} {i}", "questions": {"team": TEAM}, "labels": {"team": label}}
            )  # fmt: skip
            for i, label in enumerate(labels)
        ]
        (data / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
        record = {"source": "hf:org/support", "split": name.split("-")[-1], "rows": len(labels),
                  "role": role, "get": {"name": "support", "files": listed}}  # fmt: skip
        (data / f"{name}.source.json").write_text(json.dumps(record))


def test_report_on_the_same_gets_report_file(studio, page):
    labels = ["tech", "billing", "sales"]
    write_get(studio, {
        "help-train": ("train", labels * 20),
        "help-test": ("report", labels * 4),
        "help-eval": ("evaluate", labels * 2),
    })  # fmt: skip
    open_studio(page, studio, "datasets/help-train")
    page.get_by_role("button", name="Fit on It…").click()
    form = page.locator(".fit-form")
    report = form.locator("#fit-report")
    # The Get's report split, preselected; the file kept for Evaluate is not offered.
    expect(report).to_have_value("help-test")
    options = report.locator("option")
    expect(options).to_have_text(["A share of this dataset", "help-test (report)", "support",
                                  "tickets"])  # fmt: skip
    expect(form).to_contain_text("got with it as its report split")
    expect(form.locator("#fit-holdout")).to_have_count(0)
    legend = form.locator(".split-legend")
    expect(legend).to_contain_text("12 in help-test for the report")
    expect(form.locator(".split")).to_have_class(re.compile("other-file"))
    shot(page, "fit-report-on")
    expect(form.locator(".fit-labels .line span").first).to_contain_text("20")  # all 60 train
    # A file with other questions: refused, with the reason.
    report.select_option("support")  # the same question: fine
    expect(legend).to_contain_text("in support for the report")
    report.select_option("tickets")
    expect(form.locator(".warn-text").last).to_contain_text("tickets labels other questions")
    expect(page.locator(".run-btn")).to_be_disabled()
    # Back to the Get's file, then a share of the dataset.
    report.select_option("help-test")
    expect(page.locator(".run-btn")).to_be_enabled()
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".centre h1")).to_have_text("Fit fake-help-train")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["report"] == "help-test" and job.details["holdout"] is None
    assert job.argv[job.argv.index("--validation") + 1].endswith("help-test.jsonl")
    expect(page.locator(".centre .pane-head p")).to_contain_text("reported on help-test")
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()
    page.locator(".run-btn", has_text="New Fit").click()
    report.select_option("")
    expect(form.locator("#fit-holdout")).to_have_value("20")
    expect(legend).to_contain_text("12 kept aside for the report")


def test_report_overlap_is_said(studio, page):
    data = studio.workspace / "data"
    lines = (data / "support.jsonl").read_text().splitlines()
    (data / "copy.jsonl").write_text("\n".join(lines[:5]) + "\n")  # five of the same states
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    form.locator("#fit-data").select_option("support")
    form.locator("#fit-report").select_option("copy")
    expect(form.locator(".warn-text", has_text="also in")).to_have_text(
        "5 of 5 examples in copy have a state that is also in support: the report would "
        "score examples the fit learns from"
    )
