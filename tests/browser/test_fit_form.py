"""The fit form: data, split and labels, the base, the method cards with their
estimates, the result's name and options, and Start Fit (⌘↩) as a job."""

import json
import re

import pytest
from drive import open_studio

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
    expect(page.locator(".centre .empty")).to_contain_text("Pick training data")
    expect(page.locator(".run-btn")).to_be_disabled()
    page.locator(".sidebar .data-list .sb-item", has_text="support").click()
    form = page.locator(".fit-form")
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
    expect(form.locator(".fit-labels")).to_contain_text("tech")
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
    expect(page.locator(".toast")).to_contain_text("Fit support-calibrated started")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["method"] == "calibrate" and job.details["holdout"] == 0.25
    assert job.argv[job.argv.index("--method") + 1] == "calibrate"
    # One GPU job at a time.
    expect(run).to_have_attribute("title", re.compile("Fit support-calibrated is running"))
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()
    expect(page.locator(".sidebar .job small")).to_have_text("Stopped")


def test_a_family_without_a_main_kind_offers_calibrate(studio, page):
    open_studio(page, studio, "training/support")
    form = page.locator(".fit-form")
    form.locator("#fit-base").select_option("fake")
    expect(form.locator(".method .mt b")).to_have_text(["Calibrate"])
    expect(form.locator(".method").first).to_have_attribute("aria-checked", "true")
    expect(form.locator("#fit-name")).to_have_value("fake-support")
