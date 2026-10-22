"""A fit in the Training page: Start Fit opens its progress (the stage
stepper, the step line, the loss chart, the log in the CLI's words), the
sidebar lists it (running on top), and once done its report (before and
after, the kept epoch, where it was saved, the command, Open in
Playground); a stopped fit says so. The fit is ``tests/fake_job.py`` with
``--fit-events``: a fit's events, then a fitted model over the base."""

import json
import re

import pytest
from drive import open_studio, shot
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip


@pytest.fixture
def studio(make_studio):
    studio = make_studio(convertible=True)  # the tiny Clef: head (epochs, a loss), then calibrate
    labels = ["tech"] * 30 + ["billing"] * 12 + ["sales"] * 18
    lines = [
        json.dumps({"state": f"ticket {i}", "questions": {"team": TEAM}, "labels": {"team": label}})
        for i, label in enumerate(labels)
    ]
    (studio.workspace / "data" / "support.jsonl").write_text("\n".join(lines) + "\n")
    return studio


def start_fit(page, studio, name="support-head"):
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    form.locator("#fit-data").select_option("support")
    form.locator("#fit-base").select_option("tiny-clef")
    expect(form.locator(".method[data-method='head']")).to_have_attribute("aria-checked", "true")
    form.locator("#fit-name").fill(name)
    expect(page.locator(".run-btn")).to_be_enabled()
    page.locator(".run-btn").click()
    expect(page).to_have_url(re.compile(r"#/training/\d{8}-\d{6}-fit-[0-9a-f]{6}$"))


def test_a_fit_from_start_to_report(studio, page):
    fake_jobs(studio.app, steps=60, every=0.06, fits=True)  # about 8 s
    start_fit(page, studio)
    head = page.locator(".centre .pane-head")
    expect(head.locator("h1")).to_have_text("Fit support-head")
    expect(head.locator("p")).to_contain_text("tiny-clef · on support · head")
    # Running: the stepper, the step line, the loss chart, the log.
    stepper = page.locator(".stepper li")
    expect(stepper.locator("span:not(.dot)")).to_have_text(
        ["Check", "Score before", "Train", "Calibrate", "Score after", "Write"]
    )
    expect(page.locator(".stepper li.now")).to_have_text("Train", timeout=10_000)
    expect(page.locator(".stepper li.done")).to_have_count(2)
    expect(page.locator(".step-line")).to_contain_text(
        re.compile(r"Train · \d+ of 60 · \d+% · \d+ s left")
    )
    expect(page.locator(".loss-chart .loss-line")).to_have_count(1)
    log = page.locator(".fit-log")
    expect(log).to_contain_text("check: reading the data")
    expect(log).to_contain_text("data: 60 examples: 12 holdout, 43 fit, 5 dev")
    expect(log).to_contain_text("method: head, then calibrate")
    row = page.locator(".sidebar .fit-list .fit-row").first
    expect(row.locator(".label")).to_have_text("support-head")
    expect(row.locator(".count")).to_have_text(re.compile(r"\d+%"))
    expect(page.locator(".toolbar .activity")).to_contain_text("Fit support-head has the GPU")
    shot(page, "fit-running")

    # Done: the report.
    report = page.locator(".report-table")
    expect(report).to_be_visible(timeout=20_000)
    accuracy = report.locator("tr", has_text="Accuracy")
    expect(accuracy.locator("td")).to_have_text(["50.0%", "83.3%", re.compile(r"\+33\.3 pts")])
    expect(accuracy.locator(".change")).to_have_class(re.compile("better"))
    ece = report.locator("tr", has_text="Calibration error")
    expect(ece.locator(".change")).to_have_class(re.compile("better"))  # lower is better
    expect(page.locator(".q-report h3")).to_have_text("team · 12 rows kept aside")
    expect(page.locator(".report-note")).to_have_text("Kept epoch 2 of 3.")
    expect(page.locator(".epochs tr.kept td").first).to_have_text("2")
    expect(page.locator(".loss-chart .kept")).to_have_text("kept")
    saved = page.locator(".kv", has_text="Saved in")
    expect(saved).to_contain_text(str((studio.workspace / "models" / "support-head").resolve()))
    command = page.locator(".kv", has_text="Same in a terminal").locator("code")
    expect(command).to_contain_text("mlx-decision training fit -m")
    expect(command).to_contain_text("--method head")
    expect(page.locator(".stepper li.done")).to_have_count(6)
    expect(page.locator(".log-sec")).not_to_have_attribute("open", "")  # folded once done
    expect(row.locator(".ic.ok")).to_be_visible()
    shot(page, "fit-report")
    chart = page.locator(".loss-chart svg")
    chart.scroll_into_view_if_needed()
    box = chart.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)
    expect(page.locator(".chart-tip")).to_contain_text("loss at step")
    shot(page, "fit-loss")
    page.emulate_media(color_scheme="dark")
    shot(page, "fit-loss-dark")
    page.emulate_media(color_scheme="light")

    # The fitted model is in the list: Open in Playground asks it there.
    open_playground = page.get_by_role("link", name="Open in Playground")
    expect(open_playground).not_to_have_attribute("aria-disabled", "true", timeout=10_000)
    open_playground.click()
    expect(page).to_have_url(re.compile(r"#/playground"))
    expect(page.locator(".target .mdl")).to_contain_text("support-head")

    # A reload, or another tab: the fit is listed and opens on its report.
    page.goto(f"{studio.base}/#/training")
    page.locator(".sidebar .fit-row .sb-item").first.click()
    expect(page.locator(".report-table")).to_be_visible()


def test_a_stopped_fit_says_so(studio, page):
    fake_jobs(studio.app, steps=400, every=0.05, fits=True)
    start_fit(page, studio, "support-stop")
    expect(page.locator(".stepper li.now")).to_have_text("Train", timeout=10_000)
    expect(page.locator(".centre .pane-head button")).to_have_count(0)  # the banner has Stop
    page.locator(".banner").get_by_role("button", name="Stop").click()
    expect(page.locator(".run-outcome")).to_have_text(
        "Stopped: nothing was written", timeout=10_000
    )
    expect(page.locator(".stepper li.stopped")).to_have_text("Train")
    expect(page.locator(".report-table")).to_have_count(0)
    expect(page.locator(".log-sec")).to_have_attribute("open", "")
    row = page.locator(".sidebar .fit-row").first
    expect(row.locator(".count")).to_have_text("Stopped")
    assert not (studio.workspace / "models" / "support-stop").exists()
    # New Fit goes back to the form.
    page.locator(".run-btn").click()
    expect(page.locator(".fit-form")).to_be_visible()
