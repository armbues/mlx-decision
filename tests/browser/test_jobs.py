"""Jobs in the page: the sidebar's list with progress and Stop, the banner of a
job holding the GPU, and the same jobs after a reload and in a second tab.
The jobs are ``tests/fake_job.py`` runs (``fake_studio.fake_jobs``)."""

import re

import pytest
from drive import open_studio, pick_model, start_job
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

FIT = {"kind": "fit", "model": "fake", "data": "tickets"}


def job_row(page, job):
    return page.locator(f".sidebar li.job[data-job='{job['id']}']")


def test_a_fit_turns_runs_off_shows_its_progress_and_stops(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake")
    run = page.locator(".run-btn")
    expect(run).to_be_enabled()
    expect(page.locator(".sidebar")).to_contain_text("Nothing running")

    job = start_job(page, studio, **FIT)
    row = job_row(page, job)
    expect(row).to_contain_text("Fit fake-tickets")
    expect(row.locator("small")).to_have_text(re.compile(r"^work · \d+%"))
    banner = page.locator(".banner")
    expect(banner).to_contain_text("Fit fake-tickets has the GPU. Run is off until it is done")
    expect(page.locator(".sidebar .serving")).to_contain_text("API busy")
    expect(run).to_be_disabled()
    expect(run).to_have_attribute(
        "title", "Fit fake-tickets has the GPU: Run is off until it is done"
    )
    expect(page.locator(".activity .prog")).to_have_attribute(
        "style", re.compile(r"width: ?(0\.[1-9]|[1-9])")
    )
    # The Models page: the same banner with what it means there, Load off.
    page.goto(f"{studio.base}/#/models")
    expect(banner).to_contain_text("Load and Unload are off until it is done")

    banner.get_by_role("button", name="Stop").click()
    expect(row.locator("small")).to_have_text("Stopped")
    expect(page.locator(".toast")).to_contain_text("Fit fake-tickets stopped")
    expect(banner).to_have_count(0)
    expect(page.locator(".sidebar .serving")).not_to_contain_text("API busy")
    assert not (studio.workspace / "models" / "fake-tickets").exists()
    # An ended job stays listed until it is closed.
    row.get_by_role("button", name="Close Fit fake-tickets").click()
    expect(row).to_have_count(0)
    expect(page.locator(".sidebar")).to_contain_text("Nothing running")
    page.goto(f"{studio.base}/#/playground/ticket-triage")
    expect(run).to_be_enabled()


def test_a_download_runs_alongside_and_ends_done(studio, page):
    fake_jobs(studio.app, steps=20, every=0.1)
    open_studio(page, studio, "playground/ticket-triage")
    download = start_job(page, studio, kind="download", repo="Org/model-x")
    row = job_row(page, download)
    expect(row).to_contain_text("Download Org/model-x")
    expect(page.locator(".banner")).to_have_count(0)  # no GPU: nothing is off
    expect(page.locator(".run-btn")).to_be_enabled()
    expect(row.locator("small")).to_have_text("Done", timeout=10_000)
    expect(page.locator(".toast")).to_contain_text("Download Org/model-x is done")
    expect(row.locator(".bar")).to_have_count(0)


def test_stop_from_the_list_and_a_job_that_fails(studio, page):
    open_studio(page, studio)
    job = start_job(page, studio, kind="convert", model="fake")
    row = job_row(page, job)
    expect(row.locator("small")).to_have_text(re.compile(r"\d+%"))
    row.get_by_role("button", name="Stop Convert fake-q8").click()
    expect(row.locator("small")).to_have_text("Stopped")

    # A process that ends without a last event fails, with its stderr's end.
    studio.app.state.jobs.command[2:] = ["--steps", "3", "--end", "crash"]
    failed = start_job(page, studio, kind="convert", model="fake", name="again")
    row = job_row(page, failed)
    expect(row.locator("small")).to_have_text("Failed · RuntimeError: it broke")
    expect(row.locator("small")).to_have_attribute("title", re.compile("RuntimeError: it broke"))
    expect(page.locator(".toast").last).to_contain_text(
        "Convert again failed: RuntimeError: it broke"
    )


def test_a_reload_and_a_second_tab_show_the_job_so_far(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    job = start_job(page, studio, **FIT)
    row = job_row(page, job)
    expect(row.locator("small")).to_have_text(re.compile(r"[1-9]\d*%"))

    page.reload()
    page.wait_for_selector(".app")
    # Straight from the snapshot: the job, its latest progress, the banner.
    expect(row.locator("small")).to_have_text(re.compile(r"^work · [1-9]\d*%"))
    expect(page.locator(".banner")).to_contain_text("Fit fake-tickets has the GPU")
    expect(page.locator(".run-btn")).to_be_disabled()

    other = page.context.new_page()
    other.goto(f"{studio.base}/#/models")
    other_row = job_row(other, job)
    expect(other_row.locator("small")).to_have_text(re.compile(r"^work · [1-9]\d*%"))
    expect(other.locator(".banner")).to_contain_text("Load and Unload are off")
    # Stop in one tab ends it in both.
    other_row.get_by_role("button", name="Stop Fit fake-tickets").click()
    expect(row.locator("small")).to_have_text("Stopped")
    expect(other_row.locator("small")).to_have_text("Stopped")
    expect(page.locator(".banner")).to_have_count(0)
    expect(page.locator(".run-btn")).to_be_enabled()
    other.close()


def test_progress_lines(studio, page):
    """What the list says per kind and state (``progress`` in the page's store)."""
    open_studio(page, studio)
    lines = page.evaluate("""async () => {
      const { progress } = await import("app/store/jobs.js");
      const running = { state: "running", holds_gpu: true, stop_requested: false };
      const job = (fields) => ({ ...running, ...fields });
      return [
        job({ state: "waiting" }),
        job({ state: "waiting", holds_gpu: false }),
        job({ last: { stage: "check", message: "loading fake" } }),
        job({ last: { stage: "download", step: 5.3e9, total: 8.5e9, eta_s: 41.6 } }),
        job({ last: { stage: "before", step: 30, total: 200, eta_s: 150 } }),
        job({ last: { stage: "train", step: 3, total: 9, metrics: { loss: 0.4 } } }),
        job({ stop_requested: true, last: { stage: "train", step: 3, total: 9 } }),
        job({ state: "ended", outcome: "done" }),
        job({ state: "ended", outcome: "cancelled", message: "nothing was written" }),
        job({ state: "ended", outcome: "error", message: "the data has no labels" }),
        job({ state: "ended", outcome: "failed", message: "Traceback\\nValueError: no\\n" }),
        job({ state: "ended", outcome: "interrupted" }),
      ].map((item) => progress(item).text);
    }""")
    assert lines == [
        "Waiting for the GPU",
        "Starting",
        "Checking",
        "5.3 of 8.5 GB · 62% · 42 s left",
        "Scoring before · 15% · 3 min left",
        "Training · 33%",
        "Stopping…",
        "Done",
        "Stopped",
        "Failed · the data has no labels",
        "Failed · ValueError: no",
        "Interrupted",
    ]
