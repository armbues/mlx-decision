"""A newcomer's next steps: Run as Batch… first on a dataset, finished jobs
as links, a dataset ready after its questions are saved, a fit's report
leading to its evaluation, Settings' folders and where models live, the
Models list's units."""

import re

import pytest
from drive import open_studio, start_job
from fake_studio import fake_jobs
from test_datasets import write_tweets
from test_settings_page import _eventually

from mlx_decision.studio.routes import settings as settings_route

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_a_dataset_leads_with_run_as_batch(studio, page):
    open_studio(page, studio, "datasets/tickets")
    buttons = page.locator(".centre .pane-head .actions > .btn")
    expect(buttons.first).to_have_text("Run as Batch…")
    expect(buttons.first).to_have_class(re.compile(r"\bprimary\b"))
    page.get_by_role("button", name="More actions for tickets").click()
    expect(page.locator(".menu")).to_contain_text("Evaluate on It…")


def test_finished_jobs_link_to_what_they_made(studio, page):
    fake_jobs(studio.app, steps=3, every=0.02, fits=True)
    open_studio(page, studio, "playground")
    fit = start_job(page, studio, kind="fit", model="fake", data="tickets", name="mine")
    row = page.locator(f".sidebar li.job[data-job='{fit['id']}']")
    expect(row.locator("a.job-title")).to_have_attribute("href", f"#/training/{fit['id']}")
    expect(row.locator("small")).to_have_text("Done", timeout=15_000)
    data = start_job(page, studio, kind="data", preset="trec", name="trec",
                     splits=[{"split": "train", "role": "train"}])  # fmt: skip
    row = page.locator(f".sidebar li.job[data-job='{data['id']}']")
    expect(row.locator("a.job-title")).to_have_attribute("href", "#/datasets/trec", timeout=15_000)


def test_a_dataset_says_it_is_ready_once_its_questions_are_saved(studio, page):
    write_tweets(studio, questions=True)  # value 1 not mapped yet
    open_studio(page, studio, "playground/data:tweets")
    banner = page.locator(".ds-banner")
    expect(banner).to_contain_text("Not ready for a fit")
    page.get_by_label("Answer for 1", exact=True).select_option("out")
    # The save changes the folder's modified time, so the page reads the
    # dataset list again and then the document: the banner outlasts that.
    document = re.compile(r"/api/data/tweets/document(\?|$)")
    with page.expect_response(lambda response: document.search(response.url)):
        page.keyboard.press("Meta+s")
    expect(page.locator(".toast").last).to_contain_text("Saved: tweets is ready for a fit")
    expect(banner).to_contain_text("Ready for a fit and a batch.")
    banner.get_by_role("button", name="Run as Batch…").click()
    expect(page.locator(".sheet h2")).to_have_text("Run as Batch")
    expect(page.locator(".sheet #batch-data")).to_have_value("tweets")


def test_a_fit_report_leads_to_its_evaluation(studio, page):
    fake_jobs(studio.app, steps=3, every=0.02, fits=True)
    open_studio(page, studio, "training")
    job = start_job(page, studio, kind="fit", model="fake", data="tickets", name="mine")
    page.goto(f"{studio.base}/#/training/{job['id']}")
    evaluate = page.get_by_role("button", name="Evaluate Against fake…")
    expect(evaluate).to_be_visible(timeout=15_000)
    evaluate.click()
    expect(page).to_have_url(re.compile(r"#/training/evaluate$"))
    form = page.locator(".eval-form")
    expect(form.locator("#eval-model")).to_have_value("mine")
    expect(form.locator("#eval-compare")).to_have_value("fake")
    expect(form.locator("#eval-data")).to_have_value("tickets")


def test_settings_show_each_folder_and_where_models_live(studio, page, monkeypatch):
    shown = []
    monkeypatch.setattr(settings_route, "reveal", shown.append)  # no Finder in a test
    open_studio(page, studio, "settings")
    folder = page.locator(".kv.folder[data-folder='batches']")
    folder.get_by_role("button", name="Show in Finder").click()
    expect(page.locator(".toast")).to_have_count(0)
    assert _eventually(lambda: shown and shown[-1] == studio.workspace.resolve() / "batches")
    row = page.locator(".kv.setting[data-key='models_dir']")
    expect(row).to_contain_text("the Hugging Face cache")
    row.get_by_role("link", name="Where models live").click()
    expect(page).to_have_url(re.compile(r"#/models$"))


def test_the_models_list_says_what_reads_counts(studio, page):
    open_studio(page, studio, "models")
    header = page.locator(".mhead .c-reads")
    expect(header).to_have_attribute("title", re.compile("in tokens"))
    expect(page.locator(".mrow .c-reads").first).to_contain_text("tokens")
