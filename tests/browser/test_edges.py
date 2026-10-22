"""Edge cases the page gets right: lists read at the start and on focus, a
fit opened on what it learned from, no models yet, the split a batch reads,
a set renamed or a dataset deleted in another tab, a model over the budget
or gone, long names, a dataset's new questions."""

import json
import re
import shutil

import pytest
from drive import inspector_tab, menu_item, open_studio, pick, pick_model, put_set, start_job
from fake_studio import fake_jobs
from test_datasets import write_tweets

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def write_ready(studio):
    """data/tweets/ with a topic question whose answers map every value."""
    write_tweets(studio, questions=True)
    path = studio.workspace / "data" / "tweets" / "dataset.json"
    body = json.loads(path.read_text())
    body["answers"]["topic"]["map"] = {"0": "analyst", "1": "fed"}
    path.write_text(json.dumps(body))


def test_training_lists_earlier_fits_when_it_opens(studio, page):
    # A fit of an earlier run of the Studio, only on disk.
    folder = studio.workspace / "jobs" / "20261020-101010-fit-abc123"
    folder.mkdir(parents=True)
    summary = {
        "id": folder.name, "kind": "fit", "title": "Fit old", "state": "ended",
        "outcome": "done", "created": "2026-10-20T10:10:10Z", "details": {"name": "old"},
        "command": ["mlx-decision", "training", "fit"],
    }  # fmt: skip
    (folder / "job.json").write_text(json.dumps(summary))
    open_studio(page, studio, "training")
    expect(page.locator(".sidebar .fit-list li")).to_have_count(1)
    expect(page.locator(".sidebar .fit-list")).to_contain_text("old")


def test_a_fit_opens_on_its_dataset_beside_its_base(studio, page):
    write_ready(studio)
    fake_jobs(studio.app, steps=3, every=0.02, fits=True)
    open_studio(page, studio, "training")
    job = start_job(page, studio, kind="fit", model="fake", data="tweets", name="tw")
    page.goto(f"{studio.base}/#/training/{job['id']}")
    button = page.get_by_role("link", name="Compare with fake in the Playground")
    expect(button).not_to_have_attribute("aria-disabled", "true", timeout=15_000)
    button.click()
    expect(page).to_have_url(re.compile(r"#/playground/data(:|%3A)tweets$"))
    expect(page.locator(".target .mdl .name")).to_have_text("tw")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 1")  # fake, compared


def test_with_no_models_each_picker_points_at_download(make_studio, page):
    studio = make_studio(empty=True)
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    expect(form.locator(".no-models")).to_contain_text("No models yet")
    expect(form.get_by_role("button", name="Download a Model…")).to_be_visible()
    expect(page.locator(".run-btn")).to_have_attribute("title", "Download a model first")
    page.goto(f"{studio.base}/#/playground")
    expect(page.locator(".insp-body .empty")).to_contain_text("No models yet")


def test_run_as_batch_reads_the_evaluate_split_unless_a_tab_is_picked(studio, page):
    write_tweets(studio, questions=True)
    open_studio(page, studio, "datasets/tweets")
    page.get_by_role("button", name="More actions for tweets").click()
    menu_item(page, "Run as Batch…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("#batch-split")).to_have_value("test")  # the evaluate split
    expect(sheet.locator(".split-hint")).to_have_count(0)
    sheet.locator("#batch-split").select_option("train")
    expect(sheet.locator(".split-hint")).to_contain_text("A fit learns from these rows")


def test_a_set_renamed_in_another_tab_moves_here_with_its_edits(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    state = page.locator(".state-text")
    state.fill("Edited here, not saved")
    other = page.context.new_page()
    response = other.request.post(
        f"{studio.base}/studio/api/sets/ticket-triage/rename", data={"name": "triage"}
    )
    assert response.ok, response.text()
    expect(page).to_have_url(re.compile(r"#/playground/triage$"))
    expect(page.locator(".toast")).to_contain_text("Renamed to triage in another tab")
    expect(state).to_have_value("Edited here, not saved")
    expect(page.locator(".disk-banner")).to_have_count(0)
    other.close()


def test_a_dataset_deleted_while_its_questions_are_edited(studio, page):
    write_tweets(studio, questions=True)
    open_studio(page, studio, "playground/data:tweets")
    pick(page, ".add-bar .btn", "Noul")
    response = page.request.delete(f"{studio.base}/studio/api/data/tweets", data={})
    assert response.status == 204, response.text()
    banner = page.locator(".disk-banner")
    expect(banner).to_contain_text("data/tweets/ was deleted. Your questions are only here.")
    expect(banner.get_by_role("button", name="Save as Question Set…")).to_be_visible()


def test_a_model_over_the_budget_cannot_load_and_says_why(make_studio, page):
    studio = make_studio(sizes=True)
    open_studio(page, studio, "models/fake-big")
    load = page.locator(".idet-actions").get_by_role("button", name="Load")
    expect(load).to_be_disabled()
    expect(load).to_have_attribute("title", re.compile(r"fake-big needs about 19\.0 GB, more than"))
    page.goto(f"{studio.base}/#/playground/ticket-triage")
    pick_model(page, "fake-big")
    expect(page.locator(".run-btn")).to_be_disabled()
    expect(page.locator(".toolbar")).to_contain_text("more than the 8.0 GB budget")


def test_a_picked_model_that_goes_is_named(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake-strict")
    shutil.rmtree(studio.home / "fake-models" / "fake-strict")
    response = page.request.post(f"{studio.base}/studio/api/models/rescan", data={})
    assert response.json()["removed"] == ["fake-strict"]
    expect(page.locator(".toolbar")).to_contain_text("fake-strict is gone · Run uses")


def test_a_batch_deleted_on_disk_goes_from_the_list_on_focus(studio, page):
    open_studio(page, studio, "batches")
    folder = studio.workspace / "batches" / "made-by-hand"
    page.request.post(f"{studio.base}/studio/api/batches", data={})  # nothing: no dataset named
    write_tweets(studio, questions=True)
    started = page.request.post(
        f"{studio.base}/studio/api/batches",
        data={"dataset": "tweets", "split": "test", "model": "fake", "name": "made-by-hand"},
    )
    assert started.status == 201, started.text()
    entries = page.locator(".sidebar .batch-list li")
    expect(entries).to_have_count(1)
    expect(entries.locator(".sb-delete")).to_have_count(1, timeout=10_000)  # done
    shutil.rmtree(folder)
    page.evaluate("window.dispatchEvent(new Event('focus'))")
    expect(entries).to_have_count(0)


def test_long_names_are_cut_in_the_sidebar(studio, page):
    open_studio(page, studio, "playground")
    name = "a-question-set-with-a-very-long-name-that-goes-on-and-on"
    put_set(page, studio, name, {"state": "", "questions": {"q": {"type": "noul"}}})
    page.goto(f"{studio.base}/#/playground/{name}")
    row = page.locator(".sidebar .sb-list li", has=page.locator(f"[href='#/playground/{name}']"))
    row.hover()
    sidebar = page.locator(".sidebar").bounding_box()
    delete = row.locator(".sb-delete").bounding_box()
    assert delete["x"] + delete["width"] <= sidebar["x"] + sidebar["width"]
    label = row.locator(".label")
    assert label.evaluate("el => el.scrollWidth > el.clientWidth")  # cut with "…"


def test_a_new_question_on_a_dataset_is_answered_by_its_values(studio, page):
    write_tweets(studio)  # no questions yet
    open_studio(page, studio, "playground/data:tweets")
    pick(page, ".add-bar .btn", "Choice")
    expect(page.get_by_label("Answered by, choice")).to_have_value("label")
    options = page.locator(".question[data-q='choice'] [data-item] [data-cell='description']")
    expect(options).to_have_count(2)  # one per value: 0 and 1, named from the card


def test_clearing_a_datasets_history_names_the_dataset(studio, page):
    write_tweets(studio, questions=True)
    open_studio(page, studio, "playground/data:tweets")
    pick_model(page, "fake")
    page.keyboard.press("Meta+Enter")  # Run All
    inspector_tab(page, "History")
    page.locator(".hist-foot .btn").click(timeout=15_000)
    page.locator(".sheet-foot .btn", has_text=re.compile("^Clear History$")).click()
    expect(page.locator(".toast").last).to_have_text("Cleared the history of tweets")
