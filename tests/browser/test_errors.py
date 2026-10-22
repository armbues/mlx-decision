"""Failures and how they reach the user: a stopped and a restarted Studio,
files that do not read, refusals inside their sheet, failure toasts, New
Batch without datasets."""

import json
import re
import shutil

import pytest
from drive import menu_item, open_studio, start_job
from fake_studio import fake_jobs, serve

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def status_field(page):
    return page.locator(".toolbar .activity")


def test_a_stopped_then_restarted_studio(tmp_path, page):
    home = tmp_path / "studio"
    first = serve(home)
    studio = first.__enter__()
    port = int(studio.base.rsplit(":", 1)[1])
    open_studio(page, studio, "playground/ticket-triage")
    page.locator(".state-text").fill("Edited, not saved")
    first.__exit__(None, None, None)

    # Stopped: said within seconds, and an action fails in a toast that stays.
    expect(status_field(page)).to_contain_text("The Studio is not answering", timeout=10_000)
    expect(page.locator(".sidebar .serving")).to_contain_text("Not connected")
    page.keyboard.press("Meta+s")
    failure = page.locator(".toast.error")
    expect(failure).to_contain_text("Not saved: the Studio is not answering")
    failure.get_by_role("button", name="Close").click()
    expect(failure).to_have_count(0)

    # Started again: a new session, which this tab does not have yet.
    second = serve(home, port=port)
    again = second.__enter__()
    try:
        expect(status_field(page)).to_contain_text("The Studio restarted", timeout=15_000)
        expect(page.locator(".state-text")).to_have_value("Edited, not saved")
        # Another tab opens the new address: this one reconnects by itself.
        other = page.context.new_page()
        other.goto(again.url)
        other.wait_for_selector(".app")
        expect(status_field(page)).not_to_contain_text("restarted", timeout=15_000)
        expect(page.locator(".sidebar .serving")).to_contain_text("API at")
        page.keyboard.press("Meta+s")
        expect(page.locator(".toast").last).to_contain_text("Saved")
        assert json.loads((home / "ws" / "sets" / "ticket-triage.json").read_text())["state"] == (
            "Edited, not saved"
        )
        other.close()
    finally:
        second.__exit__(None, None, None)


def test_a_set_that_does_not_read_can_be_shown_or_deleted(studio, page):
    (studio.sets / "broken.json").write_text('{"state": "x", oops}')
    open_studio(page, studio, "playground/broken")
    empty = page.locator(".centre .empty")
    expect(empty).to_contain_text("broken.json is not valid JSON")
    expect(empty.get_by_role("button", name="Show in Finder")).to_be_visible()
    empty.get_by_role("button", name="Delete…").click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page.locator(".sheet")).to_have_count(0)
    assert not (studio.sets / "broken.json").exists()


def test_a_dataset_line_that_does_not_read_names_its_file(studio, page):
    folder = studio.workspace / "data" / "bad"
    folder.mkdir()
    (folder / "train.jsonl").write_text(
        '{"id": "a", "state": "x", "values": {"label": "0"}}\nnope\n'
    )
    body = {"columns": {"label": {"values": {"0": {"rows": 1}}}},
            "splits": {"train": {"role": "train", "examples": 2}}}  # fmt: skip
    (folder / "dataset.json").write_text(json.dumps(body))
    open_studio(page, studio, "datasets/bad")
    centre = page.locator(".centre")
    expect(centre.locator(".warn-text")).to_contain_text("train.jsonl line 2")
    expect(centre.get_by_role("button", name="Show in Finder")).to_be_visible()
    expect(centre.get_by_role("button", name="Delete…")).to_be_visible()


def test_a_dataset_a_fit_reads_cannot_be_deleted_or_cleared(studio, page):
    fake_jobs(studio.app, steps=400, every=0.05)
    open_studio(page, studio, "datasets/tickets")
    job = start_job(page, studio, kind="fit", model="fake", data="tickets")
    page.get_by_role("button", name="More actions for tickets").click()
    menu_item(page, "Delete…").click()
    sheet = page.locator(".sheet")
    expect(sheet).to_contain_text(f"{job['title']} reads it: stop it or wait until it ends")
    expect(sheet.get_by_role("button", name="Delete")).to_be_disabled()
    page.keyboard.press("Escape")
    page.goto(f"{studio.base}/#/settings")
    clear = page.locator(".kv.folder[data-folder='data']").get_by_role("button", name="Clear…")
    expect(clear).to_be_disabled()
    expect(clear).to_have_attribute("title", re.compile(f"^{job['title']} reads a dataset"))
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()


def test_a_refused_delete_says_why_in_its_sheet(studio, page):
    open_studio(page, studio, "playground/order-json")
    refusal = {"error": {"message": "the disk said no", "type": "invalid_request_error"}}
    page.route(
        "**/studio/api/sets/order-json",
        lambda route: (
            route.fulfill(status=409, json=refusal)
            if route.request.method == "DELETE"
            else route.continue_()
        ),
    )
    page.locator("li.open").hover()
    page.locator("li.open .sb-delete").click()
    sheet = page.locator(".sheet")
    sheet.get_by_role("button", name="Delete").click()
    expect(sheet.locator(".sheet-error")).to_have_text("Not deleted: the disk said no")
    expect(sheet).to_be_visible()  # it stays open, the reason in it
    assert (studio.sets / "order-json.json").exists()


def test_new_batch_without_datasets_offers_to_get_one(studio, page):
    shutil.rmtree(studio.workspace / "data")
    (studio.workspace / "data").mkdir()
    open_studio(page, studio, "batches")
    page.locator(".sidebar").get_by_role("button", name="New Batch…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator(".no-datasets")).to_contain_text("A batch answers a dataset's rows.")
    expect(sheet.get_by_role("button", name="Download a Dataset…")).to_be_visible()
    expect(sheet.get_by_role("button", name="Add a Dataset…")).to_be_visible()
