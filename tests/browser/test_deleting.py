"""Deleting from the lists: a dataset that is not picked, the Studio's own
models on the Models page, and fits and evaluations in Training's History
(a fit's model too, when asked)."""

import json
import re

import pytest
from drive import menu_item, open_studio, start_job
from fake_studio import fake_jobs

from fake_backend import write_fitted, write_model

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_a_dataset_that_is_not_picked(studio, page):
    line = (studio.workspace / "data" / "tickets.jsonl").read_text()
    (studio.workspace / "data" / "more.jsonl").write_text(line)
    open_studio(page, studio, "datasets/tickets")
    row = page.locator(".sidebar .data-list li", has=page.locator("[href='#/datasets/more']"))
    row.hover()
    row.get_by_role("button", name="Delete more…").click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page.locator(".sidebar .data-list .sb-item")).to_have_count(1)
    expect(page).to_have_url(re.compile(r"#/datasets/tickets$"))
    assert not (studio.workspace / "data" / "more.jsonl").exists()


@pytest.fixture
def made(studio, page):
    """The Studio with a converted model in its models folder (``conv``) and a
    fitted one on it (``conv-fit``), found by a rescan."""
    folder = studio.workspace / "models"
    write_model(folder / "conv")
    write_fitted(folder / "conv-fit", folder / "conv")
    open_studio(page, studio, "models")
    response = page.request.post(f"{studio.base}/studio/api/models/rescan", data={})
    assert sorted(response.json()["added"]) == ["conv", "conv-fit"]
    return studio


def test_the_studios_own_models_and_not_downloaded_ones(made, page):
    folder = made.workspace / "models"
    page.goto(f"{made.base}/#/models/fake")
    expect(page.locator(".idet h2")).to_have_text("fake")
    expect(page.locator(".idet-foot")).to_have_count(0)  # a folder named with -m

    page.goto(f"{made.base}/#/models/conv")
    delete = page.locator(".idet-foot").get_by_role("button", name="Delete…")
    expect(delete).to_be_disabled()
    expect(delete).to_have_attribute(
        "title", re.compile(r"^Fitted models are made from it \(conv-fit\)")
    )

    # The fitted one from the row's context menu.
    page.locator(".mrow", has_text="conv-fit").first.click(button="right")
    menu_item(page, "Delete…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Delete the model conv-fit?")
    expect(sheet).to_contain_text(str(folder / "conv-fit"))
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast")).to_contain_text("Deleted the model conv-fit")
    expect(page.locator(".mrow", has_text="conv-fit")).to_have_count(0)
    assert not (folder / "conv-fit").exists()

    # Now its base can go, from the inspector.
    expect(delete).to_be_enabled()
    delete.click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page).to_have_url(re.compile(r"#/models$"))
    expect(page.locator(".mrow", has_text="conv")).to_have_count(0)
    assert not (folder / "conv").exists()


def fit_entry(page, title):
    return page.locator(".sidebar .fit-list li", has=page.locator(f"[title='{title}']"))


def test_a_fit_with_its_model_and_an_evaluation(studio, page):
    fake_jobs(studio.app, steps=3, every=0.02, fits=True)
    open_studio(page, studio, "training")
    entries = page.locator(".sidebar .fit-list li")
    fit = start_job(page, studio, kind="fit", model="fake", data="tickets", name="mine")
    expect(entries.locator(".sb-delete")).to_have_count(1, timeout=15_000)  # ended
    evaluation = start_job(page, studio, kind="eval", model="fake", data="tickets")
    expect(entries).to_have_count(2)
    expect(entries.locator(".sb-delete")).to_have_count(2, timeout=15_000)
    expect(page.locator(".sidebar .sb-item", has_text="mine")).to_be_visible()
    model = studio.workspace / "models" / "mine"

    # The evaluation: only its entry, no model to offer.
    row = fit_entry(page, evaluation["title"])
    row.hover()
    row.locator(".sb-delete").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text(
        "Delete the evaluation of " + evaluation["title"].removeprefix("Evaluate ") + "?"
    )
    expect(sheet.locator("input[type=checkbox]")).to_have_count(0)
    sheet.get_by_role("button", name="Delete").click()
    expect(entries).to_have_count(1)

    # The fit, opened, with its model.
    page.goto(f"{studio.base}/#/training/{fit['id']}")
    expect(page.locator(".report-table")).to_be_visible()
    row = fit_entry(page, fit["title"])
    row.hover()
    row.locator(".sb-delete").click()
    expect(sheet).to_contain_text("Also delete its model mine")
    sheet.get_by_role("checkbox").check()
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast").last).to_contain_text("Deleted the fit mine and the model mine")
    expect(entries).to_have_count(0)
    expect(page).to_have_url(re.compile(r"#/training$"))
    assert not model.exists()
    assert not (studio.workspace / "jobs" / fit["id"]).exists()
    listed = page.request.get(f"{studio.base}/studio/api/models").json()["models"]
    assert "mine" not in [entry["name"] for entry in listed]
    assert json.loads(page.request.get(f"{studio.base}/studio/api/jobs").text())["jobs"] == []
