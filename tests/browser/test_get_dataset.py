"""Get a Dataset: the presets with their question and splits, a sample and a
name, and the job whose file joins Training Data. The job is
``tests/fake_job.py`` (no network)."""

import re

import pytest
from drive import open_studio
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_get_a_dataset(studio, page):
    fake_jobs(studio.app, steps=5, every=0.05)
    open_studio(page, studio, "training")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator(".fam b")).to_have_text(
        ["TREC", "TweetEval Offensive", "SST-5", "OpenBookQA", "ANLI round 3"]
    )
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("TREC")
    expect(sheet.locator(".ds-options li")).to_have_count(6)
    expect(sheet.locator(".crow[data-split='train']")).to_contain_text("5,452 rows")
    expect(sheet.locator(".crow[data-split='test']")).to_contain_text("keep it for training eval")
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("trec-train")

    sheet.locator(".fam[data-preset='sst5']").click()
    expect(sheet.locator(".crow[aria-checked='true']")).to_have_attribute("data-split", "train")
    expect(name).to_have_value("sst5-train")
    sheet.locator(".crow[data-split='validation']").click()
    expect(name).to_have_value("sst5-validation")
    sheet.get_by_role("spinbutton", name="Rows").fill("5000")  # more than validation's 1,101 rows
    expect(sheet.locator(".warn-text")).to_have_text("A sample is 1 to 1101 rows")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    sheet.get_by_role("spinbutton", name="Rows").fill("300")
    sheet.get_by_role("button", name="Get").click()

    expect(page.locator(".toast")).to_contain_text("Get sst5 validation (300 rows)")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details == {"preset": "sst5", "split": "validation", "sample": 300, "seed": 0,
                           "name": "sst5-validation"}  # fmt: skip
    assert job.argv[:5] == ["data", "--preset", "sst5", "--split", "validation"]
    # Done: the file joins the list, in this tab and the job's card says so.
    items = page.locator(".sidebar .data-list .sb-item")
    expect(items).to_have_text(["sst5-validation", "tickets"], timeout=15_000)
    expect(page.locator(".sidebar .job small")).to_have_text("Done")


def test_a_name_taken_is_said(studio, page):
    open_studio(page, studio, "training")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    sheet.get_by_label("Name").fill("tickets")
    expect(sheet.locator(".warn-text")).to_have_text("A data file named tickets exists already")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    expect(sheet.locator(".sheet-foot")).to_contain_text(
        re.compile("mlx-decision data hf:ORG/NAME")
    )
