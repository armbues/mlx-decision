"""The inspector per page: always on the Playground and Models (shown or
hidden remembered per page), on a pick on Datasets and Batches (over the
centre below 1,280 px), none on Training and Settings."""

import re

import pytest
from drive import open_studio
from playwright.sync_api import expect
from test_batches import done, new_batch, write_tweets

pytestmark = pytest.mark.browser

OPEN = re.compile(r"\binspector-open\b")
CLOSED = re.compile(r"\bno-inspector\b")
OVER = re.compile(r"\binsp-over\b")


def test_shown_or_hidden_is_remembered_per_page(studio, page):
    open_studio(page, studio, "models/fake")
    app = page.locator(".app")
    expect(app).to_have_class(OPEN)
    page.get_by_role("button", name="Hide Inspector").click()
    expect(app).to_have_class(CLOSED)
    open_studio(page, studio, "playground/ticket-triage")
    expect(app).to_have_class(OPEN)  # the Playground's answers stay
    open_studio(page, studio, "models/fake")
    expect(app).to_have_class(CLOSED)
    page.reload()
    page.wait_for_selector(".app")
    expect(app).to_have_class(CLOSED)


def test_training_and_settings_have_none(studio, page):
    for route in ("training", "training/evaluate", "settings"):
        open_studio(page, studio, route)
        expect(page.locator(".app")).to_have_class(CLOSED)
        expect(page.locator(".tb-right .icon-btn[aria-label*='Inspector']")).to_have_count(0)
    expect(page.locator(".config-group .kv.place")).to_contain_text("a change creates it")


def test_a_batch_row_opens_it_and_over_the_table_below_1280(studio, page):
    write_tweets(studio.workspace / "data")
    open_studio(page, studio, "batches")
    new_batch(page)
    done(page)
    app = page.locator(".app")
    expect(app).to_have_class(CLOSED)
    page.locator(".batch-rows .ex-rows .batch-row").first.click()
    expect(app).to_have_class(OPEN)
    expect(app).not_to_have_class(OVER)
    page.set_viewport_size({"width": 1200, "height": 900})
    expect(app).to_have_class(OVER)  # over the table, which keeps its width
    centre = page.locator(".centre").evaluate("e => e.getBoundingClientRect().width")
    assert centre > 900
    page.get_by_role("button", name="Close the Inspector").click()
    expect(app).to_have_class(CLOSED)
    expect(page.locator(".batch-row[aria-selected='true']")).to_have_count(0)


def test_the_evaluate_form_shows_the_labels(studio, page):
    write_tweets(studio.workspace / "data")
    open_studio(page, studio, "training/evaluate")
    form = page.locator(".eval-form")
    form.locator("#eval-data").select_option("tweets")
    expect(form.locator(".label-line")).to_contain_text("analyst 3 · fed 3 · other 3")
