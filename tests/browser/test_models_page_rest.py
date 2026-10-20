"""The rest of the Models page: filters, speed measured here (no number before
a run), the GPU working set on the memory card, estimates labelled, and the
layout in light and dark at 1,440 and 1,100 px."""

import re

import pytest
from drive import open_studio

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}


@pytest.fixture
def sized(make_studio):
    return make_studio(sizes=True)


def row(page, name: str):
    return page.locator(f".mrow[data-name='{name}']")


def test_filters(sized, page):
    open_studio(page, sized, "models")
    filters = page.locator(".pane-head .seg button")
    expect(filters).to_have_text(
        [re.compile(r"All\s+5"), re.compile(r"Loaded\s+0"), re.compile(r"Fitted\s+1")]
    )
    filters.nth(2).click()
    expect(page.locator(".msec")).to_have_text([re.compile("^Fitted models")])
    expect(page.locator(".mrow")).to_have_count(1)
    expect(row(page, "fake-fit")).not_to_have_class(re.compile("fitted"))  # not nested here
    filters.nth(1).click()
    expect(page.locator(".mempty")).to_contain_text("Nothing loaded")
    assert page.request.post(f"{sized.base}/v1/systemone", data={**BODY, "model": "fake"}).ok
    expect(page.locator(".mrow")).to_have_count(1)
    expect(row(page, "fake")).to_be_visible()
    expect(filters.nth(1)).to_have_text(re.compile(r"Loaded\s+1"))
    # Remembered in this browser.
    page.reload()
    expect(page.locator(".pane-head .seg button[aria-pressed='true']")).to_have_text(
        re.compile("Loaded")
    )
    filters.nth(0).click()
    expect(page.locator(".mrow")).to_have_count(5)


def test_speed_is_measured_only(sized, page):
    open_studio(page, sized, "models/fake")
    speed = row(page, "fake").locator(".c-speed")
    expect(speed).to_have_text("–")  # no number before a run
    expect(speed.locator("span")).to_have_attribute("title", re.compile("Not measured yet"))
    expect(page.locator(".speed-group")).to_contain_text("Not measured yet")
    for _ in range(3):
        assert page.request.post(f"{sized.base}/v1/systemone", data={**BODY, "model": "fake"}).ok
    # The list follows the answers while the page is open.
    expect(speed).to_have_text(re.compile(r"^[\d.]+ (ms|s)$"), timeout=10_000)
    expect(speed.locator("span")).to_have_attribute(
        "title", re.compile(r"Median of 3 answers here, \d+ input tokens?")
    )
    expect(page.locator(".speed-group")).to_contain_text(
        re.compile(r"3 answers, \d+ input tokens?, on ")
    )
    expect(row(page, "fake-vision").locator(".c-speed")).to_have_text("–")


def test_the_memory_card_and_the_estimates(sized, page):
    open_studio(page, sized, "models/fake")
    card = page.locator(".memcard")
    expect(card.locator(".memcard-sub")).to_contain_text("This Mac's GPU working set is")
    expect(page.locator(".mhead")).to_contain_text("Memory, est.")
    expect(page.locator(".idet")).to_contain_text("about 1.8 GB")
    expect(page.locator(".idet")).to_contain_text("An estimate: the weights plus 10%")
    card.get_by_role("link", name="Change the budget").click()
    expect(page.locator(".centre h1")).to_have_text("Settings")


@pytest.mark.parametrize("width", [1440, 1100])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_the_layout_holds(sized, chrome, width, scheme):
    context = chrome.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        open_studio(page, sized, "models/fake")
        expect(row(page, "fake")).to_be_visible()
        expect(page.locator(".places")).to_be_visible()
        assert page.evaluate(
            "document.documentElement.scrollWidth <= innerWidth"
        )  # nothing sideways
        rows_fit = "[...document.querySelectorAll('.mlist .mrow')]" + (
            ".every(r => r.scrollWidth <= r.clientWidth + 1)"
        )
        assert page.evaluate(rows_fit)
        speed_shown = row(page, "fake").locator(".c-speed").is_visible()
        assert speed_shown == (width == 1440)  # a narrow centre leaves Size, Reads and Speed out
        assert errors == []
    finally:
        context.close()
