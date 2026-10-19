"""The window around the pages: a job holding the GPU, shortcuts while a sheet
is open, the theme switch and the inspector in a narrow window."""

import pytest
from drive import answers_view, hold_gpu, open_studio, pick, pick_model, release_gpu

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_a_job_holding_the_gpu_turns_runs_off_and_says_so(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake")
    run = page.locator(".run-btn")
    expect(run).to_be_enabled()

    hold_gpu(studio, "fit-7")
    banner = page.locator(".banner")
    expect(banner).to_contain_text("fit-7 has the GPU. Run is off until it is done")
    expect(banner).to_contain_text("the API answers 503 busy")
    expect(page.locator(".activity")).to_contain_text("nothing else runs until it is done")
    expect(page.locator(".sidebar")).to_contain_text("API busy")
    expect(run).to_be_disabled()
    expect(run).to_have_attribute("title", "fit-7 has the GPU: Run is off until it is done")
    page.keyboard.press("Meta+Enter")
    # Compare too.
    answers_view(page, "Compare")
    pick_model(page, "fake-vision", "Compare With")
    expect(run).to_have_text("Compare")
    expect(run).to_be_disabled()
    expect(run).to_have_attribute("title", "fit-7 has the GPU: Compare is off until it is done")
    assert studio.history_lines("ticket-triage") == []  # nothing was sent

    # The Models page says what it means there.
    page.goto(f"{studio.base}/#/models")
    expect(banner).to_contain_text("Load and Unload are off until it is done")

    release_gpu(studio, "fit-7")
    expect(banner).to_have_count(0)
    page.goto(f"{studio.base}/#/playground/ticket-triage")
    answers_view(page, "One Model")
    expect(run).to_be_enabled()
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".acard .verdict")).to_have_count(3)


def test_page_shortcuts_do_not_reach_chrome_while_a_sheet_is_open(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    page.locator(".question[data-q='department'] .ins").fill("Which team?")
    saved = studio.set_file("ticket-triage")
    page.get_by_role("button", name="Get Code").first.click()
    expect(page.locator(".sheet")).to_be_visible()
    # A listener added after the page's sees whether it kept the key from Chrome
    # (cmd-S would open Chrome's Save Page As).
    page.evaluate(
        "window.seen = []; addEventListener('keydown', e => "
        "e.metaKey && e.code.startsWith('Key') && seen.push([e.code, e.defaultPrevented]))"
    )
    page.keyboard.press("Meta+s")
    page.keyboard.press("Meta+Shift+s")
    page.keyboard.press("Meta+o")
    assert page.evaluate("seen") == [["KeyS", True], ["KeyS", True], ["KeyO", True]]
    expect(page.locator(".sheet")).to_contain_text("Use This Request Elsewhere")  # still Get Code
    assert studio.set_file("ticket-triage") == saved  # nothing saved


def test_the_download_button_names_its_shortcut(studio, page):
    open_studio(page, studio, "models")
    button = page.locator(".run-btn")
    expect(button).to_have_text("Download Model…")
    expect(button).to_have_attribute("title", "Download Model… (⌘D)")


def test_the_theme_switch_overrides_the_system_and_is_kept(studio, page):
    page.emulate_media(color_scheme="light")
    open_studio(page, studio, "playground/ticket-triage")
    background = "getComputedStyle(document.body).backgroundColor"
    light = page.evaluate(background)
    pick(page, "button[aria-label=Appearance]", "Dark")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    dark = page.evaluate(background)
    assert dark != light
    page.reload()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    pick(page, "button[aria-label=Appearance]", "Follow the System")
    expect(page.locator("html")).not_to_have_attribute("data-theme", "dark")
    assert page.evaluate(background) == light


def test_below_1100_px_the_inspector_opens_over_the_centre(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    expect(page.locator(".inspector")).to_be_visible()
    page.set_viewport_size({"width": 1000, "height": 800})
    expect(page.locator(".inspector")).to_be_hidden()
    page.get_by_role("button", name="Show Inspector").click()
    expect(page.locator(".inspector")).to_be_visible()
    width = page.evaluate("document.documentElement.scrollWidth")
    assert width <= 1000  # over the centre, not beside it
    page.keyboard.press("Escape")
    expect(page.locator(".inspector")).to_be_hidden()
