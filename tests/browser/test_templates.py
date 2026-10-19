"""The New sheet's templates: the list, the suggested name, arrow keys, a
double-click, Cancel, a narrow window, sets made from templates (with their
examples and images, unedited), a run on one, and the image preview."""

import base64
import re

import pytest
from drive import open_studio, pick_model, put_set, shot
from fake_studio import TICKET_TRIAGE

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def open_new(page) -> None:
    """Open the New sheet with ⌥⌘N and wait for the templates."""
    page.keyboard.press("Alt+Meta+KeyN")
    expect(page.locator(".tpl[data-template='screenshot-check']")).to_be_visible()


def template(page, name: str):
    return page.locator(f".tpl[data-template='{name}']")


def name_field(page):
    return page.locator(".sheet-input")


def picked(page):
    return page.locator(".tpl[aria-checked=true]")


def chips(page):
    return page.locator(".ex-chip [role=tab]")


def create_shots(page, studio) -> None:
    """A set my-shots made from the screenshot-check template, open."""
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    name_field(page).fill("my-shots")
    template(page, "screenshot-check").click()
    expect(name_field(page)).to_have_value("my-shots")  # a typed name stays
    shot(page, "new-sheet-picked-images")
    page.keyboard.press("Enter")
    expect(page).to_have_url(re.compile("#/playground/my-shots$"))
    expect(chips(page)).to_have_count(3)


def test_the_sheet_lists_the_templates(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    titles = page.locator(".tpl .tpl-title")
    expect(titles).to_have_count(9)  # Empty set and eight templates
    expect(titles.first).to_have_text("Empty set")
    expect(picked(page)).to_have_attribute("data-template", "")
    expect(name_field(page)).to_have_value("untitled")
    expect(name_field(page)).to_be_focused()

    template(page, "support-triage").click()
    expect(name_field(page)).to_have_value("support-triage")
    expect(page.locator(".tpl.on .chip")).to_have_text(["Choice", "Score", "Noul", "4 examples"])
    for scheme in ("light", "dark"):
        page.emulate_media(color_scheme=scheme)
        shot(page, f"new-sheet-{scheme}")


def test_creating_a_set_from_a_template_and_running_it(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    template(page, "support-triage").click()
    page.keyboard.press("Enter")
    expect(page.locator(".sheet")).to_have_count(0)
    expect(page).to_have_url(re.compile("#/playground/support-triage$"))
    expect(page.locator(".question")).to_have_count(3)

    body = studio.set_file("support-triage")
    assert len(body["examples"]) == 4 and len(body["questions"]) == 3
    expect(chips(page)).to_have_text([e["name"] for e in body["examples"]])
    expect(page.locator(".edited-tag")).to_have_count(0)
    expect(page.locator(".sb-item[href='#/playground/support-triage']")).to_be_visible()

    page.keyboard.press("Meta+Enter")
    expect(page.locator(".ex-grid tbody td .v")).to_have_count(12, timeout=15_000)


def test_a_template_with_images(studio, page):
    create_shots(page, studio)
    shots = studio.set_file("my-shots")
    for example in shots["examples"]:
        assert example["images"][0].startswith("data:image/png;base64,")
    expect(page.locator(".edited-tag")).to_have_count(0)


def test_the_image_preview(studio, page, tmp_path):
    create_shots(page, studio)
    # A model that reads images, so ⌘↩ would run if the preview let it.
    pick_model(page, "fake-vision")
    lightbox = page.locator(".lightbox")
    caption = page.locator(".lightbox figcaption")

    page.locator(".thumb-open").click()
    expect(page.locator(".lightbox img")).to_be_visible()
    expect(caption).to_contain_text("640 × 400")
    expect(caption).to_contain_text("[image 1]")
    expect(caption).to_contain_text("PNG")
    expect(page.locator(".lb-step")).to_have_count(0)  # one image: no stepping
    shot(page, "preview")
    page.keyboard.press("Escape")
    expect(lightbox).to_have_count(0)

    # A second image (the next example's) on the same example.
    url = studio.set_file("my-shots")["examples"][1]["images"][0]
    second = tmp_path / "two.png"
    second.write_bytes(base64.b64decode(url.split(",", 1)[1]))
    page.locator(".drop input[type=file]").set_input_files(str(second))
    expect(page.locator(".thumb-open")).to_have_count(2)
    page.locator(".thumb-open").nth(1).click()
    expect(page.locator(".lightbox .lb-step")).to_have_count(2)
    expect(caption).to_contain_text("[image 2] of 2")
    expect(lightbox).to_be_focused()  # the preview takes the keys once it has the focus
    page.keyboard.press("ArrowRight")
    expect(caption).to_contain_text("[image 1] of 2")  # wraps round

    page.keyboard.press("Meta+Enter")
    page.wait_for_timeout(300)
    expect(page.locator(".stop-btn:not([disabled])")).to_have_count(0)
    assert studio.history_lines("my-shots") == []
    expect(lightbox).to_be_visible()

    # A click beside the image closes it.
    lightbox.click(position={"x": 5, "y": 5})
    expect(lightbox).to_have_count(0)
    expect(page.locator(".run-btn")).to_be_enabled()


def test_arrow_keys_move_the_pick(studio, page):
    open_studio(page, studio)
    put_set(page, studio, "support-triage", dict(TICKET_TRIAGE))
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    template(page, "").click()
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    expect(picked(page)).to_have_attribute("data-template", "review-rating")
    page.keyboard.press("ArrowUp")
    expect(picked(page)).to_have_attribute("data-template", "support-triage")
    expect(name_field(page)).to_have_value("support-triage-2")  # the name is taken
    page.keyboard.press("ArrowUp")
    expect(name_field(page)).to_have_value("untitled")


def test_a_double_click_creates_the_set(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    template(page, "fact-check").dblclick()
    expect(page).to_have_url(re.compile("#/playground/fact-check$"))
    assert studio.set_file("fact-check")["examples"][0]["name"] == "Supported"


def test_cancel_writes_nothing(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    before = sorted(p.name for p in studio.sets.iterdir())
    open_new(page)
    template(page, "conversation").click()
    page.locator(".sheet .btn", has_text=re.compile("^Cancel$")).click()
    expect(page.locator(".sheet")).to_have_count(0)
    assert sorted(p.name for p in studio.sets.iterdir()) == before


def test_a_narrow_window_scrolls_the_list_down_not_sideways(studio, page):
    page.set_viewport_size({"width": 900, "height": 640})
    open_studio(page, studio, "playground/ticket-triage")
    open_new(page)
    body = page.locator(".sheet-body")
    assert body.evaluate("e => e.scrollWidth <= e.clientWidth")
    shot(page, "new-sheet-small")
