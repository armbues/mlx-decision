"""The Models page: the list (fitted models under their base, the status pill
that turns into Load / Unload under the pointer), the memory card (segments,
unload marks, the preview of a load), the inspector, the row's context menu,
the sidebar's unload mark and the controls while a job holds the GPU.

The Studio here has made-up sizes against an 8 GB budget: fake 1.8 GB, its
fit fake-fit (+0.05 GB on it), fake-vision 3.2, fake-strict 4.6 and fake-big
19, over the budget."""

import re

import pytest
from drive import hold_gpu, menu_item, open_studio, release_gpu, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

MODELS = ["fake", "fake-big", "fake-fit", "fake-strict", "fake-vision"]


@pytest.fixture
def sized(make_studio):
    return make_studio(sizes=True)


def row(page, name: str):
    return page.locator(f".mrow[data-name='{name}']")


def pill(page, name: str):
    return row(page, name).locator(".pill-act")


def state(page, name: str):
    return pill(page, name).locator(".as-state")


def segments(page):
    """The memory card's segments, one per group in memory."""
    return page.locator(".membig .segs i:not(.ghost) span")


def inspector_button(page):
    """The inspector's Load / Unload button (after Open in Playground)."""
    return page.locator(".idet-actions button").nth(1)


def toast(page, text: str):
    return page.locator(".toast", has_text=text)


def load(page, studio, name: str) -> None:
    """Load a model through the Studio's API, as another tab would."""
    response = page.request.post(f"{studio.base}/studio/api/models/load", data={"model": name})
    assert response.ok, response.text()


def test_the_list_groups_models_and_shows_their_state_as_text(sized, page):
    open_studio(page, sized, "models")
    names = page.locator(".mrow .nm-txt b")
    expect(names).to_have_count(5)
    assert sorted(names.all_text_contents()) == MODELS
    # One control per row: the pill.
    for name in MODELS:
        expect(row(page, name).locator("button")).to_have_count(1)
        expect(row(page, name).locator("button.pill-act")).to_have_count(1)
    # The fit sits right under its base.
    expect(row(page, "fake-fit")).to_have_class(re.compile(r"\bfitted\b"))
    assert row(page, "fake").evaluate("r => r.nextElementSibling.dataset.name") == "fake-fit"
    expect(state(page, "fake")).to_have_text("Not loaded")
    expect(state(page, "fake-big")).to_contain_text("Over budget")
    # The memory column; a fit sharing its base shows what it adds.
    expect(row(page, "fake-vision")).to_contain_text("3.2 GB")
    expect(row(page, "fake-fit")).to_contain_text("+0.1 GB")


def test_the_pill_says_load_only_under_the_pointer(sized, page):
    open_studio(page, sized, "models")
    strict = pill(page, "fake-strict")
    expect(strict.locator(".as-state")).to_be_visible()
    expect(strict.locator(".as-state")).to_have_text("Not loaded")
    expect(strict.locator(".as-action")).to_be_hidden()
    # On the row, it still shows the state; on the pill, it says Load.
    row(page, "fake-strict").locator(".nm").hover()
    expect(strict.locator(".as-state")).to_be_visible()
    expect(strict.locator(".as-action")).to_be_hidden()
    strict.hover()
    expect(strict.locator(".as-action")).to_be_visible()
    expect(strict.locator(".as-action")).to_have_text("Load")
    expect(strict.locator(".as-state")).to_be_hidden()


def test_a_model_over_the_budget_cannot_load_from_its_pill(sized, page):
    open_studio(page, sized, "models")
    big = pill(page, "fake-big")
    big.hover()
    expect(big.locator(".as-state")).to_be_visible()
    expect(big.locator(".as-state")).to_have_text("Over budget")
    expect(big.locator(".as-action")).to_be_hidden()
    expect(big).to_have_attribute("title", re.compile("cannot load"))
    expect(big).to_have_attribute("aria-disabled", "true")
    # A click (forced: Playwright skips aria-disabled) does nothing: no toast,
    # nothing in memory.
    big.click(force=True)
    page.wait_for_timeout(300)
    expect(toast(page, "fake-big")).to_have_count(0)
    expect(page.locator(".membar i")).to_have_count(0)
    assert sized.app.state.gpu.pool.loaded() == []


def test_load_and_unload_from_the_inspector(sized, page):
    open_studio(page, sized, "models")
    row(page, "fake-vision").locator(".nm").click()
    buttons = page.locator(".idet-actions button")
    expect(buttons).to_have_text(["Open in Playground", "Load"])
    inspector_button(page).click()
    expect(segments(page)).to_have_text(["fake-vision"])
    # The row moves to Loaded and says when it was used.
    expect(state(page, "fake-vision")).to_have_text(re.compile("^used "))
    expect(toast(page, re.compile("^Loaded fake-vision in "))).to_be_visible()
    expect(inspector_button(page)).to_have_text("Unload")
    inspector_button(page).click()
    expect(segments(page)).to_have_count(0)
    expect(state(page, "fake-vision")).to_have_text("Not loaded")
    expect(inspector_button(page)).to_have_text("Load")


def test_the_pill_loads_and_unloads_without_changing_the_selection(sized, page):
    open_studio(page, sized, "models/fake")
    vision = pill(page, "fake-vision")
    vision.click()
    expect(page.locator(".membar i")).to_have_count(1)
    expect(state(page, "fake-vision")).to_have_text(re.compile("^used "))
    # Under the pointer, a loaded pill says Unload.
    page.locator(".pane-head h1").hover()
    vision.hover()
    expect(vision.locator(".as-action")).to_be_visible()
    expect(vision.locator(".as-action")).to_have_text("Unload")
    vision.click()
    expect(page.locator(".membar i")).to_have_count(0)
    expect(state(page, "fake-vision")).to_have_text("Not loaded")
    vision.click()
    expect(page.locator(".membar i")).to_have_count(1)
    expect(state(page, "fake-vision")).to_have_text(re.compile("^used "))
    assert page.evaluate("location.hash") == "#/models/fake"


def test_a_fit_loads_with_its_base_and_the_preview_shows_what_a_load_takes(sized, page):
    open_studio(page, sized, "models")
    load(page, sized, "fake-vision")
    load(page, sized, "fake-fit")
    expect(segments(page)).to_have_count(2)
    # One group for the fit and its base; the base row says so.
    expect(segments(page).filter(has_text="fake-fit (with fake)")).to_have_count(1)
    expect(state(page, "fake")).to_have_text("with fake-fit")

    # fake-strict (4.6 GB) needs the least recently used group to go.
    row(page, "fake-strict").locator(".nm").hover()
    note = page.locator(".memnote")
    expect(note).to_have_text(re.compile("^Loading fake-strict"))
    expect(note).to_contain_text("unloads fake-vision")
    # The leaving segment dims; a ghost shows the load.
    expect(page.locator(".membig .segs i.leaving")).to_have_count(1)
    expect(page.locator(".membig .segs i.ghost")).to_have_count(1)
    shot(page, "models-preview-light")
    row(page, "fake-big").locator(".nm").hover()
    expect(note).to_contain_text("cannot load")


def test_unload_from_the_memory_card_and_the_sidebar(sized, page):
    open_studio(page, sized, "models")
    load(page, sized, "fake-vision")
    load(page, sized, "fake-fit")
    expect(segments(page)).to_have_count(2)
    # The memory card's mark takes the whole group.
    page.locator(".mem-group", has_text="fake-fit").locator(".mem-x").click()
    expect(segments(page)).to_have_text(["fake-vision"])
    expect(toast(page, re.compile("^Unloaded fake-fit$"))).to_be_visible()
    # The sidebar's mark.
    page.locator(".mem-legend div", has_text="fake-vision").locator(".mem-x").click()
    expect(segments(page)).to_have_count(0)


def test_the_row_context_menu_loads_a_model(sized, page):
    open_studio(page, sized, "models")
    row(page, "fake-strict").locator(".nm").click(button="right")
    expect(page.locator(".ctx-menu .mi")).to_have_text(["Open in Playground", "Load", "Copy Name"])
    menu_item(page, "Load").click()
    expect(segments(page)).to_have_text(["fake-strict"])
    expect(page.locator(".ctx-menu")).to_have_count(0)


def test_load_and_unload_wait_while_a_job_holds_the_gpu(sized, page):
    open_studio(page, sized, "models/fake-vision")
    load(page, sized, "fake-vision")
    expect(segments(page)).to_have_text(["fake-vision"])
    expect(inspector_button(page)).to_be_enabled()
    hold_gpu(sized, "fit-7")
    # The inspector's button is disabled and names the job.
    expect(inspector_button(page)).to_be_disabled()
    expect(inspector_button(page)).to_have_attribute("title", re.compile("fit-7"))
    # So are the memory card's and the sidebar's marks.
    marks = page.locator(".mem-group .mem-x, .mem-legend .mem-x")
    expect(marks).to_have_count(2)
    for mark in marks.all():
        expect(mark).to_be_disabled()
    # The pill stays a pill under the pointer and names the job.
    vision = pill(page, "fake-vision")
    vision.hover()
    expect(vision.locator(".as-state")).to_be_visible()
    expect(vision.locator(".as-action")).to_be_hidden()
    expect(vision).to_have_attribute("title", re.compile("fit-7"))
    release_gpu(sized, "fit-7")
    expect(inspector_button(page)).to_be_enabled()
    expect(page.locator(".mem-group .mem-x")).to_be_enabled()


def test_open_in_playground_picks_the_model(sized, page):
    open_studio(page, sized, "models/fake-vision")
    page.locator(".idet-actions .btn.primary").click()
    page.wait_for_url("**/#/playground**")
    expect(page.locator(".target .mdl .name")).to_have_text("fake-vision")


def test_at_1100_px_the_names_keep_their_room(sized, page):
    open_studio(page, sized, "models/fake-vision")
    page.emulate_media(color_scheme="dark")
    shot(page, "models-1440-dark")
    page.set_viewport_size({"width": 1100, "height": 800})
    expect(page.locator(".mrow .c-size").first).to_be_hidden()
    width = page.locator(".mrow .nm").first.evaluate("e => e.getBoundingClientRect().width")
    assert width > 150
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    shot(page, "models-1100-dark")
    page.emulate_media(color_scheme="light")
    shot(page, "models-1100-light")
