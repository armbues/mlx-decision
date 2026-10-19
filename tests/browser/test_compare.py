"""Compare: the same questions on two or three models in turn, as a table.

Picking models in the Compare view and the model menu, three models loading in
turn within the 8 GB budget, the history's lines and the History's one entry,
stale marks, a model over the budget refused in its column, Stop, the notes
where models differ, a reload, and a set with several examples.
"""

import re
import time

import pytest
from drive import answers_view, inspector_tab, open_studio, pick_model, put_set, shot
from fake_studio import GB, SIZES

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

BODY = {
    "state": "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing"},
        },
        "is_urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    },
}
THREE = ["fake", "fake-vision", "fake-strict"]


def open_cmp(page, studio, model: str = "fake") -> None:
    """The set ``cmp`` run on ``model``, the Answers tab open."""
    open_studio(page, studio)
    put_set(page, studio, "cmp", BODY)
    open_studio(page, studio, "playground/cmp")
    pick_model(page, model)
    expect(page.locator(".target .mdl .name")).to_have_text(model)
    inspector_tab(page, "Answers")
    expect(page.locator(".answers-view")).to_be_visible()


def tick(page, name: str) -> None:
    """Tick or untick a model in the Compare view's list."""
    label = page.locator(".cmp-picks label").filter(
        has=page.locator(".mono", has_text=re.compile(f"^{re.escape(name)}$"))
    )
    label.locator("input").click()


def columns(page):
    return page.locator(".inspector .cmp thead th .mono")


def cells(page):
    return page.locator(".inspector .cmp tbody .cv")


def finished(page, timeout: float = 15_000) -> None:
    """The Compare is over: Stop is off and no column waits."""
    expect(page.locator(".stop-btn")).to_be_disabled(timeout=timeout)
    expect(page.locator(".inspector .cmp thead small.wait")).to_have_count(0, timeout=timeout)


def compare_three(page, studio) -> None:
    """Compare fake with fake-vision and fake-strict on ``cmp``, to the end."""
    open_cmp(page, studio)
    answers_view(page, "Compare")
    tick(page, "fake-vision")
    tick(page, "fake-strict")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 2")
    page.keyboard.press("Meta+Enter")
    expect(columns(page)).to_have_text(THREE)
    expect(cells(page)).to_have_count(6)
    finished(page)


def loaded(page, studio) -> list[str]:
    return page.request.get(f"{studio.base}/studio/api/status").json()["loaded"]


def wait_for(condition, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def test_the_compare_view_asks_for_models_and_takes_two(make_studio, page):
    studio = make_studio(sizes=True)
    open_cmp(page, studio)
    expect(page.locator(".answers-view [role=tab]")).to_have_text(["One Model", "Compare"])

    answers_view(page, "Compare")
    expect(page.locator(".cmp-picks")).to_be_visible()
    expect(page.locator(".run-btn")).to_have_text("Compare")
    expect(page.locator(".run-btn")).to_be_disabled()
    expect(page.locator(".activity")).to_contain_text("pick one or two more")
    expect(page.locator(".app")).to_have_class(re.compile(r"\bcompare\b"))  # it widens

    # Two are picked; a third is not offered.
    tick(page, "fake-vision")
    tick(page, "fake-strict")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 2")
    expect(page.locator(".cmp-picks input:not(:disabled):not(:checked)")).to_have_count(0)
    activity = page.locator(".activity")
    expect(activity).to_contain_text("fake, fake-vision and fake-strict")
    expect(activity).to_contain_text("all load first")
    shot(page, "compare-before")


def test_compare_asks_three_models_in_turn_within_the_budget(make_studio, page):
    studio = make_studio(sizes=True)
    compare_three(page, studio)
    footer = page.locator(".inspector .cmp tfoot tr").first.locator("td")
    assert all("load" in cell for cell in footer.all_inner_texts()), footer.all_inner_texts()
    now = loaded(page, studio)
    assert "fake-strict" in now and len(now) < 3, now
    assert sum(SIZES[name] for name in now) <= 8 * GB, now
    expect(page.locator(".activity")).to_contain_text("3 models answered")

    lines = studio.history_lines("cmp")
    assert [line["model"] for line in lines] == THREE
    # One Compare id on all three; each names the example and is no Run All.
    assert lines[0].get("compare") and len({line.get("compare") for line in lines}) == 1
    assert all(line.get("example") == "Example" and "batch" not in line for line in lines)
    shot(page, "compare-table")


def test_a_compare_is_one_history_entry_with_its_table(make_studio, page):
    studio = make_studio(sizes=True)
    compare_three(page, studio)
    inspector_tab(page, "History")
    rows = page.locator(".hrow")
    expect(rows).to_have_count(1)
    expect(rows).to_contain_text("Compare")
    expect(rows).to_contain_text("fake, fake-vision, fake-strict")
    expect(page.locator(".moves .cmp tbody tr")).to_have_count(2)
    shot(page, "compare-history")


def test_an_edit_marks_the_compare_stale(make_studio, page):
    studio = make_studio(sizes=True)
    compare_three(page, studio)
    page.locator(".question[data-q='is_urgent'] .ins").fill(
        "Does this need a reply within the hour?"
    )
    expect(page.locator(".stale-banner")).to_contain_text("is_urgent")
    expect(page.locator(".run-btn .badge")).to_be_visible()
    expect(page.locator(".inspector .cmp tbody tr.stale")).to_have_count(1)
    expect(page.locator(".activity")).to_contain_text("Edited since the Compare")


def test_a_model_over_the_budget_is_refused_in_its_column(make_studio, page):
    studio = make_studio(sizes=True)
    compare_three(page, studio)
    pick_model(page, "fake-strict", "Compare With")
    pick_model(page, "fake-big", "Compare With")
    # Other models than the Compare shown: the banner says whom it asked.
    expect(page.locator(".stale-banner")).to_contain_text("It asked")

    page.keyboard.press("Meta+Enter")
    expect(columns(page)).to_have_text(["fake", "fake-vision", "fake-big"])
    finished(page)
    expect(page.locator(".inspector .cmp thead small.err")).to_contain_text("budget")
    expect(cells(page)).to_have_count(4)  # the others answered
    expect(page.locator(".activity")).to_contain_text("2 of 3 models answered")
    shot(page, "compare-refused")


def test_stop_leaves_the_models_not_asked(make_studio, page):
    studio = make_studio(sizes=True, slow=True)
    open_cmp(page, studio)
    pick_model(page, "fake-vision", "Compare With")
    pick_model(page, "fake-slow", "Compare With")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 2")
    # The model run on leaves the compared ones.
    pick_model(page, "fake-slow")
    expect(page.locator(".target .mdl .name")).to_have_text("fake-slow")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 1")
    pick_model(page, "fake", "Compare With")
    expect(page.locator(".target .mdl .plus")).to_have_text("+ 2")

    page.keyboard.press("Meta+Enter")
    expect(page.locator(".stop-btn")).to_be_enabled()
    page.wait_for_timeout(300)  # fake-slow is answering
    page.locator(".stop-btn").click()
    expect(page.locator(".inspector .cmp thead small.wait")).to_have_text(
        ["Stopped while it answered", "Not asked: stopped", "Not asked: stopped"]
    )
    expect(page.locator(".activity")).to_contain_text("was stopped")
    # The Studio finishes fake-slow's answer; nothing more is sent.
    wait_for(lambda: studio.history_lines("cmp"))
    page.wait_for_timeout(500)
    assert [line["model"] for line in studio.history_lines("cmp")] == ["fake-slow"]


def test_verdicts_that_differ_are_marked(make_studio, page):
    studio = make_studio(sizes=True, slow=True)
    open_cmp(page, studio, "fake-slow")
    pick_model(page, "fake-vision", "Compare With")
    pick_model(page, "fake", "Compare With")
    answers_view(page, "Compare")
    page.keyboard.press("Meta+Enter")
    expect(cells(page)).to_have_count(6, timeout=20_000)
    finished(page, 20_000)
    expect(page.locator(".inspector .cmp .spread")).to_have_text(
        ["Verdicts differ", "Verdicts differ"]
    )
    shot(page, "compare-differ")
    page.emulate_media(color_scheme="dark")
    shot(page, "compare-differ-dark")


def test_a_reload_shows_the_latest_compare_numbered(make_studio, page):
    studio = make_studio(sizes=True)
    compare_three(page, studio)
    page.keyboard.press("Meta+Enter")
    wait_for(lambda: len(studio.history_lines("cmp")) == 6)
    finished(page)

    # Only a Compare read back from the history has its number.
    page.reload()
    expect(page.locator(".answers-view")).to_be_visible()
    answers_view(page, "Compare")
    expect(page.locator(".inspector .cmp tbody")).to_be_visible()
    expect(page.locator(".run-head h3")).to_have_text(re.compile("^Compare #2"))
    expect(columns(page)).to_have_text(THREE)


def test_with_several_examples_the_selected_one_is_compared(make_studio, page):
    studio = make_studio(sizes=True)
    open_cmp(page, studio)
    pick_model(page, "fake-vision", "Compare With")
    pick_model(page, "fake-strict", "Compare With")
    page.locator(".ex-add").click()
    expect(page.locator(".ex-chip")).to_have_count(2)
    page.locator(".state-text").fill("I was charged twice this month, please refund one of them.")
    answers_view(page, "Compare")
    expect(page.locator(".answers-view [role=tab]")).to_have_text(
        ["This Example", "All Examples", "Compare"]
    )
    expect(page.locator(".run-btn")).to_have_text("Compare")
    expect(page.locator(".activity")).to_contain_text("about Example 2")

    page.keyboard.press("Meta+Enter")
    expect(cells(page)).to_have_count(6)
    finished(page)
    lines = studio.history_lines("cmp")
    assert [line.get("example") for line in lines] == ["Example 2"] * 3

    # Option-Command-Return runs the example alone and shows its cards.
    page.keyboard.press("Alt+Meta+Enter")
    expect(page.locator(".answers-view [aria-pressed=true]")).to_have_text("This Example")
    expect(page.locator(".run-btn")).to_have_text("Run All")
    expect(page.locator(".stop-btn")).to_be_disabled()
    shot(page, "compare-examples")
