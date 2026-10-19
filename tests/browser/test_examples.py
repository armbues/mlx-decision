"""Examples in the Playground: adding, saving, renaming and removing them, Run
This Example, Run All (the grid, the History, the file's lines), stale marks,
a JSON problem at its chip, a reload and asking the selected example alone."""

import re
import time

import pytest
from drive import answers_view, inspector_tab, menu_item, open_studio, pick, put_set, shot
from fake_studio import TICKET_TRIAGE

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

REFUND = "I was charged twice this month, please refund one of them."


def chips(page):
    return page.locator(".ex-chip [role=tab]")


def chip(page, name: str):
    return page.locator(".ex-chip [role=tab]", has_text=re.compile(f"^{re.escape(name)}$"))


def example_menu(page, item: str) -> None:
    """Choose ``item`` in the selected example's chip menu."""
    pick(page, ".ex-chip.on .ex-more", item)


def save(page) -> None:
    page.keyboard.press("Meta+KeyS")
    expect(page.locator(".edited-tag")).to_have_count(0)


def open_one_example(page, studio) -> None:
    """The ticket-triage questions and state as a plain set named examples."""
    open_studio(page, studio)
    put_set(page, studio, "examples", dict(TICKET_TRIAGE))
    open_studio(page, studio, "playground/examples")
    expect(chips(page)).to_have_count(1)


def open_two_examples(page, studio) -> None:
    """A saved set examples with Example (ticket-triage's state) and Refund ask."""
    body = {
        **TICKET_TRIAGE,
        "examples": [
            {"name": "Example", "state": TICKET_TRIAGE["state"]},
            {"name": "Refund ask", "state": REFUND},
        ],
    }
    open_studio(page, studio)
    put_set(page, studio, "examples", body)
    open_studio(page, studio, "playground/examples")
    expect(chips(page)).to_have_text(["Example", "Refund ask"])


def run_all(page) -> None:
    """⌘↩ with two examples: wait for the grid's six answers."""
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".ex-grid tbody td .v")).to_have_count(6, timeout=15_000)
    expect(page.locator(".stop-btn")).to_be_disabled()


def test_adding_an_example_and_saving(studio, page):
    open_one_example(page, studio)
    expect(chips(page)).to_have_text(["Example"])
    expect(page.locator(".run-btn")).to_have_text("Run")

    page.locator(".ex-add").click()
    expect(chips(page)).to_have_text(["Example", "Example 2"])
    expect(page.locator(".ex-chip.on")).to_have_text("Example 2")
    expect(page.locator(".run-btn")).to_have_text("Run All")
    state = page.locator(".state-text")
    expect(state).to_have_value("")

    state.fill(REFUND)
    expect(page.locator(".edited-tag")).to_be_visible()
    save(page)
    saved = studio.set_file("examples")
    assert [e["name"] for e in saved.get("examples", [])] == ["Example", "Example 2"]
    # The first example is also the file's top-level state.
    assert saved["state"] == TICKET_TRIAGE["state"]

    # Switching examples is not an edit.
    chip(page, "Example").click()
    expect(state).to_have_value(TICKET_TRIAGE["state"])
    expect(page.locator(".edited-tag")).to_have_count(0)


def test_renaming_an_example(studio, page):
    open_two_examples(page, studio)
    chip(page, "Refund ask").click()
    example_menu(page, "Rename…")
    page.locator(".sheet-input").fill("Double charge")
    page.keyboard.press("Enter")
    expect(page.locator(".sheet")).to_have_count(0)
    expect(chips(page)).to_have_text(["Example", "Double charge"])


def test_run_this_example_then_run_all(studio, page):
    open_two_examples(page, studio)
    chip(page, "Example").click()
    inspector_tab(page, "Answers")
    example_menu(page, "Run This Example")
    expect(page.locator(".acard .verdict")).to_have_count(3, timeout=8_000)
    expect(page.locator(".run-head h3")).to_contain_text("Example")
    expect(page.locator(".ex-grid")).to_have_count(0)

    # Run All is the main action, ⌘↩: every example, shown as a grid.
    run_all(page)
    lines = studio.history_lines("examples")
    assert len(lines) == 3  # the single run and Run All's two
    assert [line.get("example") for line in lines] == ["Example", "Example", "Refund ask"]
    assert "batch" not in lines[0]
    assert lines[1].get("batch") and lines[1]["batch"] == lines[2].get("batch")
    expect(page.locator(".activity")).to_contain_text("answered 2 examples")
    # A row per example, a cell per question.
    expect(page.locator(".ex-grid tbody th")).to_have_text(["Example", "Refund ask"])
    expect(page.locator(".ex-grid tbody td .v")).to_have_count(6)
    shot(page, "examples-grid")


def test_a_grid_row_opens_its_example_and_edits_mark_the_grid(studio, page):
    open_two_examples(page, studio)
    run_all(page)
    page.locator(".ex-grid tbody tr", has_text=re.compile(r"^Refund ask")).click()
    expect(page.locator(".ex-chip.on")).to_have_text("Refund ask")
    expect(page.locator(".acard .verdict")).to_have_count(3)
    expect(page.locator(".run-head h3")).to_contain_text("Refund ask")

    page.locator(".question[data-q='is_urgent'] .ins").fill(
        "Does this need a reply within the hour?"
    )
    answers_view(page, "All Examples")
    expect(page.locator(".stale-banner")).to_contain_text("Questions edited")
    expect(page.locator(".run-btn .badge")).to_be_visible()


def test_run_all_is_one_history_entry(studio, page):
    open_two_examples(page, studio)
    example_menu(page, "Run This Example")
    expect(page.locator(".acard .verdict")).to_have_count(3, timeout=8_000)
    run_all(page)

    inspector_tab(page, "History")
    rows = page.locator(".hrow")
    expect(rows).to_have_count(2)  # two entries for three runs
    top = rows.first
    expect(top).to_contain_text(re.compile(r"^#2"))
    expect(top).to_contain_text("Run All")
    expect(top).to_contain_text("2 examples")
    top.click()
    expect(page.locator(".moves .ex-grid tbody tr")).to_have_count(2)
    shot(page, "examples-history")


def test_a_json_problem_marks_its_chip_and_blocks_run_all(studio, page):
    open_two_examples(page, studio)
    chip(page, "Refund ask").click()
    page.locator(".seg button", has_text=re.compile("^JSON$")).click()
    state = page.locator(".state-text")
    state.fill("{not json")
    expect(page.locator(".ex-chip .ex-problem")).to_have_count(1)
    expect(chip(page, "Refund ask").locator(".ex-problem")).to_be_visible()
    expect(page.locator(".run-btn")).to_be_disabled()

    page.locator(".seg button", has_text=re.compile("^Text$")).click()
    state.fill(REFUND)
    expect(page.locator(".ex-chip .ex-problem")).to_have_count(0)
    expect(page.locator(".run-btn")).to_be_enabled()
    shot(page, "examples-editor")


def test_a_problem_the_live_check_finds_blocks_run_all(studio, page):
    open_two_examples(page, studio)
    expect(page.locator(".run-btn")).to_have_text("Run All")
    department = page.locator(".question[data-q='department']")
    # fake takes at most 4 options: a fifth is a problem for every example.
    department.locator(".add-row .btn").click()
    department.locator(".add-row .btn").click()
    expect(department.locator(".field-problem")).to_contain_text("at most 4 options")
    expect(page.locator(".run-btn")).to_be_disabled()
    page.keyboard.press("Meta+Enter")
    page.wait_for_timeout(300)
    assert studio.history_lines("examples") == []  # nothing was sent
    department.locator(".opt-row .minus").last.click()
    expect(page.locator(".run-btn")).to_be_enabled()


def test_removing_examples_keeps_one(studio, page):
    open_two_examples(page, studio)
    chip(page, "Refund ask").click()
    example_menu(page, "Remove")
    expect(chips(page)).to_have_text(["Example"])
    expect(page.locator(".run-btn")).to_have_text("Run")

    page.locator(".ex-chip.on .ex-more").click()
    expect(menu_item(page, "Remove")).to_be_disabled()
    page.keyboard.press("Escape")
    expect(page.locator(".menu")).to_have_count(0)


def test_a_reload_brings_back_the_examples_and_their_answers(studio, page):
    open_two_examples(page, studio)
    run_all(page)
    # An unsaved edit, so the page asks before it goes.
    chip(page, "Refund ask").click()
    example_menu(page, "Remove")
    expect(chips(page)).to_have_count(1)
    page.on("dialog", lambda dialog: dialog.accept())
    page.reload()

    expect(chips(page)).to_have_text(["Example", "Refund ask"], timeout=10_000)
    chip(page, "Refund ask").click()
    answers_view(page, "This Example")
    expect(page.locator(".run-head h3")).to_contain_text("Refund ask", timeout=10_000)
    expect(page.locator(".acard .verdict")).to_have_count(3)


def test_alt_cmd_enter_asks_the_selected_example_alone(studio, page):
    open_two_examples(page, studio)
    chip(page, "Refund ask").click()
    expect(page.locator(".ex-chip.on")).to_have_text("Refund ask")
    page.keyboard.press("Alt+Meta+Enter")
    deadline = time.monotonic() + 8
    while not studio.history_lines("examples"):
        assert time.monotonic() < deadline, "no run in the History"
        time.sleep(0.1)
    time.sleep(0.5)  # a second line would be a Run All
    [line] = studio.history_lines("examples")
    assert line.get("example") == "Refund ask"
    assert "batch" not in line
