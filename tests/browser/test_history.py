"""The Playground's History, as in tuning a set: three runs with edits in
between, each run's word diff and how the answers moved, Restore, the history
file, a reload (answers from the file) and Clear History."""

import re

import pytest
from drive import inspector_tab, open_studio, pick_model, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

# Each history row as "#n what changed".
ROWS = (
    "rows => rows.map(r => r.querySelector('.n').innerText + ' '"
    " + r.querySelector('.what b').innerText)"
)
# The diff's changed words as "DEL:word" / "INS:word".
WORDS = "els => els.map(e => e.tagName + ':' + e.innerText)"
# Each moved answer as "id value delta".
MOVES = (
    "ms => ms.map(m => m.querySelector('.qid').firstChild.textContent + ' '"
    " + m.querySelector('.delta').innerText.replace('\\n', ' '))"
)


def question(page, qid: str):
    return page.locator(f".question[data-q='{qid}']")


def run(page, number: int) -> None:
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".run-head h3")).to_have_text(re.compile(f"^Run #{number}"))


def three_runs(page, studio) -> str:
    """Run ticket-triage on fake three times: as saved, with department's
    instructions reworded, then with a frustration level reworded and a fourth
    department option. Returns the saved instructions."""
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake")
    expect(page.locator(".target .mdl .name")).to_have_text("fake")
    department = question(page, "department")
    first = department.locator(".ins").input_value()
    department.locator(".ins").click()
    run(page, 1)
    department.locator(".ins").fill("Which team should take this ticket")
    run(page, 2)
    level = question(page, "frustration").locator("[data-item] [data-cell='description']").first
    level.fill("Calm, only facts")
    department.locator(".add-row .btn").click()
    run(page, 3)
    return first


def history_row(page, number: int):
    return page.locator(".hrow", has=page.locator(".n", has_text=re.compile(f"^#{number}$")))


def test_answer_cards_show_how_they_moved_and_what_was_reworded(studio, page):
    three_runs(page, studio)
    chips = page.locator(".acard[data-q='department'] .chip")
    expect(chips).to_have_text(["−0.10 since #2", "Reworded in #3"])
    expect(page.locator(".acard[data-q='is_urgent'] .chip")).to_have_count(0)


def test_history_shows_each_runs_diff_and_how_the_answers_moved(studio, page):
    three_runs(page, studio)
    # Reworded in #3 opens the History at run #3.
    page.locator(".acard[data-q='frustration'] .chip.link").click()
    rows = page.locator(".hist-list .hrow")
    expect(rows).to_have_count(3)
    assert rows.evaluate_all(ROWS) == [
        "#3 Changed department, frustration",
        "#2 Changed department",
        "#1 First run",
    ]
    expect(page.locator(".hrow[aria-selected=true] .n")).to_have_text("#3")

    lines = page.locator(".diff .dline").evaluate_all(
        "ds => ds.map(d => d.querySelector('.where').innerText + ': '"
        " + d.querySelector('.txt').innerText)"
    )
    assert any(line.startswith("frustration.level 1:") for line in lines), lines
    assert any(line.startswith("department.option_4: added") for line in lines), lines
    words = page.locator(".diff del, .diff ins").evaluate_all(WORDS)
    assert "DEL:just stating" in words and "INS:only" in words, words
    shot(page, "history-run3")
    assert page.locator(".move").evaluate_all(MOVES) == [
        "department 0.40 −0.10 vs #2",
        "frustration 1.67 +0.00 vs #2",
        "is_urgent 0.75 +0.00 vs #2",
    ]
    # The score question gets its levels per run.
    expect(page.locator(".stack-row")).to_have_count(3)

    # Run #2 reworded department's instructions.
    history_row(page, 2).click()
    expect(page.locator(".hrow[aria-selected=true] .n")).to_have_text("#2")
    expect(page.locator(".diff .dline .where")).to_have_text(["department.instructions"])
    assert page.locator(".diff del, .diff ins").evaluate_all(WORDS) == [
        "DEL:handle",
        "INS:take",
        "INS: ticket",
    ]


def test_restore_puts_a_runs_questions_back_in_the_editor(studio, page):
    first = three_runs(page, studio)
    inspector_tab(page, "History")
    history_row(page, 1).click()
    expect(page.locator(".diff h4")).to_contain_text("First run")
    page.locator(".diff-head .btn").click()

    department = question(page, "department")
    expect(department.locator(".ins")).to_have_value(first)
    expect(department.locator(".opt-key")).to_have_count(3)
    level = question(page, "frustration").locator("[data-cell='description']").first
    expect(level).to_have_value("Calm, just stating facts")
    # Run #1 asked what the saved file holds, so nothing is edited, and it is
    # in the editor now: no Restore.
    expect(page.locator(".edited-tag")).to_have_count(0)
    expect(page.locator(".diff-head .btn")).to_have_count(0)

    history_row(page, 3).click()
    page.locator(".diff-head .btn").click()
    expect(page.locator(".edited-tag")).to_be_visible()
    expect(department.locator(".opt-key")).to_have_count(4)


def test_after_a_reload_the_answers_come_from_the_history_file(studio, page):
    three_runs(page, studio)
    lines = studio.history_lines("ticket-triage")
    assert len(lines) == 3
    assert all(line["model"] == "fake" and line["response"]["answers"] for line in lines)

    # The edits are unsaved, so leaving asks first: go anyway.
    page.on("dialog", lambda dialog: dialog.accept())
    page.reload()
    expect(question(page, "department")).to_be_visible()
    inspector_tab(page, "Answers")
    expect(page.locator(".run-head h3")).to_have_text(re.compile("^Run #3"))
    expect(page.locator(".acard .verdict")).to_have_count(3)


def test_clear_history_asks_first_and_deletes_the_file(studio, page):
    three_runs(page, studio)
    inspector_tab(page, "History")
    page.locator(".hist-foot .btn").click()
    expect(page.locator(".sheet h2")).to_contain_text("Clear the history of ticket-triage")
    page.locator(".sheet-foot .btn", has_text=re.compile("^Clear History$")).click()
    expect(page.locator(".insp-body .empty")).to_contain_text("No runs of this set yet")
    assert not (studio.history / "ticket-triage.jsonl").exists()
