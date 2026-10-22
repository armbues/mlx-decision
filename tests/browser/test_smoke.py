"""The Studio's smoke test: ``web`` on the fake backend, a set made in the page,
a question edited and run, the answer seen, the files written."""

import json

import pytest

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_edit_a_question_run_it_and_see_the_answer(fake_web, page):
    page.goto(fake_web.url)
    expect(page.locator(".centre .empty").get_by_text("No question sets yet")).to_be_visible()

    # A new set, named in the sheet.
    page.get_by_role("button", name="New Question Set…").first.click()
    page.locator(".sheet-input").fill("smoke")
    page.keyboard.press("Enter")
    page.wait_for_url("**/#/playground/smoke")

    # A state and a noul question.
    page.locator(".state-text").fill("My card was charged twice this month, please refund one.")
    page.locator(".add-bar .btn").click()
    page.locator(".menu .mi", has_text="Noul").click()
    question = page.locator(".question[data-q='noul']")
    question.locator(".ins").fill("The customer asks for money back")

    # Run (⌘↩): the answer card shows the fake model's yes at 0.75.
    expect(page.locator(".run-btn")).to_be_enabled()
    page.keyboard.press("Meta+Enter")
    card = page.locator(".acard[data-q='noul']")
    expect(card.locator(".verdict")).to_contain_text("Yes")
    expect(card.locator(".verdict")).to_contain_text("0.75 probability of yes")
    expect(page.locator(".activity")).to_contain_text("fake answered")

    # Editing the question marks its card; running again answers the edit.
    question.locator(".ins").fill("Does the customer want a refund?")
    expect(page.locator(".acard.stale[data-q='noul']")).to_be_visible()
    page.locator(".run-btn").click()
    expect(page.locator(".acard.stale[data-q='noul']")).to_have_count(0)
    expect(card.locator(".verdict")).to_contain_text("Yes")

    # Save (⌘S): the set's file, and the two runs in its history.
    page.keyboard.press("Meta+s")
    expect(page.locator(".edited-tag")).to_have_count(0)
    saved = json.loads((fake_web.workspace / "sets" / "smoke.json").read_text())
    assert saved["questions"] == {
        "noul": {"type": "noul", "instructions": "Does the customer want a refund?"}
    }
    assert saved["state"].startswith("My card was charged twice")
    runs = (fake_web.workspace / "history" / "smoke.jsonl").read_text().splitlines()
    asked = [json.loads(line)["request"]["questions"]["noul"]["instructions"] for line in runs]
    assert asked == ["The customer asks for money back", "Does the customer want a refund?"]
