"""The answers map in the Playground on a dataset: "Answered by", each
value's answer (an option, yes / no, a level, left out), Options from Values
with the names a dataset card gives, pasted names, the values not mapped yet,
and Run All's grid with each row's expected answer and the agreement. The
dataset has the shape of zeroshot/twitter-financial-news-topic: a label of 20
integers whose names only its card lists."""

import json

import pytest
from drive import open_studio, pick, pick_model, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TOPICS = [
    "Analyst Update", "Fed | Central Banks", "Company | Product News",
    "Treasuries | Corporate Debt", "Dividend", "Earnings", "Energy | Oil", "Financials",
    "Currencies", "General News | Opinion",
    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro",
    "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement",
]  # fmt: skip


@pytest.fixture
def tweets(studio):
    """data/tweets/: a train split of 40 rows, two for each of the 20 values,
    named from the card; no questions yet."""
    folder = studio.workspace / "data" / "tweets"
    folder.mkdir()
    rows = [
        {
            "id": f"t{i}",
            "state": f"tweet {i} about {TOPICS[i % 20]}",
            "values": {"label": str(i % 20)},
        }
        for i in range(40)
    ]
    (folder / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    values = {str(i): {"rows": 2, "name": name} for i, name in enumerate(TOPICS)}
    body = {"columns": {"label": {"values": values}}, "splits": {"train": {"role": "train"}},
            "source": {"source": "hf:zeroshot/twitter-financial-news-topic"}}  # fmt: skip
    (folder / "dataset.json").write_text(json.dumps(body))
    studio.folder = folder
    return studio


def described(studio):
    return json.loads((studio.folder / "dataset.json").read_text())


def add_choice(page):
    pick(page, ".add-bar .btn", "Choice")
    question = page.locator(".question").last
    question.locator(".ins").fill("What is the tweet about?")
    return question


def test_options_from_the_cards_names(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    question = add_choice(page)
    answered = question.get_by_label("Answered by, choice")
    expect(answered).to_have_value("")  # asked, not trained, until a column answers it
    answered.select_option("label")
    rows = question.locator(".am-row")
    expect(rows).to_have_count(20)
    expect(question.locator(".am-open")).to_contain_text("No value is mapped yet")
    question.get_by_role("button", name="Options from Values").click()
    keys = question.get_by_label("Option key")
    expect(keys).to_have_count(20)
    expect(keys.nth(1)).to_have_value("fed_central_banks")
    expect(question.get_by_label("Description of fed_central_banks")).to_have_value(
        "Fed | Central Banks"
    )
    expect(question.get_by_label("Answer for 13", exact=True)).to_have_value(
        "o:" + keys.nth(13).evaluate("e => e.closest('[data-item]').dataset.item")
    )
    expect(question.locator(".am-open")).to_have_count(0)
    shot(page, "dataset-answers-options")
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved data/tweets/dataset.json")
    saved = described(tweets)
    assert saved["answers"]["choice"]["map"]["13"] == "m_a_investments"
    assert len(saved["questions"]["choice"]["criteria"]) == 20
    expect(page.locator(".ds-banner")).to_have_count(0)  # ready for a fit


def test_merge_leave_out_and_yes_no(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    # A yes / no question from two values, the rest no, one left out.
    pick(page, ".add-bar .btn", "Noul")
    question = page.locator(".question").last
    question.locator(".ins").fill("Is it about earnings or dividends?")
    question.get_by_label("Answered by, noul").select_option("label")
    for value in map(str, range(20)):
        question.get_by_label(f"Answer for {value}", exact=True).select_option("no")
    question.get_by_label("Answer for 4", exact=True).select_option("yes")
    question.get_by_label("Answer for 5", exact=True).select_option("yes")
    question.get_by_label("Answer for 9", exact=True).select_option("out")
    expect(question.locator(".am-open")).to_have_count(0)
    # Renaming the question keeps its map.
    question.get_by_label("Question id").fill("payout")
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved")
    answer = described(tweets)["answers"]["payout"]
    assert answer["map"]["4"] is True and answer["map"]["0"] is False
    assert answer["left_out"] == ["9"] and "9" not in answer["map"]
    # A value not mapped is said, and what a fit misses after a save.
    question.get_by_label("Answer for 7", exact=True).select_option("")
    expect(question.locator(".am-open")).to_have_text(
        "1 value is not mapped: 7. Map each to an answer or leave it out."
    )
    page.keyboard.press("Meta+s")
    expect(page.locator(".ds-banner")).to_contain_text("payout: label values not mapped")


def test_pasted_names_and_a_merge(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    question = add_choice(page)
    question.get_by_label("Answered by, choice").select_option("label")
    question.get_by_role("button", name="Paste Names…").click()
    sheet = page.locator(".sheet")
    sheet.get_by_label("Names").fill(
        "\n".join(f'"LABEL_{i}": "{name}",' for i, name in enumerate(TOPICS))
    )
    sheet.get_by_role("button", name="Make Options").click()
    keys = question.get_by_label("Option key")
    expect(keys.first).to_have_value("analyst_update")
    # Two values on one option: the stock ones.
    movement = question.get_by_label("Answer for 19", exact=True)
    commentary_option = question.get_by_label("Answer for 18", exact=True).input_value()
    movement.select_option(commentary_option)
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved")
    mapped = described(tweets)["answers"]["choice"]["map"]
    assert mapped["18"] == mapped["19"] == "stock_commentary"


def test_run_all_shows_the_expected_answers_and_agreement(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    pick_model(page, "fake")
    pick(page, ".add-bar .btn", "Noul")
    question = page.locator(".question").last
    question.locator(".ins").fill("Is it about the Fed?")
    question.get_by_label("Answered by, noul").select_option("label")
    for value in map(str, range(20)):
        question.get_by_label(f"Answer for {value}", exact=True).select_option(
            "yes" if value == "1" else "no"
        )
    page.locator(".run-btn").click()
    grid = page.locator(".ex-grid")
    expect(grid.locator("tbody tr")).to_have_count(40, timeout=20_000)
    expect(page.locator(".toolbar .activity")).to_contain_text(
        "answered 40 examples", timeout=20_000
    )
    expect(grid.locator("thead small.agree")).to_contain_text(" of 40 agree")
    expect(grid.locator("tbody tr").first.locator("td .x")).to_have_text("expects no")
    expect(grid.locator("tbody tr", has_text="t1").first.locator("td .x")).to_have_text(
        "expects yes"
    )
    assert grid.locator("td.agree, td.disagree").count() == 40
    shot(page, "dataset-answers-grid")


def test_a_score_maps_values_to_levels(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    pick(page, ".add-bar .btn", "Score")
    question = page.locator(".question").last
    question.locator(".ins").fill("How much does it move the market?")
    question.get_by_label("Answered by, score").select_option("label")
    levels = question.get_by_label("Answer for 0", exact=True).locator("option")
    expect(levels.nth(2)).to_contain_text("Level 1")  # after not mapped and Leave out
    for value in map(str, range(20)):
        question.get_by_label(f"Answer for {value}", exact=True).select_option(
            index=2 if value != "19" else 3
        )
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved")
    answer = described(tweets)["answers"]["score"]
    assert answer["map"]["0"] == 0 and answer["map"]["19"] == 1
