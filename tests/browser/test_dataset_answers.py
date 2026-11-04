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
    # Answered by the dataset's value column at once, an option per value.
    expect(answered).to_have_value("label")
    rows = question.locator(".am-row")
    expect(rows).to_have_count(20)
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
    # It was missing its answers; now it is ready, and says so.
    expect(page.locator(".toast")).to_contain_text("Saved: tweets is ready for a fit")
    saved = described(tweets)
    assert saved["answers"]["choice"]["map"]["13"] == "m_a_investments"
    assert len(saved["questions"]["choice"]["criteria"]) == 20
    expect(page.locator(".ds-banner")).to_contain_text("Ready for a fit and a batch.")


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
    expect(page.locator(".toolbar .activity")).to_contain_text("answered 40 rows", timeout=20_000)
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


def mapped_yes_no(page):
    """A yes / no question answered by label: 1 yes, the rest no (Set the rest to)."""
    pick(page, ".add-bar .btn", "Noul")
    question = page.locator(".question").last
    question.locator(".ins").fill("Is it about the Fed?")
    question.get_by_label("Answered by, noul").select_option("label")
    question.get_by_label("Answer for 1", exact=True).select_option("yes")
    expect(question.locator(".am-open")).to_contain_text("19 values are not mapped")
    question.get_by_label("Set the rest of noul to").select_option("no")
    expect(question.locator(".am-open")).to_have_count(0)
    return question


def test_set_the_rest_and_restore_keeps_the_map(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    pick_model(page, "fake")
    question = mapped_yes_no(page)
    page.locator(".run-btn").click()
    expect(page.locator(".toolbar .activity")).to_contain_text("answered 40 rows", timeout=20_000)
    question.locator(".ins").fill("Is it about central banks?")
    page.locator(".inspector").get_by_role("tab", name="History").click()
    page.locator(".hist-list .hrow").first.click()
    page.get_by_role("button", name="Restore These Questions").click()
    expect(question.locator(".ins")).to_have_value("Is it about the Fed?")
    expect(question.get_by_label("Answered by, noul")).to_have_value("label")
    expect(question.get_by_label("Answer for 1", exact=True)).to_have_value("yes")
    expect(question.get_by_label("Answer for 0", exact=True)).to_have_value("no")
    expect(page.locator(".hist-foot")).to_contain_text("history/datasets/tweets.jsonl")


def test_a_type_change_carries_the_map(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    question = mapped_yes_no(page)
    pick(page, question.locator(".gsec-head .right button"), "Choice")
    # yes and no become options, in that order: the values follow.
    yes = question.locator("[data-item]").first.get_attribute("data-item")
    expect(question.get_by_label("Answer for 1", exact=True)).to_have_value(f"o:{yes}")
    expect(question.locator(".am-open")).to_have_count(0)


def test_a_save_while_a_job_reads_the_dataset_says_so(tweets, page):
    from types import SimpleNamespace

    open_studio(page, tweets, "playground/data:tweets")
    mapped_yes_no(page)
    jobs = tweets.app.state.jobs.jobs
    jobs["fit"] = SimpleNamespace(state="running", details={"data": "tweets"}, title="Fit x")
    try:
        page.keyboard.press("Meta+s")
        expect(page.locator(".toast").last).to_contain_text("Fit x reads tweets")
        expect(page.locator(".sheet")).to_have_count(0)  # no "changed on disk" sheet
    finally:
        del jobs["fit"]


def test_get_code_on_a_dataset_names_its_file(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    mapped_yes_no(page)
    pick_model(page, "fake")
    page.get_by_role("button", name="Get Code").first.click()
    sheet = page.locator(".sheet")
    sheet.get_by_role("tab", name="Command Line").click()
    expect(sheet).to_contain_text("Reads the questions from data/tweets/dataset.json")
    expect(sheet).to_contain_text("tweets/dataset.json")


def test_an_open_dataset_follows_a_save_elsewhere(tweets, page):
    open_studio(page, tweets, "playground/data:tweets")
    expect(page.locator(".question")).to_have_count(0)
    body = {"questions": {"fed": {"type": "noul", "instructions": "Fed?"}},
            "answers": {"fed": {"column": "label", "map": {"1": True},
                                "left_out": [str(i) for i in range(20) if i != 1]}}}  # fmt: skip
    saved = page.evaluate(
        """async (body) => (await fetch('/studio/api/data/tweets/questions',
            {method: 'PUT', headers: {'Content-Type': 'application/json'},
             body: JSON.stringify(body)})).status""",
        body,
    )
    assert saved == 200
    expect(page.locator(".question[data-q='fed']")).to_be_visible(timeout=10_000)


def test_an_open_dataset_follows_its_rows_written_again(tweets, page):
    # Downloaded again with the same dataset.json: the rows shown follow.
    open_studio(page, tweets, "playground/data:tweets")
    state = page.locator(".state-text")
    expect(state).to_have_value("tweet 0 about Analyst Update")
    split = tweets.folder / "train.jsonl"
    split.write_text(split.read_text().replace("tweet ", "post "))
    expect(state).to_have_value("post 0 about Analyst Update", timeout=10_000)
