"""The Playground on a dataset: opened by Write Questions, its questions
edited as a set's and saved into its dataset.json, its rows (about two per
label value, Shuffle, a search, another split) in place of the examples,
read-only with their values; Run All and the history; Start from a Question
Set and Save as Question Set."""

import json
import re

import pytest
from drive import open_studio, pick, pick_model, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TOPIC = {"type": "choice", "instructions": "What is the tweet about?",
         "criteria": {"analyst": "Analyst update", "fed": "Central banks"}}  # fmt: skip


def write_tweets(studio, questions=False):
    """data/tweets/: train (8 rows: values 0, 1, 2, 3 twice each) and test (4);
    with ``questions``, a topic question answered by values 0 and 1."""
    folder = studio.workspace / "data" / "tweets"
    folder.mkdir()
    for split, count in (("train", 8), ("test", 4)):
        rows = [
            {"id": f"{split}{i}", "state": f"tweet {split} {i}", "values": {"label": str(i % 4)}}
            for i in range(count)
        ]
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    names = ["Analyst Update", "Fed | Central Banks", "Dividend", "Earnings"]
    values = {str(i): {"rows": 3, "name": n} for i, n in enumerate(names)}
    splits = {
        "train": {"role": "train", "examples": 8},
        "test": {"role": "evaluate", "examples": 4},
    }
    body = {"columns": {"label": {"values": values}}, "splits": splits,
            "source": {"source": "hf:zeroshot/tweets"}}  # fmt: skip
    if questions:
        body["questions"] = {"topic": TOPIC}
        body["answers"] = {"topic": {"column": "label", "map": {"0": "analyst", "1": "fed"},
                                     "left_out": ["2", "3"]}}  # fmt: skip
    (folder / "dataset.json").write_text(json.dumps(body))
    return folder


def described(folder):
    return json.loads((folder / "dataset.json").read_text())


def test_write_questions_on_a_dataset(studio, page):
    folder = write_tweets(studio)
    open_studio(page, studio, "datasets/tweets")
    page.locator(".ds-banner").get_by_role("button", name="Write Questions").click()
    expect(page).to_have_url(re.compile(r"#/playground/data%3Atweets$"))
    centre = page.locator(".centre")
    expect(centre.locator(".pane-head h1")).to_have_text("tweets")
    expect(centre.locator(".pane-head p")).to_contain_text("data/tweets/dataset.json")
    expect(centre.locator(".ds-banner")).to_contain_text("it has no questions yet")
    expect(
        page.locator(".sidebar .sb-sec", has_text="Dataset Questions").locator(".sb-item")
    ).to_have_text("tweets")
    expect(page.locator(".target .set")).to_contain_text("tweets")
    # Its rows: about two per value, each with its value; the state read-only.
    chips = page.locator(".examples.rows .ex-chip")
    expect(chips).to_have_count(8)
    expect(page.locator(".rows-bar")).to_contain_text(
        "8 rows of 8, drawn across the values of label"
    )
    expect(chips.first).to_contain_text("0 · Analyst Update")
    state = page.locator(".state-text")
    expect(state).not_to_be_editable()
    expect(state).to_have_value("tweet train 0")
    expect(page.locator(".row-values")).to_contain_text("label 0 · Analyst Update")
    chips.nth(1).locator("button").click()
    expect(state).to_have_value("tweet train 4")  # value 0's other row
    expect(page.locator(".question")).to_have_count(0)
    # A question, as in a set; Save writes it into dataset.json.
    pick(page, ".add-bar .btn", "Noul")
    page.locator(".question[data-q='noul'] .ins").fill("Is it about the Fed?")
    expect(centre.locator(".edited-tag")).to_be_visible()
    shot(page, "dataset-questions")
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved data/tweets/dataset.json")
    expect(centre.locator(".edited-tag")).to_have_count(0)
    assert described(folder)["questions"] == {
        "noul": {"type": "noul", "instructions": "Is it about the Fed?"}
    }
    # Answered by the dataset's value column at once, its values not mapped yet:
    # what a fit still needs.
    expect(centre.locator(".ds-banner")).to_contain_text(
        "noul: label values not mapped to an answer or left out: 0, 1, 2, 3"
    )
    page.goto(f"{studio.base}/#/datasets/tweets")
    expect(page.locator(".ds-question", has_text="noul")).to_be_visible()
    expect(page.locator(".ds-banner").get_by_role("button")).to_have_text("Edit Questions")


def test_rows_shuffle_search_and_split(studio, page):
    write_tweets(studio)
    open_studio(page, studio, "playground/data:tweets")
    chips = page.locator(".examples.rows .ex-chip")
    expect(chips).to_have_count(8)
    page.get_by_label("Search the rows").fill("train 3")
    expect(chips).to_have_count(1)
    expect(page.locator(".rows-bar")).to_contain_text("1 rows hold it")
    expect(page.locator(".state-text")).to_have_value("tweet train 3")
    page.get_by_role("button", name="Shuffle").click()  # the search goes
    expect(chips).to_have_count(8)
    page.get_by_label("Split").select_option("test")
    expect(chips).to_have_count(4)
    expect(page.locator(".state-text")).to_have_value("tweet test 0")


def test_run_all_on_the_rows_keeps_a_history(studio, page):
    write_tweets(studio, questions=True)
    open_studio(page, studio, "playground/data:tweets")
    pick_model(page, "fake")
    expect(page.locator(".run-btn")).to_have_text("Run All")
    page.locator(".run-btn").click()
    expect(page.locator(".ex-grid tbody tr")).to_have_count(8, timeout=15_000)
    expect(page.locator(".toolbar .activity")).to_contain_text("answered 8 rows", timeout=15_000)
    history = studio.workspace / "history" / "datasets" / "tweets.jsonl"
    lines = history.read_text().splitlines()
    assert len(lines) == 8 and json.loads(lines[0])["request"]["state"] == "tweet train 0"
    assert not (studio.workspace / "history" / "data:tweets.jsonl").exists()


def test_a_renamed_option_keeps_its_mapping_a_removed_one_drops_it(studio, page):
    folder = write_tweets(studio, questions=True)
    open_studio(page, studio, "playground/data:tweets")
    expect(page.locator(".ds-banner")).to_have_count(0)  # every value mapped or left out
    topic = page.locator(".question[data-q='topic']")
    topic.get_by_label("Option key").first.fill("analysts")
    page.keyboard.press("Meta+s")
    expect(page.locator(".toast")).to_contain_text("Saved data/tweets/dataset.json")
    assert described(folder)["answers"]["topic"]["map"] == {"0": "analysts", "1": "fed"}
    topic.get_by_role("button", name="Remove option fed").click()
    page.keyboard.press("Meta+s")
    expect(page.locator(".ds-banner")).to_contain_text(
        "topic: label values not mapped to an answer or left out: 1"
    )
    assert described(folder)["answers"]["topic"]["map"] == {"0": "analysts"}


def test_start_from_a_set_and_save_as_a_set(studio, page):
    write_tweets(studio)
    open_studio(page, studio, "playground/data:tweets")
    pick(
        page,
        page.get_by_role("button", name="More actions for this dataset"),
        "Start from a Question Set…",
    )
    sheet = page.locator(".sheet")
    sheet.get_by_label("Question set").select_option("ticket-triage")
    sheet.get_by_role("button", name="Use Its Questions").click()
    expect(page.locator(".question")).to_have_count(3)
    expect(page.locator(".centre .edited-tag")).to_be_visible()
    expect(page.locator(".examples.rows .ex-chip")).to_have_count(8)  # the rows stay
    pick(
        page,
        page.get_by_role("button", name="More actions for this dataset"),
        "Save as Question Set…",
    )
    sheet.get_by_label("Name").fill("tweets-triage")
    sheet.get_by_role("button", name="Save").click()
    expect(page).to_have_url(re.compile(r"#/playground/tweets-triage$"))
    saved = studio.set_file("tweets-triage")
    assert list(saved["questions"]) == ["department", "frustration", "is_urgent"]
    assert sorted(item["name"] for item in saved["examples"]) == [f"train{i}" for i in range(8)]
    # The dataset keeps its edits, unsaved.
    expect(
        page.locator(".sidebar .sb-sec", has_text="Dataset Questions").locator(".edited")
    ).to_have_count(1)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_the_page_light_and_dark(studio, page, theme):
    write_tweets(studio, questions=True)
    page.set_viewport_size({"width": 1100, "height": 900})
    page.emulate_media(color_scheme=theme)
    open_studio(page, studio, "playground/data:tweets")
    expect(page.locator(".examples.rows .ex-chip")).to_have_count(8)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    shot(page, f"dataset-questions-{theme}")
