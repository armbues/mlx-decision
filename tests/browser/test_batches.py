"""The Batches page: New Batch and Run as Batch…, a batch's progress, answers,
metrics and rows (filtered, searched, picked), Stop and Run the Rest from the
page and the sidebar, export, Save Answers as a Dataset, Delete."""

import json
import re

import pytest
from drive import menu_item, open_studio, put_set
from playwright.sync_api import expect

from mlx_decision.data.dataset import Dataset, SplitOut, open_dataset, write_dataset
from mlx_decision.data.format import Example

pytestmark = pytest.mark.browser

TOPIC = {
    "type": "choice",
    "instructions": "What is the tweet about?",
    "criteria": {"analyst": "Analyst update", "fed": "Central banks", "other": "Other"},
}
ANSWERS = {"topic": {"column": "label", "map": {"0": "analyst", "1": "fed", "2": "other"}}}


def write_tweets(data, name="tweets", questions=True, count=30):
    """A dataset folder: train (``count`` rows) and test (9), the label's values
    0, 1, 2 answering ``topic`` (with ``questions``)."""

    def rows(split, n):
        return [
            Example(f"{split}/{i}", {"state": f"tweet {split} {i}"}, {}, {"label": str(i % 3)})
            for i in range(n)
        ]

    values = {"label": {"values": {v: {"rows": 10} for v in "012"}}}
    dataset = Dataset(
        data / name,
        questions={"topic": TOPIC} if questions else None,
        answers=ANSWERS if questions else {},
        columns=values,
    )
    outs = [
        SplitOut("train", rows("train", count), {"role": "train", "examples": count}),
        SplitOut("test", rows("test", 9), {"role": "evaluate", "examples": 9}),
    ]
    write_dataset(data / name, dataset, outs)


def new_batch(page, dataset="tweets", split="train", model="fake", questions=None, sample=None):
    page.locator(".sidebar").get_by_role("button", name="New Batch").click()
    sheet = page.locator(".sheet")
    sheet.locator("#batch-data").select_option(dataset)
    sheet.locator("#batch-split").select_option(split)
    if questions:
        sheet.locator("#batch-questions").select_option(questions)
    if sample:
        sheet.locator("#batch-size").fill(str(sample))
    sheet.locator("#batch-model").select_option(model)
    sheet.get_by_role("button", name="Start").click()
    expect(sheet).to_have_count(0)


def done(page, text=None):
    expect(page.locator(".batch-progress")).to_have_class(re.compile(r"\bdone\b"), timeout=30_000)
    if text:
        expect(page.locator(".batch-progress")).to_contain_text(text)


@pytest.fixture
def tweets(studio):
    write_tweets(studio.workspace / "data")
    return studio


def test_an_empty_page_explains_batches(studio, page):
    open_studio(page, studio, "batches")
    expect(page.locator(".batch-empty")).to_contain_text("No batches yet")
    expect(page.locator(".toolbar")).to_contain_text("No batches yet")
    expect(page.locator(".sidebar .sb-item", has_text="Batches")).to_have_attribute(
        "aria-current", "page"
    )


def test_a_batch_shows_its_answers_metrics_and_rows(tweets, page):
    open_studio(page, tweets, "batches")
    new_batch(page)
    expect(page).to_have_url(re.compile(r"#/batches/tweets-fake$"))
    done(page, "30 answered")
    head = page.locator(".pane-head")
    expect(head).to_contain_text("tweets")
    expect(head).to_contain_text("30 rows")
    expect(head).to_contain_text("its own questions")
    question = page.locator(".batch-q[data-q=topic]")
    expect(question.locator(".tag")).to_have_text("choice · with labels")
    expect(question.locator(".metric", has_text="Accuracy")).to_be_visible()
    expect(question.locator(".confusion-table")).to_be_visible()
    expect(question.locator(".reliability svg .rel-bar").first).to_be_visible()
    rows = page.locator(".batch-rows .ex-rows .batch-row")
    expect(rows).to_have_count(30)
    expect(page.locator(".batch-rows .ex-pager")).to_contain_text("1–30 of 30")
    # A row picked: the inspector shows it whole, with each answer's probabilities.
    rows.nth(1).click()
    inspector = page.locator(".batch-inspector")
    expect(inspector.locator("h2")).to_have_text("train/1")
    expect(inspector).to_contain_text("tweet train 1")
    expect(inspector.locator(".prob")).to_have_count(3)
    expect(inspector).to_contain_text("Expected")
    # The command that answers the same rows from a terminal.
    expect(page.locator(".batch-cli")).to_contain_text(
        "-q batches/tweets-fake/questions.json --states batches/tweets-fake/rows.jsonl --json"
    )
    # The batch in the sidebar's list, done.
    expect(page.locator(".batch-list .sb-item", has_text="tweets-fake")).to_have_attribute(
        "aria-current", "true"
    )


def test_the_rows_filter_by_wrong_answer_cell_and_search(tweets, page):
    open_studio(page, tweets, "batches")
    new_batch(page)
    done(page)
    expect(page.locator(".metric", has_text="Labelled")).to_contain_text("30")  # the final results
    wrong = int(re.search(r"(\d+) wrong", page.locator(".bq-counts").inner_text()).group(1))
    table = page.locator(".batch-rows")
    table.get_by_role("tab", name="Wrong").click()
    rows = table.locator(".ex-rows .batch-row")
    expect(rows).to_have_count(wrong)
    expect(table.locator(".batch-expected.right")).to_have_count(0)
    table.get_by_role("tab", name="All").click()
    expect(rows).to_have_count(30)
    # A confusion matrix's cell: the rows with that expected answer and that answer.
    cell = page.locator(".confusion-table .cm-cell").first
    count = int(cell.inner_text())
    cell.click()
    expect(table.locator(".chip")).to_contain_text("expected")
    expect(rows).to_have_count(count)
    table.locator(".chip button").click()
    # Search in the states.
    table.locator(".ex-search").fill("train 2")
    expect(rows).to_have_count(11)  # 2, 20 to 29


def test_run_as_batch_from_the_datasets_page(tweets, page):
    open_studio(page, tweets, "datasets/tweets")
    page.locator(".ds-tabs [role=tab][data-split=test]").click()
    page.get_by_role("button", name="More actions for tweets").click()
    menu_item(page, "Run as Batch…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("#batch-data")).to_have_value("tweets")
    expect(sheet.locator("#batch-split")).to_have_value("test")
    expect(sheet.locator("#batch-questions")).to_have_value("")  # its own
    sheet.get_by_role("button", name="Start").click()
    expect(page).to_have_url(re.compile(r"#/batches/tweets-fake$"))
    done(page, "9 answered")


def test_a_dataset_without_questions_runs_a_sets_questions(studio, page):
    write_tweets(studio.workspace / "data", "raw", questions=False)
    open_studio(page, studio, "batches")
    put_set(page, studio, "topics", {"state": "", "questions": {"topic": TOPIC}})
    page.locator(".sidebar").get_by_role("button", name="New Batch").click()
    sheet = page.locator(".sheet")
    sheet.locator("#batch-data").select_option("raw")
    sheet.locator("#batch-split").select_option("test")
    expect(sheet.locator("#batch-questions option", has_text="Its own")).to_have_count(0)
    sheet.locator("#batch-questions").select_option("topics")
    expect(sheet.locator(".hint")).to_contain_text("Answers only")
    sheet.get_by_role("button", name="Start").click()
    done(page, "9 answered")
    expect(page.locator(".pane-head")).to_contain_text("the set topics")
    expect(page.locator(".batch-q .bq-note")).to_contain_text("No labels for this question")
    expect(page.locator(".batch-rows").get_by_role("tab", name="Wrong")).to_have_count(0)


def test_stop_from_the_sidebar_and_run_the_rest(make_studio, page):
    studio = make_studio(slow=True)
    write_tweets(studio.workspace / "data")
    open_studio(page, studio, "batches")
    new_batch(page, model="fake-slow", sample=3)
    job = page.locator(".sidebar .batch-job")
    expect(job).to_contain_text("Batch tweets-fake-slow")
    expect(job.locator("small")).to_contain_text("1 of 3", timeout=10_000)
    job.get_by_role("button", name="Stop the batch tweets-fake-slow").click()
    expect(page.locator(".batch-progress")).to_have_class(
        re.compile(r"\bstopped\b"), timeout=10_000
    )
    expect(page.locator(".sidebar .batch-job")).to_have_count(0)
    expect(page.locator(".batch-progress")).to_contain_text("Stopped at")
    page.locator(".pane-head").get_by_role("button", name="Run the Rest").click()
    done(page, "3 answered")
    assert (
        len(
            json.loads(
                page.request.get(f"{studio.base}/studio/api/batches/tweets-fake-slow").text()
            )["batch"]["passes"]
        )
        == 2
    )


def test_export_save_answers_and_delete(tweets, page):
    open_studio(page, tweets, "batches")
    new_batch(page, split="test")
    done(page)
    page.get_by_role("button", name="Export").click()
    with page.expect_download() as download:
        menu_item(page, "CSV").click()
    assert download.value.suggested_filename == "tweets-fake.csv"
    # Save Answers as a Dataset: a new dataset, opened on the Datasets page.
    page.get_by_role("button", name="More actions for tweets-fake").click()
    menu_item(page, "Save Answers as a Dataset…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("#save-name")).to_have_value("tweets-by-fake")
    sheet.get_by_role("button", name="Save").click()
    expect(page).to_have_url(re.compile(r"#/datasets/tweets-by-fake$"))
    saved = page.locator(".toast", has_text="Saved")
    expect(saved).to_contain_text("Saved 9 examples as tweets-by-fake")
    dataset, _ = open_dataset(tweets.workspace / "data" / "tweets-by-fake")
    assert dataset.missing() == [] and dataset.source["batch"] == "tweets-fake"
    # Delete.
    open_studio(page, tweets, "batches/tweets-fake")
    page.get_by_role("button", name="More actions for tweets-fake").click()
    menu_item(page, "Delete…").click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page).to_have_url(re.compile(r"#/batches$"))
    expect(page.locator(".batch-empty")).to_be_visible()
    assert not (tweets.workspace / "batches" / "tweets-fake").exists()
