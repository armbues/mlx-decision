"""Get a Dataset: the presets with their question and splits, a sample and a
name, and the job whose file joins the Datasets list. The job is
``tests/fake_job.py`` (no network)."""

import re

import pytest
from drive import open_studio
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_get_a_dataset(studio, page):
    fake_jobs(studio.app, steps=5, every=0.05)
    open_studio(page, studio, "datasets")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator(".fam b")).to_have_text(
        ["TREC", "TweetEval Offensive", "SST-5", "OpenBookQA", "ANLI round 3", "Another dataset…"]
    )
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("TREC")
    expect(sheet.locator(".ds-options li")).to_have_count(6)
    expect(sheet.locator(".crow[data-split='train']")).to_contain_text("5,452 rows")
    expect(sheet.locator(".crow[data-split='test']")).to_contain_text("keep it for training eval")
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("trec-train")

    sheet.locator(".fam[data-preset='sst5']").click()
    expect(sheet.locator(".crow[aria-checked='true']")).to_have_attribute("data-split", "train")
    expect(name).to_have_value("sst5-train")
    sheet.locator(".crow[data-split='validation']").click()
    expect(name).to_have_value("sst5-validation")
    sheet.get_by_role("spinbutton", name="Rows").fill("5000")  # more than validation's 1,101 rows
    expect(sheet.locator(".warn-text")).to_have_text("A sample is 1 to 1101 rows")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    sheet.get_by_role("spinbutton", name="Rows").fill("300")
    sheet.get_by_role("button", name="Get").click()

    expect(page.locator(".toast")).to_contain_text("Get sst5 validation (300 rows)")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details == {"preset": "sst5", "split": "validation", "sample": 300, "seed": 0,
                           "name": "sst5-validation"}  # fmt: skip
    assert job.argv[:5] == ["data", "--preset", "sst5", "--split", "validation"]
    # Done: the file joins the list, in this tab and the job's card says so.
    items = page.locator(".sidebar .data-list .sb-item")
    expect(items).to_have_text(["sst5-validation", "tickets"], timeout=15_000)
    expect(page.locator(".sidebar .job small")).to_have_text("Done")


def test_a_name_taken_is_said(studio, page):
    open_studio(page, studio, "datasets")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    sheet.get_by_label("Name").fill("tickets")
    expect(sheet.locator(".warn-text")).to_have_text("A data file named tickets exists already")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    expect(sheet.locator(".sheet-foot")).to_contain_text(
        re.compile("mlx-decision data does the same")
    )


@pytest.fixture
def hub(monkeypatch):
    """A Hub stand-in: org/reviews, configs en and de, splits train and test."""
    import datasets

    from mlx_decision.data import sources

    features = datasets.Features({
        "review": datasets.Value("string"),
        "sentiment": datasets.ClassLabel(names=["negative", "positive"]),
    })  # fmt: skip
    rows = {"review": [f"review {i}" for i in range(12)], "sentiment": [i % 2 for i in range(12)]}
    monkeypatch.setattr(sources, "hub_configs", lambda repo, revision=None: ["en", "de"])
    monkeypatch.setattr(
        sources, "hub_splits", lambda repo, config=None, revision=None: ["train", "test"]
    )
    monkeypatch.setattr(
        sources,
        "load_hub",
        lambda repo, config=None, split=None, revision=None: sources._from_dataset(
            datasets.Dataset.from_dict(rows, features=features), "reviews", split, {}
        ),
    )


def test_another_dataset_with_a_question_set(studio, page, hub):
    open_studio(page, studio, "datasets")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    sheet.locator(".fam[data-preset='any']").click()
    get = sheet.get_by_role("button", name="Get")
    expect(get).to_be_disabled()
    field = sheet.get_by_label("Hugging Face dataset")
    field.fill("org/reviews")
    field.press("Enter")  # reads; it does not get
    sheet.get_by_label("Config").select_option("en")
    sheet.get_by_label("Split").select_option("train")
    expect(sheet.locator(".any-table tbody tr")).to_have_count(5)
    expect(sheet.get_by_label("State column")).to_have_value("review")
    expect(sheet.get_by_label("Label column")).to_have_value("sentiment")
    sheet.get_by_label("Question set").select_option("ticket-triage")
    question = sheet.get_by_label("Question", exact=True)
    expect(question).to_have_value("department")  # the first choice or yes/no question
    expect(question.locator("option[value='frustration']")).to_have_attribute(
        "disabled", ""
    )  # a score
    # No class is named like an option: everything left out, and Get says why.
    expect(sheet.get_by_label("Option for negative")).to_have_value("")
    expect(sheet.locator(".fam-pane > .warn-text")).to_have_text(
        "Map at least one class to an option"
    )
    sheet.get_by_label("Option for negative").select_option("billing")
    sheet.get_by_label("Option for positive").select_option("technical")
    expect(sheet.get_by_label("Name")).to_have_value("reviews-train")
    expect(get).to_be_enabled()
    get.click()

    expect(page.locator(".toast")).to_contain_text("Get org/reviews train")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["label_map"] == {"negative": "billing", "positive": "technical"}
    assert job.details["question"] == "department" and job.details["set"] == "ticket-triage"
    written = (job.folder / "question.json").read_text()
    assert '"department"' in written and "Which team should handle this" in written
    assert "negative=billing,positive=technical" in job.argv


def test_a_yes_no_question_maps_by_name(studio, page, hub):
    open_studio(page, studio, "datasets")
    page.locator(".sidebar").get_by_role("button", name="Get a dataset").click()
    sheet = page.locator(".sheet")
    sheet.locator(".fam[data-preset='any']").click()
    sheet.get_by_label("Hugging Face dataset").fill("hf:org/reviews")
    sheet.get_by_role("button", name="Read").click()
    sheet.get_by_label("Config").select_option("de")
    sheet.get_by_label("Split").select_option("test")
    sheet.get_by_label("Question set").select_option("ticket-triage")
    sheet.get_by_label("Question", exact=True).select_option("is_urgent")
    # negative / positive read as no / yes.
    expect(sheet.get_by_label("Option for negative")).to_have_value("false")
    expect(sheet.get_by_label("Option for positive")).to_have_value("true")
    expect(sheet.get_by_label("Name")).to_have_value("reviews-test")
