"""Get a Dataset: the presets with their question and the table of splits
(each split's role and sample), the Get's name, and the job whose files join
the Datasets list, grouped and with their records. The job is
``tests/fake_job.py`` (no network)."""

import re

import pytest
from drive import open_studio, shot
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def open_sheet(page):
    page.locator(".sidebar").get_by_role("button", name="Download a dataset").click()
    return page.locator(".sheet")


def roles(sheet):
    """Each split's role, in the table's order."""
    return sheet.locator(".st-row select").evaluate_all("selects => selects.map(s => s.value)")


def test_get_a_dataset(studio, page):
    fake_jobs(studio.app, steps=5, every=0.05)
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.locator(".fam b")).to_have_text(
        ["TREC", "TweetEval Offensive", "SST-5", "OpenBookQA", "ANLI round 3", "Another dataset…"]
    )
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("TREC")
    expect(sheet.locator(".ds-options li")).to_have_count(6)
    expect(sheet.locator(".st-row[data-split='train']")).to_contain_text("5,452")
    test = sheet.locator(".st-row[data-split='test']")
    expect(test).to_contain_text("the coverage suite evaluates on it")
    assert roles(sheet) == ["train", "report"]  # no validation split: test reports
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("trec")

    # SST-5: train, validation reports, test for Evaluate.
    sheet.locator(".fam[data-preset='sst5']").click()
    assert roles(sheet) == ["train", "report", "evaluate"]
    expect(name).to_have_value("sst5")
    sheet.get_by_label("Role of test").select_option("out")
    # One split per role: taking train for validation leaves train out.
    sheet.get_by_label("Role of validation").select_option("train")
    assert roles(sheet) == ["out", "train", "out"]
    sheet.get_by_label("Role of train").select_option("train")
    sheet.get_by_label("Role of validation").select_option("report")
    rows = sheet.get_by_label("Rows of validation")
    expect(rows).to_have_value("1000")  # validation: 1,101 rows
    rows.fill("5000")  # more than its rows; typing ticks the sample
    expect(sheet.get_by_label("Sample validation")).to_be_checked()
    expect(sheet.locator(".warn-text")).to_have_text("validation: a sample is 1 to 1101 rows")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    rows.fill("300")
    expect(sheet.get_by_label("Seed")).to_have_value("0")
    sheet.get_by_label("Seed").fill("4")
    shot(page, "get-dataset-splits")
    sheet.get_by_role("button", name="Get").click()

    expect(page.locator(".toast")).to_contain_text("Get sst5 train, validation (300 rows)")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["splits"] == [
        {"split": "train", "role": "train", "sample": None},
        {"split": "validation", "role": "report", "sample": 300},
    ]
    assert job.details["files"] == ["sst5-train", "sst5-validation"]
    assert job.details["seed"] == 4
    assert job.argv[:7] == ["data", "--preset", "sst5", "--split", "train", "--split", "validation"]
    # Done: the files join the list under the Get's name, with their roles.
    group = page.locator(".sidebar .data-list .sb-group")
    expect(group.locator(".sb-group-name")).to_have_text("sst5", timeout=15_000)
    expect(group.locator(".sb-item .label")).to_have_text(["sst5-train", "sst5-validation"])
    expect(group.locator(".sb-item .role")).to_have_text(["train", "report"])
    expect(page.locator(".sidebar .job small")).to_have_text("Done")
    # Where it came from, on the page.
    group.locator(".sb-item", has_text="sst5-validation").click()
    source = page.locator(".ds-source")
    expect(source).to_contain_text("The sst5 preset: fake/sst5")
    expect(source).to_contain_text("a random sample of 300 of 100 rows (seed 4)")
    expect(source.locator(".kv", has_text="Role")).to_contain_text("Report")
    expect(source.locator(".ds-siblings a")).to_have_text("sst5-train")
    shot(page, "datasets-get-group")
    source.locator(".ds-siblings a").click()
    expect(page).to_have_url(f"{studio.base}/#/datasets/sst5-train")
    expect(source.locator(".kv", has_text="Split")).to_contain_text("train: all 100 rows")


def test_a_name_taken_is_said(studio, page):
    (studio.workspace / "data" / "trec-test.jsonl").write_text(
        (studio.workspace / "data" / "tickets.jsonl").read_text()
    )
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.get_by_label("Name")).to_have_value("trec-2")  # trec-test is there
    sheet.get_by_label("Name").fill("trec")
    expect(sheet.locator(".warn-text")).to_have_text("A data file named trec-test exists already")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    sheet.get_by_label("Role of test").select_option("out")  # only trec-train now: free
    expect(sheet.locator(".warn-text")).to_have_count(0)
    expect(sheet.locator(".sheet-foot")).to_contain_text(
        re.compile("mlx-decision data does the same")
    )


def test_pick_at_least_one_split(studio, page):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.get_by_label("Role of train").select_option("out")
    sheet.get_by_label("Role of test").select_option("out")
    expect(sheet.locator(".warn-text")).to_have_text("Pick a role for at least one split")
    expect(sheet.get_by_label("Rows of train")).to_be_disabled()


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
    sheet = open_sheet(page)
    sheet.locator(".fam[data-preset='any']").click()
    get = sheet.get_by_role("button", name="Get")
    expect(get).to_be_disabled()
    field = sheet.get_by_label("Hugging Face dataset")
    field.fill("org/reviews")
    field.press("Enter")  # reads; it does not get
    sheet.get_by_label("Config").select_option("en")
    sheet.get_by_label("Split to preview").select_option("train")
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
    # Every split in the table: rows known only for the one previewed.
    assert roles(sheet) == ["train", "report"]
    expect(sheet.locator(".st-row[data-split='train'] .num")).to_have_text("12")
    expect(sheet.locator(".st-row[data-split='test'] .num")).to_have_text("–")
    expect(sheet.get_by_label("Rows of train")).to_have_value("12")
    expect(sheet.get_by_label("Name")).to_have_value("reviews")
    expect(get).to_be_enabled()
    get.click()

    expect(page.locator(".toast")).to_contain_text("Get org/reviews train, test")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["label_map"] == {"negative": "billing", "positive": "technical"}
    assert job.details["question"] == "department" and job.details["set"] == "ticket-triage"
    assert job.details["files"] == ["reviews-train", "reviews-test"]
    written = (job.folder / "question.json").read_text()
    assert '"department"' in written and "Which team should handle this" in written
    assert "negative=billing,positive=technical" in job.argv


def test_a_yes_no_question_maps_by_name(studio, page, hub):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.locator(".fam[data-preset='any']").click()
    sheet.get_by_label("Hugging Face dataset").fill("hf:org/reviews")
    sheet.get_by_role("button", name="Read").click()
    sheet.get_by_label("Config").select_option("de")
    sheet.get_by_label("Split to preview").select_option("test")
    sheet.get_by_label("Question set").select_option("ticket-triage")
    sheet.get_by_label("Question", exact=True).select_option("is_urgent")
    # negative / positive read as no / yes.
    expect(sheet.get_by_label("Option for negative")).to_have_value("false")
    expect(sheet.get_by_label("Option for positive")).to_have_value("true")
    expect(sheet.get_by_label("Name")).to_have_value("reviews")


def test_a_new_question_from_the_classes(studio, page, hub):
    studio.app.state.workspace.save_set("reviews", {"state": "", "questions": {}})
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.locator(".fam[data-preset='any']").click()
    field = sheet.get_by_label("Hugging Face dataset")
    field.fill("org/reviews")
    field.press("Enter")
    sheet.get_by_label("Config").select_option("en")
    sheet.get_by_label("Split to preview").select_option("train")
    sheet.get_by_label("Question set").select_option("New question")
    # Every class an option of its own; no map to fill in.
    expect(sheet.locator(".any-map .any-row code")).to_have_text(["negative", "positive"])
    expect(sheet.get_by_label("Option for negative")).to_have_count(0)
    # The set's name: the dataset's, taken here, so -2.
    expect(sheet.get_by_label("New question set")).to_have_attribute("placeholder", "reviews-2")
    expect(sheet).to_contain_text("reviews is taken")
    get = sheet.get_by_role("button", name="Get")
    expect(sheet.locator(".fam-pane > .warn-text")).to_have_text(
        "Write the new question's instructions"
    )
    expect(get).to_be_disabled()
    sheet.get_by_label("Instructions").fill("Is the review positive or negative?")
    expect(get).to_be_enabled()
    shot(page, "get-new-question")
    get.click()

    expect(page.locator(".toast")).to_contain_text("saved as the set reviews-2")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    question = {
        "type": "choice",
        "instructions": "Is the review positive or negative?",
        "criteria": {"negative": "negative", "positive": "positive"},
    }
    assert studio.set_file("reviews-2") == {"state": "", "questions": {"reviews": question}}
    assert job.details["label_map"] == {"negative": "negative", "positive": "positive"}
    assert "negative=negative,positive=positive" in job.argv
    # It opens in the Playground, to be tried.
    page.goto(f"{studio.base}/#/playground/reviews-2")
    expect(page.locator(".question[data-q='reviews']")).to_be_visible()


def test_a_second_get_skips_the_names_a_running_get_writes(studio, page):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.get_by_label("Name")).to_have_value("trec")
    sheet.get_by_role("button", name="Get").click()
    expect(page.locator(".sidebar .job")).to_have_count(1)  # runs for 20 s
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("TREC")
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("trec-2")
    name.fill("trec")
    expect(sheet.locator(".warn-text")).to_have_text("A Get that is running writes trec-train")
    expect(sheet.get_by_role("button", name="Get")).to_be_disabled()
    name.fill("trec-2")
    sheet.get_by_role("button", name="Get").click()
    expect(page.locator(".sidebar .job")).to_have_count(2)


def test_each_splits_sample_defaults_to_its_rows_when_fewer(studio, page):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.get_by_label("Rows of train")).to_have_value("1000")  # 5,452 rows
    expect(sheet.get_by_label("Rows of test")).to_have_value("500")  # 500 rows
    expect(sheet.get_by_label("Sample train")).not_to_be_checked()  # all rows by default


def test_the_sheet_opens_again_on_the_entry_last_used(studio, page, hub):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.locator(".fam[data-preset='sst5']").click()
    sheet.get_by_label("Role of test").select_option("out")
    sheet.get_by_role("button", name="Cancel").click()
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("SST-5")
    assert roles(sheet) == ["train", "report", "out"]  # the table as it was left

    # "Another dataset…" as it was left, until a Get starts it afresh.
    sheet.locator(".fam[data-preset='any']").click()
    field = sheet.get_by_label("Hugging Face dataset")
    field.fill("org/reviews")
    field.press("Enter")
    sheet.get_by_label("Config").select_option("en")
    sheet.get_by_label("Split to preview").select_option("train")
    expect(sheet.get_by_label("Rows of train")).to_have_value("12")  # 12 rows only
    sheet.get_by_role("button", name="Cancel").click()
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("Another dataset…")
    expect(sheet.get_by_label("Hugging Face dataset")).to_have_value("hf:org/reviews")
    expect(sheet.locator(".any-table tbody tr")).to_have_count(5)
    sheet.get_by_label("Question set").select_option("ticket-triage")
    sheet.get_by_label("Option for negative").select_option("billing")
    sheet.get_by_role("button", name="Get").click()
    expect(page.locator(".toast")).to_contain_text("Get org/reviews train")
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("Another dataset…")
    expect(sheet.get_by_label("Hugging Face dataset")).to_have_value("")
    expect(sheet.locator(".any-table")).to_have_count(0)
