"""Download: the presets with their question and the table of splits (each
split's role or parts, and its sample), the dataset's name, another dataset's
columns and label values (no question), and the job whose dataset folder
joins the Datasets list. The job is ``tests/fake_job.py`` (no network)."""

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


def test_download_a_preset(studio, page):
    fake_jobs(studio.app, steps=5, every=0.05)
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.locator("h2")).to_have_text("Download a Dataset")
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
    expect(sheet.get_by_role("button", name="Download")).to_be_disabled()
    rows.fill("300")
    expect(sheet.get_by_label("Seed")).to_have_value("0")
    sheet.get_by_label("Seed").fill("4")
    shot(page, "get-dataset-splits")
    sheet.get_by_role("button", name="Download").click()

    expect(page.locator(".toast")).to_contain_text("Download sst5 train, validation (300 rows)")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["splits"] == [
        {"split": "train", "role": "train", "parts": None, "sample": None},
        {"split": "validation", "role": "report", "parts": None, "sample": 300},
    ]
    assert job.details["name"] == "sst5" and job.details["seed"] == 4
    assert job.argv[:7] == ["data", "--preset", "sst5", "--split", "train", "--split", "validation"]
    assert job.argv[-4:] == [
        "--roles",
        "train,report",
        "-o",
        str(studio.workspace.resolve() / "data" / "sst5"),
    ]
    # Done: one dataset in the list, its splits as tabs.
    item = page.locator(".sidebar .data-list .sb-item", has_text="sst5")
    expect(item).to_have_count(1, timeout=15_000)
    expect(page.locator(".sidebar .job small")).to_have_text("Done")
    item.click()
    tabs = page.locator(".ds-tabs [role=tab]")
    expect(tabs).to_have_count(2)
    expect(tabs.nth(1)).to_contain_text("validation")
    expect(tabs.nth(1)).to_contain_text("Report")
    source = page.locator(".ds-source")
    expect(source).to_contain_text("The sst5 preset: fake/sst5")
    shot(page, "datasets-downloaded")


def test_a_name_taken_is_said(studio, page):
    (studio.workspace / "data" / "trec.jsonl").write_text(
        (studio.workspace / "data" / "tickets.jsonl").read_text()
    )
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.get_by_label("Name")).to_have_value("trec-2")  # trec is there
    sheet.get_by_label("Name").fill("trec")
    expect(sheet.locator(".warn-text")).to_have_text("A dataset named trec exists already")
    expect(sheet.get_by_role("button", name="Download")).to_be_disabled()
    sheet.get_by_label("Name").fill("trec-3")
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


def test_a_split_cut_into_parts(studio, page):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.get_by_label("Role of train").select_option("parts")
    parts = sheet.get_by_role("group", name="Parts of train")
    expect(parts.get_by_label("Train share of train")).to_have_value("80")
    # Evaluate twice (a part of train, and test): said.
    sheet.get_by_label("Role of test").select_option("evaluate")
    expect(sheet.locator(".warn-text")).to_have_text("One split per role: two are evaluate")
    parts.get_by_label("Evaluate share of train").fill("0")
    parts.get_by_label("Train share of train").fill("70")
    parts.get_by_label("Report share of train").fill("40")
    expect(sheet.locator(".warn-text")).to_have_text("train: the parts add up to more than 100%")
    parts.get_by_label("Report share of train").fill("30")
    expect(sheet.get_by_label("Seed")).to_be_visible()  # the parts are drawn with it
    shot(page, "get-dataset-parts")
    sheet.get_by_role("button", name="Download").click()
    expect(page.locator(".toast")).to_contain_text("Download trec train, test")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.argv[job.argv.index("--roles") + 1] == "train:0.7+report:0.3,evaluate"
    assert job.details["splits"][0]["parts"] == [
        {"role": "train", "share": 0.7}, {"role": "report", "share": 0.3}]  # fmt: skip


@pytest.fixture
def hub(monkeypatch):
    """A Hub stand-in: org/reviews, configs en and de, splits train and test;
    its card names the stars 1 and 2."""
    import datasets

    from mlx_decision.data import sources

    features = datasets.Features({
        "review": datasets.Value("string"),
        "sentiment": datasets.ClassLabel(names=["negative", "positive"]),
        "stars": datasets.Value("int64"),
    })  # fmt: skip
    rows = {"review": [f"review {i}" for i in range(12)], "sentiment": [i % 2 for i in range(12)],
            "stars": [1 + i % 2 for i in range(12)]}  # fmt: skip
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
    card = '{"LABEL_1": "One star", "LABEL_2": "Two stars"}'
    monkeypatch.setattr(sources, "hub_card", lambda repo, revision=None: card)


def read_reviews(sheet, config="en", split="train"):
    sheet.locator(".fam[data-preset='any']").click()
    field = sheet.get_by_label("Hugging Face dataset")
    field.fill("org/reviews")
    field.press("Enter")  # reads; it does not download
    sheet.get_by_label("Config").select_option(config)
    sheet.get_by_label("Split to preview").select_option(split)


def test_another_dataset_without_a_question(studio, page, hub):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    sheet.locator(".fam[data-preset='any']").click()
    download = sheet.get_by_role("button", name="Download")
    expect(download).to_be_disabled()
    read_reviews(sheet)
    expect(sheet.locator(".any-table tbody tr")).to_have_count(5)
    expect(sheet.get_by_label("State column")).to_have_value("review")
    expect(sheet.get_by_label("Label column")).to_have_value("sentiment")
    expect(sheet.get_by_label("Question set")).to_have_count(0)  # no question here
    values = sheet.locator(".any-map .any-row")
    expect(values.locator("code")).to_have_text(["negative", "positive"])
    # The stars, named by the dataset's card.
    sheet.get_by_label("Label column").select_option("stars")
    expect(values.locator(".dim")).to_have_text(["One star", "Two stars"])
    expect(sheet).to_contain_text("Names from the dataset card")
    # Every split in the table: rows known only for the one previewed.
    assert roles(sheet) == ["train", "report"]
    expect(sheet.locator(".st-row[data-split='train'] .num")).to_have_text("12")
    expect(sheet.locator(".st-row[data-split='test'] .num")).to_have_text("–")
    expect(sheet.get_by_label("Name")).to_have_value("reviews")
    shot(page, "get-another-dataset")
    download.click()

    expect(page.locator(".toast")).to_contain_text("Download org/reviews train, test")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.details["label"] == "stars" and job.details["text"] == "review"
    assert job.argv[job.argv.index("--value-names") + 1] == "1=One star,2=Two stars"
    assert "-q" not in job.argv and "--label-map" not in job.argv


def test_one_split_is_cut_into_parts_by_default(studio, page, hub, monkeypatch):
    from mlx_decision.data import sources

    monkeypatch.setattr(sources, "hub_splits", lambda repo, config=None, revision=None: ["train"])
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    read_reviews(sheet)
    assert roles(sheet) == ["parts"]
    parts = sheet.get_by_role("group", name="Parts of train")
    expect(parts.get_by_label("Train share of train")).to_have_value("80")
    expect(parts.get_by_label("Evaluate share of train")).to_have_value("10")
    sheet.get_by_role("button", name="Download").click()
    expect(page.locator(".toast")).to_contain_text("Download org/reviews train")
    job = next(iter(studio.app.state.jobs.jobs.values()))
    assert job.argv[job.argv.index("--roles") + 1] == "train:0.8+report:0.1+evaluate:0.1"


def test_a_second_download_skips_the_names_a_running_one_writes(studio, page):
    open_studio(page, studio, "datasets")
    sheet = open_sheet(page)
    expect(sheet.get_by_label("Name")).to_have_value("trec")
    sheet.get_by_role("button", name="Download").click()
    expect(page.locator(".sidebar .job")).to_have_count(1)  # runs for 20 s
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("TREC")
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("trec-2")
    name.fill("trec")
    expect(sheet.locator(".warn-text")).to_have_text("A download that is running writes trec")
    expect(sheet.get_by_role("button", name="Download")).to_be_disabled()
    name.fill("trec-2")
    sheet.get_by_role("button", name="Download").click()
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

    # "Another dataset…" as it was left, until a download starts it afresh.
    read_reviews(sheet)
    expect(sheet.get_by_label("Rows of train")).to_have_value("12")  # 12 rows only
    sheet.get_by_role("button", name="Cancel").click()
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("Another dataset…")
    expect(sheet.get_by_label("Hugging Face dataset")).to_have_value("hf:org/reviews")
    expect(sheet.locator(".any-table tbody tr")).to_have_count(5)
    sheet.get_by_role("button", name="Download").click()
    expect(page.locator(".toast")).to_contain_text("Download org/reviews train")
    sheet = open_sheet(page)
    expect(sheet.locator(".fam[aria-selected='true'] b")).to_have_text("Another dataset…")
    expect(sheet.get_by_label("Hugging Face dataset")).to_have_value("")
    expect(sheet.locator(".any-table")).to_have_count(0)
