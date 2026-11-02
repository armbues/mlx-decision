"""Deleting from the lists: a dataset that is not picked, the Studio's own
models and downloaded ones (a Hugging Face cache repo, a folder or a link in
the models folder) on the Models page, and fits and evaluations in
Training's History (a fit's model too, when asked)."""

import json
import re

import pytest
from drive import menu_item, open_studio, start_job
from fake_studio import fake_jobs

from fake_backend import write_fitted, write_model

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def test_a_dataset_that_is_not_picked(studio, page):
    line = (studio.workspace / "data" / "tickets.jsonl").read_text()
    (studio.workspace / "data" / "more.jsonl").write_text(line)
    open_studio(page, studio, "datasets/tickets")
    row = page.locator(".sidebar .data-list li", has=page.locator("[href='#/datasets/more']"))
    row.hover()
    row.get_by_role("button", name="Delete more…").click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page.locator(".sidebar .data-list .sb-item")).to_have_count(1)
    expect(page).to_have_url(re.compile(r"#/datasets/tickets$"))
    assert not (studio.workspace / "data" / "more.jsonl").exists()


@pytest.fixture
def made(studio, page):
    """The Studio with a converted model in its models folder (``conv``) and a
    fitted one on it (``conv-fit``), found by a rescan."""
    folder = studio.workspace / "models"
    write_model(folder / "conv")
    write_fitted(folder / "conv-fit", folder / "conv")
    open_studio(page, studio, "models")
    response = page.request.post(f"{studio.base}/studio/api/models/rescan", data={})
    assert sorted(response.json()["added"]) == ["conv", "conv-fit"]
    return studio


def test_the_studios_own_models_and_not_downloaded_ones(made, page):
    folder = made.workspace / "models"
    page.goto(f"{made.base}/#/models/fake")
    expect(page.locator(".idet h2")).to_have_text("fake")
    expect(page.locator(".idet-foot")).to_have_count(0)  # a folder named with -m

    page.goto(f"{made.base}/#/models/conv")
    delete = page.locator(".idet-foot").get_by_role("button", name="Delete…")
    expect(delete).to_be_disabled()
    expect(delete).to_have_attribute(
        "title", re.compile(r"^Fitted models are made from it \(conv-fit\)")
    )

    # The fitted one from the row's context menu.
    page.locator(".mrow", has_text="conv-fit").first.click(button="right")
    menu_item(page, "Delete…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Delete the model conv-fit?")
    expect(sheet).to_contain_text(str(folder / "conv-fit"))
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast")).to_contain_text("Deleted the model conv-fit")
    expect(page.locator(".mrow", has_text="conv-fit")).to_have_count(0)
    assert not (folder / "conv-fit").exists()

    # Now its base can go, from the inspector.
    expect(delete).to_be_enabled()
    delete.click()
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page).to_have_url(re.compile(r"#/models$"))
    expect(page.locator(".mrow", has_text="conv")).to_have_count(0)
    assert not (folder / "conv").exists()


def fit_entry(page, title):
    return page.locator(".sidebar .fit-list li", has=page.locator(f"[title='{title}']"))


def test_a_fit_with_its_model_and_an_evaluation(studio, page):
    fake_jobs(studio.app, steps=3, every=0.02, fits=True)
    open_studio(page, studio, "training")
    entries = page.locator(".sidebar .fit-list li")
    fit = start_job(page, studio, kind="fit", model="fake", data="tickets", name="mine")
    expect(entries.locator(".sb-delete")).to_have_count(1, timeout=15_000)  # ended
    evaluation = start_job(page, studio, kind="eval", model="fake", data="tickets")
    expect(entries).to_have_count(2)
    expect(entries.locator(".sb-delete")).to_have_count(2, timeout=15_000)
    expect(page.locator(".sidebar .sb-item", has_text="mine")).to_be_visible()
    model = studio.workspace / "models" / "mine"

    # The evaluation: only its entry, no model to offer.
    row = fit_entry(page, evaluation["title"])
    row.hover()
    row.locator(".sb-delete").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text(
        "Delete the evaluation of " + evaluation["title"].removeprefix("Evaluate ") + "?"
    )
    expect(sheet.locator("input[type=checkbox]")).to_have_count(0)
    sheet.get_by_role("button", name="Delete").click()
    expect(entries).to_have_count(1)

    # The fit, opened, with its model.
    page.goto(f"{studio.base}/#/training/{fit['id']}")
    expect(page.locator(".report-table")).to_be_visible()
    row = fit_entry(page, fit["title"])
    row.hover()
    row.locator(".sb-delete").click()
    expect(sheet).to_contain_text("Also delete its model mine")
    sheet.get_by_role("checkbox").check()
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast").last).to_contain_text("Deleted the fit mine and the model mine")
    expect(entries).to_have_count(0)
    expect(page).to_have_url(re.compile(r"#/training$"))
    assert not model.exists()
    assert not (studio.workspace / "jobs" / fit["id"]).exists()
    listed = page.request.get(f"{studio.base}/studio/api/models").json()["models"]
    assert "mine" not in [entry["name"] for entry in listed]
    assert json.loads(page.request.get(f"{studio.base}/studio/api/jobs").text())["jobs"] == []


@pytest.fixture
def downloads(make_studio, monkeypatch, tmp_path):
    """The fake Studio with downloaded models (``downloads=True``), its Hugging
    Face cache in a scratch folder."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf"))
    studio = make_studio(downloads=True)
    studio.cache = tmp_path / "hf"
    studio.models_dir = studio.home / "models-dir"
    return studio


def location(page):
    return page.locator(".idet dt", has_text="Location").locator("xpath=following-sibling::dd[1]")


def delete_from_inspector(page, studio, name):
    page.goto(f"{studio.base}/#/models/{name}")
    expect(page.locator(".idet h2")).to_have_text(name)
    page.locator(".idet-foot").get_by_role("button", name="Delete…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text(f"Delete the model {name}?")
    return sheet


def test_a_cached_repo_every_revision(downloads, page):
    repo = downloads.cache / "models--org--fake-cached"
    open_studio(page, downloads, "models")
    row = page.locator(".mrow", has_text="fake-cached").first
    expect(row).to_contain_text("HF cache")
    page.goto(f"{downloads.base}/#/models/fake-cached")
    expect(location(page)).to_contain_text("In the Hugging Face cache")
    button = page.locator(".idet-foot").get_by_role("button", name="Delete…")
    expect(button).to_have_attribute("title", "Delete org/fake-cached from the Hugging Face cache")
    sheet = delete_from_inspector(page, downloads, "fake-cached")
    expect(sheet).to_contain_text(
        "The Hugging Face cache repo org/fake-cached (0.0 GB), every revision"
    )
    expect(sheet).to_contain_text("Other tools share the cache")
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast")).to_contain_text("Deleted the model fake-cached")
    expect(page).to_have_url(re.compile(r"#/models$"))
    expect(page.locator(".mrow", has_text="fake-cached")).to_have_count(0)
    assert not repo.exists()


def test_a_folder_and_a_link_in_the_models_folder(downloads, page):
    open_studio(page, downloads, "models")
    expect(page.locator(".mrow", has_text="fake-local").first).to_contain_text("models folder")
    page.goto(f"{downloads.base}/#/models/fake-local")
    expect(location(page)).to_contain_text("In your models folder (models_dir)")
    sheet = delete_from_inspector(page, downloads, "fake-local")
    expect(sheet).to_contain_text(f"Its folder {downloads.models_dir / 'fake-local'}")
    expect(sheet).to_contain_text("is deleted from disk")
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".mrow", has_text="fake-local")).to_have_count(0)
    assert not (downloads.models_dir / "fake-local").exists()

    # The link, from the row's context menu: only the link goes.
    page.goto(f"{downloads.base}/#/models/fake-linked")
    expect(location(page)).to_contain_text("A link in your models folder")
    button = page.locator(".idet-foot").get_by_role("button", name="Delete…")
    expect(button).to_have_attribute("title", "Delete the link fake-linked")
    page.locator(".mrow", has_text="fake-linked").first.click(button="right")
    menu_item(page, "Delete…").click()
    sheet = page.locator(".sheet")
    expect(sheet).to_contain_text(f"The link {downloads.models_dir / 'fake-linked'}")
    expect(sheet).to_contain_text("The folder it points to stays")
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".mrow", has_text="fake-linked")).to_have_count(0)
    assert not (downloads.models_dir / "fake-linked").is_symlink()
    assert (downloads.home / "elsewhere" / "linked").is_dir()


def test_a_named_folder_has_no_delete(downloads, page):
    open_studio(page, downloads, "models/fake-named")
    expect(page.locator(".idet h2")).to_have_text("fake-named")
    expect(location(page)).to_contain_text("A folder you named")
    expect(page.locator(".idet-foot")).to_have_count(0)
    page.locator(".mrow", has_text="fake-named").first.click(button="right")
    page.wait_for_selector(".menu .mi")
    expect(page.locator(".menu .mi", has_text="Delete…")).to_have_count(0)
    page.keyboard.press("Escape")
    response = page.request.delete(
        f"{downloads.base}/studio/api/models/fake-named",
        headers={"Content-Type": "application/json"},
    )
    assert response.status == 403
    assert (downloads.models_dir / "fake-named").is_dir()


def test_a_refusal_is_shown_in_the_sheet(downloads, page):
    repo = downloads.cache / "models--org--fake-cached"
    open_studio(page, downloads, "models")
    sheet = delete_from_inspector(page, downloads, "fake-cached")
    # Broken after the page listed it: the cache can no longer read the repo.
    (repo / "snapshots" / ("a" * 40) / "README.md").symlink_to(repo / "blobs" / "missing")
    sheet.get_by_role("button", name="Delete").click()
    expect(sheet).to_contain_text("Not deleted: the Hugging Face cache cannot read org/fake-cached")
    expect(sheet.locator("h2")).to_have_text("Delete the model fake-cached?")
    assert repo.is_dir()
    listed = page.request.get(f"{downloads.base}/studio/api/models").json()["models"]
    assert "fake-cached" in [entry["name"] for entry in listed]
