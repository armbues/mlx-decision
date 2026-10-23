"""Paths under the home folder show from ~, as the places do: a fit's report
(Saved in, the command) and a model's inspector (Location, the CLI's -m).
The fake Studio lives in the test's folder, which stands in for the home
folder here."""

import json
import re

import pytest
from drive import open_studio
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip


@pytest.fixture
def studio(make_studio, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    studio = make_studio(convertible=True)
    labels = ["tech"] * 30 + ["billing"] * 12 + ["sales"] * 18
    lines = [
        json.dumps({"state": f"ticket {i}", "questions": {"team": TEAM}, "labels": {"team": label}})
        for i, label in enumerate(labels)
    ]
    (studio.workspace / "data" / "support.jsonl").write_text("\n".join(lines) + "\n")
    return studio


def home_path(studio, path) -> str:
    """``path`` as the page should write it: from ~, the test's folder."""
    return "~/" + str(path.resolve().relative_to(studio.workspace.parents[1].resolve()))


def test_a_fit_report_writes_home_paths_from_tilde(studio, page):
    fake_jobs(studio.app, steps=10, every=0.05, fits=True)
    open_studio(page, studio, "training")
    form = page.locator(".fit-form")
    form.locator("#fit-data").select_option("support")
    form.locator("#fit-base").select_option("tiny-clef")
    form.locator("#fit-name").fill("support-head")
    page.locator(".run-btn").click()
    expect(page.locator(".report-table")).to_be_visible(timeout=20_000)
    saved = page.locator(".kv", has_text="Saved in").locator("code")
    model = studio.workspace / "models" / "support-head"
    folder = home_path(studio, model)
    expect(saved).to_have_text(folder)
    expect(saved).to_have_attribute("title", str(model.resolve()))
    command = page.locator(".kv", has_text="Same in a terminal").locator("code")
    expect(command).to_contain_text(f"-o {folder}")
    expect(command).not_to_contain_text(str(studio.workspace.parents[1]))


def test_a_models_location_and_cli_write_home_paths_from_tilde(studio, page):
    open_studio(page, studio, "models/fake")
    location = page.locator(".inspector .kv", has_text="Location").locator("code")
    expect(location).to_have_text(re.compile(r"^~/"))
    expect(location).to_have_attribute("title", re.compile(r"^/"))
    cli = page.locator(".inspector .kv", has_text="CLI").locator("code")
    expect(cli).not_to_contain_text(str(studio.workspace.parents[1]))
