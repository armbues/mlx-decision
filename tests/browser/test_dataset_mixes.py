"""A mix on the Datasets page: its row says what it mixes, the page shows its
sources (rows per split, caps, the held-out one) with links to them, its
splits as tabs, and offers a question set instead of writing questions."""

import json
import re

import pytest
from drive import open_studio, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip
URGENT = {"type": "noul", "instructions": "Is it urgent?"}


def write_lines(path, count, question_id, question, label):
    lines = [
        json.dumps({"id": f"r{i}", "state": f"{question_id} row {i}",
                    "questions": {question_id: question}, "labels": {question_id: label}})
        for i in range(count)
    ]  # fmt: skip
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def mixed(studio):
    """data/teams.jsonl (20 rows), data/alarms.jsonl (10) and data/blend, a mix
    of them, teams capped at 10 and alarms held out."""
    data = studio.workspace / "data"
    write_lines(data / "teams.jsonl", 20, "team", TEAM, "tech")
    write_lines(data / "alarms.jsonl", 10, "urgent", URGENT, True)
    (data / "blend").mkdir()
    sources = [{"path": "../teams.jsonl", "cap": 10}, {"path": "../alarms.jsonl", "held_out": True}]
    body = {"format": 2, "mix": {"sources": sources}}
    (data / "blend" / "dataset.json").write_text(json.dumps(body))
    return studio


def test_a_mix_shows_its_sources_and_splits(mixed, page):
    open_studio(page, mixed, "datasets/blend")
    row = page.locator(".sidebar .data-list li", has_text="blend")
    expect(row.locator(".sb-item")).to_have_attribute(
        "title", "blend: a mix of teams, alarms (held out: alarms)"
    )
    centre = page.locator(".centre")
    # The train split asks teams' question only: alarms is held out.
    expect(centre.locator(".pane-head p")).to_have_text("A mix of 2 datasets · 1 question")
    tabs = centre.locator(".ds-tabs [role=tab]")
    expect(tabs).to_have_count(3)
    for tab, name in zip(["train", "report", "evaluate"], range(3), strict=True):
        expect(tabs.nth(name)).to_contain_text(tab)
    sources = centre.locator(".ds-sources")
    expect(sources.locator(".gsec-head .tag")).to_have_text("2 datasets · seed 0")
    teams, alarms = sources.locator("[data-source=teams]"), sources.locator("[data-source=alarms]")
    expect(teams).to_contain_text("8 rows in train · at most 10 train rows")
    expect(alarms).to_contain_text("0 rows in train")
    expect(alarms.locator(".ds-q-type")).to_have_text("Held out")
    expect(sources.locator(".hint")).to_contain_text("scored in a fit's report but not trained on")
    # The report split holds both: a fifth of the capped teams rows, a fifth of alarms.
    tabs.nth(1).click()
    expect(teams).to_contain_text("2 rows in report")
    expect(alarms).to_contain_text("2 rows in report")
    expect(page.locator(".ex-rows .ex-row")).to_have_count(4)
    expect(page.locator(".ex-rows .ex-row").first.locator(".ex-id")).to_have_text(
        re.compile(r"^teams/r\d+$")
    )
    expect(page.locator(".centre .ds-files .file-group")).to_contain_text("dataset.json")
    # Its questions are its sources': no Write Questions, a question set instead.
    head = page.locator(".pane-head")
    expect(head.get_by_role("button", name="Write Questions")).to_have_count(0)
    expect(head.get_by_role("button", name="Edit Questions")).to_have_count(0)
    expect(head.get_by_role("button", name="New Question Set from It")).to_be_enabled()
    shot(page, "datasets-mix")
    # A source links to its own page.
    teams.get_by_role("link", name="teams").click()
    expect(page).to_have_url(f"{mixed.base}/#/datasets/teams")


def test_a_mix_with_a_missing_source_says_so(studio, page):
    folder = studio.workspace / "data" / "broken"
    folder.mkdir()
    body = {"format": 2, "mix": {"sources": [{"path": "../gone"}]}}
    (folder / "dataset.json").write_text(json.dumps(body))
    open_studio(page, studio, "datasets/broken")
    row = page.locator(".sidebar .data-list li", has_text="broken")
    expect(row.locator(".mark-warn")).to_have_count(1)
    expect(row.locator(".sb-item")).to_have_attribute(
        "title", "Not ready for a fit: the mix's source ../gone is missing"
    )
    expect(page.locator(".centre")).to_contain_text("the mix's source ../gone is missing")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_mix_light_and_dark(mixed, page, theme):
    page.set_viewport_size({"width": 1100, "height": 900})
    page.emulate_media(color_scheme=theme)
    open_studio(page, mixed, "datasets/blend")
    expect(page.locator(".ds-sources [data-source]")).to_have_count(2)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    centre = page.locator(".centre")
    assert centre.evaluate("e => e.scrollWidth <= e.clientWidth")
    shot(page, f"datasets-mix-{theme}")


def test_the_fit_form_mixes_datasets_and_picks_the_mix(mixed, page):
    open_studio(page, mixed, "training")
    form = page.locator(".fit-form")
    expect(form.locator("#fit-data")).to_have_value("alarms")  # the first
    form.get_by_role("button", name="Mix Datasets…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Mix Datasets")
    # Ready datasets that are not mixes, alarms (the form's) picked to start with.
    rows = sheet.locator(".mix-table .st-row")
    expect(rows.locator("code")).to_have_text(["alarms", "teams", "tickets"])
    expect(sheet.get_by_label("Mix alarms")).to_be_checked()
    save = sheet.get_by_role("button", name="Save Mix")
    expect(save).to_be_disabled()
    expect(sheet.locator(".warn-text")).to_have_text("Pick two datasets or more")
    sheet.get_by_label("Mix teams").check()
    sheet.get_by_label("Hold out alarms").check()
    sheet.get_by_label("Cap of teams").fill("12")
    expect(sheet.get_by_label("Cap of alarms")).to_have_attribute("placeholder", "2,000")
    expect(sheet.locator("#mix-name")).to_have_value("mix")
    shot(page, "fit-mix-sheet")
    save.click()
    expect(sheet).to_have_count(0)
    expect(form.locator("#fit-data")).to_have_value("mix")
    expect(form).to_contain_text("A mix of alarms, teams; held out: alarms. The report shows each")
    # The report is the mix's report split: a fifth of teams' 12 and of alarms' 10.
    expect(form.locator(".split-legend")).to_contain_text("4 in report for the report")
    body = json.loads((mixed.workspace / "data" / "mix" / "dataset.json").read_text())
    assert body["mix"] == {
        "sources": [
            {"path": "../alarms.jsonl", "held_out": True},
            {"path": "../teams.jsonl", "cap": 12},
        ],
        "cap": 2000,
        "seed": 0,
    }
    expect(page.locator(".sidebar")).to_be_visible()
