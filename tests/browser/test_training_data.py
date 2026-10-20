"""Training data on the Training page: the sidebar's list, "+" to add a
JSON-lines file (checked, errors by line, a name taken), and the inspector's
Data for the file picked."""

import json

import pytest
from drive import open_studio

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip


def line(i, team):
    return json.dumps(
        {"state": f"ticket {i}", "questions": {"team": TEAM}, "labels": {"team": team}}
    )


def add(page, path):
    with page.expect_file_chooser() as chooser:
        page.locator(".sidebar").get_by_role("button", name="Add training data").click()
    chooser.value.set_files(str(path))


def test_the_list_and_the_data_inspector(studio, page):
    open_studio(page, studio, "training")
    items = page.locator(".sidebar .data-list .sb-item")
    expect(items).to_have_text(["tickets"])  # the fake Studio's file
    items.first.click()
    inspector = page.locator(".data-inspector")
    expect(inspector.locator(".idet-head p")).to_have_text("1 example · 3 questions")
    department = inspector.locator(".igroup", has_text="department · choice")
    expect(department).to_contain_text("Which team should handle this")
    expect(department.locator(".label-counts li")).to_have_count(3)  # every option, counted or not
    expect(
        department.locator(".label-counts li", has_text="technical").locator(".num")
    ).to_have_text("1")
    expect(department).to_contain_text("1 of 1 example labelled")
    expect(inspector).to_contain_text("--data")


def test_add_a_file_errors_by_line_and_a_name_taken(studio, page, tmp_path):
    open_studio(page, studio, "training")
    good = tmp_path / "support tickets.jsonl"
    good.write_text(
        "\n".join(line(i, team) for i, team in enumerate(["tech"] * 4 + ["billing"])) + "\n"
    )
    add(page, good)
    expect(page.locator(".toast")).to_contain_text(
        "Added support tickets.jsonl as data/support-tickets.jsonl · 5 examples"
    )
    expect(page).to_have_url(f"{studio.base}/#/training/support-tickets")
    expect(page.locator(".sidebar .sb-item[aria-current='true']")).to_have_text("support-tickets")
    expect(
        page.locator(".data-inspector .label-counts li", has_text="tech").locator(".num")
    ).to_have_text("4")
    assert (studio.workspace / "data" / "support-tickets.jsonl").read_text() == good.read_text()

    # The same name again: another name is asked for.
    add(page, good)
    sheet = page.locator(".sheet")
    expect(sheet).to_contain_text("a data file named support-tickets exists already")
    expect(sheet.get_by_label("Name")).to_have_value("support-tickets-2")
    sheet.get_by_role("button", name="Add").click()
    expect(page.locator(".sidebar .data-list .sb-item")).to_have_text(
        ["support-tickets", "support-tickets-2", "tickets"]
    )

    # A file with bad lines: each named, nothing added.
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join([line(0, "tech"), line(1, "nobody"), "{not json", line(3, "sales")]))
    add(page, bad)
    expect(sheet.locator("h2")).to_have_text("bad.jsonl cannot be used")
    errors = sheet.locator(".data-errors li")
    expect(errors).to_have_count(2)
    expect(errors.nth(0)).to_contain_text("Line 2")
    expect(errors.nth(0)).to_contain_text('"nobody" is none of the options')
    expect(errors.nth(0).locator("code")).to_have_text("labels.team")
    expect(errors.nth(1)).to_contain_text("Line 3")
    sheet.get_by_role("button", name="OK").click()
    assert not (studio.workspace / "data" / "bad.jsonl").exists()


def test_a_second_tab_sees_new_data(studio, page, tmp_path):
    open_studio(page, studio, "training")
    other = page.context.new_page()
    other.goto(f"{studio.base}/#/training")
    expect(other.locator(".sidebar .data-list .sb-item")).to_have_text(["tickets"])
    good = tmp_path / "more.jsonl"
    good.write_text(line(0, "sales") + "\n")
    add(page, good)
    expect(other.locator(".sidebar .data-list .sb-item")).to_have_text(["more", "tickets"])
    other.close()
