"""The Datasets page: the sidebar's list, Add… (checked, errors by line, a
name taken) and delete; the picked dataset's questions with their labels,
its examples in a paged table (a label filter, a search in the states), the
example picked whole in the inspector, Fit on It…, Open in Playground and
the empty state."""

import json
import re

import pytest
from drive import open_studio, pick, shot

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
        page.locator(".sidebar").get_by_role("button", name="Add a dataset").click()
    chooser.value.set_files(str(path))


@pytest.fixture
def support(studio):
    """data/support.jsonl: 120 tickets, 70 tech, 30 billing, 20 sales; ticket
    7 is long, over several lines."""
    teams = (["tech"] * 7 + ["billing"] * 3 + ["sales"] * 2) * 10
    lines = []
    for i, team in enumerate(teams):
        state = f"Ticket {i}: something about {team}"
        if i == 7:
            state = "My card was charged twice.\n\nPlease refund one of them. " + "Thanks! " * 60
        lines.append(json.dumps({"id": f"t{i}", "state": state, "questions": {"team": TEAM},
                                 "labels": {"team": team}}))  # fmt: skip
    (studio.workspace / "data" / "support.jsonl").write_text("\n".join(lines) + "\n")
    return studio


def test_the_page_and_a_datasets_questions(studio, page):
    open_studio(page, studio, "datasets")
    nav = page.locator(".sidebar .sb-sec").first.locator(".sb-item .label")
    expect(nav).to_have_text(["Playground", "Models", "Datasets", "Training", "Settings"])
    items = page.locator(".sidebar .data-list .sb-item")
    expect(items).to_have_text(["tickets"])  # the fake Studio's file
    expect(page.locator(".centre .empty")).to_contain_text("Pick a dataset")
    items.first.click()
    expect(page).to_have_url(f"{studio.base}/#/datasets/tickets")
    centre = page.locator(".centre")
    expect(centre.locator(".pane-head p")).to_contain_text("1 example · 3 questions")
    department = centre.locator(".ds-question", has_text="department")
    expect(department).to_contain_text("Which team should handle this")
    expect(department.locator(".label-counts li")).to_have_count(3)  # every option, counted or not
    expect(
        department.locator(".label-counts li", has_text="technical").locator(".num")
    ).to_have_text("1")
    expect(department).to_contain_text("1 of 1 example labelled")
    # Several questions: each label with its question in the table.
    expect(centre.locator(".ex-row .lab")).to_have_text(["departmenttechnical", "is_urgenttrue"])
    expect(page.locator(".toolbar .activity")).to_contain_text("tickets · 1 example")


def test_the_table_pages_filters_and_searches(support, page):
    open_studio(page, support, "datasets/support")
    rows = page.locator(".ex-rows .ex-row")
    pager = page.locator(".ex-pager")
    expect(rows).to_have_count(50)
    expect(pager).to_contain_text("1–50 of 120")
    expect(rows.first.locator(".ex-id")).to_have_text("t0")
    expect(page.locator(".ex-head")).to_contain_text(
        "team"
    )  # one question: its id heads the column
    # A long state is cut to a line.
    long = rows.nth(7).locator(".ex-state")
    expect(long).to_contain_text("My card was charged twice. Please refund one of them.")
    assert long.evaluate("e => e.scrollWidth > e.clientWidth")
    page.get_by_role("button", name="Next page").click()
    expect(pager).to_contain_text("51–100 of 120")
    expect(rows.first.locator(".ex-id")).to_have_text("t50")
    page.get_by_role("button", name="Next page").click()
    expect(rows).to_have_count(20)
    expect(page.get_by_role("button", name="Next page")).to_be_disabled()

    # The label filter, from the menu: back to the first page.
    page.get_by_label("Label", exact=True).select_option(label="billing (30)")
    expect(pager).to_contain_text("1–30 of 30 (of 120 examples)")
    expect(rows).to_have_count(30)
    expect(rows.locator(".lab")).to_have_text(["billing"] * 30)
    # A label picked in Questions does the same; picked again, it shows all.
    sales = page.locator(".ds-question .label-counts button", has_text="sales")
    sales.click()
    expect(sales).to_have_attribute("aria-pressed", "true")
    expect(page.get_by_label("Label", exact=True)).to_have_value(
        json.dumps(["team", "sales"], separators=(",", ":"))
    )
    expect(pager).to_contain_text("of 20 ")
    sales.click()
    expect(pager).to_contain_text("1–50 of 120")

    # The search, in the states, any case; with a filter.
    search = page.get_by_label("Search the states")
    search.fill("REFUND")
    expect(rows).to_have_count(1)
    expect(rows.first.locator(".ex-id")).to_have_text("t7")
    page.get_by_label("Label", exact=True).select_option(label="tech (70)")
    expect(page.locator(".ex-none")).to_have_text("No example holds “REFUND” with the label tech.")
    page.get_by_label("Label", exact=True).select_option("")
    search.fill("")
    expect(pager).to_contain_text("1–50 of 120")


def test_the_example_whole_in_the_inspector(support, page):
    open_studio(page, support, "datasets/support")
    inspector = page.locator(".inspector")
    expect(inspector).to_contain_text("No example picked")
    expect(inspector.locator(".file-group")).to_contain_text("--data")
    page.locator(".ex-row", has_text="My card was charged twice").click()
    expect(inspector.locator(".idet-head h2")).to_have_text("t7")
    expect(inspector.locator(".idet-head p")).to_have_text("Example 8 of 120")
    whole = inspector.locator(".ex-whole")
    expect(whole).to_contain_text("Please refund one of them.")
    assert "\n\nPlease refund" in whole.inner_text()  # the state as it is, its lines kept
    labels = inspector.locator(".igroup", has_text="Labels")
    expect(labels.locator("code")).to_have_text("billing")
    expect(labels).to_contain_text("Money")  # the option in the question's words
    # ↓ and ↑ move the pick.
    page.keyboard.press("ArrowDown")
    expect(inspector.locator(".idet-head h2")).to_have_text("t8")
    expect(page.locator(".ex-row[aria-selected='true'] .ex-id")).to_have_text("t8")
    page.keyboard.press("ArrowUp")
    page.keyboard.press("ArrowUp")
    expect(inspector.locator(".idet-head h2")).to_have_text("t6")
    # Another dataset: nothing picked, its first page.
    page.locator(".sidebar .data-list .sb-item", has_text="tickets").click()
    expect(inspector).to_contain_text("No example picked")
    expect(page.locator(".ex-rows .ex-row")).to_have_count(1)


def test_fit_on_it(support, page):
    open_studio(page, support, "datasets/support")
    page.get_by_role("button", name="Fit on It…").click()
    expect(page).to_have_url(re.compile(r"#/training$"))
    form = page.locator(".fit-form")
    expect(form.locator("#fit-data")).to_have_value("support")
    expect(form.locator(".split-legend")).to_contain_text("24 kept aside for the report")
    expect(page.locator(".inspector .data-inspector h2")).to_have_text("support")
    # The form's own field picks another; the inspector follows.
    form.locator("#fit-data").select_option("tickets")
    expect(page.locator(".inspector .data-inspector h2")).to_have_text("tickets")
    page.goto(f"{support.base}/#/datasets/tickets")
    page.get_by_role("button", name="Fit on It…").click()
    expect(form.locator("#fit-data")).to_have_value("tickets")


def test_open_in_playground(support, page):
    open_studio(page, support, "datasets/support")
    # Show in Finder and Delete… are in the "…" menu, so the actions fit at 1,100 px.
    page.get_by_role("button", name="More actions for support").click()
    expect(page.locator(".menu .mi")).to_have_text(["Show in Finder", "Delete…"])
    page.set_viewport_size({"width": 1100, "height": 800})
    shot(page, "datasets-actions")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Open in Playground").click()
    # A new set named after the file, its first ten examples named by id and label.
    expect(page).to_have_url(f"{support.base}/#/playground/support")
    expect(page.locator(".toast")).to_contain_text(
        "Saved as the question set support, with 10 examples from support"
    )
    tabs = page.locator(".ex-chip [role=tab]")
    expect(tabs).to_have_count(10)
    expect(tabs.first).to_have_text("t0 · tech")
    expect(tabs.nth(8)).to_have_text("t8 · billing")
    expect(page.locator(".question[data-q='team']")).to_be_visible()
    expect(page.locator(".state-text")).to_have_value("Ticket 0: something about tech")
    # It joins the sidebar's question sets, as the one open.
    expect(page.locator(".sb-item[href='#/playground/support']")).to_have_attribute(
        "aria-current", "true"
    )
    shot(page, "playground-from-dataset")
    saved = support.set_file("support")
    assert saved["questions"] == {"team": TEAM} and len(saved["examples"]) == 10
    # Once more: the name is taken, so -2.
    page.goto(f"{support.base}/#/datasets/support")
    page.get_by_role("button", name="Open in Playground").click()
    expect(page).to_have_url(f"{support.base}/#/playground/support-2")


def test_add_a_file_errors_by_line_and_a_name_taken(studio, page, tmp_path):
    open_studio(page, studio, "datasets")
    good = tmp_path / "support tickets.jsonl"
    good.write_text(
        "\n".join(line(i, team) for i, team in enumerate(["tech"] * 4 + ["billing"])) + "\n"
    )
    add(page, good)
    expect(page.locator(".toast")).to_contain_text(
        "Added support tickets.jsonl as data/support-tickets.jsonl · 5 examples"
    )
    expect(page).to_have_url(f"{studio.base}/#/datasets/support-tickets")
    expect(page.locator(".sidebar .sb-item[aria-current='true']")).to_have_text("support-tickets")
    expect(
        page.locator(".ds-question .label-counts li", has_text="tech").locator(".num")
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
    open_studio(page, studio, "datasets")
    other = page.context.new_page()
    other.goto(f"{studio.base}/#/datasets")
    expect(other.locator(".sidebar .data-list .sb-item")).to_have_text(["tickets"])
    good = tmp_path / "more.jsonl"
    good.write_text(line(0, "sales") + "\n")
    add(page, good)
    expect(other.locator(".sidebar .data-list .sb-item")).to_have_text(["more", "tickets"])
    other.close()


def test_delete_the_picked_file(studio, page):
    open_studio(page, studio, "datasets/tickets")
    row = page.locator(".sidebar .data-list li", has_text="tickets")
    row.hover()
    row.get_by_role("button", name="Delete tickets…").click()
    sheet = page.locator(".sheet")
    expect(sheet).to_contain_text("data/tickets.jsonl (1 example) is deleted")
    sheet.get_by_role("button", name="Delete").click()
    expect(page.locator(".toast")).to_contain_text("Deleted data/tickets.jsonl")
    expect(page.locator(".sidebar .data-list .sb-item")).to_have_count(0)
    expect(page.locator(".sidebar .sb-empty", has_text="No datasets yet")).to_be_visible()
    expect(page).to_have_url(re.compile(r"#/datasets$"))
    assert not (studio.workspace / "data" / "tickets.jsonl").exists()
    # The empty state: what a dataset is, with Get a Dataset… and Add….
    empty = page.locator(".centre .ds-empty")
    expect(empty).to_contain_text("A dataset is a file of labelled examples")
    expect(empty.get_by_role("button", name="Get a Dataset…")).to_be_visible()
    expect(empty.get_by_role("button", name="Add…")).to_be_visible()
    # Training says where datasets come from.
    page.goto(f"{studio.base}/#/training")
    expect(page.locator(".centre .empty")).to_contain_text("No datasets yet")
    expect(page.locator(".sidebar .sb-empty", has_text="fits")).to_have_text(
        "No fits or evaluations yet"
    )


def test_delete_from_the_centre(studio, page):
    open_studio(page, studio, "datasets/tickets")
    pick(page, page.get_by_role("button", name="More actions for tickets"), "Delete…")
    page.locator(".sheet").get_by_role("button", name="Delete").click()
    expect(page.locator(".sidebar .data-list .sb-item")).to_have_count(0)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1440, 1100])
def test_the_page_light_and_dark(support, page, theme, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.emulate_media(color_scheme=theme)
    open_studio(page, support, "datasets/support")
    page.locator(".ex-row", has_text="My card was charged twice").click()
    expect(page.locator(".inspector .ex-whole")).to_be_visible()
    # Nothing scrolls sideways; the table's columns fit the centre.
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    centre = page.locator(".centre")
    assert centre.evaluate("e => e.scrollWidth <= e.clientWidth")
    pager = page.locator(".ex-pager").bounding_box()
    tools = page.locator(".ex-tools").bounding_box()
    assert pager["x"] + pager["width"] <= tools["x"] + tools["width"] + 1
    shot(page, f"datasets-{theme}-{width}")


def test_a_file_over_the_upload_limit_is_copied_in_instead(make_studio, page, tmp_path):
    studio = make_studio(max_body_bytes=4_000)
    open_studio(page, studio, "datasets")
    big = tmp_path / "big tickets.jsonl"
    big.write_text("\n".join(line(i, "tech") for i in range(100)) + "\n")  # about 11 kB
    add(page, big)
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("big tickets.jsonl is too large to add here")
    expect(sheet).to_contain_text("the Studio takes files up to 4 kB this way")
    expect(sheet).to_contain_text("as big-tickets.jsonl")
    expect(sheet.locator(".too-large code")).to_have_text(
        str((studio.workspace / "data").resolve())
    )
    expect(sheet.get_by_role("button", name="Show in Finder")).to_be_visible()
    shot(page, "datasets-too-large")
    sheet.get_by_role("button", name="OK").click()
    assert not (studio.workspace / "data" / "big-tickets.jsonl").exists()
    # Under the limit on disk, over it as JSON (quotes escaped): the Studio's 413, the same sheet.
    edge = tmp_path / "edge.jsonl"
    edge.write_text("\n".join(line(i, "tech") for i in range(20)) + "\n")
    assert edge.stat().st_size <= 4_000
    add(page, edge)
    expect(sheet.locator("h2")).to_have_text("edge.jsonl is too large to add here")
