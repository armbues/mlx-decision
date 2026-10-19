"""The Playground's set files: Save, a file changed or deleted by hand while
open (never overwritten), Save As, New, Import, Export, Rename, Delete and the
unsaved-changes prompt."""

import json
import re
from pathlib import Path

import pytest
from drive import menu_item, open_studio, pick, pick_model, put_set, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

MORE = "button[aria-label='More actions for this set']"


def at(page, studio, name: str) -> None:
    """Wait until the address names set ``name``."""
    expect(page).to_have_url(f"{studio.base}/#/playground/{name}")


def change_by_hand(studio, name: str, state: str) -> None:
    """Change a set's file behind the page's back, as an editor would."""
    body = studio.set_file(name)
    body["state"] = state
    (studio.sets / f"{name}.json").write_text(json.dumps(body, indent=2))


def focus_window(page) -> None:
    # The Studio looks at the files again when its window gets the focus; a
    # headless tab never loses it, so send the event.
    page.evaluate("dispatchEvent(new Event('focus'))")


def sheet_button(page, label: str):
    return page.locator(".sheet-foot button", has_text=re.compile(f"^{re.escape(label)}$"))


def edit_state(page, text: str) -> None:
    page.locator(".state-text").fill(text)
    expect(page.locator(".edited-tag")).to_be_visible()


def run_set(page, studio, name: str) -> None:
    """Open ``name`` on fake and run it once."""
    page.goto(f"{studio.base}/#/playground/{name}")
    expect(page.locator(".question").first).to_be_visible()
    pick_model(page, "fake")
    expect(page.locator(".target .mdl .name")).to_have_text("fake")
    page.locator(".question .ins").first.click()
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".acard .verdict").first).to_be_visible()


def sidebar_item(page, name: str):
    return page.locator(f".sb-item[href='#/playground/{name}']")


def test_save_with_the_keyboard(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    save = page.locator(".pane-head .actions .btn", has_text=re.compile("^Save$"))
    expect(save).to_be_disabled()

    edit_state(page, "Edited in the Studio")
    expect(page.locator(".sb-item[aria-current] .edited")).to_be_visible()
    expect(save).to_be_enabled()

    page.keyboard.press("Meta+s")
    expect(page.locator(".edited-tag")).to_have_count(0)
    expect(page.locator(".sb-item[aria-current] .edited")).to_have_count(0)
    expect(page.locator(".toast", has_text=re.compile("^Saved"))).to_be_visible()
    assert studio.set_file("ticket-triage")["state"] == "Edited in the Studio"


def test_an_unedited_set_follows_the_file_on_focus(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    expect(page.locator(".question[data-q='department']")).to_be_visible()
    change_by_hand(studio, "ticket-triage", "Changed by hand, unedited in the page")
    focus_window(page)
    expect(page.locator(".state-text")).to_have_value("Changed by hand, unedited in the page")
    expect(page.locator(".edited-tag")).to_have_count(0)


def test_saving_over_a_file_changed_on_disk_offers_reload_or_save_as(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    edit_state(page, "My edit, not saved yet")
    change_by_hand(studio, "ticket-triage", "Second hand edit")
    page.keyboard.press("Meta+s")
    expect(page.locator(".sheet h2")).to_contain_text("changed on disk")
    assert studio.set_file("ticket-triage")["state"] == "Second hand edit"
    assert page.locator(".sheet-foot button").all_inner_texts() == [
        "Cancel",
        "Reload",
        "Save As…",
    ]

    sheet_button(page, "Save As…").click()
    expect(page.locator(".sheet-input")).to_be_focused()
    expect(page.locator(".sheet-input")).to_have_value("ticket-triage-copy")
    page.keyboard.press("Enter")
    expect(page.locator(".sheet")).to_have_count(0)
    at(page, studio, "ticket-triage-copy")
    assert studio.set_file("ticket-triage-copy")["state"] == "My edit, not saved yet"
    assert studio.set_file("ticket-triage")["state"] == "Second hand edit"

    # The original now shows the file as it is, without the edits.
    page.goto(f"{studio.base}/#/playground/ticket-triage")
    expect(page.locator(".state-text")).to_have_value("Second hand edit")
    expect(page.locator(".edited-tag")).to_have_count(0)


def test_an_edited_set_changed_on_disk_gets_a_banner_and_reloads(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    edit_state(page, "Another edit")
    change_by_hand(studio, "ticket-triage", "Third hand edit")
    focus_window(page)
    banner = page.locator(".disk-banner")
    expect(banner).to_contain_text("changed on disk")
    expect(page.locator(".state-text")).to_have_value("Another edit")
    shot(page, "sets-disk-banner")

    banner.locator(".btn", has_text=re.compile("^Reload$")).click()
    expect(page.locator(".state-text")).to_have_value("Third hand edit")
    expect(banner).to_have_count(0)
    expect(page.locator(".edited-tag")).to_have_count(0)


def test_an_edited_set_deleted_on_disk_can_be_saved_again(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    edit_state(page, "Edit of a deleted set")
    (studio.sets / "ticket-triage.json").unlink()
    focus_window(page)
    expect(page.locator(".disk-banner")).to_contain_text("deleted")

    page.keyboard.press("Meta+s")
    sheet_button(page, "Save Again").click()
    expect(page.locator(".sheet")).to_have_count(0)
    expect(page.locator(".edited-tag")).to_have_count(0)
    assert studio.set_file("ticket-triage")["state"] == "Edit of a deleted set"


def test_new_question_set(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    expect(page.locator(".question[data-q='department']")).to_be_visible()
    page.keyboard.press("Alt+Meta+n")
    name = page.locator(".sheet-input")
    expect(name).to_be_focused()
    name.fill("my set")
    expect(page.locator(".sheet-problem")).to_contain_text("letters, digits")
    name.fill("my-set")
    expect(page.locator(".sheet-problem")).to_have_count(0)
    page.keyboard.press("Enter")
    expect(page.locator(".sheet")).to_have_count(0)
    at(page, studio, "my-set")
    assert studio.set_file("my-set") == {"state": "", "questions": {}}
    expect(sidebar_item(page, "my-set")).to_be_visible()


def test_the_menu_copies_with_save_as_only(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    edit_state(page, "Edited, not saved")
    page.locator(MORE).click()
    labels = page.locator(".menu .mi").all_inner_texts()
    assert any(label.startswith("Save As") for label in labels), labels
    assert not any(label.startswith("Duplicate") for label in labels), labels


def test_import_adds_a_file_as_a_set(studio, page, tmp_path):
    questions = {"spam": {"type": "noul", "instructions": "Is it spam?"}}
    file = tmp_path / "my questions (2).json"
    file.write_text(json.dumps(questions))
    open_studio(page, studio, "playground/ticket-triage")
    with page.expect_file_chooser() as chooser:
        pick(page, MORE, "Import Questions…")
    chooser.value.set_files(str(file))
    # The file's name, made a set's name.
    at(page, studio, "my-questions-2-")
    expect(page.locator(".question[data-q='spam']")).to_be_visible()
    assert studio.set_file("my-questions-2-") == {"questions": questions}


def test_export_downloads_the_saved_file_not_the_edits(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    saved = studio.set_file("ticket-triage")
    edit_state(page, "Edited, not saved")
    page.locator(MORE).click()
    export = menu_item(page, "Export")
    expect(export).to_contain_text("The saved file, without your edits")
    with page.expect_download() as download:
        export.click()
    assert download.value.suggested_filename == "ticket-triage.json"
    assert json.loads(Path(download.value.path()).read_text()) == saved


def test_rename_moves_the_file_and_its_history_and_keeps_the_answers(studio, page):
    open_studio(page, studio)
    put_set(page, studio, "to-rename", studio.set_file("ticket-triage"))
    run_set(page, studio, "to-rename")
    pick(page, MORE, "Rename…")
    page.locator(".sheet-input").fill("renamed-set")
    page.keyboard.press("Enter")
    at(page, studio, "renamed-set")
    assert (studio.sets / "renamed-set.json").exists()
    assert not (studio.sets / "to-rename.json").exists()
    assert (studio.history / "renamed-set.jsonl").exists()
    assert not (studio.history / "to-rename.jsonl").exists()
    expect(sidebar_item(page, "renamed-set")).to_be_visible()
    expect(sidebar_item(page, "to-rename")).to_have_count(0)
    expect(page.locator(".acard .verdict")).to_have_count(3)


def test_rename_keeps_unsaved_edits_unsaved(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    saved = studio.set_file("ticket-triage")["state"]
    edit_state(page, "Edited, not saved")
    pick(page, MORE, "Rename…")
    page.locator(".sheet-input").fill("triage-2")
    page.keyboard.press("Enter")
    at(page, studio, "triage-2")
    expect(page.locator(".edited-tag")).to_be_visible()
    expect(page.locator(".state-text")).to_have_value("Edited, not saved")
    assert studio.set_file("triage-2")["state"] == saved


def test_delete_asks_naming_the_runs_then_removes_the_file_and_history(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    run_set(page, studio, "ticket-triage")
    pick(page, MORE, "Delete…")
    sheet = page.locator(".sheet")
    expect(sheet).to_contain_text("Delete ticket-triage?")
    expect(sheet).to_contain_text("1 run")
    sheet.locator(".btn.danger").click()
    expect(sheet).to_have_count(0)
    expect(page).not_to_have_url(re.compile("#/playground/ticket-triage$"))
    expect(page).to_have_url(re.compile("#/playground/"))
    expect(sidebar_item(page, "ticket-triage")).to_have_count(0)
    assert not (studio.sets / "ticket-triage.json").exists()
    assert not (studio.history / "ticket-triage.jsonl").exists()


def test_cancel_keeps_the_set(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick(page, MORE, "Delete…")
    sheet = page.locator(".sheet")
    sheet.locator(".btn", has_text=re.compile("^Cancel$")).click()
    expect(sheet).to_have_count(0)
    at(page, studio, "ticket-triage")
    expect(sidebar_item(page, "ticket-triage")).to_be_visible()
    assert (studio.sets / "ticket-triage.json").exists()


def test_the_open_sets_row_has_a_delete_button_shown_on_hover(studio, page):
    open_studio(page, studio, "playground/order-json")
    delete = page.locator(".sb-delete")
    expect(delete).to_have_count(1)
    expect(page.locator("li.open .sb-delete")).to_have_count(1)
    expect(delete).to_have_css("opacity", "0")
    page.locator("li.open .sb-item").hover()
    expect(delete).to_have_css("opacity", "1")
    shot(page, "sidebar-delete")

    delete.click()
    expect(page.locator(".sheet")).to_contain_text("Delete order-json?")
    page.locator(".sheet .btn.danger").click()
    expect(page.locator(".sheet")).to_have_count(0)
    expect(page).not_to_have_url(re.compile("#/playground/order-json$"))
    assert not (studio.sets / "order-json.json").exists()


def test_leaving_with_unsaved_edits_asks_first(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    edit_state(page, "Edited, not saved")
    dialogs = []

    def answer(dialog):
        dialogs.append(dialog.type)
        dialog.accept()

    page.on("dialog", answer)
    page.reload()
    assert dialogs == ["beforeunload"]
