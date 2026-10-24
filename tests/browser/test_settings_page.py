"""Where models live (with Show in Finder) on the Models page, and the
Settings page's config file, changed through its sheets and written one line
at a time, and the workspace's folders with Clear."""

import re
import time
from pathlib import Path

import pytest
from drive import open_studio, start_job
from fake_studio import fake_jobs

from mlx_decision.studio.routes import settings as settings_route

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def _eventually(condition, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_where_models_live(studio, page, monkeypatch):
    shown = []
    monkeypatch.setattr(settings_route, "reveal", shown.append)
    open_studio(page, studio, "models")
    places = page.locator(".places .kv.place")
    expect(places.locator("dt")).to_have_text(["Hugging Face cache", "Made here", "Your folders"])
    folders = page.locator(".places .kv.place[data-place='named']")
    expect(folders).to_contain_text(str(studio.home / "fake-models"))
    expect(folders).to_contain_text("3 models")
    made = page.locator(".places .kv.place[data-place='studio']")
    expect(made.locator("small")).to_have_text("Converted copies and fitted models · 0 models")
    folders.get_by_role("button", name="Show in Finder").click()
    assert _eventually(lambda: shown == [studio.home / "fake-models"])

    # A model made here is counted once the list has it.
    fake_jobs(studio.app, steps=3, every=0.05)
    studio.app.state.jobs.command.append("--fake-model")
    start_job(page, studio, kind="fit", model="fake", data="tickets")
    expect(made.locator("small")).to_have_text(
        "Converted copies and fitted models · 1 model", timeout=15_000
    )

    # Settings has the config file, not the places.
    page.locator(".sidebar a", has_text="Settings").click()
    expect(page.locator(".centre h1")).to_have_text("Settings")
    expect(page.locator(".places")).to_have_count(0)
    page.locator(".config-group .kv.place").get_by_role("button", name="Show in Finder").click()
    assert _eventually(lambda: shown[-1] == studio.home)  # its folder until it exists


# Each key's name in words, as the rows and the sheets title them.
TITLES = {
    "models_dir": "Where Models Go",
    "models": "Other Models",
    "default_model": "Default Model",
    "memory_budget_gb": "Memory Budget",
    "host": "Address",
    "port": "Port",
}


def change(page, key):
    page.locator(f".kv.setting[data-key='{key}']").get_by_role("button", name="Change…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text(f"Change {TITLES[key]}")
    expect(sheet).to_contain_text(f"as {key}.")  # the note names the key
    return sheet


def test_change_the_config_from_the_page(studio, page):
    config = studio.home / "config.toml"
    config.write_text("# mine\n\n[server]\nport = 1\n")
    open_studio(page, studio, "settings")
    setting = page.locator(".kv.setting[data-key='memory_budget_gb']")
    expect(setting.locator("dd")).to_contain_text("1 GB")

    sheet = change(page, "memory_budget_gb")
    sheet.get_by_label("memory_budget_gb").fill("2")
    sheet.get_by_role("button", name="Save").click()
    expect(page.locator(".toast")).to_contain_text("Saved memory_budget_gb = 2 · applies now")
    expect(setting.locator("dd")).to_contain_text("2 GB")
    expect(setting.locator("small")).to_contain_text("from the config file")
    expect(page.locator(".sidebar .mem-head")).to_contain_text("of 2.0 GB")
    assert config.read_text() == "# mine\nmemory_budget_gb = 2\n\n[server]\nport = 1\n"

    sheet = change(page, "port")
    sheet.get_by_label("port").fill("9123")
    sheet.get_by_label("port").press("Enter")  # Enter saves
    expect(page.locator(".toast").last).to_contain_text("takes effect at the next start")
    port = page.locator(".kv.setting[data-key='port']")
    expect(port.locator("small")).to_contain_text("the file says 9123")

    sheet = change(page, "port")
    sheet.get_by_label("port").fill("70000")
    sheet.get_by_role("button", name="Save").click()
    expect(sheet.locator(".warn-text")).to_contain_text("must be between 1 and 65535")
    sheet.get_by_role("button", name="Take Out of the File").click()
    expect(page.locator(".toast").last).to_contain_text("Took port out of the config file")
    expect(port.locator("small")).not_to_contain_text("the file says")
    assert "port = 9123" not in config.read_text()

    # models_dir: a folder; downloads go there.
    folder = studio.home / "downloads"
    sheet = change(page, "models_dir")
    sheet.get_by_role("textbox", name="Folder").fill(str(folder))
    sheet.get_by_role("button", name="Save").click()
    expect(page.locator(".toast").last).to_contain_text(f"downloads go to {folder}")
    page.goto(f"{studio.base}/#/models")
    first = page.locator(".places .kv.place").first
    expect(first.locator("dt")).to_have_text("Your models folder")
    expect(first.locator("small")).to_contain_text("downloads go here")
    assert f'models_dir = "{folder}"' in config.read_text()


def test_the_default_model_from_a_list(studio, page):
    open_studio(page, studio, "settings")
    sheet = change(page, "default_model")
    sheet.get_by_label("default_model").select_option("fake-vision")
    sheet.get_by_role("button", name="Save").click()
    expect(page.locator(".toast")).to_contain_text("applies now")
    expect(page.locator(".kv.setting[data-key='default_model'] dd")).to_contain_text("fake-vision")
    assert studio.app.state.gpu.pool.default == "fake-vision"
    assert 'default_model = "fake-vision"' in Path(studio.home / "config.toml").read_text()


def test_paths_show_with_a_tilde_and_the_workspace_is_shown_only(studio, page, monkeypatch):
    shown = []
    monkeypatch.setattr(settings_route, "reveal", shown.append)
    monkeypatch.setenv("HOME", str(studio.home.parent))  # the test's folders are under it
    open_studio(page, studio, "settings")
    expect(page.locator(".kv.setting[data-key='models'] dd")).to_contain_text(
        "~/studio/fake-models"
    )
    expect(page.locator(".config-group .kv.place dd code")).to_have_text("~/studio/config.toml")
    workspace = page.locator(".kv.setting[data-key='workspace']")
    expect(workspace.locator("dd")).to_contain_text("~/studio/ws")
    expect(workspace.locator("small")).to_contain_text(
        "moves with --workspace or the config file, at the next start"
    )
    expect(workspace.get_by_role("button", name="Change…")).to_have_count(0)
    workspace.get_by_role("button", name="Show in Finder").click()
    assert _eventually(lambda: shown == [(studio.home / "ws").resolve()])


def test_the_memory_budget_slider(studio, page):
    open_studio(page, studio, "settings")
    sheet = change(page, "memory_budget_gb")
    slider = sheet.get_by_label("Memory budget")
    number = sheet.get_by_label("memory_budget_gb")
    working_set = float(slider.get_attribute("max"))
    assert working_set > 1
    expect(sheet.locator(".hint")).to_contain_text(f"Up to {working_set:g} GB")
    slider.focus()
    page.keyboard.press("Home")  # the slider's start: 1 GB
    expect(number).to_have_value("1")
    page.keyboard.press("ArrowRight")
    expect(number).to_have_value("1.5")
    number.fill("3")
    expect(slider).to_have_value("3")  # the two follow each other
    sheet.get_by_role("button", name="Save").click()
    expect(page.locator(".toast")).to_contain_text("Saved memory_budget_gb = 3 · applies now")
    assert studio.app.state.gpu.pool.budget == 3 * 10**9


def test_the_host_choices_and_the_api_key(studio, page):
    open_studio(page, studio, "settings")
    sheet = change(page, "host")
    expect(sheet.get_by_role("radio", name="This Mac only")).to_be_checked()
    expect(sheet).to_contain_text("0.0.0.0: other machines on your network reach the Studio")
    note = sheet.locator(".setting-editor p").first
    expect(note).to_contain_text("--api-key KEY")
    expect(note).to_contain_text("MLX_DECISION_API_KEY")
    expect(note).to_have_class("note")
    sheet.get_by_role("radio", name="Every network interface").check()
    # Without a key this Studio would not start there: said as a warning.
    expect(sheet.locator(".setting-editor .warn-text")).to_contain_text(
        "This Studio runs without one, so it would not start on such an address."
    )
    sheet.get_by_role("button", name="Save").click()
    expect(page.locator(".toast")).to_contain_text(
        'Saved host = "0.0.0.0" · takes effect at the next start'
    )
    assert 'host = "0.0.0.0"' in (studio.home / "config.toml").read_text()
    sheet = change(page, "host")
    expect(sheet.get_by_role("radio", name="Every network interface")).to_be_checked()
    sheet.get_by_label("host").fill("192.168.1.20")
    expect(sheet.get_by_role("radio", name="Another address")).to_be_checked()


def test_the_workspace_folders_and_clear(studio, page):
    open_studio(page, studio, "settings")
    group = page.locator(".workspace-group")
    expect(group.locator(".kv.folder dt")).to_have_text(
        ["sets/", "history/", "data/", "batches/", "models/", "jobs/", "speed/"]
    )
    sets = group.locator(".kv.folder[data-folder='sets']")
    expect(sets.locator("small")).to_contain_text("2 sets")
    batches = group.locator(".kv.folder[data-folder='batches']")
    expect(batches.get_by_role("button", name="Clear…")).to_be_disabled()  # empty

    sets.get_by_role("button", name="Clear…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Clear sets/?")
    expect(sheet).to_contain_text("2 sets")
    expect(sheet).to_contain_text("question sets, with their history")
    sheet.get_by_role("button", name="Clear", exact=True).click()
    expect(page.locator(".toast")).to_contain_text("Cleared sets/: 2 sets deleted")
    expect(sets.locator("small")).to_contain_text("0 sets")
    expect(sets.get_by_role("button", name="Clear…")).to_be_disabled()
    assert list(studio.sets.glob("*.json")) == []
    # The Playground's list follows.
    page.goto(f"{studio.base}/#/playground")
    expect(page.locator(".sidebar .sb-empty", has_text="No question sets yet")).to_be_visible()


def test_settings_name_each_key_in_words_and_the_sidebar_opens_the_budget(studio, page):
    open_studio(page, studio, "settings")
    row = page.locator(".kv.setting[data-key='memory_budget_gb'] dt")
    expect(row).to_contain_text("Memory budget")
    expect(row.locator("code")).to_have_text("memory_budget_gb")
    page.goto(f"{studio.base}/#/playground")
    page.locator(".sidebar .mem-link").click()
    expect(page.locator(".sheet h2")).to_have_text("Change Memory Budget")
    expect(page).to_have_url(re.compile(r"#/settings$"))
