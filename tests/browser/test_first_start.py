"""First start: an empty Studio opens on Models, asks once where models go
(written to the config file), then offers a small, a medium and a large
model; a pick opens Download Model on it."""

import re

import pytest
from drive import open_studio

from mlx_decision.download import RepoCheck
from mlx_decision.studio.routes import downloads as downloads_route

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

GB = 10**9


def fake_check(repo_id: str) -> RepoCheck:
    family = {"vllm-sr": "d3", "LiquidAI": "d1", "h2oai": "h2o"}.get(repo_id.split("/")[0], "laya")
    return RepoCheck(repo_id, GB, family, weight_bytes=GB, quantizable_bytes=GB)


@pytest.fixture
def empty(make_studio, monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(downloads_route, "check_repo", fake_check)
    studio = make_studio(empty=True)
    studio.user_home = home
    return studio


def test_an_empty_studio_asks_where_models_go_then_offers_three(empty, page):
    open_studio(page, empty)  # no route: the Studio picks Models
    expect(page).to_have_url(re.compile(r"#/models$"))
    card = page.locator(".first-card")
    expect(card.locator("h2")).to_have_text("Where should models go?")
    expect(card.locator("input[value='hf-cache']")).to_be_checked()  # preselected
    card.locator(".folder-path").fill("~/Models/Decision")
    expect(card.locator("input[value='folder']")).to_be_checked()
    card.get_by_role("button", name="Continue").click()

    expect(card.locator("h2")).to_have_text("Download your first model")
    config = empty.home / "config.toml"
    expect(card.locator(".saved")).to_contain_text(
        f'Saved models_dir = "~/Models/Decision" in {config}'
    )
    assert config.read_text() == 'models_dir = "~/Models/Decision"\n'
    assert (empty.user_home / "Models" / "Decision").is_dir()

    picks = card.locator(".first-pick")
    expect(picks.locator(".tier")).to_have_text(["Small", "Medium", "Large"])
    expect(picks.nth(0)).to_contain_text("download · about")
    repo = picks.nth(1).get_attribute("data-repo")
    picks.nth(1).click()
    sheet = page.locator(".sheet")
    expect(sheet.locator(f".crow[data-repo='{repo}']")).to_have_attribute("aria-checked", "true")
    expect(sheet).to_contain_text(f"Into {empty.user_home / 'Models' / 'Decision'}")
    sheet.get_by_role("button", name="Download", exact=True).click()
    # The download's row takes the panel's place.
    expect(page.locator(".mrow.mjob")).to_contain_text(repo.split("/")[1])
    expect(page.locator(".first-card")).to_have_count(0)
    job = next(iter(empty.app.state.jobs.jobs.values()))
    assert job.argv[-2:] == [
        "--local-dir",
        str(empty.user_home / "Models" / "Decision" / repo.split("/")[1]),
    ]
    page.locator(".sidebar .job").get_by_role("button", name=re.compile("^Stop ")).click()


def test_the_hugging_face_cache_is_the_default_answer(empty, page):
    open_studio(page, empty)
    card = page.locator(".first-card")
    card.get_by_role("button", name="Continue").click()
    expect(card.locator(".saved")).to_contain_text('Saved models_dir = "hf-cache"')
    assert (empty.home / "config.toml").read_text() == 'models_dir = "hf-cache"\n'
    # Asked once: a reload goes straight to the picks.
    page.reload()
    expect(card.locator("h2")).to_have_text("Download your first model")
    expect(card.locator(".saved")).to_have_count(0)


def test_a_folder_outside_the_home_folder_is_refused(empty, page, tmp_path):
    open_studio(page, empty)
    card = page.locator(".first-card")
    card.locator(".folder-path").fill(str(tmp_path / "elsewhere"))
    card.get_by_role("button", name="Continue").click()
    expect(page.locator(".toast")).to_contain_text("not a folder under your home folder")
    expect(card.locator("h2")).to_have_text("Where should models go?")
    assert not (empty.home / "config.toml").exists()
