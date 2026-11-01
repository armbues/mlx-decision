"""Download Model: the sheet's families, models and ways of storing one, the
check of a typed repo id, and the download that becomes a job. The check is
answered by ``fake_check`` (no network); the job is ``tests/fake_job.py``."""

import httpx
import pytest
from drive import open_studio

from mlx_decision.download import RepoCheck
from mlx_decision.studio.routes import downloads as downloads_route

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser

GB = 10**9
FAMILIES = ["Clef", "pplx", "d3", "Base Labs", "H2O-Lightning", "d1 LFM3", "d1", "Laya", "Julia"]


def fake_check(repo_id: str) -> RepoCheck:
    from huggingface_hub.errors import RepositoryNotFoundError

    if repo_id == "org/unknown":
        return RepoCheck(repo_id, 3 * GB, None, "no model family recognises it")
    if repo_id == "org/missing":
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
        raise RepositoryNotFoundError("missing", response=httpx.Response(404, request=request))
    family = {"vllm-sr": "d3", "LiquidAI": "d1"}.get(repo_id.split("/")[0], "clef")
    size = int(1.2 * GB) if repo_id == "vllm-sr/d3-edge" else 16 * GB
    return RepoCheck(repo_id, size, family, weight_bytes=size, quantizable_bytes=int(size * 0.9))


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setattr(downloads_route, "check_repo", fake_check)


def test_pick_a_model_and_download_it(studio, page, hub):
    open_studio(page, studio, "models")
    page.get_by_role("button", name="Download a Model…").click()
    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Download a Model")
    expect(sheet).to_contain_text("Into the Hugging Face cache")
    expect(sheet.locator(".fam b")).to_have_text(FAMILIES)

    sheet.locator(".fam[data-family='d3']").click()
    expect(sheet.locator(".fam-pane > p")).to_contain_text("pplx's design")  # the summary alone
    sheet.locator(".crow[data-repo='vllm-sr/d3-edge']").click()
    options = sheet.locator(".qopt")
    expect(options.locator("b")).to_have_text(["Original", "8-bit", "4-bit"])
    # The largest that fits the fake Studio's 1 GB budget is checked first:
    # 8-bit (about 0.8 GB to run); full size is over the budget, not this Mac.
    expect(sheet.locator(".qopt[aria-checked='true'] b")).to_have_text("8-bit")
    expect(options.nth(0)).to_contain_text("1.2 GB on disk")  # exact
    expect(options.nth(1)).to_contain_text("about")  # estimates say so
    expect(options.nth(0).locator(".fit")).to_have_text("Over the memory budget")
    expect(options.nth(2).locator(".fit")).to_have_text("Fits the memory budget")
    options.nth(2).click()
    expect(sheet.locator(".qopt[aria-checked='true'] b")).to_have_text("4-bit")
    options.nth(1).click()

    download = sheet.get_by_role("button", name="Download", exact=True)
    expect(download).to_be_enabled()
    download.click()
    expect(sheet).to_have_count(0)
    expect(page.locator(".toast")).to_contain_text("Started Download vllm-sr/d3-edge")
    row = page.locator(".mrow.mjob")
    expect(row).to_contain_text("d3-edge")
    expect(row).to_contain_text("vllm-sr/d3-edge · 8-bit")
    expect(page.locator(".sidebar .job")).to_contain_text("Download vllm-sr/d3-edge (8-bit)")

    # ⌘D, from another page too: the model says it is downloading.
    page.goto(f"{studio.base}/#/playground/ticket-triage")
    page.wait_for_selector(".question")
    page.keyboard.press("Meta+d")
    sheet.locator(".fam[data-family='d3']").click()
    sheet.locator(".crow[data-repo='vllm-sr/d3-edge']").click()
    expect(sheet.locator(".crow[data-repo='vllm-sr/d3-edge'] .getting")).to_have_text("Downloading")
    expect(sheet.get_by_role("button", name="Downloading")).to_be_disabled()
    page.keyboard.press("Escape")
    expect(sheet).to_have_count(0)
    # Stop leaves nothing behind.
    page.locator(".sidebar .job").get_by_role(
        "button", name="Stop Download vllm-sr/d3-edge (8-bit)"
    ).click()
    expect(page.locator(".sidebar .job small")).to_have_text("Stopped")


def test_a_typed_repo_id_is_checked_first(studio, page, hub):
    open_studio(page, studio, "models")
    page.get_by_role("button", name="Download a Model…").click()
    sheet = page.locator(".sheet")
    field = sheet.locator("#repo-id")
    download = sheet.get_by_role("button", name="Download", exact=True)

    field.fill("org/unknown")
    field.press("Enter")  # checks; it does not download
    expect(sheet.locator(".notice")).to_contain_text(
        "org/unknown: no model family recognises it. The Studio cannot load it"
    )
    expect(download).to_be_disabled()
    expect(sheet.locator(".fam[aria-selected='true']")).to_have_count(0)

    field.fill("org/missing")
    sheet.get_by_role("button", name="Check").click()
    expect(sheet.locator(".notice")).to_contain_text("org/missing: not found on the Hub")
    expect(download).to_be_disabled()

    field.fill("not a repo")
    field.press("Enter")
    expect(page.locator(".toast")).to_contain_text("Type a repo id as org/name")

    field.fill("org/clef-copy")
    field.press("Enter")
    expect(sheet.locator(".qopt b")).to_have_text(["Original", "8-bit", "4-bit"])
    expect(sheet.locator(".fam-pane")).to_contain_text("Clef model")
    expect(download).to_be_enabled()


def test_a_licence_note_comes_before_the_download(studio, page, hub):
    open_studio(page, studio, "models")
    page.keyboard.press("Meta+d")
    sheet = page.locator(".sheet")
    sheet.locator(".fam[data-family='d1']").click()
    expect(sheet.locator(".notice.info")).to_contain_text("LFM Open License")
    # d1 is not stored quantized: full size only, and the sheet says so.
    expect(sheet.locator(".qopt b")).to_have_text(["Original"])
    expect(sheet.locator(".opts .note")).to_contain_text("not stored quantized")
