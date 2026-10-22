"""Convert: the sheet's sizes and name, a job that holds the GPU, Stop, and the
copy in the list when done. The job is ``tests/fake_job.py``."""

import re

import pytest
from drive import open_studio
from fake_studio import fake_jobs

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


@pytest.fixture
def studio(make_studio):
    return make_studio(convertible=True)


def inspector_convert(page):
    return page.locator(".idet-actions").get_by_role("button", name="Convert…")


def test_convert_a_model_then_stop_it(studio, page):
    open_studio(page, studio, "models/fake")
    expect(inspector_convert(page)).to_be_disabled()
    expect(inspector_convert(page)).to_have_attribute(
        "title", "fake cannot be converted: fake models are not stored quantized"
    )
    page.goto(f"{studio.base}/#/models/tiny-clef")
    expect(inspector_convert(page)).to_be_enabled()
    inspector_convert(page).click()

    sheet = page.locator(".sheet")
    expect(sheet.locator("h2")).to_have_text("Convert tiny-clef")
    options = sheet.locator(".qopt")
    expect(options.locator("b")).to_have_text(["8-bit", "4-bit"])
    expect(options.nth(0)).to_contain_text(re.compile(r"about [\d.]+ (MB|GB) on disk"))
    expect(sheet).to_contain_text("Sizes are estimates")
    expect(sheet).to_contain_text("Converting needs the GPU to itself")
    name = sheet.get_by_label("Name")
    expect(name).to_have_value("tiny-clef-q8")
    options.nth(1).click()
    expect(name).to_have_value("tiny-clef-q4")  # follows the bits until typed
    convert = sheet.get_by_role("button", name="Convert", exact=True)
    name.fill("fake")
    expect(sheet.locator(".warn-text")).to_have_text("A model named fake exists already")
    expect(convert).to_be_disabled()
    name.fill("../x")
    expect(convert).to_be_disabled()
    name.fill("my-copy")
    options.nth(0).click()
    expect(name).to_have_value("my-copy")  # typed: kept
    convert.click()

    expect(sheet).to_have_count(0)
    expect(page.locator(".toast")).to_contain_text("Convert my-copy: the copy shows up in Models")
    row = page.locator(".mrow.mjob")
    expect(row).to_contain_text("8-bit copy of tiny-clef")
    expect(page.locator(".toolbar .activity")).to_contain_text("Convert my-copy has the GPU")
    expect(inspector_convert(page)).to_be_disabled()  # one GPU job at a time
    job = next(iter(studio.app.state.jobs.jobs.values()))
    output = (studio.workspace / "models" / "my-copy").resolve()
    assert job.argv[0] == "convert" and job.argv[3:] == ["-o", str(output), "--bits", "8"]
    assert job.details == {"model": "tiny-clef", "bits": 8, "name": "my-copy"}

    page.locator(".sidebar .job").get_by_role("button", name="Stop Convert my-copy").click()
    expect(page.locator(".sidebar .job small")).to_have_text("Stopped")
    expect(row).to_have_count(0)
    assert not any((studio.workspace / "models").iterdir())  # no copy, no .partial
    expect(inspector_convert(page)).to_be_enabled()


def test_the_copy_shows_up_when_done(studio, page):
    fake_jobs(studio.app, steps=10, every=0.05)
    studio.app.state.jobs.command.append("--fake-model")
    open_studio(page, studio, "models")
    page.locator(".mrow[data-name='tiny-clef']").click(button="right")
    page.get_by_role("menuitem", name="Convert…").click()
    page.locator(".sheet").get_by_role("button", name="Convert", exact=True).click()
    expect(page.locator(".mrow[data-name='tiny-clef-q8']")).to_be_visible(timeout=15_000)
    expect(page.locator(".sidebar .job small")).to_have_text("Done")
