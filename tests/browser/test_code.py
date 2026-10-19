"""Get Code: the sheet from the toolbar, its four snippets (curl, Python, the
command line, an MCP client's config) with the Studio's address, the model and
the set's file, Copy, another example's command, unsaved edits, and the model
run on while comparing. That the snippets run is checked by hand."""

import base64
import io
import json
import random
import re
from pathlib import Path

import pytest
from drive import open_studio, pick_model, put_set, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def noise_png() -> str:
    """A PNG big enough to be shortened on screen, as a data URL."""
    from PIL import Image

    rng = random.Random(0)
    image = Image.new("RGB", (48, 48))
    image.putdata(
        [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(48 * 48)]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


IMAGE = noise_png()
BODY = {
    "state": "Hi, I can't log in since the update.",
    "images": [IMAGE],
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payments", "technical": "Bugs"},
        },
        "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    },
    "examples": [
        {"name": "Login", "state": "Hi, I can't log in since the update.", "images": [IMAGE]},
        {"name": "Refund", "state": "Please refund the second charge, it's a mistake."},
    ],
}
SNIPPETS = {"curl": "curl", "Python": "python", "Command Line": "cli", "MCP Client": "mcp"}


def open_code_set(page, studio) -> None:
    """The set ``code`` on fake-vision (it reads images)."""
    open_studio(page, studio)
    put_set(page, studio, "code", BODY)
    open_studio(page, studio, "playground/code")
    pick_model(page, "fake-vision")
    expect(page.locator(".target .mdl .name")).to_have_text("fake-vision")


def get_code(page) -> None:
    page.locator(".get-code").click()
    expect(page.locator(".sheet pre.code")).to_be_visible()


def snippet(page, name: str) -> str:
    """Show the sheet's snippet ``name`` and return its text."""
    page.locator(".sheet [role=tab]", has_text=re.compile(f"^{re.escape(name)}$")).click()
    code = page.locator(f".sheet pre.code[data-snippet={SNIPPETS[name]}]")
    expect(code).to_be_visible()
    return code.text_content()


def heredoc(text: str):
    """The JSON between ``<<'JSON'`` and ``JSON``."""
    return json.loads(text.split("<<'JSON'\n", 1)[1].rsplit("\nJSON", 1)[0])


def server_status(page, studio) -> dict:
    response = page.request.get(f"{studio.base}/studio/api/status")
    assert response.ok, response.text()
    return response.json()


def test_curl_shows_the_request_and_copy_takes_it_whole(studio, page):
    # Headless Chrome has a clipboard of its own, not the system's.
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_code_set(page, studio)
    get_code(page)
    expect(page.locator(".sheet-head p")).to_have_text(
        "code on fake-vision: 2 questions about Login."
    )

    curl = snippet(page, "curl")
    assert curl.startswith(f"curl {studio.base}/v1/systemone"), curl[:80]
    assert "Authorization" not in curl  # the Studio has no key
    shown = heredoc(curl)
    assert list(shown)[0] == "model" and shown["model"] == "fake-vision", list(shown)
    # The image is shortened on screen.
    assert shown["images"][0].endswith("KB)") and len(shown["images"][0]) < 80

    page.locator(".sheet .btn.copy").click()
    expect(page.locator(".sheet .btn.copy")).to_have_text("Copied")
    copied = heredoc(page.evaluate("navigator.clipboard.readText()"))
    assert copied["images"] == [IMAGE]
    assert {**copied, "images": None} == {**shown, "images": None}
    shot(page, "code-curl")


def test_python_uses_the_sdk_on_the_studio(studio, page):
    open_code_set(page, studio)
    get_code(page)
    python = snippet(page, "Python")
    assert f'TypeSafeClient(base_url="{studio.base}", api_key="unused")' in python, python[:200]
    assert "from typesafe_sdk import Choice, Noul, TypeSafeClient" in python
    assert '"team": Choice(' in python
    assert '"urgent": Noul(' in python
    assert 'model="fake-vision"' in python
    assert 'extra_body={"images": [' in python


def test_the_command_line_and_mcp_name_the_file_and_the_cli(studio, page):
    open_code_set(page, studio)
    status = server_status(page, studio)
    get_code(page)
    cli = snippet(page, "Command Line")
    file = str(Path(status["sets_folder"]) / "code.json")
    assert cli.startswith("mlx-decision run -m ") and cli.endswith(f"-q {file}"), cli
    expect(page.locator(".snip-note")).to_contain_text(status["cli_path"])

    server = json.loads(snippet(page, "MCP Client"))["mcpServers"]["decision"]
    assert server["command"] == status["cli_path"], server
    assert server["args"][:2] == ["mcp", "-m"], server
    expect(page.locator(".snip-note")).to_contain_text("claude mcp add decision -- ")
    shot(page, "code-mcp")

    page.locator(".sheet-foot .btn.primary").click()
    expect(page.locator(".sheet")).to_have_count(0)


def test_another_example_is_a_states_line_without_the_first_examples_image(studio, page):
    open_code_set(page, studio)
    page.locator(".ex-chip [role=tab]", has_text=re.compile("^Refund$")).click()
    expect(page.locator(".state-text")).to_have_value(
        "Please refund the second charge, it's a mistake."
    )
    get_code(page)
    cli = snippet(page, "Command Line")
    assert "--states -" in cli, cli[-160:]
    assert heredoc(cli) == {
        "state": "Please refund the second charge, it's a mistake.",
        "images": [],
    }
    expect(page.locator(".sheet-head p")).to_have_text(re.compile(r"about Refund\.$"))
    page.keyboard.press("Escape")
    expect(page.locator(".sheet")).to_have_count(0)


def test_an_unsaved_edit_says_to_save_first(studio, page):
    open_code_set(page, studio)
    page.locator(".question[data-q='urgent'] .ins").fill("Is it urgent?")
    expect(page.locator(".edited-tag")).to_be_visible()
    get_code(page)
    snippet(page, "Command Line")
    expect(page.locator(".snip-note")).to_contain_text("save first")
    # The other snippets follow the edit.
    assert '"Is it urgent?"' in snippet(page, "Python")
    page.keyboard.press("Escape")
    expect(page.locator(".sheet")).to_have_count(0)


def test_while_comparing_the_code_is_for_the_model_run_on(studio, page):
    open_code_set(page, studio)
    pick_model(page, "fake", "Compare With")
    expect(page.locator(".run-btn")).to_contain_text("Compare")
    get_code(page)
    assert '"model": "fake-vision"' in snippet(page, "curl")
    page.emulate_media(color_scheme="dark")
    snippet(page, "Python")
    shot(page, "code-python-dark")
