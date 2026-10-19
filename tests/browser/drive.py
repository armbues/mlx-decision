"""Helpers for driving the Studio's page with Playwright in the browser tests."""

import os
import re
from pathlib import Path

from fake_studio import Served


def open_studio(page, studio: Served, route: str = "") -> None:
    """Log the tab in (the token sets the session cookie) and open ``route``,
    e.g. ``playground/ticket-triage``."""
    page.goto(studio.url)
    page.wait_for_selector(".app")
    if route:
        page.goto(f"{studio.base}/#/{route}")
        page.wait_for_selector(".app")


def put_set(page, studio: Served, name: str, body: dict) -> None:
    """Write a set through the Studio's API, as the page saves one. The tab must
    be logged in first (``open_studio``): the request uses its session cookie."""
    response = page.request.put(
        f"{studio.base}/studio/api/sets/{name}", data={"body": body, "loaded": None}
    )
    assert response.ok, response.text()


# The index of the open menu's item whose first rendered line is the label (in
# a section, the items after its header up to the next one).
FIND_ITEM = """(menu, [label, section]) => {
  let kids = [...menu.children];
  if (section !== null) {
    const header = (e) => e.classList.contains("mh") && e.innerText.trim() === section;
    kids = kids.slice(kids.findIndex(header) + 1);
    const next = kids.findIndex(e => e.classList.contains("mh"));
    if (next >= 0) kids = kids.slice(0, next);
  }
  const first = (e) => e.innerText.trim().split("\\n")[0].trim();
  const item = kids.find(e => e.classList.contains("mi") && first(e) === label);
  return [...menu.querySelectorAll(".mi")].indexOf(item);
}"""


def menu_item(page, label: str, section: str | None = None):
    """The open menu's item whose first line is ``label`` (in ``section``, a
    menu header, when given)."""
    page.wait_for_selector(".menu .mi")
    index = page.locator(".menu").evaluate(FIND_ITEM, [label, section])
    assert index >= 0, f"no menu item {label!r}" + (f" under {section!r}" if section else "")
    return page.locator(".menu .mi").nth(index)


def pick(page, opener, label: str, section: str | None = None) -> None:
    """Open a menu with ``opener`` (a selector or locator) and choose ``label``."""
    (page.locator(opener) if isinstance(opener, str) else opener).click()
    page.wait_for_selector(".menu")
    menu_item(page, label, section).click()


def pick_model(page, name: str, section: str = "Run on") -> None:
    """Choose a model in the Playground's model menu (or tick one under
    "Compare With")."""
    pick(page, ".target .mdl", name, section)


def inspector_tab(page, name: str) -> None:
    page.locator(".insp-tabs [role=tab]", has_text=re.compile(f"^{re.escape(name)}$")).click()


def answers_view(page, name: str) -> None:
    page.locator(".answers-view [role=tab]", has_text=re.compile(f"^{re.escape(name)}$")).click()


def shot(page, name: str) -> None:
    """A screenshot for review, written only when ``STUDIO_SHOTS`` names a folder."""
    if folder := os.environ.get("STUDIO_SHOTS"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(folder) / f"{name}.png"))
