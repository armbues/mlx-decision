"""The Playground's editor and answers: run, cards, staleness, the live check,
editing questions, a model's limits, a JSON state and images."""

import base64
import io
import json

import pytest
from drive import open_studio, pick, pick_model, shot

expect = pytest.importorskip("playwright.sync_api").expect
pytestmark = pytest.mark.browser


def question(page, qid: str):
    return page.locator(f".question[data-q='{qid}']")


def question_menu(page, qid: str, item: str) -> None:
    pick(page, question(page, qid).locator(".gsec-head .right button"), item)


def order(page) -> list[str]:
    return page.locator(".question").evaluate_all("qs => qs.map(q => q.dataset.q)")


def run_triage(page, studio, model: str = "fake") -> None:
    """Open ticket-triage on ``model`` and run it once."""
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, model)
    expect(page.locator(".target .mdl .name")).to_have_text(model)
    page.locator(".run-btn").click()
    expect(page.locator(".acard .verdict")).to_have_count(3)


def png(width: int = 32, height: int = 24) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 102, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_run_with_the_keyboard_answers_with_a_card_per_question(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake")
    expect(page.locator(".check span.ok")).to_be_visible()
    expect(page.locator(".check")).to_contain_text("within fake's limits")
    expect(page.locator(".activity")).to_contain_text("Run sends 3 questions to fake")

    question(page, "department").locator(".ins").click()
    page.keyboard.press("Meta+Enter")
    cards = page.locator(".acard")
    expect(cards.locator(".verdict")).to_have_count(3)
    assert cards.evaluate_all("cards => cards.map(c => c.dataset.q)") == [
        "department",
        "frustration",
        "is_urgent",
    ]
    expect(page.locator(".acard[data-q='department'] .verdict .big")).to_have_text("billing")
    expect(page.locator(".acard[data-q='is_urgent'] .verdict .big")).to_have_text("Yes")
    stats = page.locator(".run-stats")
    for words in ("ms", "input tokens", "Nothing cut", "loaded in"):
        expect(stats).to_contain_text(words)
    expect(page.locator(".activity")).to_contain_text("fake answered in")
    [line] = studio.history_lines("ticket-triage")
    assert line["model"] == "fake"
    shot(page, "playground-answers")

    # A question and its card are linked both ways.
    question(page, "frustration").locator(".ins").click()
    expect(page.locator(".acard.linked[data-q='frustration']")).to_be_visible()
    expect(page.locator(".question.linked[data-q='frustration']")).to_be_visible()
    page.locator(".acard[data-q='is_urgent'] .qid").click()
    expect(question(page, "is_urgent").locator(".ins")).to_be_focused()


def test_an_edit_marks_its_question_card_and_the_run_button(studio, page):
    run_triage(page, studio)
    question(page, "department").locator(".ins").fill("Which team should take it?")
    expect(question(page, "department").locator(".stale-tag")).to_be_visible()
    expect(page.locator(".acard.stale[data-q='department']")).to_be_visible()
    expect(page.locator(".acard.stale")).to_have_count(1)
    expect(page.locator(".run-btn .badge")).to_be_visible()
    expect(page.locator(".activity")).to_contain_text("department edited since the run")


def test_the_live_check_shows_problems_at_their_field(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake")
    department = question(page, "department")
    # fake takes at most 4 options.
    department.locator(".add-row .btn").click()
    department.locator(".add-row .btn").click()
    expect(department.locator(".field-problem")).to_contain_text("at most 4 options")
    expect(page.locator(".run-btn")).to_be_disabled()
    department.locator(".opt-row .minus").last.click()
    expect(page.locator(".field-problem")).to_have_count(0)
    expect(page.locator(".run-btn")).to_be_enabled()
    # Two questions with one id.
    ids = question(page, "frustration").locator(".qid-input")
    ids.fill("department")
    expect(page.locator(".field-problem")).to_contain_text("Two questions are named department")
    page.locator(".question .qid-input").nth(1).fill("frustration")
    expect(page.locator(".field-problem")).to_have_count(0)


def test_editing_questions(studio, page):
    run_triage(page, studio)
    department = question(page, "department")
    # Option-down moves an option, keeping the focus on it.
    keys = department.locator(".opt-key")
    before = keys.evaluate_all("keys => keys.map(k => k.value)")
    keys.first.click()
    page.keyboard.press("Alt+ArrowDown")
    after = keys.evaluate_all("keys => keys.map(k => k.value)")
    assert after[:2] == [before[1], before[0]]
    expect(page.locator(":focus")).to_have_value(before[0])

    # Changing the type keeps what carries over.
    urgent = question(page, "is_urgent")
    question_menu(page, "is_urgent", "Choice")
    expect(urgent.locator(".opt-key")).to_have_count(2)
    assert urgent.locator(".opt-key").evaluate_all("keys => keys.map(k => k.value)") == [
        "yes",
        "no",
    ]
    question_menu(page, "is_urgent", "Score")
    expect(urgent.locator(".level-n")).to_have_count(2)
    expect(urgent.locator(".ins")).to_have_value("Does this need a reply today?")
    question_menu(page, "is_urgent", "Noul")
    expect(urgent.locator("input[placeholder^='Optional: what counts as yes']")).to_be_visible()

    # Duplicate, move, remove.
    question_menu(page, "frustration", "Duplicate")
    expect(question(page, "frustration_copy")).to_be_visible()
    assert order(page) == ["department", "frustration", "frustration_copy", "is_urgent"]
    question_menu(page, "frustration_copy", "Move Down")
    assert order(page)[-2:] == ["is_urgent", "frustration_copy"]
    question_menu(page, "frustration_copy", "Remove Question")
    expect(question(page, "frustration_copy")).to_have_count(0)

    # Add Question focuses the new one; its card says the run did not ask it.
    pick(page, ".add-bar .btn", "Score")
    expect(question(page, "score").locator(".ins")).to_be_focused()
    expect(page.locator(".acard[data-q='score']")).to_contain_text("Not in this run")


def test_a_model_refuses_what_it_does_not_take_at_the_field(studio, page):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake-strict")
    # No score questions.
    expect(question(page, "frustration").locator(".field-problem")).to_contain_text(
        "does not support score questions"
    )
    question_menu(page, "frustration", "Remove Question")
    expect(page.locator(".field-problem")).to_have_count(0)
    # Instructions required.
    pick(page, ".add-bar .btn", "Noul")
    added = question(page, "noul")
    expect(added.locator(".field-problem")).to_contain_text("instructions are required")
    expect(added.locator(".ins")).to_have_attribute("placeholder", "Required by fake-strict")
    question_menu(page, "noul", "Remove Question")
    expect(page.locator(".field-problem")).to_have_count(0)


def test_a_json_state(studio, page):
    # An edit of one set is kept while another is open.
    open_studio(page, studio, "playground/ticket-triage")
    question_menu(page, "frustration", "Remove Question")
    page.goto(f"{studio.base}/#/playground/order-json")
    expect(question(page, "refund")).to_be_visible()
    pick_model(page, "fake")
    expect(page.locator(".seg [aria-pressed=true]").first).to_have_text("JSON")
    state = page.locator(".state-text")
    text = state.input_value()
    state.fill(text.replace('"total"', "total"))
    expect(page.locator(".field-problem")).to_contain_text("not valid JSON")
    state.fill(text)
    expect(page.locator(".field-problem")).to_have_count(0)
    expect(page.locator("input[placeholder^='Optional: what counts as yes']")).to_have_value(
        "the customer asks for money back"
    )
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".acard .verdict")).to_have_count(1)
    assert studio.history_lines("order-json")[0]["request"]["state"] == json.loads(text)

    page.goto(f"{studio.base}/#/playground/ticket-triage")
    expect(page.locator(".question")).to_have_count(2)


def test_images_on_a_model_that_reads_them(studio, page, tmp_path):
    open_studio(page, studio, "playground/ticket-triage")
    pick_model(page, "fake-vision")
    # The drop area names the types and the model's limits.
    limits = page.locator(".drop .limits")
    expect(limits).to_have_text("PNG, JPEG, WebP · at most 4 · shrunk to 2.1 MP")
    expect(limits).to_have_attribute(
        "title",
        "fake-vision reads at most 4 images an example. "
        "Larger images are shrunk to 2,097,152 pixels, keeping their shape.",
    )
    image = tmp_path / "a.png"
    image.write_bytes(png())
    page.locator(".drop input[type=file]").set_input_files(str(image))
    expect(page.locator(".thumb img")).to_have_count(1)
    page.keyboard.press("Meta+Enter")
    expect(page.locator(".run-stats")).to_be_visible()
    expect(page.locator(".stale-banner")).to_have_count(0)
    assert "sha256" in json.dumps(studio.history_lines("ticket-triage")[-1]["request"]["images"])
    shot(page, "playground-image")

    # A file that is not an image is refused at the images.
    broken = tmp_path / "b.png"
    broken.write_bytes(base64.b64decode("AAAA"))
    page.locator(".drop input[type=file]").set_input_files(str(broken))
    expect(page.locator(".field-problem")).to_contain_text("image")
    page.locator(".thumb .minus").last.click()
    expect(page.locator(".field-problem")).to_have_count(0)

    # More images than the model reads: the live check says so before Run.
    page.locator(".drop input[type=file]").set_input_files([str(image)] * 4)
    expect(page.locator(".thumb img")).to_have_count(5)
    expect(page.locator(".field-problem")).to_contain_text(
        "5 images; this model reads at most 4 per request"
    )
    page.locator(".thumb .minus").last.click()
    expect(page.locator(".field-problem")).to_have_count(0)

    # A model that reads text only refuses them, and the drop area says so.
    pick_model(page, "fake")
    expect(page.locator(".field-problem")).to_contain_text("does not support images")
    expect(page.locator(".drop")).to_contain_text("reads text only")
    page.set_viewport_size({"width": 1100, "height": 900})
    shot(page, "playground-problem-1100")
