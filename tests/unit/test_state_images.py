"""Images inside a request's state, lifted out for the families that read images."""

import base64
import io

import pytest
from PIL import Image

import mlx_decision
from mlx_decision import DecisionError
from mlx_decision.state_images import lift_images
from tiny_models import write_clef, write_h2o, write_marker, write_pplx

QUESTIONS = {"team": {"type": "choice", "criteria": {"billing": None, "sales": None}}}


def image(width=16, height=16, kind="PNG", color=(200, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=kind)
    return buffer.getvalue()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def data_url(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64,{b64(raw)}"


PNG = image()


def test_a_data_url_state_becomes_a_placeholder():
    state, found = lift_images(data_url(PNG))
    assert state == "[image 1]"
    assert found == [(PNG, "state")]


def test_images_anywhere_in_the_state_are_numbered_after_the_top_level_ones():
    jpeg = image(kind="JPEG")
    state = {
        "note": "two photos",
        "photos": [data_url(PNG), {"caption": "x", "file": b64(jpeg)}],
        "link": "https://example.com/a.png",  # a plain URL is text
    }
    lifted, found = lift_images(state, first=2)
    assert lifted == {
        "note": "two photos",
        "photos": ["[image 3]", {"caption": "x", "file": "[image 4]"}],
        "link": "https://example.com/a.png",
    }
    assert found == [(PNG, "state.photos.0"), (jpeg, "state.photos.1.file")]


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": data_url(PNG)}},
        {"type": "image_url", "image_url": data_url(PNG)},
        {"type": "image", "image": b64(PNG)},
        {"type": "input_image", "image_url": data_url(PNG)},
        {"type": "base64", "media_type": "image/png", "data": b64(PNG)},
    ],
)
def test_image_parts_are_lifted(part):
    assert lift_images({"attachment": part}) == (
        {"attachment": "[image 1]"},
        [(PNG, "state.attachment")],
    )


@pytest.mark.parametrize(
    "state",
    [
        "plain text",
        "iVBORw0KGgo" + "A" * 10,  # too short for bare base64
        b64(b"not an image at all, but long enough to be base64 text " * 3),
        b64(PNG)[:40] + " " + b64(PNG)[40:],  # broken by a space
        {"type": "image", "image": "a sunset"},  # shaped like a part, but data
        ["a", 1, None, {"b": True}],
    ],
)
def test_text_stays_as_it_is(state):
    lifted, found = lift_images(state)
    assert lifted is state
    assert found == []


@pytest.mark.parametrize(
    ("state", "param", "message"),
    [
        ({"a": "data:image/png;base64,!!!"}, "state.a", "valid base64"),
        (
            [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}],
            "state.0",
            "not fetched",
        ),
    ],
)
def test_bad_images_name_their_place_in_the_state(state, param, message):
    with pytest.raises(DecisionError, match=message) as error:
        lift_images(state)
    assert error.value.param == param


@pytest.fixture(scope="module")
def clef(tmp_path_factory):
    return mlx_decision.load(write_clef(tmp_path_factory.mktemp("clef") / "clef", vision=True))


@pytest.fixture(scope="module")
def pplx(tmp_path_factory):
    return mlx_decision.load(write_pplx(tmp_path_factory.mktemp("pplx") / "pplx"))


@pytest.fixture(scope="module")
def h2o(tmp_path_factory):
    return mlx_decision.load(write_h2o(tmp_path_factory.mktemp("h2o") / "h2o"))


@pytest.mark.parametrize("family", ["clef", "pplx", "h2o"])
def test_a_lifted_image_answers_as_the_same_image_in_images(family, request):
    model = request.getfixturevalue(family)
    other = image(color=(20, 200, 30))
    lifted = model.decide_request(
        {"state": {"photo": data_url(PNG)}, "questions": QUESTIONS, "images": [data_url(other)]}
    )
    given = model.decide_request(
        {"state": {"photo": "[image 2]"}, "questions": QUESTIONS, "images": [other, PNG]}
    )
    assert lifted.answers == given.answers
    assert lifted.usage == given.usage


def test_an_unreadable_lifted_image_is_named_by_its_place(clef):
    gif = data_url(image(kind="GIF"), "image/gif")
    with pytest.raises(DecisionError, match="PNG, JPEG or WebP") as error:
        clef.decide({"photos": ["fine", gif]}, QUESTIONS)
    assert error.value.param == "state.photos.1"


def test_check_returns_the_lifted_request(clef):
    checked = clef.check({"state": data_url(PNG), "questions": QUESTIONS})
    assert checked.state == "[image 1]"
    assert checked.images == [PNG]


def test_families_without_images_read_the_state_as_text(tmp_path):
    laya = mlx_decision.load(write_marker(tmp_path / "laya", "laya"))
    checked = laya.check({"state": data_url(PNG), "questions": QUESTIONS})
    assert checked.state == data_url(PNG)
    assert not checked.images
