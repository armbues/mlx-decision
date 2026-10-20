"""Training data in the Studio: uploads checked as training fit checks them,
errors by line, name clashes and escapes refused, and a file's summary."""

import json

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.data.format import check_examples
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio

HERE = "http://127.0.0.1:8000"
TEAM = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Money", "tech": "Bugs", "sales": "Plans"}}  # fmt: skip
URGENT = {"type": "noul"}


def line(i, team="tech", urgent=None, **extra):
    labels = {"team": team} | ({"urgent": urgent} if urgent is not None else {})
    return json.dumps(
        {"state": f"ticket {i}", "questions": {"team": TEAM, "urgent": URGENT}, "labels": labels}
        | extra
    )


GOOD = (
    "\n".join(
        [line(i, "tech") for i in range(6)] + [line(6, "billing", True), line(7, "sales", False)]
    )
    + "\n"
)


@pytest.fixture
def studio(tmp_path):
    write_model(tmp_path / "models" / "fake")
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(
        pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml"),
        max_body_bytes=200_000,
    )  # fmt: skip
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.workspace = tmp_path / "ws"
        client.app_state = app.state
        yield client


def upload(studio, name, text):
    return studio.post("/studio/api/data", json={"name": name, "text": text})


def test_a_good_file_is_saved_and_described(studio):
    response = upload(studio, "tickets", GOOD)
    assert response.status_code == 200, response.text
    body = response.json()
    path = studio.workspace / "data" / "tickets.jsonl"
    assert path.read_text() == GOOD and body["path"] == str(path.resolve())
    assert body["examples"] == 8
    team, urgent = body["questions"]
    assert team == {
        "id": "team", "type": "choice", "instructions": "Which team?",
        "options": ["billing", "tech", "sales"], "labelled": 8,
        "labels": [["tech", 6], ["billing", 1], ["sales", 1]],
    }  # fmt: skip
    assert urgent["labelled"] == 2 and urgent["options"] == ["true", "false"]
    assert len(body["sample"]) == 5 and body["sample"][0] == {
        "id": "1", "state": "ticket 0", "labels": {"team": "tech"},
    }  # fmt: skip
    files = studio.get("/studio/api/data").json()["files"]
    assert [(f["name"], f["bytes"]) for f in files] == [("tickets", len(GOOD.encode()))]
    assert studio.get("/studio/api/data/tickets").json() == body
    # Other tabs hear of it.
    recent = [message for _, (_, message) in studio.app_state.hub.recent]
    assert {"type": "workspace", "folder": "data", "name": "tickets", "action": "saved"} in recent


def test_bad_lines_are_named(studio):
    text = "\n".join([
        line(0),
        line(1, "nobody"),
        "{not json",
        json.dumps({"state": "x", "questions": {"team": TEAM}}),  # no labels
        line(4, urgent="maybe"),
        line(5, id="1"),
    ])  # fmt: skip
    response = upload(studio, "broken", text)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"].startswith("line 2: ")
    assert [e["line"] for e in error["errors"]] == [2, 3, 4, 5, 6]
    assert '"nobody" is none of the options' in error["errors"][0]["message"]
    assert error["errors"][0]["param"] == "labels.team"
    assert error["errors"][3]["message"] == "a noul label is true or false"
    assert error["errors"][4]["message"] == "the same id as line 1"
    assert not (studio.workspace / "data" / "broken.jsonl").exists()
    many = "\n".join("{bad" for _ in range(50))
    assert len(upload(studio, "many", many).json()["error"]["errors"]) == 20
    for text, message in (("", "the data holds no examples"),
                          (json.dumps({"state": "x", "questions": {"team": TEAM}, "labels": {}}),
                           "no example has a label")):  # fmt: skip
        error = upload(studio, "empty", text).json()["error"]
        assert error["message"] == message and error["errors"][0]["line"] is None


def test_a_name_clash_and_escapes_are_refused(studio):
    assert upload(studio, "tickets", GOOD).status_code == 200
    clash = upload(studio, "tickets", GOOD)
    assert clash.status_code == 409 and "exists already" in clash.text
    for name in ("../tickets", "/etc/x", ".hidden", "a/b", "", "x" * 81):
        assert upload(studio, name, GOOD).status_code == 400, name
    assert studio.get("/studio/api/data/..%2Fsets").status_code in (400, 404)
    assert studio.get("/studio/api/data/missing").status_code == 404
    assert sorted(p.name for p in (studio.workspace / "data").iterdir()) == ["tickets.jsonl"]


def test_the_body_limit(studio):
    big = GOOD * 200  # about 320 kB: over the 200 kB this Studio takes
    assert upload(studio, "big", big).status_code == 413


def test_a_file_changed_behind_the_studios_back(studio):
    upload(studio, "tickets", GOOD)
    (studio.workspace / "data" / "tickets.jsonl").write_text("{broken\n")
    response = studio.get("/studio/api/data/tickets")
    assert response.status_code == 422 and "line 1" in response.text


def test_check_examples_goes_on_past_errors():
    examples, errors = check_examples([line(0), "{x", line(2), "", line(3, "nobody")])
    assert [e.id for e in examples] == ["1", "3"]
    assert [e.line for e in errors] == [2, 5]
    examples, errors = check_examples([line(i) for i in range(3)])
    assert len(examples) == 3 and errors == []


def test_the_list_is_by_name(studio):
    for name in ("tickets-2", "Alpha", "tickets"):
        assert upload(studio, name, GOOD).status_code == 200
    names = [item["name"] for item in studio.get("/studio/api/data").json()["files"]]
    assert names == ["Alpha", "tickets", "tickets-2"]
