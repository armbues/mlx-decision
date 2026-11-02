"""Training data in the Studio: uploads checked as training fit checks them,
errors by line, name clashes and escapes refused, a file's summary, and its
deletion (not while a job reads it)."""

import json
from types import SimpleNamespace

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
    assert body["display"] == body["path"]  # outside the home folder: as it is
    assert body["examples"] == 8
    team, urgent = body["questions"]
    assert team == {
        "id": "team", "type": "choice", "instructions": "Which team?",
        "options": ["billing", "tech", "sales"], "labelled": 8,
        "labels": [["tech", 6], ["billing", 1], ["sales", 1]], "answered_by": None,
    }  # fmt: skip
    assert urgent["labelled"] == 2 and urgent["options"] == ["true", "false"]
    assert len(body["sample"]) == 5 and body["sample"][0] == {
        "id": "1", "state": "ticket 0", "labels": {"team": "tech"}, "values": {},
    }  # fmt: skip
    assert (body["kind"], body["split"], body["missing"]) == ("file", "tickets", [])
    listed = studio.get("/studio/api/data").json()
    files = listed["files"]
    assert [(f["name"], f["bytes"]) for f in files] == [("tickets", len(GOOD.encode()))]
    # The folder, and the most an upload holds (a larger file is copied there).
    assert listed["folder"] == str(path.parent.resolve()) == listed["display"]
    assert listed["upload_limit"] == 200_000
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
    assert error["errors"][3]["message"] == (
        "a noul label is true or false, or the probability of true"
    )
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


def remove(studio, name):
    return studio.request("DELETE", f"/studio/api/data/{name}", json={})


def test_delete(studio):
    upload(studio, "tickets", GOOD)
    upload(studio, "other", GOOD)
    # A fit that has not ended reads it: refused.
    jobs = studio.app_state.jobs.jobs
    jobs["fit"] = SimpleNamespace(state="running", details={"data": "tickets"}, title="Fit x")
    try:
        busy = remove(studio, "tickets")
        assert busy.status_code == 409 and "Fit x reads tickets" in busy.text
        jobs["fit"].state = "ended"
        assert remove(studio, "tickets").status_code == 204
    finally:
        del jobs["fit"]
    assert [item["name"] for item in studio.get("/studio/api/data").json()["files"]] == ["other"]
    recent = [message for _, (_, message) in studio.app_state.hub.recent]
    assert {"type": "workspace", "folder": "data", "name": "tickets", "action": "deleted"} in recent
    assert remove(studio, "tickets").status_code == 404
    assert remove(studio, "..%2Fother").status_code in (400, 404)
    assert (studio.workspace / "data" / "other.jsonl").exists()


def examples(studio, name, **query):
    return studio.get(f"/studio/api/data/{name}/examples", params=query)


def indexes(response):
    return [example["index"] for example in response.json()["examples"]]


def test_a_page_of_examples(studio):
    upload(studio, "tickets", GOOD)
    body = examples(studio, "tickets", offset=0, limit=3).json()
    assert (body["total"], body["matched"], body["offset"], body["limit"]) == (8, 8, 0, 3)
    assert body["examples"][0] == {
        "index": 0, "id": "1", "state": "ticket 0", "labels": {"team": "tech"}, "values": {},
    }  # fmt: skip
    assert [e["index"] for e in body["examples"]] == [0, 1, 2]
    # The next pages, and one past the end.
    assert indexes(examples(studio, "tickets", offset=6, limit=3)) == [6, 7]
    assert examples(studio, "tickets", offset=8).json()["examples"] == []
    assert examples(studio, "tickets").json()["limit"] == 50  # the default
    for query, param in (({"offset": -1}, "offset"), ({"limit": 0}, "limit"),
                         ({"limit": 201}, "limit"), ({"question": "team"}, "label"),
                         ({"label": "tech"}, "label")):  # fmt: skip
        response = examples(studio, "tickets", **query)
        assert response.status_code == 400, query
        assert response.json()["error"]["param"] == param
    assert examples(studio, "missing").status_code == 404
    assert examples(studio, "tickets", offset="x").status_code in (400, 422)


def test_the_label_filter_and_the_search(studio):
    upload(studio, "tickets", GOOD)
    body = examples(studio, "tickets", question="team", label="tech").json()
    assert body["matched"] == 6 and body["total"] == 8
    assert all(e["labels"]["team"] == "tech" for e in body["examples"])
    page = examples(studio, "tickets", question="team", label="tech", offset=4, limit=10).json()
    assert [e["index"] for e in page["examples"]] == [4, 5]
    urgent = examples(studio, "tickets", question="urgent", label="true").json()
    assert [e["index"] for e in urgent["examples"]] == [6]  # a noul label as "true"
    assert examples(studio, "tickets", question="team", label="nobody").json()["matched"] == 0
    assert examples(studio, "tickets", question="other", label="x").json()["matched"] == 0
    # The search: in the states, any case, with the filter.
    assert indexes(examples(studio, "tickets", search="TICKET 7")) == [7]
    both = examples(studio, "tickets", search="ticket", question="team", label="billing").json()
    assert [e["index"] for e in both["examples"]] == [6]
    assert examples(studio, "tickets", search="  ").json()["matched"] == 8  # blank: everything


def test_states_cut_to_a_line_and_a_whole_example(studio):
    long = "First line\n\n  second   line " + "x" * 400
    json_state = {"order": 1234, "items": ["mug", "plate"], "note": "Grüße"}
    text = "\n".join([
        json.dumps({"state": long, "questions": {"team": TEAM}, "labels": {"team": "tech"}}),
        json.dumps({"id": "order", "state": json_state, "questions": {"team": TEAM},
                    "labels": {"team": "sales"}}),
    ])  # fmt: skip
    upload(studio, "mixed", text)
    first, second = examples(studio, "mixed").json()["examples"]
    assert first["state"].startswith("First line second line xxx") and len(first["state"]) == 200
    assert first["state"].endswith("…") and "\n" not in first["state"]
    assert second["state"] == '{"order": 1234, "items": ["mug", "plate"], "note": "Grüße"}'
    # A JSON state is searched as its JSON.
    assert examples(studio, "mixed", search='"items": ["mug"').json()["matched"] == 1
    for search in ("GRÜSSE", "grüSSe", "grüße"):  # any case, ß as ss
        assert examples(studio, "mixed", search=search).json()["matched"] == 1, search
    assert examples(studio, "mixed", search="plates").json()["matched"] == 0

    whole = studio.get("/studio/api/data/mixed/examples/0").json()
    assert whole == {"index": 0, "id": "1", "state": long, "questions": {"team": TEAM},
                     "labels": {"team": "tech"}, "values": {}}  # fmt: skip
    assert studio.get("/studio/api/data/mixed/examples/1").json()["state"] == json_state
    assert studio.get("/studio/api/data/mixed/examples/2").status_code == 404
    assert studio.get("/studio/api/data/missing/examples/0").status_code == 404


def test_a_page_follows_the_file_on_disk(studio):
    upload(studio, "tickets", GOOD)
    assert examples(studio, "tickets").json()["total"] == 8
    path = studio.workspace / "data" / "tickets.jsonl"
    path.write_text(GOOD + line(8, "sales") + "\n")  # longer: read again
    assert examples(studio, "tickets").json()["total"] == 9
    path.write_text("{broken\n")
    response = examples(studio, "tickets")
    assert response.status_code == 422 and "line 1" in response.text


def test_show_in_finder(studio, monkeypatch):
    import mlx_decision.studio.routes.data as data_routes

    shown = []
    monkeypatch.setattr(data_routes, "reveal", shown.append)
    monkeypatch.setattr(data_routes.sys, "platform", "darwin")
    upload(studio, "tickets", GOOD)
    assert studio.post("/studio/api/data/tickets/reveal", json={}).status_code == 204
    assert shown == [(studio.workspace / "data" / "tickets.jsonl").resolve()]
    assert studio.post("/studio/api/data/missing/reveal", json={}).status_code == 404
    assert studio.post("/studio/api/data/..%2Fsets/reveal", json={}).status_code in (400, 404)
    assert len(shown) == 1


def test_show_in_finder_only_on_this_mac(tmp_path, monkeypatch):
    """A browser on another machine, logged in with the API key, reads the
    datasets but cannot open Finder on this Mac."""
    import mlx_decision.studio.routes.data as data_routes

    monkeypatch.setattr(data_routes, "reveal", lambda path: pytest.fail("revealed"))
    write_model(tmp_path / "models" / "fake")
    pool = ModelPool(discover([tmp_path / "models"]), budget=10**9)
    app = create_studio(pool, host="0.0.0.0", api_key="k", workspace=tmp_path / "ws",
                        config=Config(tmp_path / "config.toml"))  # fmt: skip
    (tmp_path / "ws" / "data" / "tickets.jsonl").write_text(GOOD)
    other = TestClient(app, base_url=HERE, client=("192.168.1.20", 5000))
    assert other.post("/login", json={"api_key": "k"}).status_code == 204
    assert other.get("/studio/api/data/tickets/examples").status_code == 200
    refused = other.post("/studio/api/data/tickets/reveal", json={})
    assert refused.status_code == 403 and "on this Mac" in refused.text


def test_a_path_under_the_home_folder_is_shown_with_a_tilde(studio, monkeypatch):
    from pathlib import Path

    upload(studio, "tickets", GOOD)
    monkeypatch.setattr(Path, "home", lambda: (studio.workspace / "..").resolve())
    body = studio.get("/studio/api/data/tickets").json()
    assert body["display"] == "~/ws/data/tickets.jsonl"


def test_a_question_set_from_a_file(studio):
    upload(studio, "tickets", GOOD)
    made = studio.post("/studio/api/data/tickets/set", json={})
    assert made.status_code == 200, made.text
    assert made.json() == {"name": "tickets", "examples": 8}
    body = studio.app_state.workspace.load_set("tickets")[0]
    assert body["questions"] == {"team": TEAM, "urgent": URGENT}
    assert body["state"] == "ticket 0"
    # Each example named by its id and its labels (a yes / no as the answers say it).
    names = [example["name"] for example in body["examples"]]
    assert names[0] == "1 · tech" and names[6:] == ["7 · billing, yes", "8 · sales, no"]
    assert [example["state"] for example in body["examples"]][:2] == ["ticket 0", "ticket 1"]
    # Again: the name taken, so -2.
    assert studio.post("/studio/api/data/tickets/set", json={}).json()["name"] == "tickets-2"


def test_a_set_from_a_file_holds_its_first_ten_examples(studio):
    level = {"type": "score", "criteria": ["low", "mid", "high"]}
    lines = [
        json.dumps({"id": f"r{i}", "state": {"n": i}, "questions": {"level": level},
                    "labels": {"level": i % 3}})
        for i in range(12)
    ]  # fmt: skip
    upload(studio, "levels", "\n".join(lines) + "\n")
    assert studio.post("/studio/api/data/levels/set", json={}).json()["examples"] == 10
    body = studio.app_state.workspace.load_set("levels")[0]
    names = [example["name"] for example in body["examples"]]
    # Score labels as levels counted from 1.
    assert names[:4] == ["r0 · level 1", "r1 · level 2", "r2 · level 3", "r3 · level 1"]
    assert body["state"] == {"n": 0} and body["examples"][9]["state"] == {"n": 9}
    assert studio.post("/studio/api/data/missing/set", json={}).status_code == 404
