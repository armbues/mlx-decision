"""The Studio's API for the Playground: status, models, sets, check, run, history."""

import base64
import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import mlx_decision
from fake_backend import write_fitted, write_model
from mlx_decision.cli import app as cli_app
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio, templates
from mlx_decision.studio.routes import sets as sets_routes

HERE = "http://127.0.0.1:8000"
JSON = {"Content-Type": "application/json"}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs", "billing": "Payments", "sales": "Pricing"},
    },
    "frustration": {"type": "score", "criteria": ["calm", "upset", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
}
BODY = {"state": "Stripe is down and we cannot bill anyone", "questions": QUESTIONS}


def png(color=(255, 0, 0)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def models_folder(tmp_path):
    folder = tmp_path / "models"
    write_model(folder / "fake", max_choice_options=4)
    write_model(folder / "seeing", supports_images=True, open_images=True)
    write_fitted(folder / "fake-fit", folder / "fake")
    return folder


@pytest.fixture
def studio(tmp_path, models_folder):
    """A logged-in client of a Studio with three fake models (no default)."""
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws")
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        yield client


def run(studio, request=BODY, model="fake", **extra):
    return studio.post("/studio/api/run", json={"request": request, "model": model, **extra})


# Status and models


def test_status(studio, tmp_path):
    status = studio.get("/studio/api/status").json()
    assert status["version"] == mlx_decision.__version__
    assert status["address"] == HERE
    assert status["api"] == {"url": f"{HERE}/v1/systemone", "key_required": False}
    assert status["workspace"] == str((tmp_path / "ws").resolve())
    assert status["sets_folder"] == str((tmp_path / "ws" / "sets").resolve())
    assert status["busy"] is None and status["loaded"] == []
    assert status["memory"] == {"used": 0, "budget": 10**9}
    assert status["models"] == ["fake", "fake-fit", "seeing"]
    assert status["default_model"] is None
    assert status["config"]["exists"] is False
    assert status["config"]["values"] == {}


def test_status_shows_the_config_file(tmp_path, models_folder, mlx_decision_home):
    (mlx_decision_home / "config.toml").write_text(
        f'models = ["{models_folder}"]\nport = 9000\nworkspace = "./ws"\n'
    )
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, port=9000)
    with TestClient(app, base_url="http://127.0.0.1:9000", follow_redirects=False) as client:
        client.get(f"/?token={app.state.session.token}")
        status = client.get("/studio/api/status").json()
    assert status["config"]["path"] == str(mlx_decision_home / "config.toml")
    assert status["config"]["exists"] is True
    assert status["config"]["values"] == {
        "models": [str(models_folder)],
        "port": 9000,
        "workspace": str(mlx_decision_home / "ws"),
    }
    # The config's workspace is used when none is given.
    assert status["workspace"] == str((mlx_decision_home / "ws").resolve())


def test_models(studio):
    models = {entry["name"]: entry for entry in studio.get("/studio/api/models").json()["models"]}
    assert list(models) == ["fake", "fake-fit", "seeing"]
    assert models["fake"]["capabilities"]["max_choice_options"] == 4
    assert models["fake"]["capabilities"]["supports_images"] is False
    assert models["seeing"]["capabilities"]["supports_images"] is True
    assert models["fake-fit"]["fitted"] and models["fake-fit"]["base"] == "fake"
    assert models["fake-fit"]["capabilities"] == models["fake"]["capabilities"]
    assert not any(entry["loaded"] for entry in models.values())
    run(studio)
    assert studio.get("/studio/api/models").json()["models"][0]["loaded"] is True


# Sets


def test_set_round_trip(studio):
    saved = studio.put("/studio/api/sets/tickets", json={"body": BODY, "loaded": None})
    assert saved.status_code == 200
    digest = saved.json()["sha256"]
    loaded = studio.get("/studio/api/sets/tickets").json()
    assert loaded == {"name": "tickets", "body": BODY, "sha256": digest}
    assert [entry["name"] for entry in studio.get("/studio/api/sets").json()["sets"]] == ["tickets"]
    changed = {**BODY, "state": "v2"}
    again = studio.put("/studio/api/sets/tickets", json={"body": changed, "loaded": digest})
    assert again.status_code == 200 and again.json()["sha256"] != digest


def test_a_set_changed_on_disk_gives_409_with_the_current_hash(studio):
    digest = studio.put("/studio/api/sets/tickets", json={"body": BODY}).json()["sha256"]
    path = studio.app_state.workspace.root / "sets" / "tickets.json"
    path.write_text(json.dumps({**BODY, "state": "from an editor"}))
    response = studio.put("/studio/api/sets/tickets", json={"body": BODY, "loaded": digest})
    assert response.status_code == 409
    error = response.json()["error"]
    assert "changed on disk" in error["message"]
    assert error["current"] == studio.get("/studio/api/sets/tickets").json()["sha256"]
    # Save As a new set with a taken name is a conflict too.
    assert studio.put("/studio/api/sets/tickets", json={"body": BODY}).status_code == 409


@pytest.mark.parametrize(
    "path, status",
    [
        ("/studio/api/sets/..%2Fescape", 404),  # no route takes a slash in a name
        ("/studio/api/sets/%2E%2E", 400),
        ("/studio/api/sets/.hidden", 400),
        ("/studio/api/sets/a%20b", 400),
        ("/studio/api/sets/missing", 404),
        ("/studio/api/history/.x", 400),
        ("/studio/api/sets/.hidden/export", 400),
    ],
)
def test_bad_and_missing_names(studio, tmp_path, path, status):
    (tmp_path / "escape.json").write_text("{}")
    response = studio.get(path)
    assert response.status_code == status, response.text


def test_set_body_errors(studio):
    response = studio.put("/studio/api/sets/x", json={"body": [1]})
    assert response.status_code == 400
    response = studio.put("/studio/api/sets/x", json={"loaded": None})
    assert response.status_code == 400 and response.json()["error"]["param"] == "body"
    response = studio.put("/studio/api/sets/x", json={"body": {**BODY, "images": ["/etc/hosts"]}})
    assert response.status_code == 400 and "images.0" in response.json()["error"]["message"]
    response = studio.put("/studio/api/sets/x", content=b"{nope", headers=JSON)
    assert response.status_code == 400 and "not valid JSON" in response.json()["error"]["message"]


def test_rename_and_delete(studio):
    studio.put("/studio/api/sets/tickets", json={"body": BODY})
    run(studio, set="tickets")
    response = studio.post("/studio/api/sets/tickets/rename", json={"name": "support"})
    assert response.json() == {"name": "support"}
    assert len(studio.get("/studio/api/history/support").json()["runs"]) == 1
    studio.put("/studio/api/sets/other", json={"body": BODY})
    assert studio.post("/studio/api/sets/support/rename", json={"name": "other"}).status_code == 409
    assert studio.delete("/studio/api/sets/support", headers=JSON).status_code == 204
    assert studio.get("/studio/api/sets/support").status_code == 404
    assert studio.get("/studio/api/history/support").json() == {"runs": []}
    assert studio.delete("/studio/api/sets/support", headers=JSON).status_code == 404
    # The guard wants JSON for every write, a body-less DELETE included.
    assert studio.delete("/studio/api/sets/other").status_code == 415


def test_import(studio):
    response = studio.post(
        "/studio/api/sets/import", json={"name": "questions", "text": json.dumps(QUESTIONS)}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "questions"
    assert response.json()["body"] == {"questions": QUESTIONS}
    whole = json.dumps({**BODY, "model": "fake"})
    response = studio.post("/studio/api/sets/import", json={"name": "questions", "text": whole})
    assert response.json()["name"] == "questions-2"
    assert studio.get("/studio/api/sets/questions-2").json()["body"] == {**BODY, "model": "fake"}
    response = studio.post("/studio/api/sets/import", json={"name": "bad", "text": "{nope"})
    assert response.status_code == 400 and response.json()["error"]["param"] == "text"
    response = studio.post("/studio/api/sets/import", json={"name": "../x", "text": "{}"})
    assert response.status_code == 400
    with_path = json.dumps({**BODY, "images": ["/etc/hosts"]})
    response = studio.post("/studio/api/sets/import", json={"name": "img", "text": with_path})
    assert response.status_code == 400


def test_export(studio):
    studio.put("/studio/api/sets/tickets", json={"body": BODY})
    response = studio.get("/studio/api/sets/tickets/export")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert 'filename="tickets.json"' in response.headers["content-disposition"]
    assert json.loads(response.content) == BODY
    assert studio.get("/studio/api/sets/missing/export").status_code == 404


# Templates: the sets that ship with the Studio


def test_the_templates_are_listed_in_order(studio):
    listed = studio.get("/studio/api/templates").json()["templates"]
    assert [entry["name"] for entry in listed] == list(templates.ORDER)
    for entry in listed:
        assert entry["title"] and entry["description"]
        assert entry["questions"] >= 1 and entry["examples"] >= 3
    types = {entry["name"]: entry["types"] for entry in listed}
    assert types["support-triage"] == ["choice", "score", "noul"]
    assert types["screenshot-check"] == ["choice", "score", "noul"]  # not the set's order
    assert [entry["name"] for entry in listed if entry["images"]] == ["screenshot-check"]


def test_a_set_from_a_template(studio):
    made = studio.post("/studio/api/sets/triage/from-template", json={"template": "support-triage"})
    assert made.status_code == 200, made.text
    body = made.json()["body"]
    loaded = studio.get("/studio/api/sets/triage").json()
    assert loaded["body"] == body and loaded["sha256"] == made.json()["sha256"]
    assert list(body) == ["state", "questions", "examples"]  # as the Studio saves a set
    assert body["state"] == body["examples"][0]["state"]
    again = studio.post("/studio/api/sets/triage/from-template", json={"template": "fact-check"})
    assert again.status_code == 409
    assert studio.get("/studio/api/sets/triage").json()["body"] == body
    missing = studio.post("/studio/api/sets/x/from-template", json={"template": "nope"})
    assert missing.status_code == 404
    assert studio.post("/studio/api/sets/x/from-template", json={}).status_code == 400
    bad = studio.post("/studio/api/sets/..x/from-template", json={"template": "fact-check"})
    assert bad.status_code == 400


def test_a_template_with_images_carries_them_as_data_urls(studio):
    made = studio.post(
        "/studio/api/sets/shots/from-template", json={"template": "screenshot-check"}
    ).json()["body"]
    assert made["images"] == made["examples"][0]["images"]
    for example in made["examples"]:
        [image] = example["images"]
        assert image.startswith("data:image/png;base64,iVBORw0KGgo")


@pytest.mark.parametrize("name", templates.ORDER)
def test_every_example_of_a_template_passes_the_check(studio, name):
    body = templates.template_set(name)
    for example in body["examples"]:
        request = {"state": example["state"], "questions": body["questions"]}
        if "images" in example:
            request["images"] = example["images"]
        checked = studio.post("/studio/api/check", json={"request": request, "model": "seeing"})
        assert checked.json()["errors"] == [], (name, example["name"])


def test_template_set_gives_a_fresh_copy():
    first = templates.template_set("fact-check")
    first["questions"].clear()
    assert templates.template_set("fact-check")["questions"]
    with pytest.raises(templates.UnknownTemplate):
        templates.template_set("../support-triage")


def test_reveal(studio, monkeypatch):
    revealed = []
    monkeypatch.setattr(sets_routes, "reveal", revealed.append)
    studio.put("/studio/api/sets/tickets", json={"body": BODY})
    assert studio.post("/studio/api/sets/tickets/reveal", json={}).status_code == 204
    assert revealed == [studio.app_state.workspace.root / "sets" / "tickets.json"]
    assert studio.post("/studio/api/sets/missing/reveal", json={}).status_code == 404


def test_reveal_only_on_this_mac(tmp_path, models_folder, monkeypatch):
    monkeypatch.setattr(sets_routes, "reveal", lambda path: pytest.fail("revealed"))
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, host="0.0.0.0", api_key="k", workspace=tmp_path / "ws")
    other = TestClient(app, base_url=HERE, client=("192.168.1.20", 5000))
    assert other.post("/login", json={"api_key": "k"}).status_code == 204
    app.state.workspace.save_set("tickets", BODY)
    assert other.post("/studio/api/sets/tickets/reveal", json={}).status_code == 403


# Check


def test_check_passes_a_good_request(studio):
    response = studio.post("/studio/api/check", json={"request": BODY, "model": "fake"})
    assert response.json() == {"model": "fake", "errors": []}
    assert studio.app_state.gpu.pool.loaded() == []  # nothing loaded


CHECKED = [
    ({**BODY, "images": [png()]}, "fake", "images"),
    (
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": list("abcde")}}},
        "fake",
        "questions.q.criteria",
    ),
    ({"state": "x", "questions": {"q": {"type": "maybe"}}}, "fake", "questions.q"),
    ({"questions": QUESTIONS}, "fake", "state"),
    ({**BODY, "images": ["/etc/hosts"]}, "seeing", "images.0"),
    ({**BODY, "images": ["data:image/png;base64,AAAA"]}, "seeing", "images.0"),
    ({**BODY, "state": {"photo": "data:image/png;base64,AAAA"}}, "seeing", "state"),
    (
        {**BODY, "state": {"photo": "data:image/png;base64," + "iVBORw0KGgo" + "A" * 80}},
        "seeing",
        "state",
    ),
    (BODY, "nope", "model"),
]


@pytest.mark.parametrize("request_body, model, param", CHECKED)
def test_check_gives_the_errors_run_gives(studio, request_body, model, param):
    checked = studio.post("/studio/api/check", json={"request": request_body, "model": model})
    [error] = checked.json()["errors"]
    assert error["param"] is not None and error["param"].startswith(param), error
    answered = run(studio, request_body, model)
    assert answered.status_code == 422, answered.text
    assert answered.json()["error"]["message"] == error["message"]
    assert answered.json()["error"]["param"] == error["param"]


def test_a_request_the_check_passes_is_answered(studio):
    request_body = {**BODY, "images": [png()]}
    checked = studio.post("/studio/api/check", json={"request": request_body, "model": "seeing"})
    assert checked.json()["errors"] == []
    assert run(studio, request_body, "seeing").status_code == 200


def test_check_body_errors(studio):
    assert studio.post("/studio/api/check", json={"model": "fake"}).status_code == 400
    response = studio.post("/studio/api/check", content=b"[", headers=JSON)
    assert response.status_code == 400


# Run, history and speed


def test_run_answers_as_the_model_does(studio, models_folder):
    response = run(studio, set="tickets")
    assert response.status_code == 200, response.text
    data = response.json()
    expected = mlx_decision.load(models_folder / "fake").decide_request(BODY).to_wire()
    assert data["response"] == expected
    assert data["model"] == "fake"
    assert data["latency_ms"] >= 0 and data["load_ms"] >= 0
    assert data["input_tokens"] == expected["usage"]["input_tokens"]
    assert data["truncated"] is False
    assert isinstance(data["id"], str) and data["time"].endswith("Z")
    second = run(studio, set="tickets").json()
    assert second["load_ms"] is None  # loaded already


def test_run_without_a_set_keeps_no_history(studio):
    data = run(studio).json()
    assert data["id"] is None
    assert list((studio.app_state.workspace.root / "history").glob("*.jsonl")) == []


def test_run_errors(studio):
    assert run(studio, set="../x").status_code == 400
    response = studio.post("/studio/api/run", json={"model": "fake"})
    assert response.status_code == 400 and response.json()["error"]["param"] == "request"
    response = run(studio, model=None)  # no default model, none named
    assert response.status_code == 422 and response.json()["error"]["param"] == "model"
    response = run(studio, {**BODY, "model": "seeing"}, model=None)
    assert response.status_code == 200 and response.json()["model"] == "seeing"


def test_the_history_keeps_each_run(studio):
    image = png()
    request_body = {**BODY, "images": [image]}
    first = run(studio, request_body, "seeing", set="tickets").json()
    run(studio, {**BODY, "state": "second"}, "fake", set="tickets")
    runs = studio.get("/studio/api/history/tickets").json()["runs"]
    assert [entry["model"] for entry in runs] == ["seeing", "fake"]
    entry = runs[0]
    assert entry["id"] == first["id"]
    assert entry["response"] == first["response"]
    assert len(entry["fingerprint"]) == 64
    [reference] = entry["request"]["images"]
    assert reference["type"] == "image/png"
    image_response = studio.get(f"/studio/api/images/{reference['sha256']}")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"
    assert image_response.content == base64.b64decode(image.split(",", 1)[1])
    restored = studio.get(f"/studio/api/history/tickets/{first['id']}").json()
    assert restored["request"] == request_body
    assert studio.get("/studio/api/history/tickets/nope").status_code == 404
    assert studio.get("/studio/api/images/" + "0" * 64).status_code == 404
    assert studio.delete("/studio/api/history/tickets", headers=JSON).status_code == 204
    assert studio.get("/studio/api/history/tickets").json() == {"runs": []}
    assert studio.get(f"/studio/api/images/{reference['sha256']}").status_code == 404


def speed_records(studio, name="fake"):
    from mlx_decision.fitted import fingerprint

    pool = studio.app_state.gpu.pool
    return studio.app_state.workspace.read_speed(fingerprint(pool.specs[name].path))


def test_runs_and_api_answers_record_their_speed(studio):
    run(studio)
    [record] = speed_records(studio)
    assert record["source"] == "studio" and record["model"] == "fake"
    assert record["latency_ms"] >= 0 and record["input_tokens"] > 0
    assert "chip" in record
    response = studio.post("/v1/systemone", json={**BODY, "model": "fake"})
    assert response.status_code == 200
    deadline = time.monotonic() + 5
    while len(speed_records(studio)) < 2:
        assert time.monotonic() < deadline, "the API's answer was not recorded"
        time.sleep(0.01)
    assert speed_records(studio)[-1]["source"] == "api"


def test_busy_gives_503_with_retry_after(studio):
    gpu = studio.app_state.gpu
    studio.portal.call(gpu.acquire, "fit-1")
    for response in (
        run(studio),
        studio.post("/v1/systemone", json={**BODY, "model": "fake"}),
    ):
        assert response.status_code == 503
        assert response.headers["retry-after"] == "10"
        assert response.json()["error"]["code"] == "busy"
        assert "fit-1" in response.json()["error"]["message"]
    # The check needs no GPU.
    checked = studio.post("/studio/api/check", json={"request": BODY, "model": "fake"})
    assert checked.json()["errors"] == []
    gpu.release("fit-1")
    assert run(studio).status_code == 200


# Load and Unload


def test_load_and_unload(studio):
    load = studio.post("/studio/api/models/load", json={"model": "fake-fit"})
    assert load.status_code == 200
    body = load.json()
    assert (body["model"], body["loaded"], body["unloaded"]) == ("fake-fit", ["fake-fit"], [])
    assert body["load_ms"] >= 0
    again = studio.post("/studio/api/models/load", json={"model": "fake-fit"}).json()
    assert again["load_ms"] is None
    studio.post("/studio/api/models/load", json={"model": "fake"})
    assert studio.get("/studio/api/status").json()["loaded"] == ["fake-fit", "fake"]
    # The fit and its base share their weights: they go together.
    unload = studio.post("/studio/api/models/unload", json={"model": "fake"})
    assert unload.json() == {"model": "fake", "unloaded": ["fake-fit", "fake"]}
    assert studio.get("/studio/api/status").json()["loaded"] == []
    nothing = studio.post("/studio/api/models/unload", json={"model": "seeing"})
    assert nothing.json() == {"model": "seeing", "unloaded": []}


def test_load_reports_what_made_room(tmp_path, models_folder):
    pool = ModelPool(discover([models_folder]), budget=10**9)
    pool.budget = pool.size("fake") + pool.size("seeing") - 1  # room for one of them
    app = create_studio(pool, workspace=tmp_path / "ws")
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        client.post("/studio/api/models/load", json={"model": "fake"})
        body = client.post("/studio/api/models/load", json={"model": "seeing"}).json()
    assert (body["loaded"], body["unloaded"]) == (["seeing"], ["fake"])


def test_load_and_unload_errors(studio):
    for action in ("load", "unload"):
        path = f"/studio/api/models/{action}"
        missing = studio.post(path, json={"model": "nope"})
        assert missing.status_code == 404 and missing.json()["error"]["param"] == "model"
        assert studio.post(path, json={}).status_code == 400
        plain = studio.post(path, content="fake", headers={"Content-Type": "text/plain"})
        assert plain.status_code == 415


def test_load_and_unload_while_busy(studio):
    gpu = studio.app_state.gpu
    studio.portal.call(gpu.acquire, "fit-1")
    for action in ("load", "unload"):
        response = studio.post(f"/studio/api/models/{action}", json={"model": "fake"})
        assert response.status_code == 503
        assert response.headers["retry-after"] == "10"
        assert "fit-1" in response.json()["error"]["message"]
    gpu.release("fit-1")
    assert studio.post("/studio/api/models/load", json={"model": "fake"}).status_code == 200


def test_a_set_saved_through_the_api_answers_the_same_with_run_q(studio, models_folder):
    studio.put("/studio/api/sets/tickets", json={"body": BODY})
    body = studio.get("/studio/api/sets/tickets").json()["body"]
    through_studio = run(studio, body, "fake").json()["response"]
    path = studio.app_state.workspace.root / "sets" / "tickets.json"
    result = CliRunner().invoke(
        cli_app, ["run", "-m", str(models_folder / "fake"), "-q", str(path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == through_studio


# Examples (several states per set) and Run All


def test_a_run_records_its_example_and_batch(studio):
    first = run(studio, BODY, "fake", set="tickets", example="Stripe outage", batch="b1x-2")
    assert first.status_code == 200
    assert (first.json()["example"], first.json()["batch"]) == ("Stripe outage", "b1x-2")
    run(studio, {**BODY, "state": "plain"}, "fake", set="tickets")
    lines = studio.get("/studio/api/history/tickets").json()["runs"]
    assert (lines[0]["example"], lines[0]["batch"]) == ("Stripe outage", "b1x-2")
    assert "example" not in lines[1] and "batch" not in lines[1]
    for extra, param in (
        ({"example": ""}, "example"),
        ({"example": "x" * 81}, "example"),
        ({"batch": "no spaces"}, "batch"),
        ({"batch": "x" * 41}, "batch"),
    ):
        refused = run(studio, BODY, "fake", set="tickets", **extra)
        assert refused.status_code == 400 and refused.json()["error"]["param"] == param


def test_a_set_with_examples_answers_its_first_with_run_q(studio, models_folder):
    """The file keeps the first example as its top-level state, so run -q and
    the API read it as before; the other examples ride along unread."""
    examples = [
        {"name": "Stripe outage", "state": BODY["state"]},
        {"name": "Refund ask", "state": "I was charged twice, please refund one"},
    ]
    saved = {**BODY, "examples": examples}
    assert studio.put("/studio/api/sets/many", json={"body": saved}).status_code == 200
    assert studio.get("/studio/api/sets/many").json()["body"] == saved
    path = studio.app_state.workspace.root / "sets" / "many.json"
    result = CliRunner().invoke(
        cli_app, ["run", "-m", str(models_folder / "fake"), "-q", str(path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == run(studio, BODY, "fake").json()["response"]
    through_api = studio.post("/v1/systemone", json={**saved, "model": "fake"})
    assert through_api.status_code == 200


def test_the_api_needs_the_session(tmp_path, models_folder):
    pool = ModelPool(discover([models_folder]), budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws")
    client = TestClient(app, base_url=HERE)
    for method, path in [
        ("GET", "/studio/api/status"),
        ("GET", "/studio/api/sets"),
        ("POST", "/studio/api/run"),
        ("POST", "/studio/api/check"),
        ("GET", "/studio/api/history/x"),
    ]:
        response = client.request(method, path, json={})
        assert response.status_code == 401, path
