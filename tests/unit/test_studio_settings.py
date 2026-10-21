"""The Settings page's API: the places with their models, Show in Finder by a
place's id only, the values the Studio uses, and changes to the config file."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio import settings as studio_settings
from mlx_decision.studio.catalog import Catalog, studio_models
from mlx_decision.studio.routes import settings as settings_route

HERE = "http://127.0.0.1:8000"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Models in every place: models_dir, the cache, the workspace, a folder
    of models and a single model named with -m."""
    from huggingface_hub import constants

    cache = tmp_path / "hf"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    write_model(tmp_path / "models-dir" / "laya")
    write_model(cache / "models--org--cached" / "snapshots" / ("a" * 40))
    write_model(tmp_path / "ws" / "models" / "made-here")
    write_model(tmp_path / "kept" / "one")
    write_model(tmp_path / "kept" / "two")
    write_model(tmp_path / "single")
    return tmp_path


def make(layout, client=("127.0.0.1", 50000), sources=None, **config):
    config_path = layout / "config.toml"
    references = [str(layout / "kept"), str(layout / "single"), "org/not-downloaded"]
    catalog = Catalog(
        references[:2], layout / "ws" / "models", include_cache=False,
        models_dir=layout / "models-dir",
    )  # fmt: skip
    named = [*catalog.references, str(layout / "models-dir")]
    specs = studio_models(named, catalog.workspace_models, include_cache=False)
    specs += discover([layout / "hf" / "models--org--cached" / "snapshots" / ("a" * 40)])
    catalog.references = references  # a repo id among them, as in the config's models
    pool = ModelPool(specs, budget=8 * 10**9)
    app = create_studio(
        pool,
        workspace=layout / "ws",
        config=Config(config_path, models_dir=layout / "models-dir", **config),
        catalog=catalog,
        sources=sources,
    )
    studio = TestClient(app, base_url=HERE, client=client)
    studio.__enter__()
    studio.get(f"/?token={app.state.session.token}")
    return studio


@pytest.fixture
def studio(layout):
    client = make(layout, sources={"models": "flag", "host": "default", "port": "flag",
                                   "workspace": "config", "memory_budget_gb": "flag",
                                   "default_model": "default"})  # fmt: skip
    yield client
    client.__exit__(None, None, None)


def test_the_places_and_their_models(studio, layout):
    listed = studio.get("/studio/api/places").json()
    assert listed["editable"] is True  # a browser on this Mac
    places = listed["places"]
    by_kind = [
        (place["kind"], place["path"], place["models"], place["downloads"]) for place in places
    ]
    assert by_kind == [
        ("models_dir", str(layout / "models-dir"), 1, True),
        ("cache", str(layout / "hf"), 1, False),
        ("studio", str(layout / "ws" / "models"), 1, False),
        ("named", str(layout / "kept"), 2, False),
        ("named", str(layout / "single"), 1, False),
    ]  # the repo id is no folder: its model would be in the cache
    assert all(place["exists"] for place in places)
    assert len({place["id"] for place in places}) == len(places)


def test_show_in_finder_takes_a_place_id_only(studio, layout, monkeypatch):
    shown = []
    monkeypatch.setattr(settings_route, "reveal", shown.append)
    monkeypatch.setattr(settings_route.sys, "platform", "darwin")
    places = studio.get("/studio/api/places").json()["places"]
    for place in places:
        response = studio.post("/studio/api/places/reveal", json={"place": place["id"]})
        assert response.status_code == 204, response.text
    assert shown == [Path(place["path"]) for place in places]
    # The config file: its folder before it exists, then the file.
    assert studio.post("/studio/api/places/reveal", json={"place": "config"}).status_code == 204
    assert shown[-1] == layout
    (layout / "config.toml").write_text("")
    studio.post("/studio/api/places/reveal", json={"place": "config"})
    assert shown[-1] == layout / "config.toml"
    # The workspace's datasets folder (where a file too large to upload goes).
    assert studio.post("/studio/api/places/reveal", json={"place": "data"}).status_code == 204
    assert shown[-1] == (layout / "ws" / "data").resolve()
    count = len(shown)
    # Anything else is refused: paths, unknown ids, other types.
    for place in (str(layout / "kept"), "/", "../etc", "named-99", "", 3):
        response = studio.post("/studio/api/places/reveal", json={"place": place})
        assert response.status_code in (400, 404), (place, response.text)
    assert len(shown) == count
    assert studio.post(
        "/studio/api/places/reveal", content=b"place=cache",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    ).status_code == 415  # fmt: skip


def test_show_in_finder_only_for_a_browser_on_this_mac(layout, monkeypatch):
    """A browser on another machine, logged in with the API key, sees the
    places but cannot open Finder on this Mac."""
    monkeypatch.setattr(settings_route, "reveal", lambda path: pytest.fail("revealed"))
    pool = ModelPool(discover([layout / "kept"]), budget=10**9)
    app = create_studio(pool, host="0.0.0.0", api_key="k", workspace=layout / "ws",
                        config=Config(layout / "config.toml"))  # fmt: skip
    other = TestClient(app, base_url=HERE, client=("192.168.1.20", 5000))
    assert other.post("/login", json={"api_key": "k"}).status_code == 204
    assert other.get("/studio/api/places").status_code == 200
    refused = other.post("/studio/api/places/reveal", json={"place": "cache"})
    assert refused.status_code == 403 and "on this Mac" in refused.text


def test_a_place_that_is_gone_is_said_so(studio, layout, monkeypatch):
    import shutil

    monkeypatch.setattr(settings_route, "reveal", lambda path: None)
    shutil.rmtree(layout / "kept")
    places = studio.get("/studio/api/places").json()["places"]
    assert [p["kind"] for p in places].count("named") == 1  # a folder no longer there drops out
    shutil.rmtree(layout / "ws" / "models")
    studio_place = next(p for p in places if p["kind"] == "studio")
    response = studio.post("/studio/api/places/reveal", json={"place": studio_place["id"]})
    assert response.status_code == 404 and "does not exist yet" in response.text


def test_the_config_and_the_values_the_studio_uses(studio, layout):
    listed = studio.get("/studio/api/places").json()
    assert listed["home"] == str(Path.home())  # the page writes paths under it with ~
    config = listed["config"]
    assert config["path"] == str(layout / "config.toml") and config["exists"] is False
    settings = {item["key"]: item for item in config["settings"]}
    assert list(settings) == list(studio_settings.KEYS)
    assert settings["models_dir"] == {
        "key": "models_dir", "value": str(layout / "models-dir"), "source": "config",
        "in_file": str(layout / "models-dir"), "live": True, "editable": True,
    }  # fmt: skip
    assert settings["memory_budget_gb"]["value"] == 8.0
    assert settings["memory_budget_gb"]["source"] == "flag"
    assert settings["memory_budget_gb"]["in_file"] is None
    assert settings["port"] == {
        "key": "port", "value": 8000, "source": "flag", "in_file": None, "live": False,
        "editable": True,
    }  # fmt: skip
    assert settings["workspace"]["value"] == str((layout / "ws").resolve())
    assert settings["models"]["value"][:2] == [str(layout / "kept"), str(layout / "single")]


def test_setting_sources():
    config = Config(Path("config.toml"), port=9000, host="0.0.0.0")
    sources = studio_settings.setting_sources(config, {"port": 8001, "models": [], "host": None})
    assert sources["port"] == "flag"
    assert sources["host"] == "config"
    assert sources["models"] == "default" and sources["workspace"] == "default"


def test_paths_under_the_home_folder_show_with_a_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert settings_route.tilde(str(tmp_path / "Models")) == "~/Models"
    assert settings_route.tilde(str(tmp_path)) == "~"
    assert settings_route.tilde(str(tmp_path) + "-other/x") == str(tmp_path) + "-other/x"
    assert settings_route.tilde("/opt/models") == "/opt/models"


# Changes to the config file


@pytest.fixture
def live(layout):
    """A Studio whose settings all come from the file or the defaults (no flags)."""
    client = make(layout, sources={})
    yield client
    client.__exit__(None, None, None)


def put(studio, key, value):
    return studio.put(f"/studio/api/settings/{key}", json={"value": value})


def test_the_memory_budget_applies_at_once(live, layout):
    config = layout / "config.toml"
    config.write_text("# mine\n[server]\nport = 1\n")
    changed = put(live, "memory_budget_gb", 6)
    assert changed.status_code == 200, changed.text
    body = changed.json()
    assert body["line"] == "memory_budget_gb = 6" and body["applies"] == "now"
    assert body["setting"] == {
        "key": "memory_budget_gb", "value": 6.0, "source": "config", "in_file": 6.0, "live": True,
        "editable": True,
    }  # fmt: skip
    assert config.read_text() == "# mine\nmemory_budget_gb = 6\n\n[server]\nport = 1\n"
    assert live.app.state.gpu.pool.budget == 6 * 10**9
    assert live.get("/studio/api/status").json()["memory"]["budget"] == 6 * 10**9
    # Taken out: the built-in default again (the GPU's working set here).
    taken = put(live, "memory_budget_gb", None).json()
    assert taken["line"] is None and taken["setting"]["source"] == "default"
    assert live.app.state.gpu.pool.budget is None
    assert config.read_text() == "# mine\n\n[server]\nport = 1\n"


def test_the_default_model(live, layout):
    pool = live.app.state.gpu.pool
    assert put(live, "default_model", "laya").json()["applies"] == "now"
    assert pool.default == "laya"
    assert live.get("/studio/api/status").json()["default_model"] == "laya"
    later = put(live, "default_model", "not-listed").json()
    assert later["applies"] == "next start" and "no model the Studio lists" in later["note"]
    assert pool.default == "laya"  # unchanged until then
    put(live, "default_model", None)
    assert pool.default is None  # several models: none


def test_models_and_models_dir_apply_with_a_rescan(live, layout):
    write_model(layout / "more" / "extra")
    before = list(live.app.state.rescans.catalog.references)
    added = put(live, "models", [*before, str(layout / "more")]).json()
    assert added["applies"] == "now" and added["note"] is None
    names = [model["name"] for model in live.get("/studio/api/models").json()["models"]]
    assert "extra" in names
    narrowed = put(live, "models", [str(layout / "more")]).json()
    assert narrowed["note"] == "models taken out leave the list at the next start"
    assert "one" in live.app.state.gpu.pool.specs  # still listed this run

    cache = put(live, "models_dir", "hf-cache").json()
    assert cache["applies"] == "now" and cache["note"] == "downloads go to the Hugging Face cache"
    places = live.get("/studio/api/places").json()["places"]
    assert [place["kind"] for place in places][:1] == ["cache"] and places[0]["downloads"]
    write_model(layout / "elsewhere" / "fresh")
    moved = put(live, "models_dir", str(layout / "elsewhere")).json()
    assert moved["note"] == f"downloads go to {layout / 'elsewhere'}"
    assert "fresh" in live.app.state.gpu.pool.specs
    assert live.app.state.rescans.catalog.models_dir == layout / "elsewhere"


def test_host_and_port_wait_for_the_next_start(live, layout):
    for key, value in (("port", 9001), ("host", "0.0.0.0")):
        body = put(live, key, value).json()
        assert body["applies"] == "next start", (key, body)
        assert body["setting"]["in_file"] is not None and body["setting"]["live"] is False
    assert live.get("/studio/api/places").json()["config"]["settings"][5]["value"] == 8000


def test_a_flag_wins_until_the_next_start(studio, layout):
    body = put(studio, "memory_budget_gb", 3).json()
    assert (
        body["applies"] == "next start" and body["note"] == "--memory-budget sets it for this run"
    )
    assert body["setting"]["value"] == 8.0 and body["setting"]["in_file"] == 3.0
    assert body["setting"]["source"] == "flag"


def test_changes_are_refused(live, layout):
    config = layout / "config.toml"
    config.write_text("port = 8001 # kept\n")
    for key, value, status in [
        ("port", "x", 400),
        ("port", 70000, 400),
        ("memory_budget_gb", -2, 400),
        ("models", "one", 400),
        ("host", "", 400),
    ]:
        response = put(live, key, value)
        assert response.status_code == status, (key, value, response.text)
    assert put(live, "api_key", "x").status_code == 404
    assert live.put("/studio/api/settings/port", json={}).status_code == 400
    assert config.read_text() == "port = 8001 # kept\n"
    config.write_text('models = [\n["a"]\n]\n')  # a nested list it cannot place a key after
    assert put(live, "port", 1).status_code == 409


def test_other_tabs_hear_of_a_change(live):
    hub = live.app.state.hub
    put(live, "port", 9002)
    assert ("settings", "port") in [(m["type"], m.get("key")) for _, (_, m) in hub.recent]


def test_off_this_mac_the_settings_are_read_only(layout):
    pool = ModelPool(discover([layout / "kept"]), budget=10**9)
    app = create_studio(pool, host="0.0.0.0", api_key="k", workspace=layout / "ws",
                        config=Config(layout / "config.toml"))  # fmt: skip
    other = TestClient(app, base_url=HERE, client=("192.168.1.20", 5000))
    assert other.post("/login", json={"api_key": "k"}).status_code == 204
    assert other.get("/studio/api/places").json()["editable"] is False
    refused = other.put("/studio/api/settings/port", json={"value": 9000})
    assert refused.status_code == 403 and "on this Mac" in refused.text
    assert not (layout / "config.toml").exists()


def test_the_workspace_is_shown_not_changed(live, layout, monkeypatch):
    """It moves with --workspace or the config file edited by hand."""
    item = next(
        item
        for item in live.get("/studio/api/places").json()["config"]["settings"]
        if item["key"] == "workspace"
    )
    assert item["editable"] is False and item["value"] == str((layout / "ws").resolve())
    refused = put(live, "workspace", "~/Studio")
    assert refused.status_code == 403 and "--workspace" in refused.text
    assert not (layout / "config.toml").exists()
    shown = []
    monkeypatch.setattr(settings_route, "reveal", shown.append)
    monkeypatch.setattr(settings_route.sys, "platform", "darwin")
    assert live.post("/studio/api/places/reveal", json={"place": "workspace"}).status_code == 204
    assert shown == [(layout / "ws").resolve()]


def test_models_leaves_out_models_dir(live, layout):
    """models_dir is a key of its own: the models row shows only what models adds."""
    settings = live.app.state.settings
    catalog = live.app.state.rescans.catalog
    catalog.references = [str(layout / "models-dir"), *catalog.references]  # as web lists it
    listed = live.get("/studio/api/places").json()
    item = next(i for i in listed["config"]["settings"] if i["key"] == "models")
    assert str(layout / "models-dir") not in item["value"]
    assert item["value"] == settings.named_models() == catalog.references[1:]
    assert listed["working_set_gb"] > 0  # the budget slider's end
    assert listed["api_key"] is False  # whether one is needed, never the key
