"""Download Model: the families offered, the check on a fake Hub, the sizes and
the fit check per way of storing a model."""

import random
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import fake_hub
import mlx_decision.download as download
from fake_backend import write_model
from mlx_decision.config import HF_CACHE, Config
from mlx_decision.download import RepoCheck
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.routes import downloads as downloads_route

HERE = "http://127.0.0.1:8000"
GB = 10**9


@pytest.fixture
def make_studio(tmp_path, monkeypatch):
    """A logged-in client of a Studio over ``models`` (paths) with ``config``,
    the fake Hub installed (``org/fake``, 3 MB of weights)."""
    fake_hub.install({"model.safetensors": 3_000_000}, patch=monkeypatch.setattr)
    clients = []

    def make(models=(), budget=10 * GB, **config):
        pool = ModelPool(discover(list(models)) if models else [], budget=budget)
        app = create_studio(
            pool, workspace=tmp_path / "ws", config=Config(tmp_path / "config.toml", **config)
        )
        client = TestClient(app, base_url=HERE, client=("127.0.0.1", 50000))
        client.__enter__()
        client.get(f"/?token={app.state.session.token}")
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def check(studio, repo):
    return studio.post("/studio/api/downloads/check", json={"repo": repo})


def test_the_families_in_the_public_order_with_their_models(make_studio):
    offered = make_studio().get("/studio/api/downloads").json()
    titles = [family["title"] for family in offered["families"]]
    assert titles == ["Clef", "pplx", "d3", "H2O-Lightning", "d1", "Laya", "Julia"]
    clef = offered["families"][0]
    assert clef["maker"] and clef["summary"]
    assert clef["models"][0]["repo"] == "Cloudflare/clef-flash"
    assert clef["models"][0]["size"].endswith("GB") and clef["models"][0]["description"]
    assert all(m["on_this_mac"] is None for f in offered["families"] for m in f["models"])
    d1 = next(family for family in offered["families"] if family["name"] == "d1")
    assert d1["licence"]  # its terms of use come before a download
    assert offered["destination"]["kind"] == "cache"
    assert offered["destination"]["setting"] is None


def test_where_downloads_go_and_what_is_here_already(make_studio, tmp_path):
    folder = tmp_path / "models"
    write_model(folder / "d3-edge")  # as download puts vllm-sr/d3-edge there
    cache = tmp_path / "hf" / "models--Cloudflare--clef-flash" / "snapshots" / ("a" * 40)
    write_model(cache)
    studio = make_studio([folder, cache], models_dir=folder)
    offered = studio.get("/studio/api/downloads").json()
    assert offered["destination"] == {
        "kind": "folder",
        "path": str(folder),
        "setting": "models_dir",
    }
    here = {m["repo"]: m["on_this_mac"] for f in offered["families"] for m in f["models"]}
    assert here["vllm-sr/d3-edge"] == "d3-edge"
    assert here["Cloudflare/clef-flash"] is not None
    assert here["Cloudflare/clef"] is None
    cached = make_studio(models_dir=HF_CACHE).get("/studio/api/downloads").json()
    assert (
        cached["destination"]["kind"] == "cache"
        and cached["destination"]["setting"] == "models_dir"
    )


def test_the_check_on_the_hub(make_studio):
    found = check(make_studio(), "org/fake")
    assert found.status_code == 200, found.text
    body = found.json()
    assert body["repo"] == "org/fake" and body["family"] == "fake" and body["reason"] is None
    assert body["revision"] == fake_hub.SHA
    # The fake family cannot store a model quantized: full size only, exact.
    [full] = body["options"]
    assert full["bits"] == 0 and full["estimate"] is False
    assert full["bytes"] == body["size_bytes"] > 3_000_000
    assert full["memory_bytes"] == round(3_000_000 * 1.1) and full["fit"] == "ok"
    assert body["default_bits"] == 0
    assert body["on_this_mac"] is None


def test_the_check_refuses(make_studio, monkeypatch):
    studio = make_studio()
    missing = check(studio, "org/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["message"] == (
        "org/missing: not found on the Hub (or private: set HF_TOKEN)"
    )
    for repo in ("org", "../etc", "org/x/y", "", "org/x y"):
        assert check(studio, repo).status_code == 400, repo

    def offline(repo_id):
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided\nmore")

    monkeypatch.setattr(downloads_route, "check_repo", offline)
    down = check(studio, "org/fake")
    assert down.status_code == 502
    assert down.json()["error"]["message"] == "org/fake: [Errno 8] nodename nor servname provided"

    # A format no family knows: the reason, nothing to offer.
    monkeypatch.setattr(
        downloads_route,
        "check_repo",
        lambda repo_id: RepoCheck(repo_id, 5 * GB, None, "no model family recognises it"),
    )
    unknown = check(studio, "org/other").json()
    assert unknown["family"] is None and unknown["reason"] == "no model family recognises it"
    assert unknown["options"] == [] and unknown["default_bits"] is None


def test_sizes_and_the_fit_check_per_way_of_storing():
    """A 16 GB model of a family that converts: full size, 8-bit and 4-bit (estimates)."""
    weights = 16 * GB
    found = RepoCheck(
        "Cloudflare/clef-flash", weights + 10**7, "clef", weight_bytes=weights,
        quantizable_bytes=15 * GB,
    )  # fmt: skip
    body = downloads_route.describe(found, budget=10 * GB, mac=12 * GB)
    options = {option["bits"]: option for option in body["options"]}
    assert list(options) == [0, 8, 4]
    assert options[0]["bytes"] == weights + 10**7 and not options[0]["estimate"]
    assert options[8]["estimate"] and options[4]["estimate"]
    assert options[0]["bytes"] > options[8]["bytes"] > options[4]["bytes"]
    assert options[8]["bytes"] == found.quantized_bytes(8)
    assert options[0]["fit"] == "too_large"  # 17.6 GB to run on a 12 GB Mac
    assert options[8]["fit"] == "ok"  # about 9.4 GB: within the 10 GB budget
    assert options[4]["fit"] == "ok"
    assert body["default_bits"] == 8  # the largest that fits the budget
    # Over the budget, still within the Mac: said so, and taken when nothing fits the budget.
    tight = downloads_route.describe(found, budget=4 * GB, mac=12 * GB)
    fits = {option["bits"]: option["fit"] for option in tight["options"]}
    assert fits == {0: "too_large", 8: "over_budget", 4: "over_budget"}
    assert tight["default_bits"] == 8
    nothing = downloads_route.describe(found, budget=GB, mac=2 * GB)
    assert nothing["default_bits"] == 4  # the smallest when nothing fits


def test_the_download_itself_is_a_job(make_studio):
    studio = make_studio()
    fake_job = Path(__file__).resolve().parents[1] / "fake_job.py"
    studio.app.state.jobs.command = [sys.executable, str(fake_job), "--steps", "500"]
    job = studio.post("/studio/api/jobs", json={"kind": "download", "repo": "org/fake"})
    assert job.status_code == 202
    assert job.json()["job"]["details"] == {"repo": "org/fake", "bits": None}
    studio.post(f"/studio/api/jobs/{job.json()['job']['id']}/stop", json={})


def test_the_licence_comes_with_the_check(make_studio, monkeypatch):
    monkeypatch.setattr(
        downloads_route,
        "check_repo",
        lambda repo_id: RepoCheck(repo_id, 7 * GB, "d1", weight_bytes=7 * GB),
    )
    body = check(make_studio(), "LiquidAI/d1-3B").json()
    assert body["family_title"] == "d1" and body["licence"]
    assert [option["bits"] for option in body["options"]] == [0]  # d1 does not convert


def test_download_module_is_the_one_the_route_checks_with():
    assert downloads_route.check_repo is download.check_repo


# First start


def test_first_picks_one_small_one_medium_one_large_that_fit():
    budget = int(12.7 * GB)  # this travel Mac's working set
    seen = {"small": set(), "medium": set(), "large": set()}
    for seed in range(200):
        picks = downloads_route.first_picks(budget, random.Random(seed))
        assert [pick["tier"] for pick in picks] == ["small", "medium", "large"]
        sizes = [pick["bytes"] for pick in picks]
        assert sizes == sorted(sizes)
        assert all(pick["memory_bytes"] <= budget for pick in picks)
        assert all(pick["memory_bytes"] == round(pick["bytes"] * 1.1) for pick in picks)
        assert len({pick["family"] for pick in picks}) == 3  # three families here
        for pick in picks:
            seen[pick["tier"]].add(pick["repo"])
    # A random choice in each size class: below 2 GB, 2 to 8 GB, above.
    assert seen["small"] == {
        "SupersonicLabs/Julia-1", "convaiinnovations/laya", "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions", "vllm-sr/d3-edge", "vllm-sr/d3-lite",
    }  # fmt: skip
    assert seen["medium"] == {"vllm-sr/d3-nano", "LiquidAI/d1-3B"}
    assert seen["large"] == {
        "vllm-sr/d3-mini", "h2oai/h2o-lightning-4b", "perplexity-ai/pplx-decider-flash-v1.2"
    }  # fmt: skip
    assert "Cloudflare/clef-flash" not in set().union(*seen.values())  # 19 GB does not fit
    # A big Mac sees the big models; a small one gets thirds of what fits; a
    # tiny budget fewer than three.
    big = {pick["repo"] for seed in range(100) for pick in
           downloads_route.first_picks(64 * GB, random.Random(seed))[2:]}  # fmt: skip
    assert "Cloudflare/clef-flash" in big and "perplexity-ai/pplx-decider-v1.2" in big
    small_mac = downloads_route.first_picks(int(5.4 * GB), random.Random(1))
    assert [pick["tier"] for pick in small_mac] == ["small", "medium", "large"]
    assert small_mac[-1]["memory_bytes"] <= 5.4 * GB
    tiny = downloads_route.first_picks(int(0.8 * GB), random.Random(1))
    assert [pick["repo"] for pick in tiny] == [
        "SupersonicLabs/Julia-1", "convaiinnovations/laya-multilingual"
    ]  # fmt: skip
    assert [pick["tier"] for pick in tiny] == ["small", "medium"]
    assert downloads_route.first_picks(GB // 10, random.Random(1)) == []


def test_first_start_asks_where_models_go(make_studio, tmp_path):
    studio = make_studio(budget=int(12.7 * GB))
    first = studio.get("/studio/api/downloads/first").json()
    assert first["models_dir_set"] is False
    assert first["elsewhere"] is False  # this Mac: the page asks where models go
    assert first["config_path"] == str(tmp_path / "config.toml")
    assert first["suggestion"] == "~/Models"
    assert [pick["tier"] for pick in first["picks"]] == ["small", "medium", "large"]


def test_the_hugging_face_cache_answer_is_added_to_the_config(make_studio, tmp_path):
    config = tmp_path / "config.toml"
    original = "# my settings\nmemory_budget_gb = 8\n\n[server]\n  port = 9000 # kept\n"
    config.write_text(original)
    studio = make_studio()
    answer = studio.post("/studio/api/downloads/destination", json={"place": "hf-cache"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["line"] == 'models_dir = "hf-cache"'
    assert answer.json()["destination"]["kind"] == "cache"
    assert answer.json()["destination"]["setting"] == "models_dir"
    # Before the first table; every other byte kept.
    assert config.read_text() == (
        '# my settings\nmemory_budget_gb = 8\nmodels_dir = "hf-cache"\n\n[server]\n'
        "  port = 9000 # kept\n"
    )
    assert studio.get("/studio/api/downloads/first").json()["models_dir_set"] is True
    again = studio.post("/studio/api/downloads/destination", json={"place": "hf-cache"})
    assert again.status_code == 409 and "sets models_dir already" in again.text


def test_a_folder_answer_is_created_and_downloads_go_there(make_studio, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    studio = make_studio()
    answer = studio.post(
        "/studio/api/downloads/destination", json={"place": "folder", "path": "~/Models"}
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["line"] == 'models_dir = "~/Models"'
    assert (home / "Models").is_dir()
    assert (tmp_path / "config.toml").read_text() == 'models_dir = "~/Models"\n'
    assert answer.json()["destination"] == {
        "kind": "folder", "path": str(home / "Models"), "setting": "models_dir"
    }  # fmt: skip
    # A download now goes there, and what lands there joins the list.
    fake_job = Path(__file__).resolve().parents[1] / "fake_job.py"
    studio.app.state.jobs.command = [sys.executable, str(fake_job), "--steps", "500"]
    job = studio.post("/studio/api/jobs", json={"kind": "download", "repo": "org/fake"})
    assert job.json()["job"]["command"][-2:] == ["--local-dir", str(home / "Models" / "fake")]
    studio.post(f"/studio/api/jobs/{job.json()['job']['id']}/stop", json={})
    write_model(home / "Models" / "laya")
    added = studio.post("/studio/api/models/rescan", json={}).json()
    assert added == {"added": ["laya"], "removed": []}


def test_the_folder_must_be_under_the_home_folder(make_studio, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "notes.txt").parent.mkdir()
    (home / "notes.txt").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    studio = make_studio()

    def refused(**body):
        response = studio.post("/studio/api/downloads/destination", json=body)
        assert response.status_code == 400, (body, response.text)
        return response.json()["error"]["message"]

    assert "not a folder under your home folder" in refused(place="folder", path=str(tmp_path))
    assert "not a folder under your home folder" in refused(place="folder", path="~")
    assert "not a folder under your home folder" in refused(place="folder", path="~/../x")
    assert "e.g. ~/Models" in refused(place="folder", path="Models")
    assert "a file is there" in refused(place="folder", path="~/notes.txt")
    assert "place must be" in refused(place="elsewhere")
    assert not (tmp_path / "config.toml").exists()  # nothing written


def test_another_repo_of_the_same_name_is_not_taken_for_it(make_studio, tmp_path):
    from mlx_decision.download import record_source

    folder = tmp_path / "models"
    write_model(folder / "d3-edge")
    record_source(folder / "d3-edge", "someone/d3-edge", "abc")  # downloaded from another repo
    studio = make_studio([folder], models_dir=folder)
    families = studio.get("/studio/api/downloads").json()["families"]
    here = {m["repo"]: m["on_this_mac"] for f in families
            for m in f["models"]}  # fmt: skip
    assert here["vllm-sr/d3-edge"] is None
    refused = studio.post("/studio/api/jobs", json={"kind": "download", "repo": "vllm-sr/d3-edge"})
    assert refused.status_code == 409 and "holds someone/d3-edge" in refused.text


def test_a_download_into_a_folder_records_its_repo(make_studio, tmp_path):  # the fake Hub
    from typer.testing import CliRunner

    from mlx_decision.cli import app
    from mlx_decision.download import recorded_repo

    result = CliRunner().invoke(app, ["download", "org/fake", "--local-dir", str(tmp_path / "m")])
    assert result.exit_code == 0, result.output
    assert recorded_repo(tmp_path / "m") == "org/fake"


def test_a_download_is_listed_also_when_models_dir_is_not_scanned(tmp_path):
    from mlx_decision.studio.catalog import Catalog

    folder = tmp_path / "models"
    write_model(folder / "fake")
    catalog = Catalog(references=[])  # web -m ...: models_dir not scanned
    catalog.add_download(folder / "fake")
    assert [spec.path for spec in catalog.scan()] == [folder / "fake"]
    scanned = Catalog(references=[], models_dir=folder)  # without -m: already scanned
    scanned.add_download(folder / "fake")
    assert scanned.references == []
