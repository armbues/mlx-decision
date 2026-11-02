"""Deleting models the Studio made, and fit and evaluation entries."""

import dataclasses
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_fitted, write_model, write_sized_model
from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.registry import MARKER_FILE
from mlx_decision.studio import create_studio
from mlx_decision.studio.catalog import Catalog, studio_models
from mlx_decision.studio.jobs import Job

TESTS = Path(__file__).resolve().parents[1]
HERE = "http://127.0.0.1:8000"
JSON = {"Content-Type": "application/json"}


@pytest.fixture
def studio(tmp_path):
    """A Studio with a downloaded model (``named``), a converted one in the
    workspace (``conv``) with a fitted model on it (``conv-fit``)."""
    workspace = tmp_path / "ws"
    made = workspace / "models"
    write_model(tmp_path / "named")
    write_model(made / "conv")
    write_fitted(made / "conv-fit", made / "conv")
    pool = ModelPool(discover([tmp_path / "named", made]), budget=10**9, default="named")
    app = create_studio(pool, workspace=workspace, config=Config(tmp_path / "config.toml"))
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        client.made = made
        yield client


def delete(studio, name):
    return studio.delete(f"/studio/api/models/{name}", headers=JSON)


def listed(studio):
    return [model["name"] for model in studio.get("/studio/api/models").json()["models"]]


def test_a_converted_model_after_its_fitted_ones(studio):
    response = delete(studio, "conv")
    assert response.status_code == 409
    assert "fitted models are made from conv (conv-fit)" in response.json()["error"]["message"]
    # A loaded model is unloaded first.
    assert studio.post("/studio/api/models/load", json={"model": "conv-fit"}).status_code == 200
    response = delete(studio, "conv-fit")
    assert response.status_code == 200, response.text
    assert response.json() == {"model": "conv-fit", "path": str(studio.made / "conv-fit")}
    assert not (studio.made / "conv-fit").exists()
    assert [path.name for path in studio.made.iterdir()] == ["conv"]  # nothing left aside
    assert "conv-fit" not in studio.app_state.gpu.pool.specs
    assert "conv-fit" not in listed(studio)
    assert delete(studio, "conv").status_code == 200
    assert listed(studio) == ["named"]


def test_a_folder_elsewhere_unknown_and_default_models_stay(studio, tmp_path):
    response = delete(studio, "named")
    assert response.status_code == 403
    message = response.json()["error"]["message"]
    assert message.startswith("named is not in models_dir or the Hugging Face cache")
    assert (tmp_path / "named").is_dir()
    assert delete(studio, "nope").status_code == 404
    studio.app_state.gpu.pool.default = "conv-fit"
    response = delete(studio, "conv-fit")
    assert response.status_code == 409 and "default model" in response.json()["error"]["message"]


def test_a_model_in_use_stays(studio):
    jobs = studio.app_state.jobs
    job = Job(
        id="j1",
        kind="eval",
        title="Evaluate conv-fit",
        argv=["training", "eval", "-m", "conv-fit"],
        holds_gpu=True,
        folder=studio.made.parent / "jobs" / "j1",
    )
    job.state = "running"
    jobs.jobs[job.id] = job
    response = delete(studio, "conv-fit")
    assert response.status_code == 409
    assert response.json()["error"]["message"].startswith("Evaluate conv-fit uses conv-fit")
    del jobs.jobs[job.id]
    # A batch keeps it loaded: busy, naming the batch.
    pool = studio.app_state.gpu.pool
    studio.post("/studio/api/models/load", json={"model": "conv-fit"})
    pool.pin("conv-fit", "batch tweets")
    response = delete(studio, "conv-fit")
    assert response.status_code == 503 and "batch tweets" in response.json()["error"]["message"]
    assert (studio.made / "conv-fit").is_dir() and "conv-fit" in pool.specs
    pool.unpin("conv-fit")


def wait_for(studio, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while (job := studio.get(f"/studio/api/jobs/{job_id}").json()["job"])["state"] != "ended":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return job


def test_a_fit_entry_with_or_without_its_model(studio):
    from test_studio_jobs_api import write_data

    write_data(studio.made.parent / "data" / "tickets.jsonl")
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(TESTS / "fake_job.py"), "--steps", "2", "--fit-events"]
    started = []
    for name in ("one", "two"):
        body = {"kind": "fit", "model": "conv", "data": "tickets", "name": name}
        response = studio.post("/studio/api/jobs", json=body)
        assert response.status_code == 202, response.text
        started.append(wait_for(studio, response.json()["job"]["id"]))
    deadline = time.monotonic() + 10
    while not {"one", "two"} <= set(studio.app_state.gpu.pool.specs):
        assert time.monotonic() < deadline, "the fitted models were not listed"
        time.sleep(0.02)

    one, two = started
    # The entry alone: its model stays.
    response = studio.delete(f"/studio/api/jobs/{one['id']}", headers=JSON)
    assert response.json() == {"job": one["id"], "model": None}
    assert not (studio.made.parent / "jobs" / one["id"]).exists()
    assert "one" in listed(studio)
    # With its model.
    response = studio.delete(f"/studio/api/jobs/{two['id']}?model=true", headers=JSON)
    assert response.json() == {"job": two["id"], "model": "two"}
    assert "two" not in listed(studio) and not (studio.made / "two").exists()
    assert [job["id"] for job in studio.get("/studio/api/jobs").json()["jobs"]] == []
    assert studio.delete(f"/studio/api/jobs/{one['id']}", headers=JSON).status_code == 404


def test_a_job_that_has_not_ended_stays(studio):
    jobs = studio.app_state.jobs
    job = Job(
        id="j2",
        kind="fit",
        title="Fit x",
        argv=["training", "fit"],
        holds_gpu=True,
        folder=studio.made.parent / "jobs" / "j2",
    )
    job.state = "running"
    jobs.jobs[job.id] = job
    response = studio.delete("/studio/api/jobs/j2", headers=JSON)
    assert response.status_code == 409 and "has not ended" in response.json()["error"]["message"]
    del jobs.jobs[job.id]


def test_a_fit_whose_model_was_deleted_on_disk_says_so(studio):
    import shutil

    from test_studio_jobs_api import write_data

    write_data(studio.made.parent / "data" / "tickets.jsonl")
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(TESTS / "fake_job.py"), "--steps", "2", "--fit-events"]
    body = {"kind": "fit", "model": "conv", "data": "tickets", "name": "gone"}
    response = studio.post("/studio/api/jobs", json=body)
    job = wait_for(studio, response.json()["job"]["id"])
    assert studio.get(f"/studio/api/jobs/{job['id']}/report").status_code == 200
    shutil.rmtree(studio.made / "gone")  # in Finder, say
    report = studio.get(f"/studio/api/jobs/{job['id']}/report")
    assert report.status_code == 404
    error = report.json()["error"]
    assert error["code"] == "model_gone"
    assert error["message"] == "The fitted model gone was deleted; its report went with it."


def test_a_folder_that_cannot_move_aside_keeps_its_model(studio, monkeypatch):
    def refuse(self, target):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "rename", refuse)
    assert delete(studio, "conv-fit").status_code == 500
    monkeypatch.undo()
    assert "conv-fit" in listed(studio)
    assert (studio.made / "conv-fit").is_dir()


def test_clear_leaves_a_linked_model_as_delete_does(studio, tmp_path):
    elsewhere = write_model(tmp_path / "mine")
    (studio.made / "linked").symlink_to(elsewhere)
    studio.post("/studio/api/models/rescan", json={})
    if "linked" in listed(studio):
        assert delete(studio, "linked").status_code == 403
    cleared = studio.post("/studio/api/workspace/folders/models/clear", json={})
    assert cleared.status_code == 200, cleared.text
    assert (studio.made / "linked").is_symlink() and elsewhere.is_dir()


def test_an_old_fit_entry_does_not_take_a_later_fit_of_its_name(studio):
    from test_studio_jobs_api import write_data

    write_data(studio.made.parent / "data" / "tickets.jsonl")
    jobs = studio.app_state.jobs
    jobs.command = [sys.executable, str(TESTS / "fake_job.py"), "--steps", "2", "--fit-events"]
    pool = studio.app_state.gpu.pool

    def fit(seed):
        body = {"kind": "fit", "model": "conv", "data": "tickets", "name": "mine", "seed": seed}
        response = studio.post("/studio/api/jobs", json=body)
        assert response.status_code == 202, response.text
        job = wait_for(studio, response.json()["job"]["id"])
        deadline = time.monotonic() + 10
        while "mine" not in pool.specs:  # the rescan after done
            assert time.monotonic() < deadline
            time.sleep(0.02)
        return job

    old = fit(1)
    assert old["outcome"] == "done" and old["wrote"] and old["replaced"] is False
    assert delete(studio, "mine").status_code == 200
    new = fit(2)
    old = studio.get(f"/studio/api/jobs/{old['id']}").json()["job"]
    assert old["replaced"] is True
    report = studio.get(f"/studio/api/jobs/{old['id']}/report")
    assert report.status_code == 404 and report.json()["error"]["code"] == "model_gone"
    assert "replaced by a later fit" in report.json()["error"]["message"]
    response = studio.delete(f"/studio/api/jobs/{old['id']}?model=true", headers=JSON)
    assert response.json() == {"job": old["id"], "model": None}  # the new model stays
    assert (studio.made / "mine").is_dir() and "mine" in listed(studio)
    assert studio.get(f"/studio/api/jobs/{new['id']}/report").status_code == 200


# Downloads: a repo in the Hugging Face cache, a folder in models_dir.


def write_cached(
    cache: Path, repo: str, commit: str, main: bool = True, extra=(), weights: int = 64
) -> Path:
    """A fake model in the Hub cache's layout: files in ``blobs/`` (named by
    their hash), links to them in ``snapshots/<commit>``, ``refs/main``.
    ``extra`` names links that point nowhere (a broken snapshot)."""
    import hashlib

    folder = cache / f"models--{repo.replace('/', '--')}"
    staged = write_sized_model(folder / "staged", weights)
    snapshot = folder / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (folder / "blobs").mkdir(exist_ok=True)
    for file in sorted(staged.iterdir()):
        data = file.read_bytes()
        blob = folder / "blobs" / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        (snapshot / file.name).symlink_to(Path("../../blobs") / blob.name)
    for name in extra:
        (snapshot / name).symlink_to(Path("../../blobs") / ("0" * 64))
    shutil.rmtree(staged)
    if main:
        (folder / "refs").mkdir(exist_ok=True)
        (folder / "refs" / "main").write_text(commit)
    return snapshot


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    """A Studio started without -m: models_dir holds ``laya``, ``kept`` (named
    in the config's models) and ``linked`` (a link to a folder elsewhere);
    the cache ``org/cached`` (two revisions), ``org/other`` and ``org/named``
    (named in the config's models); ``single`` elsewhere is named there too
    and is the default."""
    from huggingface_hub import constants

    cache = tmp_path / "hf"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    write_cached(cache, "org/cached", "b" * 40, main=False, weights=4096)
    snapshot = write_cached(cache, "org/cached", "a" * 40)
    write_cached(cache, "org/other", "c" * 40)
    write_cached(cache, "org/named", "d" * 40)
    models_dir = tmp_path / "models-dir"
    write_model(models_dir / "laya")
    write_model(models_dir / "kept")
    (models_dir / "linked").symlink_to(write_model(tmp_path / "elsewhere" / "target"))
    write_model(tmp_path / "single")
    workspace = tmp_path / "ws"
    config = Config(
        tmp_path / "config.toml",
        models_dir=models_dir,
        models=(str(models_dir / "kept"), str(tmp_path / "single"), "org/named"),
        default_model="single",
    )
    catalog = Catalog(
        [str(models_dir), str(tmp_path / "single")], workspace / "models", models_dir=models_dir
    )
    specs = studio_models(catalog.references, catalog.workspace_models)
    pool = ModelPool(specs, budget=10**9, default="single")
    app = create_studio(pool, workspace=workspace, config=config, catalog=catalog)
    with TestClient(
        app, base_url=HERE, client=("127.0.0.1", 50000), follow_redirects=False
    ) as client:
        assert client.get(f"/?token={app.state.session.token}").status_code == 303
        client.app_state = app.state
        client.cache, client.models_dir, client.snapshot = cache, models_dir, snapshot
        yield client


def test_the_places_are_listed_as_expected(downloads):
    names = set(listed(downloads))
    assert {"laya", "kept", "linked", "single", "cached", "other", "named"} <= names
    removal = downloads.app_state.removal
    kinds = {name: removal.kind(name) for name in names}
    assert kinds == {
        "laya": "models",
        "linked": "models",
        "cached": "cache",
        "other": "cache",
        "kept": None,
        "single": None,
        "named": None,
    }


def models_by_name(studio):
    return {model["name"]: model for model in studio.get("/studio/api/models").json()["models"]}


def test_the_page_is_told_what_may_go_and_what_it_frees(downloads, tmp_path):
    models = models_by_name(downloads)
    seen = {
        name: (model["place"], model["deletable"], model["link"]) for name, model in models.items()
    }
    assert seen == {
        "laya": ("models", True, False),
        "linked": ("models", True, True),
        "cached": ("cache", True, False),
        "other": ("cache", True, False),
        "kept": ("folder", False, False),  # named, though in models_dir
        "single": ("folder", False, False),
        "named": ("cache", False, False),
    }
    # A cache repo frees every revision's files (the older one is larger).
    repo = downloads.cache / "models--org--cached"
    blobs = sum(path.stat().st_size for path in (repo / "blobs").iterdir())
    refs = (repo / "refs" / "main").stat().st_size
    assert blobs > 4096 and models["cached"]["freed_bytes"] == blobs + refs
    assert models["cached"]["disk_bytes"] < blobs  # the main revision alone
    assert models["laya"]["freed_bytes"] == models["laya"]["disk_bytes"] > 0
    assert models["linked"]["freed_bytes"] == 0  # only the link goes
    assert models["linked"]["disk_bytes"] > 0
    for name in ("kept", "single", "named"):
        assert models[name]["freed_bytes"] is None
    # Named in the config's models from now on: no Delete any more.
    settings = downloads.app_state.settings
    settings.config = dataclasses.replace(
        settings.config, models=(*settings.config.models, str(downloads.models_dir / "laya"))
    )
    laya = models_by_name(downloads)["laya"]
    assert (laya["place"], laya["deletable"], laya["freed_bytes"]) == ("folder", False, None)


def test_a_studio_model_says_it_may_go(studio):
    models = models_by_name(studio)
    for name in ("conv", "conv-fit"):
        model = models[name]
        assert (model["place"], model["deletable"]) == ("studio", True)
        assert model["freed_bytes"] == model["disk_bytes"]
    assert (models["named"]["deletable"], models["named"]["freed_bytes"]) == (False, None)


def test_a_cached_repo_goes_with_every_revision(downloads):
    repo = downloads.cache / "models--org--cached"
    assert sorted(path.name for path in (repo / "snapshots").iterdir()) == ["a" * 40, "b" * 40]
    assert downloads.post("/studio/api/models/load", json={"model": "cached"}).status_code == 200
    response = delete(downloads, "cached")
    assert response.status_code == 200, response.text
    assert response.json() == {"model": "cached", "path": str(downloads.snapshot)}
    assert not repo.exists()
    assert (downloads.cache / "models--org--other").is_dir()
    assert "cached" not in downloads.app_state.gpu.pool.specs
    assert "cached" not in listed(downloads)
    downloads.post("/studio/api/models/rescan", json={})
    assert "cached" not in listed(downloads) and "other" in listed(downloads)


def test_a_folder_in_models_dir_goes(downloads):
    response = delete(downloads, "laya")
    assert response.status_code == 200, response.text
    assert response.json() == {"model": "laya", "path": str(downloads.models_dir / "laya")}
    assert sorted(path.name for path in downloads.models_dir.iterdir()) == ["kept", "linked"]
    assert "laya" not in listed(downloads)


def test_a_link_in_models_dir_loses_only_the_link(downloads, tmp_path):
    response = delete(downloads, "linked")
    assert response.status_code == 200, response.text
    assert not (downloads.models_dir / "linked").exists()
    assert not (downloads.models_dir / "linked").is_symlink()
    assert (tmp_path / "elsewhere" / "target" / MARKER_FILE).is_file()
    assert "linked" not in listed(downloads)


def test_models_named_themselves_stay(downloads, tmp_path):
    for name, place in (
        ("kept", downloads.models_dir / "kept"),  # in models_dir, named in models
        ("named", downloads.cache / "models--org--named"),  # a repo id in models
    ):
        response = delete(downloads, name)
        assert response.status_code == 403, name
        error = response.json()["error"]
        assert error["type"] == "permission_error"
        assert error["message"].startswith(f"{name} was named for the Studio"), error
        assert place.is_dir()
    assert "Hugging Face cache" in delete(downloads, "named").json()["error"]["message"]
    # Given with -m: a cached repo by its id, a folder in models_dir.
    catalog = downloads.app_state.rescans.catalog
    catalog.references += ["org/other", str(downloads.models_dir / "laya")]
    for name in ("other", "laya"):
        assert delete(downloads, name).status_code == 403, name
    assert (downloads.cache / "models--org--other").is_dir()
    assert (downloads.models_dir / "laya").is_dir()


def test_a_download_with_web_m_can_go(downloads):
    """With -m the Studio's downloads into models_dir are references too, but
    not named ones."""
    catalog = downloads.app_state.rescans.catalog
    catalog.references = [str(downloads.models_dir.parent / "single")]  # web -m single
    catalog.models_dir = None
    fetched = write_model(downloads.models_dir / "fetched")
    catalog.add_download(fetched)
    downloads.post("/studio/api/models/rescan", json={})
    assert "fetched" in listed(downloads)
    assert catalog.named() == [str(downloads.models_dir.parent / "single")]
    response = delete(downloads, "fetched")
    assert response.status_code == 200, response.text
    assert not fetched.exists()


def test_downloads_are_refused_as_the_studios_models(downloads):
    pool = downloads.app_state.gpu.pool
    pool.default = "cached"
    response = delete(downloads, "cached")
    assert response.status_code == 409 and "default model" in response.json()["error"]["message"]
    pool.default = "single"
    # A fitted model on another revision of the repo keeps all of it.
    older = downloads.cache / "models--org--cached" / "snapshots" / ("b" * 40)
    fitted = write_fitted(downloads.app_state.workspace.root / "models" / "on-b", older)
    downloads.post("/studio/api/models/rescan", json={})
    assert "on-b" in listed(downloads)
    response = delete(downloads, "cached")
    assert response.status_code == 409
    assert "fitted models are made from cached (on-b)" in response.json()["error"]["message"]
    assert delete(downloads, "on-b").status_code == 200 and not fitted.exists()
    # A job naming the repo (a download of it).
    jobs = downloads.app_state.jobs
    job = Job(
        id="j3",
        kind="download",
        title="Download org/cached",
        argv=["download", "org/cached"],
        holds_gpu=False,
        folder=downloads.app_state.workspace.root / "jobs" / "j3",
    )
    job.state = "running"
    jobs.jobs[job.id] = job
    response = delete(downloads, "cached")
    assert response.status_code == 409
    assert response.json()["error"]["message"].startswith("Download org/cached uses cached")
    del jobs.jobs[job.id]
    # A batch keeps it loaded.
    downloads.post("/studio/api/models/load", json={"model": "laya"})
    pool.pin("laya", "batch tweets")
    response = delete(downloads, "laya")
    assert response.status_code == 503 and "batch tweets" in response.json()["error"]["message"]
    pool.unpin("laya")
    assert (downloads.models_dir / "laya").is_dir()
    assert (downloads.cache / "models--org--cached").is_dir()
    assert {"laya", "cached"} <= set(listed(downloads))


def test_a_repo_the_cache_cannot_read_stays(downloads):
    write_cached(downloads.cache, "org/broken", "e" * 40, extra=["README.md"])
    downloads.post("/studio/api/models/rescan", json={})
    assert "broken" in listed(downloads)
    response = delete(downloads, "broken")
    assert response.status_code == 409, response.text
    assert "the Hugging Face cache cannot read org/broken" in response.json()["error"]["message"]
    assert "broken" in listed(downloads)
    assert (downloads.cache / "models--org--broken").is_dir()


def test_a_repo_left_in_the_cache_is_listed_again(downloads, monkeypatch):
    from mlx_decision.studio import removal

    monkeypatch.setattr(removal, "_delete_revisions", lambda repo, revisions: True)
    response = delete(downloads, "other")
    assert response.status_code == 500, response.text
    error = response.json()["error"]
    assert error["type"] == "api_error"
    assert "hf cache rm model/org/other" in error["message"]
    deadline = time.monotonic() + 10
    while "other" not in listed(downloads):  # the rescan after it
        assert time.monotonic() < deadline, "not listed again"
        time.sleep(0.02)
