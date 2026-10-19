"""What the Models page says about a model, read from its files: parameters,
precision, place and memory, and the pool's groups and last-use times."""

import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from fake_backend import write_fitted, write_model, write_sized_model
from mlx_decision.convert import convert
from mlx_decision.memory import weights_summary
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.catalog import model_facts
from tiny_models import write_clef

HERE = "http://127.0.0.1:8000"
BODY = {"state": "x", "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}}


@pytest.fixture(scope="module")
def clef_copies(tmp_path_factory):
    """A tiny Clef release and its 8-bit, 4-bit and mixed copies."""
    root = tmp_path_factory.mktemp("clef")
    release = write_clef(root / "release")
    return {
        "release": release,
        "8": convert(release, root / "q8", bits=8),
        "4": convert(release, root / "q4", bits=4),
        "mixed": convert(release, root / "mixed", target_bits=4.5, group_size=32),
    }


def test_quantized_copies_count_the_parameters_of_the_release(clef_copies):
    parameters, precision = weights_summary(clef_copies["release"])
    assert parameters > 0 and precision in ("bf16", "fp16", "fp32")
    assert weights_summary(clef_copies["8"]) == (parameters, "8-bit")
    assert weights_summary(clef_copies["4"]) == (parameters, "4-bit")
    mixed_parameters, mixed = weights_summary(clef_copies["mixed"])
    assert mixed_parameters == parameters
    assert mixed.startswith("mixed ") and mixed.endswith("-bit") and "4" in mixed


def test_weights_without_a_float_type_have_no_precision(tmp_path):
    folder = write_sized_model(tmp_path / "bytes", 1000)
    assert weights_summary(folder) == (1000, None)  # uint8 only: not a float type


def test_model_facts_say_where_a_model_lives(clef_copies, tmp_path):
    [spec] = discover([clef_copies["8"]])
    facts = model_facts(spec, workspace_models=tmp_path / "ws" / "models")
    assert facts["family_title"] == "Clef" and facts["maker"] == "Cloudflare"
    assert facts["precision"] == "8-bit" and facts["parameters"] > 0
    assert facts["place"] == "folder" and facts["repo"] is None
    assert facts["converted_from"] == "release"  # the source the copy was made from
    assert facts["disk_bytes"] > 0
    studio = model_facts(spec, workspace_models=clef_copies["8"].parent)
    assert studio["place"] == "studio"
    cache = tmp_path / "hub" / "models--org--tiny" / "snapshots" / "abc123"
    cache.parent.mkdir(parents=True)
    cache.symlink_to(clef_copies["8"], target_is_directory=True)
    [cached] = discover([cache])
    facts = model_facts(cached, None)
    assert (facts["place"], facts["repo"], facts["revision"]) == ("cache", "org/tiny", "abc123")


def test_pool_groups_last_use_and_footprint(tmp_path):
    root = tmp_path / "models"
    base = write_sized_model(root / "base", 1000)
    write_fitted(root / "fit", base)
    write_sized_model(root / "other", 1000)
    pool = ModelPool(discover([root]), budget=10**6)
    used = []
    pool.on_use = lambda: used.append(pool.loaded())
    alone, added, group = pool.footprint("fit")
    assert group == "base" and alone == pool.size("fit") and 0 < added < alone
    assert pool.footprint("base") == (1100, 1100, "base")
    before = time.time()
    pool.get("fit")
    pool.get("other")
    assert [g["root"] for g in pool.groups()] == ["base", "other"]
    assert pool.groups()[0]["members"] == ["fit"] and pool.groups()[0]["bytes"] == alone
    assert used == []  # loads are changes, not uses
    pool.get("fit")
    assert used == [["other", "fit"]]
    assert [g["root"] for g in pool.groups()] == ["other", "base"]
    assert set(pool.last_used()) == {"fit", "other"}
    assert all(when >= before for when in pool.last_used().values())
    pool.unload("fit")
    assert set(pool.last_used()) == {"other"} and [g["root"] for g in pool.groups()] == ["other"]


def test_the_models_route_and_status_carry_the_pages_facts(tmp_path):
    root = tmp_path / "models"
    write_model(root / "fake")
    write_sized_model(root / "huge", 10**7)
    pool = ModelPool(discover([root]), budget=10**6)
    app = create_studio(pool, workspace=tmp_path / "ws")
    with TestClient(app, base_url=HERE, client=("127.0.0.1", 50000)) as client:
        client.get(f"/?token={app.state.session.token}")
        models = {m["name"]: m for m in client.get("/studio/api/models").json()["models"]}
        fake, huge = models["fake"], models["huge"]
        assert fake["memory"]["over_budget"] is False and fake["memory"]["group"] == "fake"
        assert huge["memory"]["over_budget"] is True and huge["memory"]["bytes"] > 10**6
        assert {"maker", "precision", "parameters", "place", "path", "disk_bytes"} <= set(fake)
        assert fake["place"] == "folder"
        assert (
            client.post("/studio/api/run", json={"request": BODY, "model": "fake"}).status_code
            == 200
        )
        status = client.get("/studio/api/status").json()
        assert status["groups"] == [
            {"root": "fake", "members": ["fake"], "bytes": status["memory"]["used"]}
        ]
        assert set(status["used"]) == {"fake"} and isinstance(status["used"]["fake"], int)


# How a terminal names a model (Get Code, the Models page)


def test_cli_arguments_use_the_name_the_config_file_resolves(tmp_path):
    from mlx_decision.config import Config
    from mlx_decision.studio.catalog import cli_arguments

    folder = tmp_path / "models"
    write_model(folder / "fake")
    write_fitted(folder / "fake-fit", folder / "fake")
    elsewhere = write_model(tmp_path / "other" / "loose")
    specs = {spec.name: spec for spec in discover([folder, elsewhere])}
    config = Config(tmp_path / "config.toml", models_dir=folder)
    named = cli_arguments(specs["fake"], config, None)
    assert named == {"short": ["-m", "fake"], "full": ["-m", str(folder / "fake")]}
    assert cli_arguments(specs["fake-fit"], config, None)["short"] == ["-m", "fake-fit"]
    # Not in the config file: the folder, for both.
    loose = cli_arguments(specs["loose"], config, None)
    assert loose["short"] == loose["full"] == ["-m", str(elsewhere)]
    assert cli_arguments(specs["fake"], Config(tmp_path / "none.toml"), None)["short"] == [
        "-m",
        str(folder / "fake"),
    ]


def test_cli_arguments_of_a_cached_model_name_its_repo(tmp_path):
    from mlx_decision.config import Config
    from mlx_decision.studio.catalog import cli_arguments

    [spec] = discover([write_model(tmp_path / "snapshot")])
    spec = dataclasses.replace(spec, name="d3-edge")
    plain = cli_arguments(spec, Config(tmp_path / "config.toml"), "vllm-sr/d3-edge")
    assert plain == {"short": ["-m", "vllm-sr/d3-edge"], "full": ["-m", "vllm-sr/d3-edge"]}
    listed = Config(tmp_path / "config.toml", models=("vllm-sr/d3-edge",))
    assert cli_arguments(spec, listed, "vllm-sr/d3-edge")["short"] == ["-m", "d3-edge"]


def test_cli_arguments_carry_the_load_options(tmp_path):
    from mlx_decision.config import Config
    from mlx_decision.studio.catalog import cli_arguments

    [spec] = discover([write_model(tmp_path / "fake")])
    options = {
        "max_image_pixels": 2 * 2**20,
        "max_input_tokens": 2048,
        "prefix_cache_gb": 0.5,
        "noul_floor": 0.01,
    }
    spec = dataclasses.replace(spec, options=options)
    arguments = cli_arguments(spec, Config(tmp_path / "config.toml"), None)["short"]
    assert arguments[2:] == [
        "--max-image-mp",
        "2",
        "--max-input-tokens",
        "2048",
        "--prefix-cache",
        "0.5",
        "--noul-floor",
        "0.01",
    ]
    spec = dataclasses.replace(spec, options={"max_image_pixels": None})
    assert cli_arguments(spec, Config(tmp_path / "config.toml"), None)["full"][2:] == [
        "--max-image-mp",
        "0",
    ]
