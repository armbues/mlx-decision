"""Model discovery and the model pool, with fake models of made-up sizes."""

import logging
import weakref
from pathlib import Path

import pytest

import mlx_decision.memory as memory
import mlx_decision.pool as pool_module
from fake_backend import write_model, write_sized_model
from mlx_decision import DecisionError, ModelTooLargeError
from mlx_decision.hub import ModelNotFoundError
from mlx_decision.pool import ModelPool, discover, prefix_cache_bytes
from tiny_models import write_clef, write_marker

BODY = {"state": "Stripe is down", "questions": {"urgent": {"type": "noul"}}}


def fake(folder: Path, weights: int = 0) -> Path:
    return write_sized_model(folder, weights) if weights else write_model(folder)


@pytest.fixture
def models(tmp_path) -> Path:
    """A folder of three fake models of 1,000 bytes each (1,100 counted) and two non-models."""
    root = tmp_path / "models"
    for name in ("beta", "Alpha", "gamma"):
        fake(root / name, 1000)
    (root / "encoder-base").mkdir()
    (root / "encoder-base" / "config.json").write_text("{}")
    (root / ".cache").mkdir()
    (root / "notes.txt").write_text("not a model")
    return root


@pytest.fixture
def working_set(monkeypatch):
    monkeypatch.setattr(memory, "working_set", lambda: 10**12)


def test_a_model_folder_is_one_model_named_after_it(tmp_path):
    path = fake(tmp_path / "my-model")
    [spec] = discover([path])
    assert (spec.name, spec.path, spec.family.name, spec.options) == ("my-model", path, "fake", {})


def test_a_folder_of_models_lists_its_models_alphabetically_and_skips_the_rest(models):
    skipped = []
    specs = discover([models], on_skip=lambda path, reason: skipped.append((path.name, reason)))
    assert [spec.name for spec in specs] == ["Alpha", "beta", "gamma"]
    assert [name for name, _ in skipped] == ["encoder-base"]
    assert "not a supported decision model" in skipped[0][1]


def test_skipped_folders_are_logged_by_default(models, caplog):
    with caplog.at_level(logging.INFO, logger="mlx_decision.pool"):
        discover([models])
    assert "skipped" in caplog.text and "encoder-base" in caplog.text


def test_several_references_keep_their_order(models, tmp_path):
    extra = fake(tmp_path / "zeta")
    assert [spec.name for spec in discover([extra, models])] == ["zeta", "Alpha", "beta", "gamma"]


def test_a_folder_without_models_is_an_error(tmp_path):
    (tmp_path / "empty" / "sub").mkdir(parents=True)
    with pytest.raises(ValueError, match="none of its subfolders is one"):
        discover([tmp_path / "empty"])


def test_a_missing_folder_is_not_found(tmp_path):
    with pytest.raises(ModelNotFoundError):
        discover([tmp_path / "missing"])


def test_two_models_with_the_same_name_are_an_error(models, tmp_path):
    with pytest.raises(ValueError, match="two models are named beta"):
        discover([models, fake(tmp_path / "other" / "beta")])


def test_a_repo_id_is_named_after_the_repo(tmp_path, monkeypatch):
    path = fake(tmp_path / "snapshots" / "0123abcd")
    monkeypatch.setattr(pool_module, "resolve_model_path", lambda reference: path)
    [spec] = discover(["someone/some-model"])
    assert (spec.name, spec.path) == ("some-model", path)


def test_options_go_to_the_families_that_take_them(tmp_path):
    clef = write_clef(tmp_path / "models" / "clef")
    write_marker(tmp_path / "models" / "laya", "laya")
    specs = discover([tmp_path / "models"], {"prefix_cache_gb": 1.0, "max_input_tokens": 64})
    options = {spec.name: spec.options for spec in specs}
    assert options == {
        "clef": {"prefix_cache_gb": 1.0, "max_input_tokens": 64},
        "laya": {"max_input_tokens": 64},
    }
    with pytest.raises(ValueError, match="laya models do not take prefix_cache_gb"):
        discover([tmp_path / "models" / "laya"], {"prefix_cache_gb": 1.0})
    with pytest.raises(ValueError, match="clef, laya models do not take colour"):
        discover([tmp_path / "models"], {"colour": "red"})
    assert discover([clef], {"vision": True})[0].options == {"vision": True}


def test_the_prefix_cache_limit_is_the_option_or_the_loaders_default(tmp_path):
    clef = write_clef(tmp_path / "clef")
    laya = write_marker(tmp_path / "laya", "laya")
    assert prefix_cache_bytes(discover([clef])[0]) == 2 * 10**9
    assert prefix_cache_bytes(discover([clef], {"prefix_cache_gb": 0.5})[0]) == 5 * 10**8
    assert prefix_cache_bytes(discover([clef], {"prefix_cache_gb": 0})[0]) == 0
    assert prefix_cache_bytes(discover([laya])[0]) == 0


class Loads:
    """A load function that counts loads and keeps no references to the models."""

    def __init__(self):
        self.names: list[str] = []

    def __call__(self, spec):
        self.names.append(spec.name)
        return pool_module._load(spec)


def make_pool(models, budget=None, **kwargs):
    loads = Loads()
    return ModelPool(discover([models]), budget=budget, load=loads, **kwargs), loads


def test_one_model_is_the_default(tmp_path, working_set):
    pool = ModelPool(discover([fake(tmp_path / "only")]))
    assert pool.default == "only"
    for name in (None, "jev-latest", "only"):
        assert pool.get(name).name == "only"


def test_several_models_have_no_default_unless_named(models, working_set):
    pool, _ = make_pool(models)
    assert pool.default is None
    for name in (None, "jev-latest"):
        with pytest.raises(DecisionError) as error:
            pool.get(name)
        assert error.value.param == "model"
        assert "Alpha, beta, gamma" in error.value.message
    assert pool.get("beta").name == "beta"
    named, _ = make_pool(models, default="gamma")
    assert named.get("jev-latest").name == "gamma"
    assert named.get("Alpha").name == "Alpha"


def test_an_unknown_default_is_an_error(models):
    with pytest.raises(ValueError, match="no model is named delta"):
        ModelPool(discover([models]), default="delta")


def test_a_model_loads_once_and_answers_under_its_name(models, working_set):
    pool, loads = make_pool(models)
    first = pool.get("beta")
    assert pool.get("beta") is first
    assert loads.names == ["beta"]
    assert pool.loaded() == ["beta"]
    assert first.decide_request(BODY).model == "beta"


def test_the_least_recently_used_model_is_unloaded_to_fit_the_budget(models):
    pool, loads = make_pool(models, budget=2200)  # two models of 1,100
    alpha = weakref.ref(pool.get("Alpha"))
    pool.get("beta")
    pool.get("Alpha")
    beta = weakref.ref(pool.get("beta"))
    assert pool.loaded() == ["Alpha", "beta"]
    pool.get("gamma")
    assert pool.loaded() == ["beta", "gamma"]
    assert alpha() is None  # released, not only forgotten
    assert beta() is not None
    assert loads.names == ["Alpha", "beta", "gamma"]


def test_the_memory_is_released_only_after_an_unload(models, monkeypatch):
    released = []
    monkeypatch.setattr(pool_module, "_release_memory", lambda: released.append(True))
    pool, _ = make_pool(models, budget=1100)
    pool.get("Alpha")
    assert released == []
    pool.get("beta")
    assert released == [True]


def test_the_working_set_is_the_default_budget(models, monkeypatch):
    monkeypatch.setattr(memory, "working_set", lambda: 1100)
    pool, _ = make_pool(models)
    pool.get("Alpha")
    pool.get("beta")
    assert pool.loaded() == ["beta"]


def test_a_model_larger_than_the_budget_is_refused_without_unloading(models, tmp_path):
    big = fake(tmp_path / "big", 5000)
    pool = ModelPool(discover([models, big]), budget=2200, load=Loads())
    pool.get("Alpha")
    with pytest.raises(ModelTooLargeError, match="big needs about .* the memory budget is"):
        pool.get("big")
    assert pool.loaded() == ["Alpha"]


def test_without_the_memory_check_a_large_model_loads_alone(models, tmp_path):
    big = fake(tmp_path / "big", 5000)
    pool = ModelPool(discover([models, big]), budget=2200, load=Loads(), check_memory=False)
    pool.get("Alpha")
    pool.get("big")
    assert pool.loaded() == ["big"]


def test_a_model_counts_its_prefix_cache_limit(tmp_path, working_set):
    clef = write_clef(tmp_path / "clef")
    required = memory.required_bytes(clef)
    pool = ModelPool(discover([clef]))
    assert pool.size("clef") == required + 2 * 10**9
    small = ModelPool(discover([clef], {"prefix_cache_gb": 0}))
    assert small.size("clef") == required
    tight = ModelPool(discover([clef]), budget=required + 10**9)
    with pytest.raises(ModelTooLargeError, match="plus 2.0 GB for kept prefixes"):
        tight.get(None)


def test_a_failed_load_leaves_the_pool_unchanged(models):
    def load(spec):
        if spec.name == "beta":
            raise FileNotFoundError("weights missing")
        return pool_module._load(spec)

    pool = ModelPool(discover([models]), budget=10**9, load=load)
    pool.get("Alpha")
    with pytest.raises(FileNotFoundError):
        pool.get("beta")
    assert pool.loaded() == ["Alpha"]
