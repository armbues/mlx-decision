"""Model discovery and the model pool, with fake models of made-up sizes."""

import gc
import logging
import weakref
from pathlib import Path

import pytest

import mlx_decision.memory as memory
import mlx_decision.pool as pool_module
from fake_backend import TEAM, write_fitted, write_model, write_sized_model
from mlx_decision import DecisionError, ModelTooLargeError
from mlx_decision.fitted import FIT_FILE, FittedBackend
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


def test_unused_options_name_the_families_in_their_usual_order(tmp_path):
    write_marker(tmp_path / "models" / "a-laya", "laya")
    write_clef(tmp_path / "models" / "b-clef")
    with pytest.raises(ValueError, match="clef, laya models do not take colour"):
        discover([tmp_path / "models"], {"colour": "red"})


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


def test_unloaded_models_are_gone_before_the_memory_release_and_the_next_load(
    models, tmp_path, monkeypatch
):
    big = fake(tmp_path / "big", 1800)  # 1,980 counted: both loaded models must go
    refs, alive = {}, []

    def living():
        return [name for name, ref in refs.items() if ref() is not None]

    def load(spec):
        alive.append(("load", living()))
        model = pool_module._load(spec)
        refs[spec.name] = weakref.ref(model)
        return model

    def release():
        gc.collect()
        alive.append(("release", living()))

    monkeypatch.setattr(pool_module, "_release_memory", release)
    pool = ModelPool(discover([models, big]), budget=2200, load=load)
    pool.get("Alpha")
    pool.get("beta")
    alive.clear()
    pool.get("big")
    assert alive == [("release", []), ("load", [])]


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


def test_the_convert_hint_names_the_model_folder(tmp_path):
    clef = write_clef(tmp_path / "clef")
    pool = ModelPool(
        discover([clef], {"prefix_cache_gb": 0}), budget=memory.required_bytes(clef) - 1
    )
    with pytest.raises(ModelTooLargeError) as error:
        pool.get(None)
    assert str(error.value).startswith("clef needs about ")
    assert f"convert -m {clef} -q" in str(error.value)


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


# Fitted models


@pytest.fixture
def fits(tmp_path) -> Path:
    """A folder with the 1,000-byte model "base", two calibrations over it ("fit-a",
    "fit-b") and the 1,000-byte models "other" and "third"."""
    root = tmp_path / "models"
    base = fake(root / "base", 1000)
    write_fitted(root / "fit-a", base)
    write_fitted(root / "fit-b", base)
    fake(root / "other", 1000)
    fake(root / "third", 1000)
    return root


def fit_bytes(folder: Path) -> int:
    return round(memory.tensor_bytes(folder / FIT_FILE) * memory.MARGIN)


def test_fitted_models_are_listed_beside_their_base_in_the_base_family(fits):
    specs = {spec.name: spec for spec in discover([fits])}
    assert list(specs) == ["base", "fit-a", "fit-b", "other", "third"]
    fit = specs["fit-a"]
    assert (fit.family.name, fit.path, fit.fit.base) == ("fake", fits / "fit-a", fits / "base")
    assert (fit.base.name, fit.base.path) == ("base", fits / "base")
    assert specs["base"].fit is None and specs["base"].base is None


def test_a_fitted_model_takes_the_options_of_its_base_family(tmp_path):
    root = tmp_path / "models"
    write_clef(root / "clef")
    write_fitted(root / "laya-fit", write_marker(tmp_path / "laya", "laya"))
    specs = discover([root], {"prefix_cache_gb": 1.0, "max_input_tokens": 64})
    assert {spec.name: spec.options for spec in specs} == {
        "clef": {"prefix_cache_gb": 1.0, "max_input_tokens": 64},
        "laya-fit": {"max_input_tokens": 64},
    }
    with pytest.raises(ValueError, match="laya models do not take prefix_cache_gb"):
        discover([root / "laya-fit"], {"prefix_cache_gb": 1.0})


def test_a_fitted_model_without_its_base_is_skipped_in_a_folder_and_refused_alone(fits, tmp_path):
    (fits / "base").rename(tmp_path / "moved")
    skipped = []
    specs = discover([fits], on_skip=lambda path, reason: skipped.append((path.name, reason)))
    assert [spec.name for spec in specs] == ["other", "third"]
    assert [name for name, _ in skipped] == ["fit-a", "fit-b"]
    assert "its base model" in skipped[0][1] and "is missing" in skipped[0][1]
    with pytest.raises(ValueError, match="fit-a: its base model .* is missing"):
        discover([fits / "fit-a"])


def test_fitted_models_share_their_loaded_base(fits, working_set):
    pool, loads = make_pool(fits)
    fit_a = pool.get("fit-a")
    assert loads.names == ["base"]  # loaded as the base, not once per fitted model
    base = pool.get("base")
    fit_b = pool.get("fit-b")
    assert loads.names == ["base"]
    assert isinstance(fit_a.backend, FittedBackend)
    assert fit_a.backend.base is base.backend is fit_b.backend.base
    assert pool.loaded() == ["fit-a", "base", "fit-b"]
    body = {"state": "x", "questions": {"team": TEAM}}
    assert [pool.get(name).decide_request(body).model for name in pool.loaded()] == [
        "fit-a",
        "base",
        "fit-b",
    ]
    fitted = fit_a.decide_request(body).answers["team"].probabilities
    plain = base.decide_request(body).answers["team"].probabilities
    assert fitted != pytest.approx(plain)


def test_a_shared_base_is_counted_once_and_unloaded_with_its_fitted_models(fits, caplog):
    # Room for two models of 1,100 bytes and two fits.
    pool, loads = make_pool(fits, budget=2200 + 2 * fit_bytes(fits / "fit-a"))
    assert pool.size("fit-a") == 1100 + fit_bytes(fits / "fit-a")  # alone: with its base
    base = weakref.ref(pool.get("fit-a").backend.base)
    pool.get("other")
    pool.get("fit-b")
    pool.get("base")
    assert pool.loaded() == ["fit-a", "other", "fit-b", "base"]
    pool.get("third")  # "other" is the least recently used
    assert pool.loaded() == ["fit-a", "fit-b", "base", "third"]
    with caplog.at_level(logging.INFO, logger="mlx_decision.pool"):
        pool.get("other")  # now the base, with everything that shares it
    assert pool.loaded() == ["third", "other"]
    assert "unloaded fit-a, fit-b, base to make room for other" in caplog.text
    assert base() is None
    assert loads.names == ["base", "other", "third", "other"]


def test_a_base_that_is_not_listed_loads_under_its_own_name(fits, caplog, working_set):
    loads = Loads()
    pool = ModelPool(discover([fits / "fit-a", fits / "fit-b"]), budget=1500, load=loads)
    with caplog.at_level(logging.INFO, logger="mlx_decision.pool"):
        model = pool.get("fit-a")
        pool.get("fit-b")
    assert loads.names == ["base"]
    assert model.backend.base.name == "base"
    assert pool.loaded() == ["fit-a", "fit-b"]
    assert "loading fit-a (with its base base)" in caplog.text
    assert "loading fit-b (sharing the loaded base)" in caplog.text
    tight = ModelPool(discover([fits / "fit-a"]), budget=1100, load=Loads())
    with pytest.raises(ModelTooLargeError, match="fit-a needs about"):
        tight.get(None)  # the base alone would fit, with the fit it does not


def test_a_fit_with_trained_weights_loads_a_base_of_its_own(fits, working_set):
    write_fitted(fits / "tuned", fits / "base", kinds=("finetune", "calibrate"))
    pool, loads = make_pool(fits)
    tuned = pool.get("tuned")
    base = pool.get("base")
    assert loads.names == ["tuned", "base"]
    assert tuned.backend.base is not base.backend
    assert tuned.backend.base.fit_weights == fits / "tuned" / FIT_FILE
    assert base.backend.fit_weights is None
