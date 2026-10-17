"""Static capabilities (read from a model's files) against the loaded backend's."""

from pathlib import Path

import mlx.core as mx
import pytest

from fake_backend import write_fitted, write_model
from mlx_decision.convert import convert
from mlx_decision.registry import load_backend, static_capabilities
from tiny_models import write_clef, write_d1, write_d3, write_h2o, write_marker, write_pplx


def _models(root: Path) -> dict[str, Path]:
    models = {
        "clef": write_clef(root / "clef"),
        "clef-vision": write_clef(root / "clef-vision", vision=True),
        "pplx": write_pplx(root / "pplx"),
        "d3": write_d3(root / "d3"),
        "h2o": write_h2o(root / "h2o"),
        "d1": write_d1(root / "d1"),
        "laya": write_marker(root / "laya", "laya"),
        "julia": write_marker(root / "julia", "julia"),
        "fake": write_model(root / "fake", max_input_tokens=8, supports_images=True),
    }
    models["clef-8bit"] = convert(models["clef-vision"], root / "clef-8bit", bits=8)
    models["d3-8bit"] = convert(models["d3"], root / "d3-8bit", bits=8)
    models["laya-fit"] = write_fitted(root / "laya-fit", models["laya"])
    models["h2o-fit"] = write_fitted(root / "h2o-fit", models["h2o"])
    return models


@pytest.fixture(scope="module")
def models(tmp_path_factory) -> dict[str, Path]:
    return _models(tmp_path_factory.mktemp("models"))


# Per model: load options, each set compared on its own ({} is the defaults).
CASES = [
    ("clef", {}),
    ("clef", {"max_input_tokens": 64, "prefix_cache_gb": 0}),
    ("clef-vision", {}),
    ("clef-vision", {"vision": True, "max_input_tokens": 64}),
    ("clef-8bit", {}),
    ("pplx", {}),
    ("pplx", {"max_input_tokens": 100, "vision": True}),
    ("d3", {}),
    ("d3", {"max_input_tokens": 100, "max_image_pixels": 70000}),
    ("d3-8bit", {}),
    ("h2o", {}),
    ("h2o", {"max_input_tokens": 1000, "noul_floor": 0, "prefix_cache_gb": 0}),
    ("d1", {}),
    ("d1", {"max_input_tokens": 1000}),
    ("laya", {}),
    ("laya", {"max_input_tokens": 32, "dtype": "float32"}),
    ("julia", {}),
    ("julia", {"max_input_tokens": 32, "strict_encoding": False}),
    ("laya-fit", {}),
    ("laya-fit", {"max_input_tokens": 32}),
    ("h2o-fit", {}),
    ("fake", {}),
    ("fake", {"max_choice_options": 4, "truncates_input": False}),
]


@pytest.mark.parametrize(("name", "options"), CASES)
def test_static_capabilities_are_the_loaded_ones(models, name, options):
    path = models[name]
    assert static_capabilities(path, **options) == load_backend(path, **options).capabilities


def test_options_change_the_static_capabilities(models):
    """The table above is not passing on defaults alone."""
    assert static_capabilities(models["d1"], max_input_tokens=1000).max_input_tokens == 1000
    julia = static_capabilities(models["julia"], strict_encoding=False)
    assert julia.truncates_input and not static_capabilities(models["julia"]).truncates_input
    assert static_capabilities(models["clef-vision"]).supports_images
    assert not static_capabilities(models["clef"]).supports_images
    fitted = static_capabilities(models["laya-fit"], max_input_tokens=32)
    assert fitted == static_capabilities(models["laya"], max_input_tokens=32)


def test_no_weights_are_read(models, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("weights read")

    monkeypatch.setattr(mx, "load", refuse)
    for name, options in CASES:
        static_capabilities(models[name], **options)


# Options the loader refuses: the static capabilities refuse them with the same message.
REFUSED = [
    ("clef", {"vision": True}),
    ("pplx", {"max_input_tokens": 9000}),
    ("d3", {"max_input_tokens": 0}),
    ("h2o", {"max_input_tokens": 50000}),
    ("d1", {"max_input_tokens": 40000}),
    ("laya", {"max_input_tokens": 8}),
    ("julia", {"max_input_tokens": 100000}),
    ("laya", {"noul_floor": 0.5}),
    ("laya-fit", {"noul_floor": 0.5}),
]


@pytest.mark.parametrize(("name", "options"), REFUSED)
def test_refused_options_give_the_loaders_error(models, name, options):
    path = models[name]
    with pytest.raises(ValueError) as loaded:
        load_backend(path, **options)
    with pytest.raises(ValueError) as static:
        static_capabilities(path, **options)
    assert str(static.value) == str(loaded.value)
