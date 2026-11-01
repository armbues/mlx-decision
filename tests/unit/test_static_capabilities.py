"""Static capabilities (read from a model's files) against the loaded backend's."""

import base64
from functools import partial
from pathlib import Path

import mlx.core as mx
import pytest

from fake_backend import write_fitted, write_model
from mlx_decision import DecisionError
from mlx_decision.convert import convert
from mlx_decision.model import DecisionModel, check_request
from mlx_decision.registry import load_backend, static_capabilities
from tiny_models import (
    write_baselabs,
    write_clef,
    write_d1,
    write_d3,
    write_h2o,
    write_marker,
    write_pplx,
)


def _models(root: Path) -> dict[str, Path]:
    models = {
        "clef": write_clef(root / "clef"),
        "clef-vision": write_clef(root / "clef-vision", vision=True),
        "pplx": write_pplx(root / "pplx"),
        "d3": write_d3(root / "d3"),
        "baselabs": write_baselabs(root / "baselabs"),
        "h2o": write_h2o(root / "h2o"),
        "d1": write_d1(root / "d1"),
        "laya": write_marker(root / "laya", "laya"),
        "julia": write_marker(root / "julia", "julia"),
        "fake": write_model(root / "fake", max_input_tokens=8, supports_images=True),
    }
    models["clef-8bit"] = convert(models["clef-vision"], root / "clef-8bit", bits=8)
    models["d3-8bit"] = convert(models["d3"], root / "d3-8bit", bits=8)
    models["baselabs-8bit"] = convert(models["baselabs"], root / "baselabs-8bit", bits=8)
    models["laya-fit"] = write_fitted(root / "laya-fit", models["laya"])
    models["h2o-fit"] = write_fitted(root / "h2o-fit", models["h2o"])
    models["baselabs-fit"] = write_fitted(root / "baselabs-fit", models["baselabs"])
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
    ("clef-vision", {"max_image_pixels": 2048}),
    ("clef-vision", {"max_image_pixels": None}),
    ("clef-8bit", {}),
    ("pplx", {}),
    ("pplx", {"max_input_tokens": 100, "vision": True}),
    ("d3", {}),
    ("d3", {"max_input_tokens": 100, "max_image_pixels": 70000}),
    ("d3-8bit", {}),
    ("baselabs", {}),
    ("baselabs", {"max_input_tokens": 100_000, "prefix_cache_gb": 0}),
    ("baselabs-8bit", {}),
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
    ("baselabs-fit", {"max_input_tokens": 32}),
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


def test_image_limits(models):
    """The pixels larger images are scaled down to, and the images a request takes."""

    def limits(name, **options):
        capabilities = static_capabilities(models[name], **options)
        return capabilities.max_image_pixels, capabilities.max_images

    # Clef: its default (2**21) or the option, never over the processor's 4,096.
    assert limits("clef-vision") == (4096, None)
    assert limits("clef-vision", max_image_pixels=2048) == (2048, None)
    assert limits("clef-vision", max_image_pixels=None) == (4096, None)
    assert limits("clef") == (None, None)
    assert limits("pplx") == (262144, None)
    assert limits("d3") == (1638400, None)
    assert limits("d3", max_image_pixels=70000) == (70000, None)
    # H2O: the serve config's image block.
    assert limits("h2o") == (1048576, 4)
    assert limits("baselabs") == limits("d1") == limits("laya") == (None, None)


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
    ("baselabs", {"max_input_tokens": 262145}),
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


# The checks as a function of capabilities: the same outcome on the static
# capabilities as ``check`` on the loaded model, for every request below.
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(32)).decode()


def _choice(n: int) -> dict:
    return {"type": "choice", "criteria": {f"o{index}": None for index in range(n)}}


def _score(n: int) -> dict:
    return {"type": "score", "criteria": [f"level {index}" for index in range(n)]}


NOUL = {"type": "noul", "instructions": "Is it urgent?"}
REQUESTS = {
    "valid": {
        "state": "my card was charged twice",
        "questions": {"urgent": NOUL, "team": _choice(3), "tone": _score(5)},
    },
    "invalid body": {"state": "x", "questions": {"q": {"type": "maybe"}}},
    "no questions": {"state": "x"},
    "videos": {"state": "x", "questions": {"q": NOUL}, "videos": ["data:video/mp4;base64,AA=="]},
    "images": {"state": "x", "questions": {"q": NOUL}, "images": [PNG]},
    "5 images": {"state": "x", "questions": {"q": NOUL}, "images": [PNG] * 5},
    "image in state": {"state": {"photo": PNG}, "questions": {"q": NOUL}},
    "bad image in state": {"state": "data:image/png;base64,!!!", "questions": {"q": NOUL}},
    "21 options": {"state": "x", "questions": {"q": _choice(21)}},
    "256 options": {"state": "x", "questions": {"q": _choice(256)}},
    "11 levels": {"state": "x", "questions": {"q": _score(11)}},
    "21 levels": {"state": "x", "questions": {"q": _score(21)}},
    "no instructions": {"state": "x", "questions": {"q": {"type": "noul"}}},
    "empty instructions": {"state": "x", "questions": {"q": {**NOUL, "instructions": ""}}},
}

# Defaults per family, plus a fake model that limits everything the checks cover.
STRICT = {
    "question_types": frozenset({"noul", "choice"}),
    "requires_instructions": True,
    "max_choice_options": 2,
    "max_score_levels": 3,
    "max_images": 4,
}
CHECKED = [(name, {}) for name in dict.fromkeys(name for name, _ in CASES)] + [("fake", STRICT)]


def _outcome(check, body):
    try:
        return "ok", check(body)
    except DecisionError as error:
        return "error", error.message, error.param, error.code


@pytest.mark.parametrize(("name", "options"), CHECKED)
def test_the_checks_on_static_capabilities_are_the_loaded_models(models, name, options):
    path = models[name]
    loaded = DecisionModel(load_backend(path, **options))
    capabilities = static_capabilities(path, **options)
    for request_name, body in REQUESTS.items():
        static = _outcome(partial(check_request, capabilities=capabilities, name=loaded.name), body)
        assert static == _outcome(loaded.check, body), request_name


def test_the_requests_cover_every_check(models):
    """Each check refuses at least one request on one of the models above."""
    params = set()
    for name, options in CHECKED:
        capabilities = static_capabilities(models[name], **options)
        check = partial(check_request, capabilities=capabilities, name=name)
        for body in REQUESTS.values():
            outcome = _outcome(check, body)
            if outcome[0] == "error":
                params.add(outcome[2].rsplit(".", 1)[-1] if outcome[2] else None)
    assert {"type", "instructions", "criteria", "images", "videos", "state"} <= params
