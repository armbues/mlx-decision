"""The package's public names and how its calls take their arguments."""

import dataclasses

import pytest

import mlx_decision
from fake_backend import write_model
from mlx_decision.data.format import DataError
from mlx_decision.errors import PlatformError
from mlx_decision.memory import ModelTooLargeError
from mlx_decision.training.evaluation import EvalResult


def test_the_errors_are_exported():
    assert mlx_decision.DataError is DataError
    assert mlx_decision.PlatformError is PlatformError
    assert mlx_decision.ModelTooLargeError is ModelTooLargeError
    for name in ("DataError", "DecisionError", "ModelTooLargeError", "PlatformError"):
        assert name in mlx_decision.__all__


def test_a_model_too_large_is_a_decision_error_and_still_a_value_error():
    error = ModelTooLargeError("m needs about 20 GB", required=20, available=10)
    assert isinstance(error, mlx_decision.DecisionError) and isinstance(error, ValueError)
    assert str(error) == "m needs about 20 GB" and error.param is None
    assert (error.required, error.available) == (20, 10)


def test_check_memory_and_images_go_by_keyword(tmp_path):
    path = write_model(tmp_path / "fake")
    with pytest.raises(TypeError):
        mlx_decision.load(path, False)
    model = mlx_decision.load(path, check_memory=False)
    questions = {"q": {"type": "noul"}}
    with pytest.raises(TypeError):
        model.decide("state", questions, [])
    assert model.decide("state", questions, images=[]).answers["q"]


def test_an_eval_result_is_built_whole():
    fields = {f.name: f for f in dataclasses.fields(EvalResult)}
    assert fields["models"].default_factory is dataclasses.MISSING  # given, not appended to
