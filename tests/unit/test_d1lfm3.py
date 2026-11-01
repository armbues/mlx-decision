"""d1 LFM3 (d1-2B) on tiny random folders: detection, settings, limits and loading."""

import json
import shutil

import pytest

from mlx_decision.convert import convert
from mlx_decision.models.d1lfm3.model import INPUT_TOKEN_LIMIT, MAX_INPUT_TOKENS
from mlx_decision.registry import (
    check_options,
    detect_family,
    families,
    get_family,
    load_backend,
    static_capabilities,
)
from tiny_models import write_baselabs, write_clef, write_d1, write_d1lfm3, write_h2o, write_pplx


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_d1lfm3(tmp_path_factory.mktemp("d1lfm3") / "tiny-d1lfm3")


@pytest.fixture
def copy(folder, tmp_path):
    """A copy of the tiny folder to change."""
    return shutil.copytree(folder, tmp_path / "copy")


def edit(file, change) -> None:
    data = json.loads(file.read_text())
    change(data)
    file.write_text(json.dumps(data))


def test_detected_by_its_config(folder, tmp_path):
    assert detect_family(folder).name == "d1lfm3"
    # The JSON alone is enough (download checks a repo before fetching weights).
    (tmp_path / "config.json").write_text((folder / "config.json").read_text())
    assert get_family("d1lfm3").detect(tmp_path)


def test_other_families_are_not_detected_as_d1lfm3(tmp_path):
    writers = {
        "d1": write_d1,
        "clef": write_clef,
        "h2o": write_h2o,
        "pplx": write_pplx,
        "baselabs": write_baselabs,
    }
    for name, write in writers.items():
        assert detect_family(write(tmp_path / name)).name == name


def test_an_lfm3_model_without_the_d1_class_is_not_d1lfm3(copy):
    edit(copy / "config.json", lambda config: config.pop("auto_map"))
    assert not get_family("d1lfm3").detect(copy)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.update(d1_attention="causal"), "d1_attention 'causal' is not supported"),
        (lambda c: c.update(d1_codes=1), "d1_codes 1 is not supported"),
        (lambda c: c.update(d1_set_size=0), "d1_set_size 0 is not a whole number from 1"),
        (lambda c: c.pop("d1_set_size"), "d1_set_size None"),
        (lambda c: c.update(d1_vision_max_pixels=-5), "d1_vision_max_pixels -5"),
    ],
)
def test_other_serving_settings_are_refused_naming_them(copy, change, message):
    edit(copy / "config.json", change)
    with pytest.raises(ValueError, match=message):
        load_backend(copy)
    with pytest.raises(ValueError, match=message):
        static_capabilities(copy)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("default"), "no positive default temperature"),
        (lambda c: c.update(default=0), "no positive default temperature"),
        (lambda c: c.update(temperatures={"choice": 2.0}), "own temperature for choice"),
    ],
)
def test_calibration_other_than_the_releases_is_refused(copy, change, message):
    edit(copy / "calibration.json", change)
    with pytest.raises(ValueError, match=message):
        load_backend(copy)


def test_temperatures_for_other_answer_types_are_fine(copy):
    edit(copy / "calibration.json", lambda c: c.update(temperatures={"number": 1.5}))
    assert load_backend(copy).settings.temperature == pytest.approx(1.337398453966105)


@pytest.mark.parametrize("name", ["tokenizer.json", "calibration.json"])
def test_missing_files_are_named(copy, name):
    (copy / name).unlink()
    with pytest.raises((FileNotFoundError, ValueError), match=name):
        load_backend(copy)


def test_the_input_limit(folder):
    assert load_backend(folder).capabilities.max_input_tokens == MAX_INPUT_TOKENS == 32768
    assert (
        load_backend(folder, max_input_tokens=INPUT_TOKEN_LIMIT).capabilities.max_input_tokens
        == 128000
    )
    for wrong in (0, INPUT_TOKEN_LIMIT + 1):
        with pytest.raises(ValueError, match="between 1 and 128000"):
            load_backend(folder, max_input_tokens=wrong)


def test_the_picture_cap(folder):
    capabilities = load_backend(folder).capabilities
    assert capabilities.supports_images and capabilities.max_image_pixels == 1638400
    assert load_backend(folder, max_image_pixels=100000).capabilities.max_image_pixels == 100000
    for wrong in (65535, 1638401):
        with pytest.raises(ValueError, match="between 65536 and 1638400"):
            load_backend(folder, max_image_pixels=wrong)


def test_the_vision_tower_loads_on_first_use(folder):
    backend = load_backend(folder)
    assert backend._vision is None
    assert load_backend(folder, vision=True)._vision is not None


def test_nothing_is_kept_between_requests(folder):
    with pytest.raises(ValueError, match="prefix_cache_gb"):
        check_options(get_family("d1lfm3"), {"prefix_cache_gb": 1})


def test_it_is_not_converted(folder, tmp_path):
    with pytest.raises(ValueError, match="d1lfm3 models cannot be converted"):
        convert(folder, tmp_path / "copy")


def test_the_download_lists_it_with_d1s_licence_and_its_files():
    family = get_family("d1lfm3")
    assert [m.repo_id for m in family.known_models] == ["LiquidAI/d1-2B"]
    assert family.listing.title == "d1 LFM3" and family.listing.maker == "Liquid AI"
    assert family.licence == get_family("d1").licence and "LFM Open License" in family.licence
    assert {
        "config.json",
        "calibration.json",
        "processor_config.json",
        "model*.safetensors",
    } <= set(family.files)
    assert not any(pattern.endswith((".py", ".jinja")) for pattern in family.files)
    names = [f.name for f in families()]
    assert names.index("d1lfm3") == names.index("d1") - 1
