"""d1 LFM3 (d1-2B) on tiny random folders: detection, settings, limits and loading."""

import json
import shutil

import mlx.core as mx
import pytest
from PIL import Image

import mlx_decision
from mlx_decision.convert import convert
from mlx_decision.errors import DecisionError
from mlx_decision.models.d1lfm3 import model as module
from mlx_decision.models.d1lfm3.model import INPUT_TOKEN_LIMIT, MAX_INPUT_TOKENS, question_dict
from mlx_decision.models.d1lfm3.prompt import suffix
from mlx_decision.registry import (
    check_options,
    detect_family,
    families,
    get_family,
    load_backend,
    static_capabilities,
)
from mlx_decision.types import parse_request
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


# The backend on the word-level tiny folder (the parity tests use the real tokenizer).

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments", "technical": None, "sales": "Plans"},
    },
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
}
STATE = "my invoice is wrong please refund the billing"


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def many(count: int) -> dict:
    """``count`` questions, the three kinds in turn."""
    kinds = list(QUESTIONS.values())
    return {f"q{i:02d}": kinds[i % 3] for i in range(count)}


def test_answers_every_question_type(model):
    result = model.decide(STATE, QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert 0 < result.answers["refund"].noul < 1
    assert list(result.answers["team"].probabilities) == ["billing", "technical", "sales"]
    assert sum(result.answers["urgency"].probabilities.values()) == pytest.approx(1)
    assert not result.truncated


def test_input_tokens_count_every_set(model):
    request = parse_request({"state": STATE, "questions": many(17)})
    encoded = model.backend.encode(request)
    assert [len(p.question_ids) for p in encoded.sets] == [16, 1]
    result = model.decide(STATE, many(17))
    assert result.usage.input_tokens == sum(len(p.ids) for p in encoded.sets)


def test_batches_within_the_token_budget(folder, monkeypatch):
    model = mlx_decision.load(folder)
    with mx.stream(mx.cpu):
        together = model.decide(STATE, many(33))
        monkeypatch.setattr(module, "BATCH_TOKENS", 1)  # one set per batch
        apart = model.decide(STATE, many(33))
    for qid, answer in together.answers.items():
        a, b = answer.model_dump(), apart.answers[qid].model_dump()
        assert a.keys() == b.keys()
        for key in ("noul", "probabilities"):
            if key in a:
                got = a[key] if key == "noul" else list(a[key].values())
                want = b[key] if key == "noul" else list(b[key].values())
                assert got == pytest.approx(want, abs=1e-5), qid


@pytest.mark.parametrize(
    ("question", "same_as"),
    [
        ({"type": "noul"}, {"type": "noul", "instructions": "refund"}),
        ({"type": "noul", "instructions": "  "}, {"type": "noul", "instructions": "refund"}),
        (
            {"type": "score", "instructions": "x", "criteria": [{"level": "low"}, ["high"]]},
            {"type": "score", "instructions": "x", "criteria": ["{'level': 'low'}", "['high']"]},
        ),
    ],
)
def test_requests_the_release_refuses_are_adapted(model, question, same_as):
    asked = parse_request({"state": STATE, "questions": {"refund": question}})
    meant = parse_request({"state": STATE, "questions": {"refund": same_as}})
    tokenizer = model.backend.tokenizer
    assert suffix(question_dict("refund", asked.questions["refund"]), tokenizer) == suffix(
        question_dict("refund", meant.questions["refund"]), tokenizer
    )


def test_a_long_state_is_cut_to_fit_keeping_its_start_and_end(folder):
    model = mlx_decision.load(folder, max_input_tokens=200)
    # Words the tiny vocabulary has (unknown ones decode as "[UNK]", three tokens).
    words = " ".join(["billing"] * 150 + ["technical"] * 150)
    assert model.decide(words, QUESTIONS).truncated
    encoded = model.backend.encode(parse_request({"state": words, "questions": QUESTIONS}))
    assert encoded.truncated and all(len(p.ids) <= 200 for p in encoded.sets)
    kept = model.backend.tokenizer.decode(encoded.sets[0].ids[: encoded.sets[0].state_length])
    assert "billing" in kept.split()[:10] and "technical" in kept.split()[-10:]
    assert model.decide("billing technical", QUESTIONS).truncated is False


def test_a_set_too_long_without_its_state_is_refused_naming_its_first_question(folder):
    model = mlx_decision.load(folder, max_input_tokens=60)
    with pytest.raises(DecisionError, match="the limit is 60") as error:
        model.decide(STATE, many(3))
    assert error.value.param == "questions.q00"
    with pytest.raises(DecisionError, match="questions read together"):
        model.decide(None, many(3))


def test_pictures_are_read(model):
    picture = Image.new("RGB", (300, 200), (200, 30, 30))
    other = Image.new("RGB", (300, 200), (30, 30, 200))
    with_red = model.decide(STATE, QUESTIONS, images=[picture])
    with_blue = model.decide(STATE, QUESTIONS, images=[other])
    without = model.decide(STATE, QUESTIONS)
    assert with_red.usage.input_tokens > without.usage.input_tokens
    assert with_red.answers["refund"].noul != with_blue.answers["refund"].noul
    assert model.backend._vision is not None


def test_pictures_are_capped_before_the_processor(folder):
    model = mlx_decision.load(folder, max_image_pixels=65536)
    picture = Image.new("RGB", (1000, 500))
    _, pictures = model.backend.open_pictures(
        parse_request({"state": STATE, "questions": QUESTIONS, "images": [picture]})
    )
    capped = mlx_decision.load(folder).backend.open_pictures(
        parse_request(
            {"state": STATE, "questions": QUESTIONS, "images": [picture.resize((362, 181))]}
        )
    )[1]
    assert pictures.tokens == capped.tokens
