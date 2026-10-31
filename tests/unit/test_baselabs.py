"""Base Labs (elephant-1, mammoth-1) on tiny random models: detection, refusals and loading."""

import json
import shutil

import mlx.core as mx
import pytest

from mlx_decision.models.baselabs.model import (
    INPUT_TOKEN_LIMIT,
    MAX_INPUT_TOKENS,
    head_file,
    read_rows,
)
from mlx_decision.models.baselabs.prompt import CODES
from mlx_decision.registry import detect_family, load_backend, static_capabilities
from tiny_models import (
    BASELABS_CODES,
    write_baselabs,
    write_clef,
    write_d3,
    write_h2o,
    write_pplx,
)


@pytest.fixture(scope="module", params=["elephant", "mammoth"])
def folder(request, tmp_path_factory):
    return write_baselabs(tmp_path_factory.mktemp(request.param) / "tiny", shape=request.param)


@pytest.fixture
def copy(folder, tmp_path):
    """A copy of the tiny folder to change."""
    return shutil.copytree(folder, tmp_path / "copy")


def edit_config(folder, change) -> None:
    file = folder / "config.json"
    config = json.loads(file.read_text())
    change(config)
    file.write_text(json.dumps(config))


def test_the_answer_codes_are_the_releases():
    assert len(CODES) == 256 and len(set(CODES)) == 256
    assert CODES[:26] == tuple(BASELABS_CODES[:26])
    assert CODES[26:29] == ("AA", "AB", "AC") and "BQ" not in CODES


def test_detected_by_its_modeling_overrides(folder):
    assert detect_family(folder).name == "baselabs"


def test_other_families_are_not_detected_as_base_labs(tmp_path):
    folders = {
        "clef": write_clef(tmp_path / "clef"),
        "pplx": write_pplx(tmp_path / "pplx"),
        "d3": write_d3(tmp_path / "d3"),
        "h2o": write_h2o(tmp_path / "h2o"),
    }
    for name, path in folders.items():
        assert detect_family(path).name == name


def test_overrides_on_another_architecture_are_not_base_labs(copy):
    edit_config(copy, lambda c: c.update(model_type="llama"))
    with pytest.raises(ValueError, match="not a supported decision model"):
        detect_family(copy)


def test_loads_the_answer_code_rows_in_float32(folder):
    backend = load_backend(folder)
    assert backend.family == "baselabs"
    assert backend.precision == "float32"
    head = mx.load(str(head_file(folder)))["lm_head.weight"]
    assert backend.rows.dtype == mx.float32
    assert mx.array_equal(backend.rows, head[mx.array(backend.code_ids)])
    width = backend.backbone.language_model.args.hidden_size
    assert backend.rows.shape == (256, width)
    assert not hasattr(backend.backbone.language_model, "lm_head")
    assert not backend.capabilities.supports_images
    assert not backend.capabilities.truncates_input


def test_the_head_is_found_in_a_shard(folder):
    index = folder / "model.safetensors.index.json"
    expected = "model-00001-of-00002.safetensors" if index.exists() else "model.safetensors"
    assert head_file(folder).name == expected


def test_rows_are_read_from_bfloat16(tmp_path):
    values = mx.random.normal((40, 8)).astype(mx.bfloat16)
    file = tmp_path / "model.safetensors"
    mx.save_safetensors(str(file), {"other": mx.zeros(3), "lm_head.weight": values})
    rows = read_rows(file, "lm_head.weight", [0, 39, 7])
    assert rows.dtype == mx.float32
    assert mx.array_equal(rows, values[mx.array([0, 39, 7])].astype(mx.float32))
    with pytest.raises(ValueError, match="has 40 rows"):
        read_rows(file, "lm_head.weight", [40])


def test_the_input_limit(folder):
    assert static_capabilities(folder).max_input_tokens == MAX_INPUT_TOKENS == 16384
    limit = static_capabilities(folder, max_input_tokens=INPUT_TOKEN_LIMIT).max_input_tokens
    assert limit == 262144
    for value in (0, INPUT_TOKEN_LIMIT + 1):
        with pytest.raises(ValueError, match="between 1 and 262144"):
            static_capabilities(folder, max_input_tokens=value)
    with pytest.raises(ValueError, match="prefix_cache_gb"):
        load_backend(folder, prefix_cache_gb=-1)


def overrides(change):
    return lambda config: change(config["modeling_overrides"])


REFUSED_CONFIGS = {
    "attention.full_attention is 'causal'": overrides(
        lambda o: o["attention"].update(full_attention="causal")
    ),
    "attention.linear_attention is 'block_bidirectional'": overrides(
        lambda o: o["attention"].update(linear_attention="block_bidirectional")
    ),
    "attention.sliding_attention is 'causal'; no such layer type": overrides(
        lambda o: o["attention"].update(sliding_attention="causal")
    ),
    "attention.linear_attention is None": overrides(
        lambda o: o["attention"].pop("linear_attention")
    ),
    "attention must be an object": overrides(lambda o: o.update(attention="block_bidirectional")),
    "modeling_overrides.readout is 'head'": overrides(lambda o: o.update(readout="head")),
}


@pytest.mark.parametrize("message", REFUSED_CONFIGS)
def test_other_training_settings_are_refused_naming_them(copy, message):
    edit_config(copy, REFUSED_CONFIGS[message])
    assert detect_family(copy).name == "baselabs"
    for load in (static_capabilities, load_backend):
        with pytest.raises(ValueError, match=message.replace("(", r"\(")):
            load(copy)


@pytest.mark.parametrize(
    "readout, message",
    [
        ({"readout": "head"}, "readout.json's readout is 'head'"),
        ({"attention": "causal"}, "readout.json's attention is 'causal'"),
    ],
)
def test_a_readout_json_that_disagrees_is_refused(copy, readout, message):
    (copy / "readout.json").write_text(json.dumps({"readout": "slot_lm", **readout}))
    with pytest.raises(ValueError, match=message):
        load_backend(copy)


@pytest.mark.parametrize(
    "name, what",
    [("readout_head.pt", "pointer-head readout"), ("calibration.json", "calibration file")],
)
def test_other_readout_files_are_refused(copy, name, what):
    (copy / name).write_bytes(b"{}")
    for load in (static_capabilities, load_backend):
        with pytest.raises(ValueError, match=f"{name} is a .*{what}"):
            load(copy)


def test_missing_files_are_named(copy):
    (copy / "readout.json").unlink()
    with pytest.raises(FileNotFoundError, match="readout.json is missing"):
        load_backend(copy)


def test_a_tokenizer_without_a_code_is_refused(copy):
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[UNK]", *(code for code in BASELABS_CODES if code != "AB")]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(copy / "tokenizer.json"))
    with pytest.raises(ValueError, match="answer code 'AB' is not a single token"):
        load_backend(copy)


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "refund please"},
    "team": {
        "type": "choice",
        "instructions": "billing",
        "criteria": {"billing": None, "technical": "crash"},
    },
    "level": {"type": "score", "instructions": "how", "criteria": ["a", "b", "c", "d"]},
    "many": {"type": "choice", "criteria": {f"o{i}": f"option {i}" for i in range(30)}},
}


def probabilities(backend, questions=QUESTIONS, state="billing refund please now"):
    from mlx_decision.types import parse_request

    request = parse_request({"state": state, "questions": questions})
    return backend.score(request).probabilities


def test_batched_questions_answer_as_one_by_one(folder, monkeypatch):
    import mlx_decision.models.baselabs.model as model

    backend = load_backend(folder, prefix_cache_gb=0)
    with mx.stream(mx.cpu):
        batched = probabilities(backend)
        assert len(backend.batches(10, backend.encode(parse(QUESTIONS))[1])) == 1
        monkeypatch.setattr(model, "BATCH_TOKENS", 1)
        alone = probabilities(backend)
    for question_id, values in batched.items():
        for option, value in values.items():
            assert value == pytest.approx(alone[question_id][option], abs=1e-6)
    assert list(batched["refund"]) == ["true", "false"]
    assert list(batched["level"]) == ["0", "1", "2", "3"]


def parse(questions):
    from mlx_decision.types import parse_request

    return parse_request({"state": "billing refund please now", "questions": questions})


def test_a_kept_prefix_answers_as_a_fresh_one(folder):
    cold = load_backend(folder, prefix_cache_gb=0)
    warm = load_backend(folder, prefix_cache_gb=1)
    with mx.stream(mx.cpu):
        fresh = probabilities(cold)
        first = probabilities(warm)
        assert len(warm.prefix_cache) == 1
        # Other questions on the same state continue from the kept prefix.
        again = probabilities(warm, {"team": QUESTIONS["team"], "x": QUESTIONS["refund"]})
        assert len(warm.prefix_cache) == 1
    for got, want in [
        (first["team"], fresh["team"]),
        (first["refund"], fresh["refund"]),
        (again["team"], fresh["team"]),
        (again["x"], fresh["refund"]),
    ]:
        assert got == pytest.approx(want, abs=1e-6)


def test_batches_stay_under_the_token_budget(folder, monkeypatch):
    import mlx_decision.models.baselabs.model as model
    from mlx_decision.models.baselabs.model import Encoded

    backend = load_backend(folder, prefix_cache_gb=0)
    monkeypatch.setattr(model, "BATCH_TOKENS", 100)
    encoded = [Encoded(str(i), ["a"], [1] * n) for i, n in enumerate([10, 20, 5, 60, 70, 1])]
    # Rows times (prefix 20 + the widest) up to 100: 2 x 40; then a third row
    # would make 3 x 40; 60 and 70 alone; 1 after 70 would make 2 x 90.
    assert backend.batches(20, encoded) == [[0, 1], [2], [3], [4], [5]]
    # A question over the budget still runs, alone.
    assert backend.batches(200, encoded[:2]) == [[0], [1]]


@pytest.fixture(scope="module")
def model(folder):
    from mlx_decision.model import DecisionModel

    return DecisionModel(load_backend(folder))


def test_answers_and_tokens(model):
    result = model.decide("billing refund please now", QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert 0 < result.answers["refund"].noul < 1
    assert result.answers["team"].choice in ("billing", "technical")
    assert 0 <= result.answers["level"].score <= 3
    prefix, encoded = model.backend.encode(parse(QUESTIONS))
    assert result.usage.input_tokens == len(prefix) + sum(len(q.suffix) for q in encoded)


def test_over_the_input_limit_is_refused_never_cut(folder, model):
    from mlx_decision.errors import DecisionError
    from mlx_decision.model import DecisionModel

    questions = {"short": QUESTIONS["refund"], "long": QUESTIONS["many"]}
    prefix, encoded = model.backend.encode(parse(questions))
    short, long = (len(prefix) + len(q.suffix) for q in encoded)
    assert short < long
    exact = DecisionModel(load_backend(folder, max_input_tokens=long))
    exact.decide("billing refund please now", questions)
    small = DecisionModel(load_backend(folder, max_input_tokens=long - 1))
    with pytest.raises(
        DecisionError, match=f"needs {long} tokens; the limit is {long - 1}"
    ) as error:
        small.decide("billing refund please now", questions)
    assert error.value.param == "questions.long"
    assert small.decide("billing refund please now", {"short": QUESTIONS["refund"]}).answers


def test_option_level_and_image_limits(model):
    from mlx_decision.errors import DecisionError

    many = {f"o{i}": None for i in range(256)}
    with pytest.raises(DecisionError, match="at most 255 options"):
        model.decide("x", {"q": {"type": "choice", "criteria": many}})
    model.check(
        {
            "state": "x",
            "questions": {"q": {"type": "choice", "criteria": dict(list(many.items())[:255])}},
        }
    )
    levels = [f"l{i}" for i in range(11)]
    with pytest.raises(DecisionError, match="at most 10 levels"):
        model.decide("x", {"q": {"type": "score", "criteria": levels}})
    with pytest.raises(DecisionError, match="does not support images") as error:
        model.decide_request({"state": "x", "questions": QUESTIONS, "images": ["data:x"]})
    assert error.value.param == "images"


def test_images_inside_the_state_are_read_as_text(model):
    """Text only, as d1: nothing is lifted out of the state, so nothing is refused."""
    picture = "data:image/png;base64," + "iVBORw0KGgo" + "A" * 80
    result = model.decide({"note": "see", "photo": picture}, {"refund": QUESTIONS["refund"]})
    assert 0 < result.answers["refund"].noul < 1


def test_the_release_rows_as_a_fitted_readout_answer_as_the_release(folder):
    from mlx_decision.backend import Readable
    from mlx_decision.models.readout import Readout

    backend = load_backend(folder, prefix_cache_gb=0)
    assert isinstance(backend, Readable)
    request = parse(QUESTIONS)
    features = backend.features(request)
    readouts = {
        qid: Readout(f.option_ids, f.rows, mx.zeros(len(f.option_ids)))
        for qid, f in features.items()
    }
    with mx.stream(mx.cpu):
        plain = backend.score(request).probabilities
        fitted = backend.score(request, readouts=readouts).probabilities
    for qid, values in plain.items():
        assert fitted[qid] == pytest.approx(values, abs=1e-6)
    assert backend.finish(request.questions["refund"], plain["refund"]) == plain["refund"]
