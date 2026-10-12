"""H2O-Lightning on a tiny random model: loading, prompt, readout, limits and the serve config."""

import json

import pytest

import mlx_decision
from mlx_decision.backbones.qwen3_5.load import has_vision_weights, load_text_model
from mlx_decision.backbones.qwen3_5.preprocess import ImageConfig
from mlx_decision.backbones.qwen3_5.vision import load_vision_model
from mlx_decision.errors import DecisionError
from mlx_decision.models.h2o.model import commit_noul
from mlx_decision.models.h2o.prompt import PromptSettings, cut_state, render, render_state
from mlx_decision.registry import detect_family
from mlx_decision.types import parse_request
from tiny_models import H2O_SERVE_CONFIG, H2O_TEXT, PPLX_VISION, write_h2o

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "billing or technical",
        "criteria": {"billing": "refund", "technical": None},
    },
    "urgency": {"type": "score", "instructions": "now", "criteria": ["low", "mid", "high"]},
    "refund": {"type": "noul", "instructions": "refund"},
}
SETTINGS = PromptSettings.from_config(H2O_SERVE_CONFIG["prompt"])


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o")


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def prompt_of(state, question, settings=SETTINGS) -> str:
    request = parse_request({"state": state, "questions": {"q": question}})
    state_text = render_state(request.state, settings.state_compact)
    return render(state_text, request.questions["q"], ["A", "B", "C"], settings)[0]


def user_of(text: str) -> str:
    return text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


def test_the_single_file_release_loads_with_its_vision_tower(folder):
    assert not (folder / "model.safetensors.index.json").exists()
    backbone = load_text_model(folder, lm_head=False)
    assert backbone.language_model.args.tie_word_embeddings
    assert backbone.language_model.args.vocab_size == H2O_TEXT["vocab_size"]
    assert has_vision_weights(folder)
    vision = load_vision_model(folder)
    assert vision.patch_embed.proj.weight.shape[0] == PPLX_VISION["hidden_size"]


def test_the_flat_processor_config_is_read(folder, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    size = {"shortest_edge": 65536, "longest_edge": 1048576}
    flat = {"patch_size": 16, "merge_size": 2, "size": size}
    (copy / "preprocessor_config.json").write_text(json.dumps(flat))
    assert ImageConfig.from_folder(copy).max_pixels == 1048576


def test_detected_by_its_serve_config(folder, tmp_path):
    assert detect_family(folder).name == "h2o"
    # The Nemotron sibling's serve config has no labels: another contract.
    (tmp_path / "serve_config.json").write_text(json.dumps({"prompt": {"contract": "cygnet"}}))
    with pytest.raises(ValueError, match="not a supported decision model"):
        detect_family(tmp_path)


def test_info(model):
    info = model.info()
    assert info["family"] == "h2o"
    assert info["max_input_tokens"] == 40960
    assert info["truncates_input"]
    assert info["max_choice_options"] == info["max_score_levels"] == 255
    assert info["precision"] == "float32"
    assert info["supports_images"]


def test_answers_every_question(model):
    result = model.decide("please refund", QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert set(result.answers["team"].probabilities) == {"billing", "technical"}
    assert sum(result.answers["urgency"].probabilities.values()) == pytest.approx(1, abs=1e-5)
    noul = result.answers["refund"].noul
    assert noul >= 0.801 or noul <= 1 - 0.801  # the release's floor
    assert result.usage.input_tokens > 0
    assert not result.truncated


def test_prompt_layout():
    text = prompt_of({"ticket": "Grüße", "n": 3}, QUESTIONS["team"])
    assert text == (
        "<|im_start|>system\nYou are a decision engine.<|im_end|>\n"
        '<|im_start|>user\nrecord: {"ticket":"Grüße","n":3}\n'
        "question: billing or technical\noptions:\nA) billing: refund\nB) technical: technical"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
    )


def test_prompt_defaults():
    noul = user_of(prompt_of("s", {"type": "noul", "criteria": {"true": ""}}))
    assert noul == (
        "record: s\nquestion: Answer the question below.\noptions:\n"
        "A) true: the statement holds\nB) false: it does not"
    )
    blank = user_of(prompt_of("s", {"type": "noul", "instructions": "  "}))
    assert "question: Answer the question below.\n" in blank
    # An explicit null is JSON, as the release reads the request
    null = user_of(prompt_of("s", {"type": "noul", "instructions": None}))
    assert "question: null\n" in null
    score = user_of(
        prompt_of("s", {"type": "score", "instructions": {"q": "ü"}, "criteria": ["a", {"b": "ü"}]})
    )
    assert 'question: {\n "q": "\\u00fc"\n}\noptions:\nA) 0: a\nB) 1: {"b": "ü"}' in score
    choice = user_of(prompt_of("s", {"type": "choice", "criteria": {"x": "", "y": None}}))
    assert choice.endswith("A) x: \nB) y: y")


def test_the_state_cut_keeps_head_and_tail():
    def encode(text):
        return text.split()

    def decode(words):
        return " ".join(words)

    text = " ".join(f"w{i}" for i in range(10))
    assert cut_state(text, 10, encode, decode) == (text, False)
    assert cut_state(text, 4, encode, decode) == ("w0 w1\n...\nw8 w9", True)


def test_commit_noul():
    assert commit_noul(0.9, 0.801) == 0.9
    assert commit_noul(0.6, 0.801) == 0.801
    assert commit_noul(0.5, 0.801) == 0.801  # a tie goes to true
    assert commit_noul(0.3, 0.801) == pytest.approx(0.199)
    assert commit_noul(0.6, 0) == 0.6


def test_255_options_answered_256_refused(model):
    criteria = {f"o{i}": None for i in range(255)}
    result = model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert len(result.answers["many"].probabilities) == 255
    criteria["o255"] = None
    with pytest.raises(DecisionError, match="at most 255") as error:
        model.decide("please", {"many": {"type": "choice", "criteria": criteria}})
    assert error.value.param == "questions.many.criteria"


def test_input_over_the_limit_is_refused_before_any_pass(model, folder, monkeypatch):
    short = {"short": QUESTIONS["refund"]}
    (encoded,), _ = model.backend.encode(parse_request({"state": "please", "questions": short}))
    limit = len(encoded.input_ids) + 5
    model = mlx_decision.load(folder, max_input_tokens=limit)
    monkeypatch.setattr(model.backend, "probabilities", lambda *_: pytest.fail("ran a pass"))
    questions = {**short, "long": {**QUESTIONS["refund"], "instructions": "refund " * 60}}
    with pytest.raises(DecisionError, match=f"the limit is {limit}") as error:
        model.decide("please", questions)
    assert error.value.param == "questions.long"
    monkeypatch.undo()
    assert model.decide("please", short).answers


@pytest.mark.parametrize("limit", [0, 40961])
def test_the_limit_can_only_be_lowered(folder, limit):
    with pytest.raises(ValueError, match="between 1 and 40960"):
        mlx_decision.load(folder, max_input_tokens=limit)


def edit_config(folder, tmp_path, **changes):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    config = json.loads((copy / "serve_config.json").read_text())
    (copy / "serve_config.json").write_text(json.dumps({**config, **changes}))
    return copy


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"temperature": {"choice": 1.0}}, "a temperature per type"),
        ({"temperature": 0}, "positive"),
        ({"noul_floor": 0.4}, "noul_floor"),
        ({"labels": ["A", "A"]}, "distinct"),
        ({"labels": ["A", "not one token"]}, "not a single token"),
        ({"prompt": {**H2O_SERVE_CONFIG["prompt"], "layout": "markdown"}}, "layout"),
        ({"prompt": {**H2O_SERVE_CONFIG["prompt"], "enable_thinking": True}}, "thinking off"),
    ],
)
def test_serve_config_is_checked(folder, tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        mlx_decision.load(edit_config(folder, tmp_path, **changes))


def test_one_temperature_for_every_type(folder, tmp_path):
    model = mlx_decision.load(edit_config(folder, tmp_path, temperature=0.5, noul_floor=0))
    assert model.backend.config.temperatures == {"choice": 0.5, "noul": 0.5, "score": 0.5}
    assert model.backend.noul_floor == 0


def raw_noul(model, state="please refund") -> float:
    """P(true) before the floor, from the backend's probabilities."""
    request = parse_request({"state": state, "questions": {"refund": QUESTIONS["refund"]}})
    (encoded,), _ = model.backend.encode(request)
    return model.backend.probabilities(encoded.input_ids, 2, "noul")[0]


def test_the_floor_can_be_changed_or_turned_off(folder, model):
    raw = raw_noul(model)
    assert max(raw, 1 - raw) < 0.801  # the tiny model is unsure: the floor applies
    question = {"refund": QUESTIONS["refund"]}
    floored = model.decide("please refund", question).answers["refund"].noul
    assert floored == pytest.approx(0.801 if raw >= 0.5 else 0.199)
    off = mlx_decision.load(folder, noul_floor=0)
    assert off.decide("please refund", question).answers["refund"].noul == pytest.approx(raw)
    high = mlx_decision.load(folder, noul_floor=0.95)
    assert high.decide("please refund", question).answers["refund"].noul == pytest.approx(
        0.95 if raw >= 0.5 else 0.05
    )


@pytest.mark.parametrize("floor", [0.3, 0.5, 1.0])
def test_the_floor_is_checked(folder, floor):
    with pytest.raises(ValueError, match="noul_floor must be between 0.5 and 1, or 0"):
        mlx_decision.load(folder, noul_floor=floor)


def test_other_families_refuse_the_floor(tmp_path):
    from tiny_models import write_marker

    laya = write_marker(tmp_path / "laya", "laya")
    with pytest.raises(ValueError, match="laya models do not take noul_floor"):
        mlx_decision.load(laya, noul_floor=0)


def test_the_floor_flag(folder, tmp_path):
    from typer.testing import CliRunner

    from mlx_decision.cli import app
    from tiny_models import write_marker

    def run(model, *flags):
        args = ["run", "-m", str(model), "-s", "please refund", "--noul", "refund", "--json"]
        return CliRunner().invoke(app, [*args, *flags])

    def noul(result):
        return json.loads(result.stdout)["answers"]["noul_1"]["noul"]

    floored, off = run(folder), run(folder, "--noul-floor", "0")
    assert floored.exit_code == off.exit_code == 0
    assert noul(floored) in (0.801, 0.199)
    assert noul(off) not in (0.801, 0.199)
    refused = run(write_marker(tmp_path / "laya", "laya"), "--noul-floor", "0")
    assert refused.exit_code == 1
    assert "laya models do not take noul_floor" in refused.stderr


def test_in_a_pool_only_h2o_models_take_the_floor(folder, tmp_path):
    from mlx_decision.pool import discover
    from tiny_models import write_marker

    laya = write_marker(tmp_path / "laya", "laya")
    specs = discover([folder, laya], {"noul_floor": 0})
    assert {spec.name: spec.options for spec in specs} == {
        "tiny-h2o": {"noul_floor": 0},
        "laya": {},
    }


def test_a_lowered_limit_cuts_long_states_to_fit(folder):
    model = mlx_decision.load(folder, max_input_tokens=200)
    state = " ".join(["please"] * 300 + ["refund"] * 300)
    result = model.decide(state, QUESTIONS)
    assert result.truncated
    request = parse_request({"state": state, "questions": QUESTIONS})
    encoded, cut = model.backend.encode(request)
    assert cut
    assert 150 < max(len(e.input_ids) for e in encoded) <= 200
    # The cut keeps both ends.
    state_text, _ = model.backend.state_text(request)
    assert state_text.startswith("please") and state_text.endswith("refund")
    assert "\n...\n" in state_text
    # At the release's limit the same state is short enough to keep whole.
    assert not mlx_decision.load(folder).decide(state, QUESTIONS).truncated


def test_questions_share_their_prefix_within_and_across_requests(folder):
    import mlx.core as mx

    model = mlx_decision.load(folder)
    cache = model.backend.prefix_cache
    request = parse_request({"state": "please refund now", "questions": QUESTIONS})
    encoded, _, prefix = model.backend.encode_with_prefix(request)
    assert prefix and all(e.input_ids[: len(prefix)] == prefix for e in encoded)
    with mx.stream(mx.cpu):
        first = model.decide_request(request)
        again = model.decide_request(
            {"state": "please refund now", "questions": {"q": QUESTIONS["team"]}}
        )
        whole = {
            e.question_id: model.backend.probabilities(
                e.input_ids, len(e.option_ids), e.question_type
            )
            for e in encoded
        }
    assert (cache.misses, cache.hits, len(cache)) == (1, 1, 1)
    assert again.answers["q"].probabilities == pytest.approx(
        first.answers["team"].probabilities, abs=1e-6
    )
    assert list(first.answers["team"].probabilities.values()) == pytest.approx(
        whole["team"], abs=1e-6
    )
    assert list(first.answers["urgency"].probabilities.values()) == pytest.approx(
        whole["urgency"], abs=1e-6
    )


def test_without_kept_prefixes_a_request_still_shares_its_prefix(folder, monkeypatch):
    model = mlx_decision.load(folder, prefix_cache_gb=0)
    assert model.backend.prefix_cache is None
    calls = []
    original = model.backend.prefix_state
    monkeypatch.setattr(
        model.backend, "prefix_state", lambda *args: calls.append(args) or original(*args)
    )
    model.decide("please refund", QUESTIONS)
    assert len(calls) == 1
    model.decide("please refund", {"refund": QUESTIONS["refund"]})  # one question: one pass
    assert len(calls) == 1


def test_the_prefix_cache_size_is_checked(folder):
    with pytest.raises(ValueError, match="prefix_cache_gb must be 0"):
        mlx_decision.load(folder, prefix_cache_gb=-1)


def png(width: int, height: int, color=(200, 30, 30)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_the_vision_tower_loads_on_request(folder):
    model = mlx_decision.load(folder)
    model.decide("please", {"refund": QUESTIONS["refund"]})
    assert model.backend.vision is None
    model.decide("please", {"refund": QUESTIONS["refund"]}, images=[png(64, 64)])
    assert model.backend.vision is not None
    assert mlx_decision.load(folder, vision=True).backend.vision is not None


def test_images_come_first_and_take_their_tokens(model):
    question = {"refund": QUESTIONS["refund"]}
    plain = model.decide("please", question).usage.input_tokens
    one = model.decide("please", question, images=[png(256, 256)]).usage.input_tokens
    # 256 x 256 pixels in patches of 16 merged 2 x 2: 64 tokens, plus the two markers
    assert one - plain == 64 + 2
    request = parse_request({"state": "please", "questions": question})
    text, _ = render("please", request.questions["refund"], ["A", "B"], SETTINGS, images=2)
    slot = "<|vision_start|><|image_pad|><|vision_end|>"
    assert user_of(text).startswith(slot * 2 + "record: please\n")


def test_large_images_are_scaled_to_the_serve_configs_pixels(model):
    request = parse_request(
        {"state": "please", "questions": QUESTIONS, "images": [png(4000, 3000)]}
    )
    _, images = model.backend.open_images(request)
    assert images.tokens[0] * 32 * 32 <= 1048576


def test_at_most_four_images(model):
    with pytest.raises(DecisionError, match="5 images; this model reads at most 4") as error:
        model.decide("please", QUESTIONS, images=[png(32, 32)] * 5)
    assert error.value.param == "images"


def test_image_markers_in_text_are_refused_with_images(model):
    question = {"refund": {"type": "noul", "instructions": "refund"}}
    with pytest.raises(DecisionError, match="image markers") as error:
        model.decide("look at <|image_pad|> here", question, images=[png(64, 64)])
    assert error.value.param == "state"
    assert model.decide("look at <|image_pad|> here", question).answers


def test_requests_with_images_use_the_image_temperature(folder, tmp_path):
    question = {"team": QUESTIONS["team"]}
    image = png(64, 64)
    model = mlx_decision.load(folder)
    hot = mlx_decision.load(
        edit_config(
            folder, tmp_path, image={"max_images": 4, "max_pixels": 1048576, "temperature": 5}
        )
    )
    assert hot.decide("please", question).answers == model.decide("please", question).answers
    a = model.decide("please", question, images=[image]).answers["team"].probabilities
    b = hot.decide("please", question, images=[image]).answers["team"].probabilities
    assert max(b.values()) < max(a.values())  # flatter at 5 than at 0.65


def test_without_an_image_block_images_are_refused(folder, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(folder, copy)
    config = json.loads((copy / "serve_config.json").read_text())
    del config["image"]
    (copy / "serve_config.json").write_text(json.dumps(config))
    model = mlx_decision.load(copy)
    assert not model.info()["supports_images"]
    with pytest.raises(DecisionError, match="does not support images"):
        model.decide("please", QUESTIONS, images=[png(32, 32)])


def test_images_share_the_prefix_and_are_kept_with_it(folder):
    import mlx.core as mx

    model = mlx_decision.load(folder)
    image = png(64, 96)
    body = {"state": "please refund", "questions": QUESTIONS, "images": [image]}
    request = parse_request(body)
    _, images = model.backend.open_images(request)
    encoded, _, prefix = model.backend.encode_with_prefix(request, images.tokens)
    assert prefix and all(e.input_ids[: len(prefix)] == prefix for e in encoded)
    alone = mlx_decision.load(folder, prefix_cache_gb=0)
    with mx.stream(mx.cpu):
        first = model.decide_request(body)
        again = model.decide_request({**body, "questions": {"q": QUESTIONS["urgency"]}})
        other = model.decide_request({**body, "images": [png(64, 96, (0, 0, 255))]})
        whole = alone.decide_request({**body, "questions": {"q": QUESTIONS["urgency"]}})
    cache = model.backend.prefix_cache
    assert (cache.misses, cache.hits) == (2, 1)  # another image is another prefix
    assert again.answers["q"] == first.answers["urgency"]
    assert whole.answers["q"].probabilities == pytest.approx(
        first.answers["urgency"].probabilities, abs=1e-5
    )
    assert other.answers != first.answers


def test_a_turned_image_is_another_prefix(folder):
    import mlx.core as mx
    from test_prefix_cache import turned_images

    model = mlx_decision.load(folder)
    portrait, landscape = png(64, 96), png(96, 64)
    turned_images(model.backend.image_config(), portrait, landscape)
    alone = mlx_decision.load(folder, prefix_cache_gb=0)
    body = {"state": "please refund", "questions": QUESTIONS}
    with mx.stream(mx.cpu):
        model.decide_request({**body, "images": [portrait]})
        warm = model.decide_request({**body, "images": [landscape]})
        cold = alone.decide_request({**body, "images": [landscape]})
    assert model.backend.prefix_cache.hits == 0
    for question_id, answer in warm.answers.items():
        expected = cold.answers[question_id]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(expected.noul, abs=1e-5)
        else:
            assert answer.probabilities == pytest.approx(expected.probabilities, abs=1e-5)


def test_a_lowered_limit_leaves_room_for_the_images(folder):
    model = mlx_decision.load(folder, max_input_tokens=300)
    state = " ".join(["please"] * 400)
    result = model.decide(state, {"refund": QUESTIONS["refund"]}, images=[png(256, 256)])
    assert result.truncated
    assert result.usage.input_tokens <= 300


@pytest.mark.parametrize("bits", [8, 4])
def test_a_quantized_copy_answers_like_the_release(folder, tmp_path, bits):
    from mlx_decision.convert import convert
    from mlx_decision.registry import MARKER_FILE

    out = convert(folder, tmp_path / f"q{bits}", bits=bits)
    marker = json.loads((out / MARKER_FILE).read_text())
    assert marker["family"] == "h2o" and marker["vision"]
    assert marker["quantization"]["bits"] == bits
    for name in ("serve_config.json", "tokenizer.json", "LICENSE"):
        assert (out / name).read_bytes() == (folder / name).read_bytes(), name
    quantized = mlx_decision.load(out)
    info = quantized.info()
    assert info["family"] == "h2o" and info["precision"] == f"{bits}-bit"
    assert info["supports_images"]
    # The tied embeddings are quantized; the label rows come out dequantized.
    embed = quantized.backend.backbone.language_model.model.embed_tokens
    assert embed.bits == bits
    original = mlx_decision.load(folder)
    a = original.decide("please refund", QUESTIONS, images=[png(64, 64)]).answers
    b = quantized.decide("please refund", QUESTIONS, images=[png(64, 64)]).answers
    error = max(abs(a["team"].probabilities[k] - p) for k, p in b["team"].probabilities.items())
    assert error < (0.02 if bits == 8 else 0.15)


def test_mixed_precision_sweeps_without_the_floor(folder, tmp_path, monkeypatch):
    from mlx_decision.convert import convert
    from mlx_decision.models import qwen_convert
    from mlx_decision.models.h2o import model as h2o_model
    from mlx_decision.models.qwen_convert import SENSITIVITY_FILE

    monkeypatch.setattr(
        qwen_convert,
        "calibration_requests",
        lambda: [{"state": "please refund", "questions": QUESTIONS}],
    )
    loaded = []
    original = h2o_model.load
    monkeypatch.setattr(h2o_model, "load", lambda *a, **k: loaded.append(k) or original(*a, **k))
    out = convert(folder, tmp_path / "mixed", target_bits=4.5, group_size=32)
    assert loaded == [{"prefix_cache_gb": 0, "noul_floor": 0}]
    report = json.loads((out / SENSITIVITY_FILE).read_text())
    assert report["average_bits"] <= 4.5
    model = mlx_decision.load(out)
    assert model.info()["precision"] == "mixed 4.5-bit"
    assert model.backend.noul_floor == 0.801  # the release's, in the copy
    assert model.decide("please refund", QUESTIONS).answers
