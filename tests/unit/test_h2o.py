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
    assert not info["supports_images"]


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
