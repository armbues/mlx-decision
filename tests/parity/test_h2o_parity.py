"""H2O-Lightning on MLX against the release's own serving code, on the parity set.

``h2o/reference.json`` is written by ``scripts/make_h2o_reference.py``: the
release's shim with transformers in vLLM's place, so the token ids of every
question (its real tokenizer) and the answers of the tiny random model of
``tiny_models.write_h2o`` (CPU, float32). The token and end-to-end tests
need the release's small files in the Hugging Face cache (the script
fetches them) and skip otherwise; the tiny-model test feeds the stored ids
and always runs. ``h2o/reference-bf16.json`` is the same script with
``--model``: the release's weights in bfloat16 on MPS; the slow test
compares the release folder in ``MLX_DECISION_MODELS`` with it.

The model runs on the CPU here: on the GPU, float32 matmuls use TF32, which
moves the tiny model's probabilities by up to about 1e-3.
"""

import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import pytest
from parity_images import open_image
from test_parity import MARGIN, MEAN_TOLERANCE
from test_parity import TOLERANCE as BF16_TOLERANCE

import mlx_decision
from mlx_decision.backbones.qwen3_5.load import load_text_model
from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.h2o.model import H2OBackend, ServeConfig
from mlx_decision.models.h2o.prompt import PromptSettings
from mlx_decision.state_images import lift_images
from mlx_decision.types import parse_request
from tiny_models import write_h2o

HERE = Path(__file__).parent
CASES = []
for name in ("requests.json", "marker/requests.json", "h2o/requests.json"):
    CASES += json.loads((HERE / name).read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "h2o" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
TEXT_IDS = [case["id"] for case in CASES if not case.get("images")]
IMAGE_IDS = [case["id"] for case in CASES if case.get("images")]
# Largest allowed difference of any probability against the reference (CPU,
# float32).
TOLERANCE = 1e-5
# For the full path with the real tokenizer: one request per prompt shape.
END_TO_END = [
    "ticket",
    "single_noul_no_instructions",
    "noul_only_true_criterion",
    "choice_null_descriptions",
    "score_object_levels",
    "instructions_object",
    "instructions_array",
    "nested_state",
    "array_state_unicode",
    "japanese",
    "lang_hi",
    "special_tokens_in_state",
    "number_state",
    "bool_state",
    "empty_string_state",
    "whitespace_and_newlines",
    "h2o_empty_instructions",
    "h2o_choice_empty_description",
    # several questions: the shared prefix once, each question from a copy
    "ten_nouls",
    "twenty_mixed",
    "log_1k",
    "thread_2k",
    "ledger_3k",
]


@pytest.fixture(scope="module")
def release() -> Path:
    from huggingface_hub import snapshot_download

    meta = FIXTURE["meta"]
    try:
        path = Path(
            snapshot_download(
                meta["repo_id"],
                revision=meta["revision"],
                # What the script fetches: a snapshot without them counts as complete.
                ignore_patterns=["model.safetensors", "adapter/*", "assets/*"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("H2O release files not cached (scripts/make_h2o_reference.py fetches them)")
    if not all((path / name).exists() for name in ("tokenizer.json", "serve_config.json")):
        pytest.skip("H2O release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def encoder(release) -> H2OBackend:
    """A backend without weights: enough to render and tokenize."""
    config = ServeConfig.read(release)
    return H2OBackend("tokens", None, load_tokenizer(release), config, 40960, release)


def request(case_id: str):
    """The case as a request, images in its state lifted out as ``load`` does."""
    case = REQUESTS[case_id]
    body = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [open_image(spec) for spec in case["images"]]
    images = body.get("images", [])
    body["state"], found = lift_images(case["state"], len(images))
    if found:
        body["images"] = images + [image for image, _ in found]
    return parse_request(body)


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def test_label_ids_equal_the_release(encoder):
    assert dict(zip(encoder.labels, encoder.label_ids, strict=True)) == FIXTURE["meta"]["label_ids"]


@pytest.mark.parametrize("case_id", TEXT_IDS)
def test_token_ids_equal_the_release(encoder, case_id):
    expected = REFERENCE[case_id]
    if "refused" in expected:
        # Refused by the wire format before the release would see it.
        with pytest.raises(DecisionError):
            request(case_id)
        return
    encoded, cut = encoder.encode(request(case_id))
    assert cut == any(e.get("state_cut") for e in expected.values())
    for question in encoded:
        reference = expected[question.question_id]
        assert len(question.input_ids) == reference["tokens"], question.question_id
        assert digest(question.input_ids) == reference["ids_sha256"], question.question_id
        assert len(question.option_ids) == reference["options"]


@pytest.mark.parametrize("case_id", IMAGE_IDS)
def test_image_token_ids_equal_the_release(encoder, case_id):
    """Token ids and image grids as the release's processor makes them."""
    from mlx_decision.models.qwen import prepare

    expected = REFERENCE[case_id]
    body = request(case_id)
    if "refused" in expected:
        with pytest.raises(DecisionError, match="at most 4") as error:
            encoder.open_images(body)
        assert error.value.param == "images"
        return
    opened, images = encoder.open_images(body)
    config = encoder.image_config()
    grids = [list(prepare(image, index, config).grid) for index, image in opened]
    encoded, cut = encoder.encode(body, images.tokens)
    assert cut == any(e.get("state_cut") for e in expected.values())
    for question in encoded:
        reference = expected[question.question_id]
        assert grids == reference["image_grids"]
        assert len(question.input_ids) == reference["tokens"], question.question_id
        assert digest(question.input_ids) == reference["ids_sha256"], question.question_id


@pytest.mark.parametrize("case_id", TEXT_IDS)
def test_every_prompt_starts_with_the_shared_prefix(encoder, case_id):
    if "refused" in REFERENCE[case_id]:
        return
    encoded, _, prefix = encoder.encode_with_prefix(request(case_id))
    assert prefix
    assert all(len(prefix) < len(e.input_ids) for e in encoded)


@pytest.mark.parametrize("limit", [1024, 8192])
def test_a_lowered_limit_cuts_states_so_prompts_fit(release, limit):
    backend = H2OBackend("tokens", None, load_tokenizer(release), ServeConfig.read(release), limit)

    def tokens(case_id):
        return max((e["tokens"] for e in REFERENCE[case_id].values() if "tokens" in e), default=0)

    long = [
        case_id
        for case_id in TEXT_IDS
        if "refused" not in REFERENCE[case_id]
        and tokens(case_id) > limit
        # long for its options, which are never cut: checked below
        and case_id != "choice_255_options"
    ]
    assert len(long) >= 5
    for case_id in long:
        encoded, cut = backend.encode(request(case_id))
        assert cut, case_id
        assert limit - 16 < max(len(e.input_ids) for e in encoded) <= limit, case_id
    # The questions alone over the limit: refused, not cut away.
    if limit == 1024:
        with pytest.raises(DecisionError, match="the limit is 1024"):
            backend.encode(request("choice_255_options"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> H2OBackend:
    """The tiny model with the release's labels, temperatures and floor, no tokenizer."""
    meta = FIXTURE["meta"]
    folder = write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o")
    config = ServeConfig(
        labels=list(meta["label_ids"]),
        temperatures=meta["temperature"],
        noul_floor=meta["noul_floor"],
        prompt=PromptSettings.from_config({}),
    )
    backbone = load_text_model(folder, lm_head=False)
    ids = list(meta["label_ids"].values())
    return H2OBackend("tiny", backbone, None, config, 40960, folder, label_ids=ids)


def answer_values(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [answer["noul"]]
    return list(answer["probabilities"].values())


def test_tiny_model_answers_as_the_release(tiny):
    """Every text question with stored ids: one pass each, about 20 ms on the CPU."""
    checked = 0
    for case_id in TEXT_IDS:
        for question_id, expected in REFERENCE[case_id].items():
            if "ids" not in expected:
                continue
            kind = expected["answer"]["type"]
            with mx.stream(mx.cpu):
                values = tiny.probabilities(expected["ids"], expected["options"], kind)
            # A yes/no answer is P(true), the first option; before the floor.
            got = values[:1] if kind == "noul" else values
            want = answer_values(expected["answer_no_floor"])
            assert got == pytest.approx(want, abs=TOLERANCE), f"{case_id}.{question_id}"
            checked += 1
    assert checked > 150


@pytest.fixture(scope="module")
def tiny_with_tokenizer(release, tmp_path_factory):
    folder = write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o", files_from=release)
    return mlx_decision.load(folder)


@pytest.mark.parametrize("case_id", END_TO_END)
def test_tiny_model_end_to_end(tiny_with_tokenizer, case_id):
    """The whole path: prompt, real tokenizer, passes, the floor, answers."""
    with mx.stream(mx.cpu):
        result = tiny_with_tokenizer.decide_request(request(case_id))
    assert not result.truncated
    tokens = 0
    for question_id, answer in result.answers.items():
        expected = REFERENCE[case_id][question_id]
        tokens += expected["tokens"]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(expected["answer"]["noul"], abs=TOLERANCE)
        else:
            assert answer.probabilities == pytest.approx(
                expected["answer"]["probabilities"], abs=TOLERANCE
            )
            if answer.type == "choice":
                assert answer.choice == expected["answer"]["choice"]
    assert result.usage.input_tokens == tokens


IMAGE_END_TO_END = [
    "image_dog_claim",
    "image_tall_painting",
    "image_two",
    "image_three",
    "image_no_state_text",
    "image_enlarged",
    "image_tiny",
    "h2o_state_images",
]


@pytest.mark.parametrize("case_id", IMAGE_END_TO_END)
def test_tiny_model_answers_images_as_the_release(tiny_with_tokenizer, case_id):
    """Images through the whole path (measured within 3.4e-6 of the release)."""
    with mx.stream(mx.cpu):
        result = tiny_with_tokenizer.decide_request(request(case_id))
    tokens = 0
    for question_id, answer in result.answers.items():
        expected = REFERENCE[case_id][question_id]
        tokens += expected["tokens"]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(expected["answer"]["noul"], abs=TOLERANCE)
        else:
            assert answer.probabilities == pytest.approx(
                expected["answer"]["probabilities"], abs=TOLERANCE
            )
    assert result.usage.input_tokens == tokens


BF16 = json.loads((HERE / "h2o" / "reference-bf16.json").read_text())


def test_the_bf16_reference_is_for_the_same_prompts():
    """Every request, with the token ids of the tiny reference (same tokenizer)."""
    assert BF16["meta"]["revision"] == FIXTURE["meta"]["revision"]
    assert BF16["meta"]["dtype"] == "torch.bfloat16"
    assert set(BF16["results"]) == set(REFERENCE)
    for case_id, expected in BF16["results"].items():
        tiny = REFERENCE[case_id]
        assert ("refused" in expected) == ("refused" in tiny), case_id
        if "refused" in expected:
            continue
        for question_id, entry in expected.items():
            assert entry["ids_sha256"] == tiny[question_id]["ids_sha256"], case_id
            assert ("answer_no_floor" in entry) == ("refused" not in entry), case_id


def option_probabilities(answer: dict) -> dict[str, float]:
    """An answer (the shim's or ours, as JSON) as probabilities by option id."""
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


@pytest.mark.slow
@pytest.mark.weights
def test_bf16_answers_as_the_release():
    """About 2 minutes: the release folder answers the whole parity set.

    Measured 2026-10-10: max 0.0195, mean 0.0014; 3 changed top answers,
    all where the reference's top-two margin is within 0.04.
    """
    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / "h2o-lightning-4b" if root else None
    if path is None or not path.is_dir():
        pytest.skip("no h2o-lightning-4b in MLX_DECISION_MODELS")
    # Before the floor, as the reference's answer_no_floor.
    model = mlx_decision.load(path, noul_floor=0)
    differences, worst, decided_flips = [], (0.0, None), []
    for case_id, expected in BF16["results"].items():
        if "refused" in expected:
            continue
        result = model.decide_request(request(case_id))
        assert result.usage.input_tokens == sum(e["tokens"] for e in expected.values()), case_id
        for question_id, entry in expected.items():
            want = option_probabilities(entry["answer_no_floor"])
            got = option_probabilities(result.answers[question_id].model_dump())
            pairs = [(got[option], p) for option, p in want.items()]
            differences += [abs(a - b) for a, b in pairs]
            worst = max(worst, (max(abs(a - b) for a, b in pairs), f"{case_id}.{question_id}"))
            first, second = sorted(want.values(), reverse=True)[:2]
            top = max(want, key=want.get)
            if first - second > MARGIN and max(got, key=got.get) != top:
                decided_flips.append(f"{case_id}.{question_id}")
    mean = sum(differences) / len(differences)
    assert worst[0] <= BF16_TOLERANCE, f"max |dp| {worst[0]:.4f} at {worst[1]}"
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f}"
    assert not decided_flips
