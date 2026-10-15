"""d3 on MLX against the release's own runtime, on the parity set.

``d3/reference.json`` is written by ``scripts/make_d3_reference.py``: the
release's token ids for every question (its real tokenizer and processor)
and the answers of the tiny random model of ``tiny_models.write_d3`` (CPU,
float32). The token tests need the release's small files in the Hugging
Face cache (the script fetches them) and skip otherwise; the tiny-model
test feeds the stored ids and always runs.

The model runs on the CPU here: on the GPU, float32 matmuls use TF32, which
moves the tiny model's probabilities by up to about 1e-3.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest
from parity_images import open_image

import mlx_decision
from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.d3.model import PRESET
from mlx_decision.models.letter import LetterBackend, read_decision_config
from mlx_decision.models.qwen import count_image_tokens, open_images, prepare
from mlx_decision.types import parse_request
from tiny_models import write_d3

HERE = Path(__file__).parent
CASES = []
for name in ("requests.json", "marker/requests.json", "d3/requests.json"):
    CASES += json.loads((HERE / name).read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "d3" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
CASE_IDS = [case["id"] for case in CASES]
TEXT_IDS = [case["id"] for case in CASES if not case.get("images")]
# Largest allowed difference of any probability against the reference (CPU,
# float32; measured about 7e-7 from the stored ids, 5e-6 through the whole path).
TOLERANCE = 1e-5
# For the full path with the real tokenizer: one request per prompt shape,
# d3's own edge cases and the image layouts.
END_TO_END = [
    "ticket",
    "noul_only_true_criterion",
    "choice_null_descriptions",
    "score_object_levels",
    "instructions_array",
    "nested_state",
    "japanese",
    "special_tokens_in_state",
    "number_state",
    "empty_string_state",
    "d3_null_state",
    "d3_empty_object_state",
    "d3_falsy_instructions",
    "d3_noul_empty_false",
    "d3_choice_empty_description",
    "d3_score_mixed_levels",
    "d3_image_null_state",
    "image_dog_claim",
    "image_tall_painting",
    "image_three",
    "image_no_state_text",
    "image_enlarged",
    "image_tiny",
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
                allow_patterns=["*.json", "*.py", "*.jinja", "LICENSE", "NOTICE"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("d3 release files not cached (scripts/make_d3_reference.py fetches them)")
    if not all((path / name).exists() for name in ("tokenizer.json", "decision_config.json")):
        pytest.skip("d3 release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def encoder(release) -> LetterBackend:
    """A backend without weights: enough to tokenize and prepare images."""
    codes, temperature, _ = read_decision_config(release)
    tokenizer = load_tokenizer(release)
    return LetterBackend("tokens", None, None, tokenizer, codes, temperature, PRESET, path=release)


def request(case_id: str):
    case = REQUESTS[case_id]
    body = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [open_image(spec) for spec in case["images"]]
    return parse_request(body)


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def test_codes_are_the_releases(release):
    codes, temperature, causal = read_decision_config(release)
    assert len(codes) == 255 and codes[:3] == ["A", "B", "C"] and codes[26] == "AA"
    assert (temperature, causal) == (1.0, False)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_token_ids_equal_the_release(encoder, case_id):
    """Token ids as the release's runtime makes them, image grids included."""
    expected = REFERENCE[case_id]
    body = request(case_id)
    config = encoder.image_config()
    opened = open_images(body.images or [])
    counts = [count_image_tokens(image, index, config) for index, image in opened]
    grids = [list(prepare(image, index, config).grid) for index, image in opened]
    for question_id, reference in expected.items():
        if "refused" in reference:
            # The release refuses the question; so does mlx-decision, the level
            # count through the model's limits, the rest while rendering.
            question = body.questions.pop(question_id)
            if "2 to 10 levels" in reference["refused"]:
                assert len(question.criteria) > encoder.capabilities.max_score_levels
            else:
                with pytest.raises(DecisionError, match=reference["refused"]) as error:
                    encoder.encode(
                        parse_request({"state": "", "questions": {question_id: question}})
                    )
                assert error.value.param == f"questions.{question_id}.criteria"
            continue
        assert grids == reference.get("image_grids", [])
    for question in encoder.encode(body, counts):
        reference = expected[question.question_id]
        assert len(question.input_ids) == reference["tokens"], question.question_id
        assert digest(question.input_ids) == reference["ids_sha256"], question.question_id
        assert len(question.option_ids) == reference["options"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return mlx_decision.load(write_d3(tmp_path_factory.mktemp("d3") / "tiny-d3"))


def sample():
    """The first text question of each request up to 1,024 tokens, and the 255-option one.

    About 40 ms a question on the CPU; every question is one more pass.
    """
    for case_id in TEXT_IDS:
        if "refused" in REFERENCE[case_id]:
            continue
        question_id, expected = next(iter(REFERENCE[case_id].items()))
        if "ids" in expected and (expected["tokens"] <= 1024 or expected["options"] == 255):
            yield case_id, question_id, expected


def test_tiny_model_answers_as_the_release(tiny):
    assert tiny.info()["family"] == "d3"
    checked = 0
    for case_id, question_id, expected in sample():
        with mx.stream(mx.cpu):
            values = tiny.backend.probabilities(expected["ids"], expected["options"])
        assert len(values) == expected["options"]
        error = max(abs(a - b) for a, b in zip(values, expected["probabilities"], strict=True))
        assert error < TOLERANCE, f"{case_id}.{question_id}: {error:.2e}"
        checked += 1
    assert checked > 60


@pytest.fixture(scope="module")
def tiny_with_tokenizer(release, tmp_path_factory):
    folder = write_d3(tmp_path_factory.mktemp("d3") / "tiny-d3", files_from=release)
    return mlx_decision.load(folder)


@pytest.mark.parametrize("case_id", END_TO_END)
def test_tiny_model_end_to_end(tiny_with_tokenizer, case_id):
    """The whole path: prompt, real tokenizer, images, passes, answers."""
    with mx.stream(mx.cpu):
        result = tiny_with_tokenizer.decide_request(request(case_id))
    assert not result.truncated
    tokens = 0
    for question_id, answer in result.answers.items():
        expected = REFERENCE[case_id][question_id]
        tokens += expected["tokens"]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(expected["probabilities"][1], abs=TOLERANCE)
        else:
            values = list(answer.probabilities.values())
            assert values == pytest.approx(expected["probabilities"], abs=TOLERANCE)
    assert result.usage.input_tokens == tokens
