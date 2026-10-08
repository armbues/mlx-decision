"""pplx-decider on MLX against the release's own code, on the parity set.

``pplx/reference.json`` is written by ``scripts/make_pplx_reference.py``: the
release's token ids for every question (its real tokenizer) and the answers
of the tiny random model of ``tiny_models.write_pplx`` (CPU, float32). The
token tests need the release's small files in the Hugging Face cache (the
script fetches them) and skip otherwise; the tiny-model test feeds the
stored ids and always runs.

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
from mlx_decision.models.pplx.model import PplxBackend, read_decision_config
from mlx_decision.models.qwen import count_image_tokens, open_images, prepare
from mlx_decision.types import parse_request
from tiny_models import write_pplx

HERE = Path(__file__).parent
CASES = json.loads((HERE / "requests.json").read_text())
CASES += json.loads((HERE / "marker" / "requests.json").read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "pplx" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
CASE_IDS = [case["id"] for case in CASES]
TEXT_IDS = [case["id"] for case in CASES if not case.get("images")]
# Largest allowed difference of any probability against the reference (CPU,
# float32; measured about 1e-6).
TOLERANCE = 1e-5
# For the full path with the real tokenizer: one request per prompt shape.
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
                # What the script fetches: a snapshot without them counts as complete.
                ignore_patterns=["model-*.safetensors", "source/uv.lock"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("pplx release files not cached (scripts/make_pplx_reference.py fetches them)")
    if not all((path / name).exists() for name in ("tokenizer.json", "decision_config.json")):
        pytest.skip("pplx release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def encoder(release) -> PplxBackend:
    """A backend without weights: enough to tokenize and prepare images."""
    codes, temperature, _ = read_decision_config(release)
    tokenizer = load_tokenizer(release)
    return PplxBackend("tokens", None, None, tokenizer, codes, temperature, 8192, release)


def request(case_id: str):
    case = REQUESTS[case_id]
    body = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [open_image(spec) for spec in case["images"]]
    return parse_request(body)


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_token_ids_equal_the_release(encoder, case_id):
    """Token ids as the release's processor makes them, image grids included."""
    expected = REFERENCE[case_id]
    body = request(case_id)
    config = encoder.image_config()
    opened = open_images(body.images or [])
    counts = [count_image_tokens(image, index, config) for index, image in opened]
    grids = [list(prepare(image, index, config).grid) for index, image in opened]
    for reference in expected.values():
        assert grids == reference.get("image_grids", [])
    over = [qid for qid, e in expected.items() if "refused" in e]
    if over:
        # The whole request is refused, naming the first question over the limit.
        with pytest.raises(DecisionError, match="the limit is 8192") as error:
            encoder.encode(body, counts)
        assert error.value.param == f"questions.{over[0]}"
        body.questions = {q: v for q, v in body.questions.items() if q not in over}
        if not body.questions:
            return
    for question in encoder.encode(body, counts):
        reference = expected[question.question_id]
        assert len(question.input_ids) == reference["tokens"], question.question_id
        assert digest(question.input_ids) == reference["ids_sha256"], question.question_id
        assert len(question.option_ids) == reference["options"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return mlx_decision.load(write_pplx(tmp_path_factory.mktemp("pplx") / "tiny-pplx"))


def sample():
    """The first text question of each request up to 1,024 tokens, and the 255-option one.

    About 40 ms a question on the CPU; every question is one more pass.
    """
    for case_id in TEXT_IDS:
        question_id, expected = next(iter(REFERENCE[case_id].items()))
        if "ids" in expected and (expected["tokens"] <= 1024 or expected["options"] == 255):
            yield case_id, question_id, expected


def test_tiny_model_answers_as_the_release(tiny):
    checked = 0
    worst = 0.0
    for case_id, question_id, expected in sample():
        with mx.stream(mx.cpu):
            values = tiny.backend.probabilities(expected["ids"], expected["options"])
        assert len(values) == expected["options"]
        error = max(abs(a - b) for a, b in zip(values, expected["probabilities"], strict=True))
        assert error < TOLERANCE, f"{case_id}.{question_id}: {error:.2e}"
        worst = max(worst, error)
        checked += 1
    assert checked > 60


@pytest.fixture(scope="module")
def tiny_with_tokenizer(release, tmp_path_factory):
    folder = write_pplx(tmp_path_factory.mktemp("pplx") / "tiny-pplx", files_from=release)
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
