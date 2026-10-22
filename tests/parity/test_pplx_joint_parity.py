"""pplx-decider v1.2 (joint requests) on MLX against the release's own code.

``pplx/reference-joint.json`` is written by
``scripts/make_pplx_joint_reference.py``: for every request of the parity
set plus ``pplx/v1.2-requests.json``, the release's chunks (question ids,
token ids, marker positions, image grids) at the input limit, and the
answers of two tiny random models in the shapes of the two releases
(``tiny_models.write_pplx(joint=...)``, CPU, float32). The token tests
need the release's small files in the Hugging Face cache (the script
fetches them) and skip otherwise; the tiny-model test feeds the stored ids
and always runs.

The model runs on the CPU here: on the GPU, float32 matmuls use TF32, which
moves the tiny model's probabilities by up to about 1e-3.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from parity_images import open_image

import mlx_decision
from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.letter import read_decision_config
from mlx_decision.models.pplx.backend import JointBackend
from mlx_decision.models.pplx.joint import escaped_tokens
from mlx_decision.models.pplx.model import JOINT_FORMATS, joint_preset
from mlx_decision.models.qwen import count_image_tokens, open_images, prepare
from mlx_decision.models.readout import read_out
from mlx_decision.types import parse_request
from tiny_models import write_pplx

HERE = Path(__file__).parent
CASES = json.loads((HERE / "requests.json").read_text())
CASES += json.loads((HERE / "marker" / "requests.json").read_text())
CASES += json.loads((HERE / "pplx" / "v1.2-requests.json").read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "pplx" / "reference-joint.json").read_text())
REFERENCE = FIXTURE["results"]
CASE_IDS = [case["id"] for case in CASES]
SHAPES = ["27b", "flash"]
# Largest allowed difference of any probability against the reference (CPU,
# float32).
TOLERANCE = 1e-5
# For the full path with the real tokenizer: one request per prompt shape.
END_TO_END = [
    "ticket",
    "special_tokens_in_state",
    "image_three",
    "image_no_state_text",
    "card_example",
    "escaped_tokens_in_state",
    "marker_in_instructions_and_options",
    "image_markers_in_text_with_image",
    "json_everything",
    "noul_criteria_variants",
    "split_five_questions",
    "split_with_image",
]


def cached_release(shape: str) -> Path:
    from huggingface_hub import snapshot_download

    meta = FIXTURE["meta"]["releases"][shape]
    try:
        path = Path(
            snapshot_download(
                meta["repo_id"],
                revision=meta["revision"],
                # What the script fetches: a snapshot without them counts as complete.
                ignore_patterns=["model-*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip(
            "pplx v1.2 release files not cached (make_pplx_joint_reference.py fetches them)"
        )
    if not all((path / name).exists() for name in ("tokenizer.json", "joint_config.json")):
        pytest.skip("pplx v1.2 release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def release() -> Path:
    """Flash's small files (the 27B's tokenizer and prompt are the same)."""
    return cached_release("flash")


@pytest.fixture(scope="module")
def encoders(release):
    """Backends without weights, by input limit: enough to tokenize and prepare images."""
    codes, temperature, _ = read_decision_config(release, JOINT_FORMATS)
    tokenizer = load_tokenizer(release)
    escapes = escaped_tokens(release)
    made = {}

    def encoder(limit: int) -> JointBackend:
        if limit not in made:
            made[limit] = JointBackend(
                "tokens",
                None,
                None,
                tokenizer,
                codes,
                temperature,
                joint_preset(release),
                limit,
                release,
                escapes=escapes,
            )
        return made[limit]

    return encoder


def request(case_id: str):
    case = REQUESTS[case_id]
    body = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        body["images"] = [open_image(spec) for spec in case["images"]]
    return parse_request(body)


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def test_escaped_tokens_are_the_releases(release):
    assert sorted(escaped_tokens(release)) == FIXTURE["meta"]["escapes"]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_chunks_equal_the_release(encoders, case_id):
    """Token ids, markers and chunks as the release's encoder makes them, image grids included."""
    expected = REFERENCE[case_id]
    if "limit" not in expected:
        # Refused by the wire format, before the release's code is reached.
        with pytest.raises(DecisionError):
            request(case_id)
        return
    body = request(case_id)
    encoder = encoders(expected["limit"])
    config = encoder.image_config()
    opened = open_images(body.images or [])
    counts = [count_image_tokens(image, index, config) for index, image in opened]
    grids = [list(prepare(image, index, config).grid) for index, image in opened]
    if "refused" in expected:
        with pytest.raises(DecisionError, match=f"the limit is {expected['limit']}") as error:
            encoder.encode(body, counts)
        question = expected["refused"].split("'")[1]
        assert error.value.param == f"questions.{question}"
        return
    chunks = encoder.encode(body, counts)
    assert [c.question_ids for c in chunks] == [e["questions"] for e in expected["chunks"]]
    for chunk, reference in zip(chunks, expected["chunks"], strict=True):
        assert len(chunk.input_ids) == reference["tokens"]
        assert digest(chunk.input_ids) == reference["ids_sha256"]
        assert chunk.markers == reference["markers"]
        assert grids == reference.get("image_grids", [])


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pplx-joint")
    return {shape: mlx_decision.load(write_pplx(folder / shape, joint=shape)) for shape in SHAPES}


def text_cases():
    """Requests answered by the tiny models, without images (their ids are stored)."""
    for case_id in CASE_IDS:
        expected = REFERENCE[case_id]
        if "answers" in expected and not REQUESTS[case_id].get("images"):
            yield case_id, expected


@pytest.mark.parametrize("shape", SHAPES)
def test_tiny_model_answers_as_the_release(tiny, shape):
    backend = tiny[shape].backend
    checked = 0
    for case_id, expected in text_cases():
        answers = expected["answers"][shape]
        for chunk in expected["chunks"]:
            with mx.stream(mx.cpu):
                states = backend.hidden_states(chunk["ids"])[mx.array(chunk["markers"])]
                states = states.astype(mx.float32)
            for row, question_id in enumerate(chunk["questions"]):
                reference = answers[question_id]["probabilities"]
                option_ids = [str(i) for i in range(len(reference))]
                with mx.stream(mx.cpu):
                    values = read_out(backend.question_features(option_ids, states[row]))
                error = max(abs(a - b) for a, b in zip(values, reference, strict=True))
                assert error < TOLERANCE, f"{case_id}.{question_id}: {error:.2e}"
                checked += 1
    assert checked > 200


@pytest.fixture(scope="module")
def tiny_with_tokenizer(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pplx-joint")
    made = {}

    def model(shape: str, limit: int):
        if (shape, limit) not in made:
            if shape not in made:
                files = cached_release(shape)
                made[shape] = write_pplx(folder / shape, files_from=files, joint=shape)
            made[shape, limit] = mlx_decision.load(made[shape], max_input_tokens=limit)
        return made[shape, limit]

    return model


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("case_id", END_TO_END)
def test_tiny_model_end_to_end(tiny_with_tokenizer, shape, case_id):
    """The whole path: prompt, real tokenizer, images, chunks, passes, answers."""
    expected = REFERENCE[case_id]
    model = tiny_with_tokenizer(shape, expected["limit"])
    with mx.stream(mx.cpu):
        result = model.decide_request(request(case_id))
    assert not result.truncated
    answers = expected["answers"][shape]
    for question_id, answer in result.answers.items():
        reference = answers[question_id]["probabilities"]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(reference[1], abs=TOLERANCE)
        else:
            assert list(answer.probabilities.values()) == pytest.approx(reference, abs=TOLERANCE)
    assert result.usage.input_tokens == sum(c["tokens"] for c in expected["chunks"])


BF16_FILE = HERE / "pplx" / "reference-flash-bf16.json"


def test_the_flash_bf16_reference_is_for_the_same_prompts():
    """The real-weight reference's chunks are the tiny reference's (same tokenizer)."""
    if not BF16_FILE.exists():
        pytest.skip("no reference-flash-bf16.json (make_pplx_joint_reference.py --model)")
    bf16 = json.loads(BF16_FILE.read_text())
    assert bf16["meta"]["release"] == FIXTURE["meta"]["releases"]["flash"]
    assert bf16["meta"]["dtype"] == "torch.bfloat16"
    for case_id, expected in bf16["results"].items():
        tiny = REFERENCE[case_id]["chunks"]
        assert [(c["questions"], c["ids_sha256"]) for c in expected["chunks"]] == [
            (c["questions"], c["ids_sha256"]) for c in tiny
        ], case_id


@pytest.mark.slow
@pytest.mark.weights
def test_flash_bf16_answers_as_the_release():
    """About 4 minutes on an M2: the release folder (pplx-decider-flash-v1.2,
    bf16) answers the requests of the bf16 reference (every part up to 4,096
    tokens, 102 requests, 299 questions) as the release's code does in bf16
    on MPS.

    Measured 2026-10-22 on an M2: max 0.058 (large_image_two_questions,
    a 1.6 MP image), mean 0.0023; three changed top answers, all near ties
    (margins 0.03, 0 and 0). The largest difference is over the 0.04 of the
    other families: the release's own reference path differs by up to 0.059
    between two runs (its evaluation/inference-verification.json), so the
    tolerance here is 0.06.
    """
    import os

    from test_parity import MARGIN, MEAN_TOLERANCE

    BF16_TOLERANCE = 0.06

    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / "pplx-decider-flash-v1.2" if root else None
    if path is None or not path.is_dir():
        pytest.skip("no pplx-decider-flash-v1.2 in MLX_DECISION_MODELS")
    if not BF16_FILE.exists():
        pytest.skip("no reference-flash-bf16.json (make_pplx_joint_reference.py --model)")
    reference = json.loads(BF16_FILE.read_text())["results"]
    model = mlx_decision.load(path)
    backend, default = model.backend, model.backend.capabilities
    differences, worst, decided_flips = [], (0.0, None), []
    for case_id, expected in reference.items():
        # The split requests' lowered limits, without loading the model again.
        limit = REQUESTS[case_id].get("max_input_tokens", default.max_input_tokens)
        backend.capabilities = replace(default, max_input_tokens=limit)
        result = model.decide_request(request(case_id))
        assert result.usage.input_tokens == sum(c["tokens"] for c in expected["chunks"]), case_id
        for question_id, entry in expected["answers"].items():
            answer = result.answers[question_id]
            if answer.type == "noul":
                got = [1 - answer.noul, answer.noul]
            else:
                got = list(answer.probabilities.values())
            want = entry["probabilities"]
            pairs = list(zip(got, want, strict=True))
            differences += [abs(a - b) for a, b in pairs]
            worst = max(worst, (max(abs(a - b) for a, b in pairs), f"{case_id}.{question_id}"))
            # A one-option choice has no runner-up.
            first, second = [*sorted(want, reverse=True), 0.0][:2]
            if first - second > MARGIN and got.index(max(got)) != want.index(first):
                decided_flips.append(f"{case_id}.{question_id}")
    mean = sum(differences) / len(differences)
    assert worst[0] <= BF16_TOLERANCE, f"max |dp| {worst[0]:.4f} at {worst[1]}"
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f}"
    assert not decided_flips
