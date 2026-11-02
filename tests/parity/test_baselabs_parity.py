"""Base Labs (elephant-1, mammoth-1, joey-1) on MLX against the release's own serving code.

``baselabs/reference.json`` is written by ``scripts/make_baselabs_reference.py``:
OneJev's ``DecisionEngine`` with Base Labs' ``serve.patch``, so the prompt
text, token ids and shared prefix of every question (the releases' real
tokenizer) and the answers of the tiny random models of
``tiny_models.write_baselabs`` in the three shapes (CPU, float32). The tiny
joey ties its head to the embeddings yet stores a head that differs from
them; the release reads the stored one, as the loader does. The prompt
tests need the releases' small files in the Hugging Face cache (the script
fetches them) and skip otherwise.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.baselabs.model import code_token_ids, head_source, read_rows
from mlx_decision.models.baselabs.prompt import layout, render, render_question
from mlx_decision.types import parse_request

HERE = Path(__file__).parent
CASES = [c for c in json.loads((HERE / "requests.json").read_text()) if not c.get("images")]
CASES += json.loads((HERE / "baselabs" / "requests.json").read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "baselabs" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
IDS = [case["id"] for case in CASES]
# Requests the wire format refuses before a model sees them (the release
# takes a score with one level).
WIRE_REFUSED = {"bl_score_one_level"}
SHAPES = ["elephant", "mammoth", "joey"]


def cached_release(shape: str) -> Path:
    """The release's small files in the Hugging Face cache; skips without them."""
    from huggingface_hub import snapshot_download

    meta = FIXTURE["meta"]["releases"][shape]
    try:
        path = Path(
            snapshot_download(
                meta["repo_id"],
                revision=meta["revision"],
                # What the script fetches: a snapshot without them counts as complete.
                ignore_patterns=["*.safetensors", "assets/*"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("Base Labs release files not cached (make_baselabs_reference.py fetches them)")
    if not (path / "tokenizer.json").exists():
        pytest.skip("Base Labs release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def release() -> Path:
    return cached_release("elephant")


@pytest.fixture(scope="module")
def tokenizer(release):
    return load_tokenizer(release)


def sha256(value) -> str:
    data = value if isinstance(value, str) else json.dumps(value)
    return hashlib.sha256(data.encode()).hexdigest()


def request(case_id: str):
    case = REQUESTS[case_id]
    return parse_request({"state": case["state"], "questions": case["questions"]})


def test_the_fixture_covers_every_request():
    assert set(REFERENCE) == set(IDS)


def test_answer_codes_are_the_releases_tokens(tokenizer):
    assert code_token_ids(tokenizer) == FIXTURE["meta"]["slot_ids"]


@pytest.mark.parametrize("case_id", IDS)
def test_prompts_and_tokens_equal_the_release(tokenizer, case_id):
    expected = REFERENCE[case_id]
    if case_id in WIRE_REFUSED:
        with pytest.raises(DecisionError):
            request(case_id)
        return
    if "questions" not in expected:  # refused by the release's schema
        return
    body = request(case_id)
    suffixes = [render_question(q) for q in body.questions.values()]
    found = layout(tokenizer, body.state, suffixes)
    assert len(found.prefix) == expected["prefix_tokens"]
    for (qid, entry), suffix, rest in zip(
        expected["questions"].items(), suffixes, found.suffixes, strict=True
    ):
        prompt = render(body.state, suffix)
        ids = found.prefix + rest
        if "prompt" in entry:
            assert prompt == entry["prompt"], qid
            assert ids == entry["ids"], qid
        assert sha256(prompt) == entry["prompt_sha256"], qid
        assert sha256(ids) == entry["ids_sha256"], qid
        assert len(ids) == entry["tokens"], qid


# Images: ``baselabs/reference-images.json`` (``make_baselabs_reference.py
# --images``), the releases' training-side scoring of the image requests.
IMAGE_FIXTURE = json.loads((HERE / "baselabs" / "reference-images.json").read_text())
IMAGE_CASES = {c["id"]: c for c in json.loads((HERE / "requests.json").read_text())}
IMAGE_CASES = {key: case for key, case in IMAGE_CASES.items() if case.get("images")}
for case in json.loads((HERE / "baselabs" / "image-requests.json").read_text()):
    IMAGE_CASES[case["id"]] = case


def image_request(case_id: str):
    """The image request as a model gets it: its state's images lifted out."""
    from parity_images import with_data_urls

    from mlx_decision.model import _lift_state_images

    return _lift_state_images(parse_request(with_data_urls(IMAGE_CASES[case_id])))[0]


def expand(ids: list[int], pad: int, grids: list[list[int]]) -> list[int]:
    """Each image pad as its patches after the 2 x 2 merge, as the processor expands it."""
    counts = iter(t * h * w // 4 for t, h, w in grids)
    return [token for i in ids for token in ([pad] * next(counts) if i == pad else [i])]


def test_the_image_fixture_covers_every_image_request():
    assert list(IMAGE_FIXTURE["results"]) == list(IMAGE_CASES)


@pytest.mark.parametrize("case_id", list(IMAGE_CASES))
def test_image_prompts_and_tokens_equal_the_reference(tokenizer, case_id):
    """Images lifted out of the state get their vision block in their place,
    top-level images at the start of the state block; the prompt and, with
    each image pad expanded to its patches, the tokens are the reference's."""
    body = image_request(case_id)
    expected = IMAGE_FIXTURE["results"][case_id]["questions"]
    count = len(body.images)
    suffixes = [render_question(q) for q in body.questions.values()]
    found = layout(tokenizer, body.state, suffixes, images=count)
    pad = tokenizer.token_to_id("<|image_pad|>")
    for (qid, entry), suffix, rest in zip(expected.items(), suffixes, found.suffixes, strict=True):
        assert render(body.state, suffix, images=count) == entry["prompt"], qid
        assert len(entry["grids"]) == count, qid
        prefix = expand(found.prefix, pad, entry["grids"])
        ids = prefix + rest
        assert len(prefix) == entry["prefix_tokens"], qid
        assert sha256(ids) == entry["ids_sha256"], qid
        if "ids" in entry:
            assert ids == entry["ids"], qid


def test_typed_image_markers_stay_text():
    """A value that reads [image 1] is text unless an image was lifted there."""
    from mlx_decision.models.baselabs.prompt import IMAGE, state_text
    from mlx_decision.state_images import LiftedImage

    state = {"note": "[image 1]", "photo": LiftedImage(1)}
    text = state_text(state, images=2)
    assert text.startswith(IMAGE + "\n{")
    assert '"note": "[image 1]"' in text and f'"photo": "{IMAGE}"' in text
    with pytest.raises(ValueError, match="do not follow"):
        state_text({"photo": LiftedImage(0)}, images=2)


# Largest allowed difference of a logit against the reference (CPU, float32).
LOGIT_TOLERANCE = 1e-4
# Largest allowed difference of a probability.
TOLERANCE = 1e-5
STORED = [
    case_id
    for case_id in IDS
    if "refused" not in REFERENCE[case_id]
    and all("ids" in q for q in REFERENCE[case_id].get("questions", {}).values())
    and REFERENCE[case_id].get("questions")
]


@pytest.fixture(scope="module", params=SHAPES)
def tiny(request, tmp_path_factory):
    """The reference's tiny model (same seed), with a word-level tokenizer: the
    stored token ids are fed in, so no release files are needed."""
    from mlx_decision.registry import load_backend
    from tiny_models import write_baselabs

    folder = write_baselabs(tmp_path_factory.mktemp(request.param) / "tiny", shape=request.param)
    backend = load_backend(folder, prefix_cache_gb=0)
    # The word-level tokenizer numbers the codes otherwise: the release's ids.
    backend.rows = read_rows(*head_source(folder), FIXTURE["meta"]["slot_ids"])
    return request.param, backend


def test_tiny_models_answer_as_the_release(tiny):
    """Every question of at most 4,096 tokens: the stored prefix runs once, the
    questions in batches off it, and the answer codes' logits are the release's."""
    import mlx.core as mx

    from mlx_decision.models.baselabs.model import Encoded

    shape, backend = tiny
    with mx.stream(mx.cpu):
        for case_id in STORED:
            expected = REFERENCE[case_id]
            count = expected["prefix_tokens"]
            questions = expected["questions"]
            prefix = next(iter(questions.values()))["ids"][:count]
            encoded = [
                Encoded(qid, [str(i) for i in range(q["options"])], q["ids"][count:])
                for qid, q in questions.items()
            ]
            assert all(q["ids"][:count] == prefix for q in questions.values())
            hidden = backend.last_hidden(prefix, encoded)
            for question, last in zip(encoded, hidden, strict=True):
                entry = questions[question.question_id]
                rows = backend.rows[: entry["options"]]
                got = rows @ last
                want = mx.array(entry["logits"][shape])
                error = mx.abs(got - want).max().item()
                assert error < LOGIT_TOLERANCE, f"{case_id}.{question.question_id}: {error:.2e}"
                p_error = mx.abs(mx.softmax(got) - mx.softmax(want)).max().item()
                assert p_error < TOLERANCE, f"{case_id}.{question.question_id}: {p_error:.2e}"
    assert len(STORED) > 50


# The full path with the real tokenizer: prompt shapes, several questions,
# questions over 4,096 tokens (no stored ids) and a request near the default limit.
END_TO_END = [
    "ticket",
    "bl_every_type_twice",
    "bl_choice_27_options",
    "special_tokens_in_state",
    "twenty_mixed",
    "ledger_6k",
    "log_16k_near_limit",
]


@pytest.fixture(scope="module", params=SHAPES)
def released(request, tmp_path_factory):
    """The reference's tiny model with its release's tokenizer."""
    from mlx_decision.registry import load_backend
    from tiny_models import write_baselabs

    folder = tmp_path_factory.mktemp(request.param) / "tiny"
    write_baselabs(folder, shape=request.param, files_from=cached_release(request.param))
    return request.param, load_backend(folder, max_input_tokens=32768, prefix_cache_gb=0)


@pytest.mark.parametrize("case_id", END_TO_END)
def test_end_to_end_as_the_release(released, case_id):
    import mlx.core as mx

    shape, backend = released
    expected = REFERENCE[case_id]
    with mx.stream(mx.cpu):
        output = backend.score(request(case_id))
    assert output.input_tokens == expected["tokens"]
    for qid, entry in expected["questions"].items():
        want = mx.softmax(mx.array(entry["logits"][shape])).tolist()
        got = list(output.probabilities[qid].values())
        assert got == pytest.approx(want, abs=TOLERANCE), qid


@pytest.mark.parametrize("case_id", ["ticket", "bl_every_type_twice", "twenty_mixed"])
def test_answers_as_the_release(released, case_id):
    """The answers' formulas on top (noul, choice, score and their confidence);
    the release rounds them to 6 decimals."""
    import mlx.core as mx

    from mlx_decision.model import DecisionModel

    shape, backend = released
    with mx.stream(mx.cpu):
        result = DecisionModel(backend).decide_request(REQUESTS[case_id])
    assert result.usage.input_tokens == REFERENCE[case_id]["tokens"]
    for qid, entry in REFERENCE[case_id]["questions"].items():
        want = entry["answer"][shape]
        got = result.answers[qid].model_dump()
        assert got["type"] == want["type"], qid
        for key, value in want.items():
            if isinstance(value, float):
                assert got[key] == pytest.approx(value, abs=2e-6), f"{qid}.{key}"
            elif key == "probabilities":
                assert got[key] == pytest.approx(value, abs=2e-6), qid
            else:
                assert got[key] == value, f"{qid}.{key}"


# Per release: its folder in MLX_DECISION_MODELS, the fixture
# make_baselabs_reference.py --model writes, the largest allowed difference
# of a probability (measured: see the test's docstring).
@pytest.mark.parametrize("case_id", list(IMAGE_CASES))
def test_tiny_models_answer_images_as_the_reference(released, case_id):
    """Every image question: the prefix with its images runs once (the vision
    tower fills the pads, the full-attention layers see the image tokens both
    ways), the questions continue from it; the answer codes' logits and the
    probabilities are the releases' training-side scoring's."""
    import mlx.core as mx

    shape, backend = released
    request = image_request(case_id)
    expected = IMAGE_FIXTURE["results"][case_id]["questions"]
    with mx.stream(mx.cpu):
        prefix, encoded, hidden = backend.passes(request)
    for question, last in zip(encoded, hidden, strict=True):
        entry = expected[question.question_id]
        assert len(prefix) + len(question.suffix) == entry["tokens"], question.question_id
        got = backend.rows[: entry["options"]] @ last
        want = mx.array(entry["logits"][shape])
        error = mx.abs(got - want).max().item()
        assert error < LOGIT_TOLERANCE, f"{case_id}.{question.question_id}: {error:.2e}"
        p_error = mx.abs(mx.softmax(got) - mx.softmax(want)).max().item()
        assert p_error < TOLERANCE, f"{case_id}.{question.question_id}: {p_error:.2e}"


def test_image_questions_batched_answer_as_one_by_one(released, monkeypatch):
    import mlx.core as mx

    from mlx_decision.models.baselabs import model as baselabs_model

    _, backend = released
    request = image_request("image_error_ticket")
    with mx.stream(mx.cpu):
        _, _, batched = backend.passes(request)
        monkeypatch.setattr(baselabs_model, "BATCH_TOKENS", 1)
        _, _, single = backend.passes(request)
    for a, b in zip(batched, single, strict=True):
        assert mx.abs(a - b).max().item() < 1e-5


def test_a_kept_prefix_with_images_answers_as_a_fresh_one(released, tmp_path):
    import mlx.core as mx

    from mlx_decision.registry import load_backend

    shape, backend = released
    keeping = load_backend(backend.path, max_input_tokens=32768, prefix_cache_gb=1)
    request = image_request("bl_top_level_and_state_images")
    with mx.stream(mx.cpu):
        fresh = backend.score(request).probabilities
        first = keeping.score(request).probabilities
        kept = keeping.score(request).probabilities
        other = keeping.score(image_request("bl_image_value")).probabilities
    assert len(keeping.prefix_cache) == 2  # other images: another prefix
    for qid, values in fresh.items():
        assert first[qid] == pytest.approx(values, abs=1e-6), qid
        assert kept[qid] == pytest.approx(values, abs=1e-6), qid
    assert other


def test_image_markers_in_the_text_of_a_request_with_images_are_refused(released):
    from mlx_decision.errors import DecisionError

    _, backend = released
    request = image_request("bl_image_value")
    marked = request.model_copy(update={"state": {**request.state, "note": "<|image_pad|>"}})
    with pytest.raises(DecisionError, match="image markers") as error:
        backend.score(marked)
    assert error.value.param == "state"
    question = next(iter(request.questions.values()))
    asked = question.model_copy(update={"instructions": "Is <|vision_start|> shown?"})
    with pytest.raises(DecisionError, match="image markers") as error:
        backend.score(request.model_copy(update={"questions": {"q": asked}}))
    assert error.value.param == "questions.q"


BF16 = {
    "elephant": ("elephant-1", "reference-bf16.json", 0.05),
    "joey": ("joey-1", "reference-joey-bf16.json", 0.03),
}


@pytest.mark.slow
@pytest.mark.weights
@pytest.mark.parametrize("shape", list(BF16))
def test_bf16_answers_as_the_release(shape):
    """The release folder in bf16 answers the text parity set and the Base
    Labs set (68 requests, 166 questions) as the release's engine does in
    bf16 on MPS (the LM head in float32 on both sides), at the release's
    per-question limit (32,768).

    elephant-1, about 2 minutes. Measured 2026-10-31 on an M5 Pro (MLX
    0.32.4): max 0.039 (ledger_3k.largest_vendor, a near tie: margin 0.003),
    next 0.026, mean 0.0010; no changed top answers, near ties included.
    112 s, 23.4 GB peak (the release's engine: 398 s, 46.5 GB). The limit is
    0.05, as d3's: the worst case is a near tie, and those move with MLX
    versions.

    joey-1, under a minute. Measured 2026-11-02 on an M5 Pro (MLX 0.32.4):
    max 0.010 (review_mixed.sentiment, margin 0.094), next 0.009, mean
    0.0007; no changed top answers. 27 s, 8.5 GB peak (the release's engine:
    about 2 minutes). The limit is 0.03: three times the worst case, which
    is no near tie.
    """
    import os

    from test_parity import MARGIN, MEAN_TOLERANCE

    import mlx_decision

    name, file, tolerance = BF16[shape]
    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / name if root else None
    if path is None or not path.is_dir():
        pytest.skip(f"no {name} in MLX_DECISION_MODELS")
    fixture = HERE / "baselabs" / file
    if not fixture.exists():
        pytest.skip(f"no {file} (make_baselabs_reference.py --model)")
    reference = json.loads(fixture.read_text())["results"]
    model = mlx_decision.load(path, max_input_tokens=32768)
    differences, worst, decided_flips = [], (0.0, None), []
    for case_id, expected in reference.items():
        if "refused" in expected or case_id in WIRE_REFUSED:
            continue
        result = model.decide_request(REQUESTS[case_id])
        assert result.usage.input_tokens == expected["tokens"], case_id
        for qid, entry in expected["questions"].items():
            logits = entry["logits"][shape]
            top = max(logits)
            exps = [math.exp(x - top) for x in logits]
            want = [e / sum(exps) for e in exps]
            answer = result.answers[qid]
            if answer.type == "noul":
                got = [answer.noul, 1 - answer.noul]
            else:
                got = list(answer.probabilities.values())
            pairs = list(zip(got, want, strict=True))
            differences += [abs(a - b) for a, b in pairs]
            worst = max(worst, (max(abs(a - b) for a, b in pairs), f"{case_id}.{qid}"))
            first, second = [*sorted(want, reverse=True), 0.0][:2]
            if first - second > MARGIN and got.index(max(got)) != want.index(first):
                decided_flips.append(f"{case_id}.{qid}")
    mean = sum(differences) / len(differences)
    assert worst[0] <= tolerance, f"max |dp| {worst[0]:.4f} at {worst[1]}"
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f}"
    assert not decided_flips
