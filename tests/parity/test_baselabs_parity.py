"""Base Labs (elephant-1, mammoth-1) on MLX against the release's own serving code.

``baselabs/reference.json`` is written by ``scripts/make_baselabs_reference.py``:
OneJev's ``DecisionEngine`` with Base Labs' ``serve.patch``, so the prompt
text, token ids and shared prefix of every question (the releases' real
tokenizer) and the answers of the tiny random models of
``tiny_models.write_baselabs`` in both shapes (CPU, float32). The prompt
tests need the release's small files in the Hugging Face cache (the script
fetches them) and skip otherwise.
"""

import hashlib
import json
import math
from pathlib import Path

import pytest

from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.baselabs.model import code_token_ids, head_file, read_rows
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


@pytest.fixture(scope="module")
def release() -> Path:
    from huggingface_hub import snapshot_download

    meta = FIXTURE["meta"]["releases"]["elephant"]
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


@pytest.fixture(scope="module", params=["elephant", "mammoth"])
def tiny(request, tmp_path_factory):
    """The reference's tiny model (same seed), with a word-level tokenizer: the
    stored token ids are fed in, so no release files are needed."""
    from mlx_decision.registry import load_backend
    from tiny_models import write_baselabs

    folder = write_baselabs(tmp_path_factory.mktemp(request.param) / "tiny", shape=request.param)
    backend = load_backend(folder, prefix_cache_gb=0)
    # The word-level tokenizer numbers the codes otherwise: the release's ids.
    backend.rows = read_rows(head_file(folder), "lm_head.weight", FIXTURE["meta"]["slot_ids"])
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


@pytest.fixture(scope="module", params=["elephant", "mammoth"])
def released(request, tmp_path_factory, release):
    """The reference's tiny model with the release's tokenizer."""
    from mlx_decision.registry import load_backend
    from tiny_models import write_baselabs

    folder = tmp_path_factory.mktemp(request.param) / "tiny"
    write_baselabs(folder, shape=request.param, files_from=release)
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


BF16_FILE = HERE / "baselabs" / "reference-bf16.json"
# Measured: see the docstring.
BF16_TOLERANCE = 0.05


@pytest.mark.slow
@pytest.mark.weights
def test_bf16_answers_as_the_release():
    """About 2 minutes: the release folder (elephant-1, bf16) answers the text
    parity set and the Base Labs set (68 requests, 166 questions) as the
    release's engine does in bf16 on MPS (the LM head in float32 on both
    sides), at the release's per-question limit (32,768).

    Measured 2026-10-31 on an M5 Pro (MLX 0.32.4): max 0.039
    (ledger_3k.largest_vendor, a near tie: margin 0.003), next 0.026, mean
    0.0010; no changed top answers, near ties included. 112 s, 23.4 GB peak
    (the release's engine: 398 s, 46.5 GB). The limit is 0.05, as d3's: the
    worst case is a near tie, and those move with MLX versions.
    """
    import os

    from test_parity import MARGIN, MEAN_TOLERANCE

    import mlx_decision

    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / "elephant-1" if root else None
    if path is None or not path.is_dir():
        pytest.skip("no elephant-1 in MLX_DECISION_MODELS")
    if not BF16_FILE.exists():
        pytest.skip("no reference-bf16.json (make_baselabs_reference.py --model)")
    reference = json.loads(BF16_FILE.read_text())["results"]
    model = mlx_decision.load(path, max_input_tokens=32768)
    differences, worst, decided_flips = [], (0.0, None), []
    for case_id, expected in reference.items():
        if "refused" in expected or case_id in WIRE_REFUSED:
            continue
        result = model.decide_request(REQUESTS[case_id])
        assert result.usage.input_tokens == expected["tokens"], case_id
        for qid, entry in expected["questions"].items():
            logits = entry["logits"]["elephant"]
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
    assert worst[0] <= BF16_TOLERANCE, f"max |dp| {worst[0]:.4f} at {worst[1]}"
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f}"
    assert not decided_flips
