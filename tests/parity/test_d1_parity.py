"""d1-3B (``models/d1``) against the release's own code, on the text parity sets.

``scripts/make_d1_reference.py`` writes the fixtures in ``d1/``:
``reference.json`` (per question the prompt text, the token ids and the
token ids each option is scored by, as the release produces them),
``reference-tiny.json`` (the release's answers for the tiny random model of
``tiny_models.write_d1`` with its real tokenizer, CPU, float32) and, with
``--model``, ``reference-bf16.json`` (its weights in bfloat16 on MPS, every
question alone). The tests need the release's small files in the Hugging
Face cache (the script fetches them) and skip otherwise; the slow test
compares the release folder in ``MLX_DECISION_MODELS`` with the bf16
fixture.

The tiny model runs on the CPU here: on the GPU, float32 matmuls use TF32.
"""

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_decision
from mlx_decision.models.d1.prompt import prefix, suffix
from mlx_decision.models.d1.readout import option_token_ids, probabilities
from tiny_models import write_d1

HERE = Path(__file__).parent
CASES = []
for name in ("requests.json", "marker/requests.json", "d1/requests.json"):
    CASES += [c for c in json.loads((HERE / name).read_text()) if not c.get("images")]
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "d1" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
ANSWERED = [
    (rid, qid)
    for rid, entry in REFERENCE.items()
    if "questions" in entry
    for qid in entry["questions"]
]
REFUSED = [rid for rid, entry in REFERENCE.items() if "error" in entry]


TINY = json.loads((HERE / "d1" / "reference-tiny.json").read_text())["results"]
BF16_FILE = HERE / "d1" / "reference-bf16.json"
# Largest allowed difference of any probability against the tiny reference (CPU, float32).
TOLERANCE = 1e-5
# Against the release in bf16 (D1-7): its own two code paths differ by up to 0.061.
BF16_MAX, BF16_MEAN, BF16_MARGIN = 0.05, 0.0075, 0.04


@pytest.fixture(scope="module")
def release() -> Path:
    from huggingface_hub import snapshot_download

    meta = FIXTURE["meta"]
    try:
        return Path(
            snapshot_download(
                meta["repo_id"],
                revision=meta["revision"],
                ignore_patterns=["model.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("d1-3B release files not cached (scripts/make_d1_reference.py fetches them)")


@pytest.fixture(scope="module")
def tokenizer(release):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(release / "tokenizer.json"))


def answer_values(answer) -> list[float]:
    """An answer's probabilities in the release's order (noul: yes, then no)."""
    if answer.type == "noul":
        return [answer.noul, 1 - answer.noul]
    return list(answer.probabilities.values())


@pytest.fixture(scope="module")
def tiny(release, tmp_path_factory):
    with mx.stream(mx.cpu):
        folder = write_d1(tmp_path_factory.mktemp("d1") / "tiny-d1", files_from=release)
        yield mlx_decision.load(folder, prefix_cache_gb=0, check_memory=False)


TINY_IDS = [rid for rid, entry in TINY.items() if "skipped" not in entry]
MAX_SCORE_LEVELS = 10  # the model card's range; the release itself takes more


def over_our_limits(case: dict) -> bool:
    return any(
        q["type"] == "score" and len(q["criteria"]) > MAX_SCORE_LEVELS
        for q in case["questions"].values()
    )


@pytest.mark.parametrize("rid", TINY_IDS)
def test_tiny_model_answers_as_the_release(tiny, rid):
    from mlx_decision.errors import DecisionError

    case = REQUESTS[rid]
    if over_our_limits(case):
        with pytest.raises(DecisionError, match=f"at most {MAX_SCORE_LEVELS} levels"):
            tiny.decide(case["state"], case["questions"])
        return
    with mx.stream(mx.cpu):
        result = tiny.decide(case["state"], case["questions"])
    assert not result.truncated
    for qid, expected in TINY[rid].items():
        got = answer_values(result.answers[qid])
        assert max(abs(a - b) for a, b in zip(got, expected, strict=True)) < TOLERANCE, qid


def test_the_fixtures_are_for_the_same_requests():
    assert set(TINY) == {rid for rid, entry in REFERENCE.items() if "questions" in entry}
    if BF16_FILE.exists():
        bf16 = json.loads(BF16_FILE.read_text())
        assert bf16["meta"]["revision"] == FIXTURE["meta"]["revision"]
        assert set(bf16["results"]) == set(TINY)


@pytest.mark.slow
@pytest.mark.weights
def test_bf16_answers_as_the_release():
    """A few minutes: the release folder answers every question alone, against the
    release's own code in bf16 on MPS."""
    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / "d1-3B" if root else None
    if path is None or not path.is_dir():
        pytest.skip("no d1-3B in MLX_DECISION_MODELS")
    if not BF16_FILE.exists():
        pytest.skip("no reference-bf16.json (scripts/make_d1_reference.py --model)")
    reference = json.loads(BF16_FILE.read_text())["results"]
    model = mlx_decision.load(path, prefix_cache_gb=0)
    differences, worst, decided = [], (0.0, None), []
    for rid, entry in reference.items():
        case = REQUESTS[rid]
        if "skipped" in entry or over_our_limits(case):
            continue
        for qid, expected in entry.items():
            answer = model.decide(case["state"], {qid: case["questions"][qid]}).answers[qid]
            got = answer_values(answer)
            difference = max(abs(a - b) for a, b in zip(got, expected, strict=True))
            differences.append(difference)
            worst = max(worst, (difference, f"{rid}.{qid}"))
            ranked = sorted(expected, reverse=True) + [0.0]  # one option: no runner-up
            margin = ranked[0] - ranked[1]
            if got.index(max(got)) != expected.index(max(expected)) and margin > BF16_MARGIN:
                decided.append(f"{rid}.{qid}")
    mean = statistics.mean(differences)
    print(f"{len(differences)} questions: max {worst[0]:.4f} ({worst[1]}), mean {mean:.4f}")
    assert worst[0] <= BF16_MAX, worst
    assert mean <= BF16_MEAN, mean
    assert not decided, decided


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def matches(value, expected) -> bool:
    """A stored text or id list, or its sha256 and length (``make_d1_reference``)."""
    if not isinstance(expected, dict):
        return value == expected
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode()).hexdigest()
    else:
        digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()
    return len(value) == expected["length"] and digest == expected["sha256"]


def case(rid: str, qid: str):
    request = REQUESTS[rid]
    entry = REFERENCE[rid]
    return request, request["questions"][qid], entry, entry["questions"][qid]


@pytest.mark.parametrize(("rid", "qid"), ANSWERED, ids=[f"{r}.{q}" for r, q in ANSWERED])
def test_prompt_and_token_ids_equal_the_release(tokenizer, rid, qid):
    request, question, entry, expected = case(rid, qid)
    head, tail = prefix(request["state"]), suffix(question, tokenizer)
    assert matches(head + tail, expected["prompt"])
    if entry["trunk_length"] is None:
        ids = encode(tokenizer, head + tail)
    else:
        shared = encode(tokenizer, head)
        assert len(shared) == entry["trunk_length"]
        ids = shared + encode(tokenizer, tail)
    assert matches(ids, expected["ids"])


@pytest.mark.parametrize(("rid", "qid"), ANSWERED, ids=[f"{r}.{q}" for r, q in ANSWERED])
def test_scored_tokens_equal_the_release(tokenizer, rid, qid):
    _, question, _, expected = case(rid, qid)
    assert option_token_ids(question, tokenizer) == expected["option_ids"]


@pytest.mark.parametrize(("rid", "qid"), ANSWERED[::7], ids=[f"{r}.{q}" for r, q in ANSWERED[::7]])
def test_readout_is_a_softmax_over_each_options_best_token(tokenizer, rid, qid):
    _, question, _, expected = case(rid, qid)
    rng = np.random.default_rng(len(rid) * 31 + len(qid))
    logprobs = rng.normal(-8, 3, tokenizer.get_vocab_size())
    scores = [max(logprobs[i] for i in group) for group in expected["option_ids"]]
    weights = [math.exp(s - max(scores)) for s in scores]
    want = [w / sum(weights) for w in weights]
    got = probabilities(question, tokenizer, logprobs)
    assert max(abs(a - b) for a, b in zip(got, want, strict=True)) < 1e-12


@pytest.mark.parametrize("rid", REFUSED)
def test_the_release_refuses_only_questions_without_instructions(tokenizer, rid):
    """The backend fills them in from the question id (as for Laya and Julia)."""
    missing = [q for q in REQUESTS[rid]["questions"].values() if q.get("instructions") is None]
    assert missing, REFERENCE[rid]["error"]
    for question in missing:
        with pytest.raises(ValueError):
            suffix(question, tokenizer)


@pytest.mark.parametrize("limit", [32768, 8192, 1024])
def test_long_states_are_cut_to_fit_and_keep_their_ends(tokenizer, limit):
    """The backend without weights, on every text request of the parity sets."""
    from mlx_decision.errors import DecisionError
    from mlx_decision.models.d1.model import CUT_MARK, D1Backend
    from mlx_decision.types import parse_request

    backend = D1Backend("tokens", None, tokenizer, limit)
    cuts = refused = 0
    for case in CASES:
        request = parse_request({"state": case["state"], "questions": case["questions"]})
        try:
            encoded, shared, cut = backend.encode(request)
        except DecisionError as error:
            # Only a question too long even with an empty state.
            question = REQUESTS[case["id"]]["questions"][error.param.split(".", 1)[1]]
            alone = encode(
                tokenizer, prefix("") + suffix({"instructions": "x", **question}, tokenizer)
            )
            assert len(alone) > limit - 16, (case["id"], error)
            refused += 1
            continue
        assert all(len(q.input_ids) <= limit for q in encoded), case["id"]
        whole = encode(tokenizer, prefix(case["state"]))
        if not cut:
            continue
        cuts += 1
        head = tokenizer.decode(encoded[0].input_ids, skip_special_tokens=False)
        start, end = head.split(CUT_MARK, 1)
        original = tokenizer.decode(whole, skip_special_tokens=False)
        # The kept text is the original's start and end (but for a token at each cut).
        assert original.startswith(start[:-8]), case["id"]
        tail = end.split("\n\n\nQUESTION:\n")[0]
        assert original.split("\n\n\nQUESTION:\n")[0].endswith(tail[8:]), case["id"]
        # Not cut far below what the longest question leaves room for.
        assert max(len(q.input_ids) for q in encoded) > limit - 64, case["id"]
    assert cuts > 0
    print(f"limit {limit}: {cuts} cut, {refused} refused")
