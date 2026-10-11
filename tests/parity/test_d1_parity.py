"""d1-3B's prompt and readout (``models/d1``) against the release's own prompt code.

``d1/reference.json`` is written by ``scripts/make_d1_reference.py``: per
question of the text parity sets, the prompt text, the token ids and the
token ids each option is scored by, as the release produces them. The tests
need the release's small files in the Hugging Face cache (the script fetches
them) and skip otherwise.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mlx_decision.models.d1.prompt import prefix, suffix
from mlx_decision.models.d1.readout import option_token_ids, probabilities

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


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    meta = FIXTURE["meta"]
    try:
        path = Path(
            snapshot_download(
                meta["repo_id"],
                revision=meta["revision"],
                ignore_patterns=["model.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("d1-3B release files not cached (scripts/make_d1_reference.py fetches them)")
    return Tokenizer.from_file(str(path / "tokenizer.json"))


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
