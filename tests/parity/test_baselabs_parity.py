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
from pathlib import Path

import pytest

from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.baselabs.model import code_token_ids
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
