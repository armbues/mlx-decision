"""Clef 27B: mlx-decision against Cloudflare's reference, both in bf16.

Neither fits this Mac in bf16, so ``reference-clef.json`` comes from a
larger one: ``scripts/clef_parity_bundle.py`` wrote the runner and the
parity set, the run there wrote the reference's results and mlx-decision's
next to each other. The comparison needs no weights; the token check needs
only the 27B tokenizer (a JSON file).
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
from test_parity import MARGIN, MEAN_TOLERANCE, REQUESTS, TOLERANCE, request

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / "reference-clef.json").read_text())
REFERENCE = FIXTURE["reference"]["results"]
MLX = FIXTURE["mlx"]["results"]


def probability_pairs():
    for case_id, reference in REFERENCE.items():
        for question in reference["questions"]:
            ours = [
                MLX[case_id]["probabilities"][question["id"]][o] for o in question["option_ids"]
            ]
            yield case_id, question, ours


def test_the_fixture_is_for_the_current_parity_set():
    digest = hashlib.sha256((HERE / "requests.json").read_bytes()).hexdigest()
    assert FIXTURE["requests_sha256"] == digest
    assert set(REFERENCE) == set(MLX) == set(REQUESTS)
    assert FIXTURE["mlx"]["meta"]["precision"] == "bfloat16"
    assert FIXTURE["reference"]["meta"]["dtype"] == "bfloat16"


def test_inputs_match():
    for case_id, reference in REFERENCE.items():
        assert MLX[case_id]["input_tokens"] == len(reference["input_ids"]), case_id
        assert MLX[case_id]["truncated"] == reference["truncated"], case_id


def test_probabilities_match():
    for case_id, question, ours in probability_pairs():
        expected = question["probabilities"]
        worst = max(abs(a - b) for a, b in zip(ours, expected, strict=True))
        assert worst <= TOLERANCE, f"{case_id} {question['id']}: max |dp| {worst:.4f}"
        first, second = sorted(expected, reverse=True)[:2]
        if first - second > MARGIN:
            assert ours.index(max(ours)) == expected.index(first), (case_id, question["id"])


def test_mean_difference_over_the_set():
    differences = [
        abs(a - b)
        for _, question, ours in probability_pairs()
        for a, b in zip(ours, question["probabilities"], strict=True)
    ]
    mean = sum(differences) / len(differences)
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f}"


def tokenizer_folder() -> Path | None:
    """Clef 27B's folder (the tokenizer is all this needs), or None."""
    root = os.environ.get("MLX_DECISION_MODELS")
    if root and (Path(root) / "clef" / "tokenizer.json").exists():
        return Path(root) / "clef"
    try:
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download("Cloudflare/clef", allow_patterns=["*.json"], local_files_only=True)
        )
    except Exception:
        return None
    return path if (path / "tokenizer.json").exists() else None


def test_tokens_match():
    from tokenizers import Tokenizer

    from mlx_decision.models.clef.encode import encode_request

    folder = tokenizer_folder()
    if folder is None:
        pytest.skip("Clef 27B's tokenizer is not here (set MLX_DECISION_MODELS)")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    for case_id, reference in REFERENCE.items():
        if REQUESTS[case_id].get("images"):
            pytest.importorskip("PIL")
            from mlx_decision.models.clef.model import prepare_images

            images = prepare_images(request(case_id).images, folder, None)
            counts = [image.tokens for image in images]
        else:
            counts = []
        encoded = encode_request(tokenizer, request(case_id), image_tokens=counts)
        assert list(encoded.input_ids) == reference["input_ids"], case_id
        spans = [list(q.question_span) for q in encoded.questions]
        assert spans == [q["span"] for q in reference["questions"]], case_id


def quantized_copy(name: str) -> Path | None:
    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / name if root else None
    return path if path is not None and path.is_dir() else None


# Measured 2026-10-07 for the uniform 8-bit copy against the reference: max
# 0.113, mean 0.0020, one changed top answer where the reference's margin is
# above 0.04 (0.41 / 0.59 became 0.525 / 0.475). The limits leave a little room.
EIGHT_BIT_MAX = 0.15
EIGHT_BIT_MEAN = 0.003
EIGHT_BIT_DECIDED_FLIPS = 1


@pytest.mark.slow
@pytest.mark.weights
def test_the_8bit_copy_stays_close_to_the_reference():
    """About 8 minutes: the 8-bit copy answers the whole parity set."""
    import mlx_decision

    path = quantized_copy("clef-8bit")
    if path is None:
        pytest.skip("no clef-8bit in MLX_DECISION_MODELS (convert -m Cloudflare/clef -q)")
    # No image cap beyond the processor's own, as the reference.
    model = mlx_decision.load(path, prefix_cache_gb=0, max_image_pixels=None)
    differences, decided_flips = [], 0
    for case_id, reference in REFERENCE.items():
        output = model.backend.score(request(case_id))
        assert output.input_tokens == len(reference["input_ids"]), case_id
        for question in reference["questions"]:
            expected = question["probabilities"]
            ours = [output.probabilities[question["id"]][o] for o in question["option_ids"]]
            differences += [abs(a - b) for a, b in zip(ours, expected, strict=True)]
            first, second = sorted(expected, reverse=True)[:2]
            if first - second > MARGIN and ours.index(max(ours)) != expected.index(first):
                decided_flips += 1
    mean = sum(differences) / len(differences)
    assert max(differences) <= EIGHT_BIT_MAX, f"max |dp| {max(differences):.4f}"
    assert mean <= EIGHT_BIT_MEAN, f"mean |dp| {mean:.5f}"
    assert decided_flips <= EIGHT_BIT_DECIDED_FLIPS
