"""Laya and Julia on MLX against their own PyTorch code, on the marker parity set.

The set is the text requests of ``requests.json`` plus ``marker/requests.json``.
``marker/reference-<model>.json`` is written by
``scripts/make_marker_reference.py`` (CPU, float32). The models run here in
their default precision.
"""

import hashlib
import json
from pathlib import Path

import pytest

from mlx_decision.errors import DecisionError
from mlx_decision.models.marker.encode import encode_request

HERE = Path(__file__).parent
CASES = [c for c in json.loads((HERE / "requests.json").read_text()) if not c.get("images")]
CASES += json.loads((HERE / "marker" / "requests.json").read_text())
REQUESTS = {case["id"]: case for case in CASES}
MODELS = {
    "laya": "laya_path",
    "laya-multilingual": "laya_multilingual_path",
    "Julia-1": "julia_path",
}

# Largest allowed difference of any probability; largest mean difference over
# the set; reference top-two margin above which the top option must agree.
TOLERANCE = 0.04
MEAN_TOLERANCE = 0.0025
MARGIN = 0.04

pytestmark = pytest.mark.weights


def reference(name: str) -> dict:
    return json.loads((HERE / "marker" / f"reference-{name}.json").read_text())["results"]


@pytest.mark.parametrize("name", list(MODELS))
def test_every_request_has_a_reference(name):
    assert set(reference(name)) == set(REQUESTS)


@pytest.fixture(scope="module", params=list(MODELS))
def model(request):
    import mlx_decision

    path = request.getfixturevalue(MODELS[request.param])
    return request.param, mlx_decision.load(path)


def body(case_id: str) -> dict:
    case = REQUESTS[case_id]
    return {"state": case["state"], "questions": case["questions"]}


def test_encoding_matches(model):
    name, loaded = model
    backend = loaded.backend
    mismatches = []
    for case_id, expected in reference(name).items():
        try:
            request = loaded.check(body(case_id))
            encoded = encode_request(backend.tokenizer, backend.special, backend.settings, request)
        except DecisionError:
            if "refused" not in expected:
                mismatches.append(f"{case_id}: refused here only")
            continue
        if "refused" in expected:
            mismatches.append(f"{case_id}: refused by the reference only")
            continue
        for ours, theirs in zip(encoded, expected["questions"], strict=True):
            digest = hashlib.sha256(json.dumps(ours.input_ids).encode()).hexdigest()
            if (len(ours.input_ids), digest, ours.markers, ours.truncated) != (
                theirs["length"],
                theirs["ids_sha256"],
                theirs["markers"],
                theirs["truncated"],
            ):
                mismatches.append(f"{case_id}.{ours.question_id}: tokens differ")
    assert not mismatches


def test_the_set_covers_refusing_and_cutting(model):
    name, _ = model
    results = reference(name).values()
    answered = [r for r in results if "questions" in r]
    assert len(answered) >= 30
    if name.startswith("laya"):
        assert any(q["truncated"] for r in answered for q in r["questions"])
    else:
        assert sum("refused" in r for r in results) >= 10


@pytest.fixture(scope="module")
def outputs(model):
    name, loaded = model
    return name, {
        case_id: loaded.backend.score(loaded.check(body(case_id)))
        for case_id, expected in reference(name).items()
        if "questions" in expected
    }


def test_probabilities_match(outputs):
    name, results = outputs
    failures, differences = [], []
    for case_id, output in results.items():
        for question in reference(name)[case_id]["questions"]:
            ours = list(output.probabilities[question["id"]].values())
            expected = question["probabilities"]
            pairs = list(zip(ours, expected, strict=True))
            differences += [abs(a - b) for a, b in pairs]
            worst = max(abs(a - b) for a, b in pairs)
            where = f"{case_id}.{question['id']}"
            if worst > TOLERANCE:
                failures.append(f"{where}: max |dp| {worst:.4f}")
            first, second = sorted([*expected, 0.0], reverse=True)[:2]
            if first - second > MARGIN and ours.index(max(ours)) != expected.index(first):
                failures.append(f"{where}: top option differs")
    mean = sum(differences) / len(differences)
    assert not failures
    assert mean <= MEAN_TOLERANCE, f"mean |dp| {mean:.5f} over {len(differences)} probabilities"
