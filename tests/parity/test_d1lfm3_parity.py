"""d1-2B (``models/d1lfm3``) against the release's own code, on the parity sets.

``scripts/make_d1lfm3_reference.py`` writes ``d1lfm3/reference.json``: per
request of the parity set (text and pictures), the marker text requests and
``d1lfm3/requests.json`` (edge cases), the release's error if it refuses it;
otherwise the prompt texts, the packed sets (token ids, state length, each
question's span and answer slot), each question's scored token groups, each
picture's size after the cap with its patch grid and patch means, and the
answers of the tiny random model of ``tiny_models.write_d1lfm3`` with the
release's tokenizer (CPU, float32; sets over 16,384 tokens not run). The
tests need the release's small files in the Hugging Face cache (the script
fetches them) and skip otherwise. With ``--model`` the script runs the
release's weights in bf16 on MPS over the same requests and writes
``d1lfm3/reference-bf16.json`` (answers only); the slow test compares the
release folder ``$MLX_DECISION_MODELS/d1-2B`` with it.

The tiny model runs on the CPU here: on the GPU, float32 matmuls use TF32.
"""

import hashlib
import json
import os
import statistics
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from parity_images import open_image

import mlx_decision
from mlx_decision.errors import DecisionError
from mlx_decision.models.d1lfm3.config import settings
from mlx_decision.models.d1lfm3.model import (
    CUT_MARK,
    INPUT_TOKEN_LIMIT,
    D1Lfm3Backend,
    _capabilities,
)
from mlx_decision.models.d1lfm3.prompt import picture_size, prefix, suffix
from mlx_decision.models.d1lfm3.readout import option_token_ids
from mlx_decision.types import parse_request
from tiny_models import write_d1lfm3

HERE = Path(__file__).parent
CASES = {}
for name in ("requests.json", "marker/requests.json", "d1lfm3/requests.json"):
    for case in json.loads((HERE / name).read_text()):
        if name == "marker/requests.json" and case.get("images"):
            continue
        CASES[case["id"]] = case
FIXTURE = json.loads((HERE / "d1lfm3" / "reference.json").read_text())
RESULTS = FIXTURE["requests"]
ACCEPTED = [rid for rid, entry in RESULTS.items() if "refused" not in entry]
ANSWERED = [rid for rid in ACCEPTED if "probabilities" in RESULTS[rid]]
REFUSED = [rid for rid, entry in RESULTS.items() if "refused" in entry]
PICTURES = [rid for rid in ACCEPTED if RESULTS[rid]["pictures"]]
# Largest allowed difference of any probability against the tiny reference (CPU, float32).
TOLERANCE = 1e-5
# Pixels within one 8-bit level of the release's processor (torchvision there,
# Pillow here; mean and std 0.5), plus the fixture's rounding.
PIXEL_TOLERANCE = 2 / 255 + 1e-4
BF16_FILE = HERE / "d1lfm3" / "reference-bf16.json"
# Against the release in bf16 (LF-8, R26's family tolerance).
BF16_MAX, BF16_MEAN, BF16_MARGIN = 0.05, 0.005, 0.04


@pytest.fixture(scope="module")
def release() -> Path:
    from huggingface_hub import snapshot_download

    source = FIXTURE["source"]
    try:
        return Path(
            snapshot_download(
                source["repo"],
                revision=source["revision"],
                ignore_patterns=["*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip(
            "d1-2B release files not cached (scripts/make_d1lfm3_reference.py fetches them)"
        )


@pytest.fixture(scope="module")
def encoder(release) -> D1Lfm3Backend:
    """The backend without a model, at the context's limit (the release cuts nothing)."""
    from tokenizers import Tokenizer

    found = settings(release)
    tokenizer = Tokenizer.from_file(str(release / "tokenizer.json"))
    capabilities = _capabilities(INPUT_TOKEN_LIMIT, True, found.max_pixels)
    return D1Lfm3Backend("tokens", None, tokenizer, found, capabilities, release)


@pytest.fixture(scope="module")
def tiny(release, tmp_path_factory):
    with mx.stream(mx.cpu):
        folder = write_d1lfm3(tmp_path_factory.mktemp("d1lfm3") / "tiny-d1lfm3", files_from=release)
        yield mlx_decision.load(folder, check_memory=False)


def body(rid: str) -> dict:
    case = CASES[rid]
    data = {"state": case["state"], "questions": case["questions"]}
    if case.get("images"):
        data["images"] = [open_image(spec) for spec in case["images"]]
    return data


def matches(value, expected) -> bool:
    """A stored text or id list, or its sha256 and length (``make_d1lfm3_reference``)."""
    if not isinstance(expected, dict):
        return value == expected
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode()).hexdigest()
        return len(value) == expected["chars"] and digest == expected["sha256"]
    digest = hashlib.sha256(json.dumps(value).encode()).hexdigest()
    return len(value) == expected["tokens"] and digest == expected["sha256"]


def answer_values(answer) -> list[float]:
    """An answer's probabilities in the release's order (noul: yes, then no)."""
    if answer.type == "noul":
        return [answer.noul, 1 - answer.noul]
    return list(answer.probabilities.values())


@pytest.mark.parametrize("rid", ACCEPTED)
def test_prompts_and_sets_equal_the_release(encoder, rid):
    expected = RESULTS[rid]
    request = parse_request(body(rid))
    _, pictures = encoder.open_pictures(request)
    assert pictures.tokens == [p["grid"][1] * p["grid"][2] // 4 for p in expected["pictures"]]
    encoded = encoder.encode(request, pictures.tokens)
    assert not encoded.truncated
    assert matches(prefix(request.state, pictures.tokens), expected["prefix"])
    for qid, text in zip(request.questions, expected["suffixes"], strict=True):
        assert matches(suffix(encoded.questions[qid], encoder.tokenizer), text), qid
    assert len(encoded.sets) == len(expected["sets"])
    ids = list(request.questions)
    for packed, want in zip(encoded.sets, expected["sets"], strict=True):
        first = int(want["first"])
        assert packed.question_ids == ids[first : first + len(want["spans"])]
        assert packed.state_length == want["state_length"]
        assert packed.spans == want["spans"]
        assert packed.ends == want["ends"]
        assert matches(packed.ids, want["ids"])
    for qid, groups in zip(request.questions, expected["groups"], strict=True):
        assert option_token_ids(encoded.questions[qid], encoder.tokenizer) == groups, qid
    if "input_tokens" in expected:
        assert encoded.input_tokens == expected["input_tokens"]


@pytest.mark.parametrize("rid", PICTURES)
def test_pictures_are_cut_into_the_releases_patches(encoder, rid):
    opened, pictures = encoder.open_pictures(parse_request(body(rid)))
    encoder.prepare_pictures(opened, pictures)
    for prepared, want in zip(pictures.prepared, RESULTS[rid]["pictures"], strict=True):
        assert list(prepared.grid) == want["grid"]
        means = prepared.pixel_values.mean(-1)
        assert np.abs(means - np.array(want["patch_means"])).max() < PIXEL_TOLERANCE


def test_picture_caps_equal_the_releases():
    for case in FIXTURE["picture_caps"]:
        assert list(picture_size(*case["size"], case["max_pixels"])) == case["capped"], case


@pytest.mark.parametrize("rid", ANSWERED)
def test_tiny_model_answers_as_the_release(tiny, rid):
    with mx.stream(mx.cpu):
        result = tiny.decide_request(body(rid))
    assert not result.truncated
    assert result.usage.input_tokens == RESULTS[rid]["input_tokens"]
    for qid, expected in RESULTS[rid]["probabilities"].items():
        got = answer_values(result.answers[qid])
        assert max(abs(a - b) for a, b in zip(got, expected, strict=True)) < TOLERANCE, qid


@pytest.mark.parametrize("rid", ["lf_thirty_three", "image_two"])
def test_sets_in_one_batch_answer_as_each_set_alone(tiny, rid):
    backend = tiny.backend
    with mx.stream(mx.cpu):
        request = parse_request(body(rid))
        opened, pictures = backend.open_pictures(request)
        encoded = backend.encode(request, pictures.tokens)
        backend.prepare_pictures(opened, pictures)
        features = backend.picture_features(pictures)
        together = backend.hidden_states(encoded.sets, features)
        alone = [h for packed in encoded.sets for h in backend.hidden_states([packed], features)]
    assert len(together) == len(alone) == len(request.questions)
    for a, b in zip(together, alone, strict=True):
        assert mx.abs(a - b).max().item() < 1e-4


def test_seventeen_questions_make_two_sets_each_with_the_state(encoder):
    encoded = encoder.encode(parse_request(body("lf_seventeen")))
    first, second = encoded.sets
    assert len(first.question_ids) == 16 and len(second.question_ids) == 1
    assert second.ids[: second.state_length] == first.ids[: first.state_length]


def adapted(question_id: str, question: dict) -> dict:
    """The question as the backend reads it (instructions from the id, levels as text)."""
    question = dict(question)
    if not str(question.get("instructions") or "").strip():
        question["instructions"] = question_id
    if question["type"] == "score":
        question["criteria"] = [
            level if isinstance(level, str) else str(level) for level in question["criteria"]
        ]
    return question


@pytest.mark.parametrize("rid", REFUSED)
def test_what_the_release_refuses_is_adapted_or_refused_by_our_checks(tiny, rid):
    """Missing or blank instructions become the question id and object levels their
    text, as d1-3B reads them; fewer than two levels fail the request's checks."""
    case = CASES[rid]
    if any(
        q["type"] == "score" and len(q.get("criteria") or []) < 2
        for q in case["questions"].values()
    ):
        with pytest.raises(DecisionError, match="criteria"):
            tiny.decide_request(body(rid))
        return
    explicit = {qid: adapted(qid, q) for qid, q in case["questions"].items()}
    with mx.stream(mx.cpu):
        served = tiny.decide_request(body(rid))
        expected = tiny.decide_request({**body(rid), "questions": explicit})
    # The legend repeats the levels as sent, so only the probabilities compare.
    for qid, answer in served.answers.items():
        assert answer_values(answer) == answer_values(expected.answers[qid]), qid


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("limit", [32768, 8192, 1024])
def test_long_states_are_cut_to_fit_and_keep_their_ends(release, encoder, limit):
    """The backend without weights, on every text request of the parity sets."""
    backend = D1Lfm3Backend(
        "tokens",
        None,
        encoder.tokenizer,
        encoder.settings,
        _capabilities(limit, True, encoder.settings.max_pixels),
        release,
    )
    cuts = refused = 0
    for rid in ACCEPTED:
        if CASES[rid].get("images"):
            continue
        request = parse_request(body(rid))
        try:
            encoded = backend.encode(request)
        except DecisionError as error:
            # Only a set too long even with an empty state.
            assert error.param.startswith("questions."), (rid, error)
            refused += 1
            continue
        assert all(len(packed.ids) <= limit for packed in encoded.sets), rid
        if not encoded.truncated:
            continue
        cuts += 1
        packed = max(encoded.sets, key=lambda p: len(p.ids))
        state = backend.tokenizer.decode(
            packed.ids[: packed.state_length], skip_special_tokens=False
        )
        start, end = state.split(CUT_MARK, 1)
        whole = backend.tokenizer.decode(
            encode(backend.tokenizer, prefix(request.state)), skip_special_tokens=False
        )
        # The kept text is the original's start and end (but for a token at each cut).
        assert whole.startswith(start[:-8]), rid
        assert whole.split("\n\n\nQUESTION:\n")[0].endswith(end.split("\n\n\nQUESTION:\n")[0][8:])
        # Not cut far below what the largest set leaves room for.
        assert len(packed.ids) > limit - 64, rid
    assert cuts > 0
    print(f"limit {limit}: {cuts} cut, {refused} refused")


def test_the_bf16_fixture_is_for_the_same_requests():
    if not BF16_FILE.exists():
        pytest.skip("no reference-bf16.json (scripts/make_d1lfm3_reference.py --model)")
    bf16 = json.loads(BF16_FILE.read_text())
    assert bf16["source"]["revision"] == FIXTURE["source"]["revision"]
    assert set(bf16["requests"]) == set(RESULTS)


@pytest.mark.slow
@pytest.mark.weights
def test_bf16_answers_as_the_release():
    """About two minutes: the release folder answers every request as served (sets
    of 16), against the release's own code in bf16 on MPS."""
    root = os.environ.get("MLX_DECISION_MODELS")
    path = Path(root) / "d1-2B" if root else None
    if path is None or not path.is_dir():
        pytest.skip("no d1-2B in MLX_DECISION_MODELS")
    if not BF16_FILE.exists():
        pytest.skip("no reference-bf16.json (scripts/make_d1lfm3_reference.py --model)")
    reference = json.loads(BF16_FILE.read_text())["requests"]
    model = mlx_decision.load(path)
    differences, worst, decided = [], (0.0, None), []
    for rid, entry in reference.items():
        if "probabilities" not in entry:
            continue
        result = model.decide_request(body(rid))
        assert not result.truncated, rid
        for qid, expected in entry["probabilities"].items():
            got = answer_values(result.answers[qid])
            difference = max(abs(a - b) for a, b in zip(got, expected, strict=True))
            differences.append(difference)
            worst = max(worst, (difference, f"{rid}.{qid}"))
            ranked = sorted(expected, reverse=True)
            if got.index(max(got)) != expected.index(max(expected)) and (
                ranked[0] - ranked[1] > BF16_MARGIN
            ):
                decided.append(f"{rid}.{qid}")
    mean = statistics.mean(differences)
    print(f"{len(differences)} questions: max {worst[0]:.4f} ({worst[1]}), mean {mean:.4f}")
    assert worst[0] <= BF16_MAX, worst
    assert mean <= BF16_MEAN, mean
    assert not decided, decided
