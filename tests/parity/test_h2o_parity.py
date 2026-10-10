"""H2O-Lightning on MLX against the release's own serving code, on the parity set.

``h2o/reference.json`` is written by ``scripts/make_h2o_reference.py``: the
release's shim with transformers in vLLM's place, so the token ids of every
question (its real tokenizer) and the answers of the tiny random model of
``tiny_models.write_h2o`` (CPU, float32). The token and end-to-end tests
need the release's small files in the Hugging Face cache (the script
fetches them) and skip otherwise; the tiny-model test feeds the stored ids
and always runs.

The model runs on the CPU here: on the GPU, float32 matmuls use TF32, which
moves the tiny model's probabilities by up to about 1e-3.
"""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_decision
from mlx_decision.backbones.qwen3_5.load import load_text_model
from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
from mlx_decision.errors import DecisionError
from mlx_decision.models.h2o.model import H2OBackend, ServeConfig
from mlx_decision.models.h2o.prompt import PromptSettings
from mlx_decision.types import parse_request
from tiny_models import write_h2o

HERE = Path(__file__).parent
CASES = []
for name in ("requests.json", "marker/requests.json", "h2o/requests.json"):
    CASES += json.loads((HERE / name).read_text())
REQUESTS = {case["id"]: case for case in CASES}
FIXTURE = json.loads((HERE / "h2o" / "reference.json").read_text())
REFERENCE = FIXTURE["results"]
TEXT_IDS = [case["id"] for case in CASES if not case.get("images")]
# Largest allowed difference of any probability against the reference (CPU,
# float32).
TOLERANCE = 1e-5
# For the full path with the real tokenizer: one request per prompt shape.
END_TO_END = [
    "ticket",
    "single_noul_no_instructions",
    "noul_only_true_criterion",
    "choice_null_descriptions",
    "score_object_levels",
    "instructions_object",
    "instructions_array",
    "nested_state",
    "array_state_unicode",
    "japanese",
    "lang_hi",
    "special_tokens_in_state",
    "number_state",
    "bool_state",
    "empty_string_state",
    "whitespace_and_newlines",
    "h2o_empty_instructions",
    "h2o_choice_empty_description",
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
                ignore_patterns=["model.safetensors", "adapter/*", "assets/*"],
                local_files_only=True,
            )
        )
    except Exception:
        pytest.skip("H2O release files not cached (scripts/make_h2o_reference.py fetches them)")
    if not all((path / name).exists() for name in ("tokenizer.json", "serve_config.json")):
        pytest.skip("H2O release files incomplete in the cache")
    return path


@pytest.fixture(scope="module")
def encoder(release) -> H2OBackend:
    """A backend without weights: enough to render and tokenize."""
    config = ServeConfig.read(release)
    return H2OBackend("tokens", None, load_tokenizer(release), config, 40960, release)


def request(case_id: str):
    case = REQUESTS[case_id]
    return parse_request({"state": case["state"], "questions": case["questions"]})


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def test_label_ids_equal_the_release(encoder):
    assert dict(zip(encoder.labels, encoder.label_ids, strict=True)) == FIXTURE["meta"]["label_ids"]


@pytest.mark.parametrize("case_id", TEXT_IDS)
def test_token_ids_equal_the_release(encoder, case_id):
    expected = REFERENCE[case_id]
    if "refused" in expected:
        # Refused by the wire format before the release would see it.
        with pytest.raises(DecisionError):
            request(case_id)
        return
    encoded, cut = encoder.encode(request(case_id))
    assert cut == any(e.get("state_cut") for e in expected.values())
    for question in encoded:
        reference = expected[question.question_id]
        assert len(question.input_ids) == reference["tokens"], question.question_id
        assert digest(question.input_ids) == reference["ids_sha256"], question.question_id
        assert len(question.option_ids) == reference["options"]


@pytest.mark.parametrize("limit", [1024, 8192])
def test_a_lowered_limit_cuts_states_so_prompts_fit(release, limit):
    backend = H2OBackend("tokens", None, load_tokenizer(release), ServeConfig.read(release), limit)

    def tokens(case_id):
        return max((e["tokens"] for e in REFERENCE[case_id].values() if "tokens" in e), default=0)

    long = [
        case_id
        for case_id in TEXT_IDS
        if "refused" not in REFERENCE[case_id]
        and tokens(case_id) > limit
        # long for its options, which are never cut: checked below
        and case_id != "choice_255_options"
    ]
    assert len(long) >= 5
    for case_id in long:
        encoded, cut = backend.encode(request(case_id))
        assert cut, case_id
        assert limit - 16 < max(len(e.input_ids) for e in encoded) <= limit, case_id
    # The questions alone over the limit: refused, not cut away.
    if limit == 1024:
        with pytest.raises(DecisionError, match="the limit is 1024"):
            backend.encode(request("choice_255_options"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> H2OBackend:
    """The tiny model with the release's labels, temperatures and floor, no tokenizer."""
    meta = FIXTURE["meta"]
    folder = write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o")
    config = ServeConfig(
        labels=list(meta["label_ids"]),
        temperatures=meta["temperature"],
        noul_floor=meta["noul_floor"],
        prompt=PromptSettings.from_config({}),
    )
    backbone = load_text_model(folder, lm_head=False)
    ids = list(meta["label_ids"].values())
    return H2OBackend("tiny", backbone, None, config, 40960, folder, label_ids=ids)


def answer_values(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [answer["noul"]]
    return list(answer["probabilities"].values())


def test_tiny_model_answers_as_the_release(tiny):
    """Every text question with stored ids: one pass each, about 20 ms on the CPU."""
    checked = 0
    for case_id in TEXT_IDS:
        for question_id, expected in REFERENCE[case_id].items():
            if "ids" not in expected:
                continue
            kind = expected["answer"]["type"]
            with mx.stream(mx.cpu):
                values = tiny.probabilities(expected["ids"], expected["options"], kind)
            # A yes/no answer is P(true), the first option; before the floor.
            got = values[:1] if kind == "noul" else values
            want = answer_values(expected["answer_no_floor"])
            assert got == pytest.approx(want, abs=TOLERANCE), f"{case_id}.{question_id}"
            checked += 1
    assert checked > 150


@pytest.fixture(scope="module")
def tiny_with_tokenizer(release, tmp_path_factory):
    folder = write_h2o(tmp_path_factory.mktemp("h2o") / "tiny-h2o", files_from=release)
    return mlx_decision.load(folder)


@pytest.mark.parametrize("case_id", END_TO_END)
def test_tiny_model_end_to_end(tiny_with_tokenizer, case_id):
    """The whole path: prompt, real tokenizer, passes, the floor, answers."""
    with mx.stream(mx.cpu):
        result = tiny_with_tokenizer.decide_request(request(case_id))
    assert not result.truncated
    tokens = 0
    for question_id, answer in result.answers.items():
        expected = REFERENCE[case_id][question_id]
        tokens += expected["tokens"]
        if answer.type == "noul":
            assert answer.noul == pytest.approx(expected["answer"]["noul"], abs=TOLERANCE)
        else:
            assert answer.probabilities == pytest.approx(
                expected["answer"]["probabilities"], abs=TOLERANCE
            )
            if answer.type == "choice":
                assert answer.choice == expected["answer"]["choice"]
    assert result.usage.input_tokens == tokens
