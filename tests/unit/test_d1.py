"""The d1 family on a tiny random release (word-level tokenizer)."""

import json

import mlx.core as mx
import numpy as np
import pytest
from typer.testing import CliRunner

import mlx_decision
from mlx_decision.cli import app
from mlx_decision.errors import DecisionError
from mlx_decision.models.d1.prompt import prefix, suffix
from mlx_decision.models.d1.readout import probabilities
from mlx_decision.registry import detect_family
from mlx_decision.types import parse_request
from tiny_models import write_d1

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Charges", "technical": "Faults", "sales": None},
    },
    "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "angry"]},
}
STATE = "I was charged twice , please refund ."


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return write_d1(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def model(folder):
    return mlx_decision.load(folder)


def test_detected_as_d1_and_not_other_lfm2_vl_folders(folder, tmp_path):
    assert detect_family(folder).name == "d1"
    config = json.loads((folder / "config.json").read_text())
    del config["auto_map"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not a supported decision model"):
        detect_family(tmp_path)


def test_info(model):
    info = model.info()
    assert info["family"] == "d1"
    assert info["precision"] == "float32"
    assert info["max_input_tokens"] == 32768
    assert info["max_choice_options"] == 255
    assert info["max_score_levels"] == 10
    assert info["supports_images"] is False
    assert info["requires_instructions"] is False


def full_vocabulary_answer(model, state, question_id, question):
    """The readout over the whole vocabulary's log-probabilities, as the release reads it."""
    backend = model.backend
    tokenizer = backend.tokenizer
    data = {**question, "instructions": question.get("instructions") or question_id}
    ids = tokenizer.encode(prefix(state) + suffix(data, tokenizer), add_special_tokens=False).ids
    hidden = backend.model(mx.array([ids]))[0, -1]
    logits = (backend.model.model.embed_tokens.as_linear(hidden)).astype(mx.float32)
    logprobs = np.array(logits - mx.logsumexp(logits))
    return probabilities(data, tokenizer, logprobs)


@pytest.mark.parametrize("question_id", list(QUESTIONS))
def test_one_question_reads_the_option_tokens_as_over_the_whole_vocabulary(model, question_id):
    question = QUESTIONS[question_id]
    result = model.decide(STATE, {question_id: question})
    answer = result.answers[question_id]
    got = list(answer.probabilities.values()) if question["type"] != "noul" else [answer.noul]
    want = full_vocabulary_answer(model, STATE, question_id, question)
    assert np.allclose(got, want[: len(got)], atol=1e-4)


def test_several_questions_share_the_tokenized_prefix(model):
    backend = model.backend
    encoded, length, cut = backend.encode(parse_request({"state": STATE, "questions": QUESTIONS}))
    assert not cut
    tokenizer = backend.tokenizer
    shared = tokenizer.encode(prefix(STATE), add_special_tokens=False).ids
    assert length == len(shared)
    for question in encoded:
        tail = tokenizer.encode(suffix(question.question, tokenizer), add_special_tokens=False).ids
        assert question.input_ids == shared + tail
    result = model.decide(STATE, QUESTIONS)
    assert result.usage.input_tokens == sum(len(q.input_ids) for q in encoded)
    assert list(result.answers["team"].probabilities) == ["billing", "technical", "sales"]
    assert sum(result.answers["anger"].probabilities.values()) == pytest.approx(1)


def test_missing_instructions_become_the_question_id(model):
    backend = model.backend
    question = {"type": "noul"}
    (encoded,), _, _ = backend.encode(
        parse_request({"state": STATE, "questions": {"urgent": question}})
    )
    assert encoded.question["instructions"] == "urgent"
    assert "QUESTION:\nurgent\n" in prefix(STATE) + suffix(encoded.question, backend.tokenizer)
    assert 0 < model.decide(STATE, {"urgent": question}).answers["urgent"].noul < 1


def test_limits(model):
    many = {f"o{i}": f"option {i}" for i in range(256)}
    with pytest.raises(DecisionError, match="at most 255 options"):
        model.decide(STATE, {"q": {"type": "choice", "instructions": "Pick", "criteria": many}})
    levels = [f"level {i}" for i in range(11)]
    with pytest.raises(DecisionError):
        model.decide(STATE, {"q": {"type": "score", "instructions": "Rate", "criteria": levels}})
    with pytest.raises(DecisionError, match="does not support images") as error:
        model.decide_request({"state": STATE, "questions": QUESTIONS, "images": ["data:x"]})
    assert error.value.param == "images"


def test_a_long_state_is_cut_to_fit_keeping_its_start_and_end(folder):
    model = mlx_decision.load(folder, max_input_tokens=80)
    words = ["refund"] * 40 + ["billing"] * 60 + ["angry"] * 40
    state = " ".join(words)
    text = model.backend.cut(state, 20)
    start, end = text.split("\n...\n")
    assert start.split() == ["refund"] * 10 and end.split() == ["angry"] * 10
    assert model.backend.cut(state, 200) is None  # fits
    encoded, shared, cut = model.backend.encode(
        parse_request({"state": state, "questions": QUESTIONS})
    )
    assert cut
    assert all(len(q.input_ids) <= 80 for q in encoded)
    # One cut state for every question.
    assert len({tuple(q.input_ids[:shared]) for q in encoded}) == 1
    assert model.decide(state, QUESTIONS).truncated
    assert not model.decide(STATE, QUESTIONS).truncated


def test_only_a_question_too_long_on_its_own_is_refused(folder):
    model = mlx_decision.load(folder, max_input_tokens=30)
    many = {f"o{i}": "billing technical sales refund" for i in range(12)}
    question = {"type": "choice", "instructions": "Which team ?", "criteria": many}
    with pytest.raises(DecisionError, match="the limit is 30") as error:
        model.decide(" ".join(["refund"] * 100), {"q": question})
    assert error.value.param == "questions.q"
    # A short question on the same state: the state is cut instead.
    model = mlx_decision.load(folder, max_input_tokens=60)
    result = model.decide(" ".join(["refund"] * 100), {"refund": QUESTIONS["refund"]})
    assert result.truncated
    assert result.usage.input_tokens <= 60


def test_load_options(folder):
    for bad in (0, 32769):
        with pytest.raises(ValueError, match="max_input_tokens"):
            mlx_decision.load(folder, max_input_tokens=bad)
    with pytest.raises(ValueError, match="d1 models do not take noul_floor"):
        mlx_decision.load(folder, noul_floor=0.9)


def test_run(folder):
    result = CliRunner().invoke(
        app, ["run", "-m", str(folder), "-s", STATE, "--noul", "refund=Wants money back", "--json"]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["model"] == folder.name
    assert 0 < body["answers"]["refund"]["noul"] < 1


LONG_STATE = " ".join(
    ["refund", "billing", "angry", "sales"] * 160
)  # over KEEP_PREFIX and SHARE_TOKENS


def as_list(answer) -> list[float]:
    return [answer.noul] if answer.type == "noul" else list(answer.probabilities.values())


def one_at_a_time(model, state) -> dict[str, list[float]]:
    return {
        qid: as_list(model.decide(state, {qid: q}).answers[qid]) for qid, q in QUESTIONS.items()
    }


@pytest.fixture
def cpu():
    """Float32 on the CPU: no TF32 (load with check_memory=False, which needs the GPU)."""
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("state", [STATE, LONG_STATE], ids=["short", "long"])
def test_questions_together_answer_as_one_at_a_time(folder, cpu, state):
    """Short: every question's whole prompt a row of one batch. Long: the prefix once,
    the questions' own tokens as rows continuing it."""
    model = mlx_decision.load(folder, prefix_cache_gb=0, check_memory=False)
    alone = one_at_a_time(model, state)
    together = model.decide(state, QUESTIONS).answers
    for qid, values in alone.items():
        assert np.allclose(as_list(together[qid]), values, atol=1e-5), qid


def test_batches_within_the_token_budget(folder, cpu, monkeypatch):
    import mlx_decision.models.d1.model as d1

    model = mlx_decision.load(folder, prefix_cache_gb=0, check_memory=False)
    want = model.decide(LONG_STATE, QUESTIONS).answers
    monkeypatch.setattr(d1, "BATCH_TOKENS", 1)  # one row per pass
    got = model.decide(LONG_STATE, QUESTIONS).answers
    for qid in QUESTIONS:
        assert np.allclose(as_list(got[qid]), as_list(want[qid]), atol=1e-5), qid


def test_a_long_prefix_is_kept_for_the_next_request(folder, cpu):
    model = mlx_decision.load(folder, check_memory=False)
    cache = model.backend.prefix_cache
    first = {"refund": QUESTIONS["refund"]}
    second = {k: v for k, v in QUESTIONS.items() if k != "refund"}
    model.decide(LONG_STATE, first)
    assert (cache.hits, cache.misses, len(cache)) == (0, 1, 1)
    kept = model.decide(LONG_STATE, second).answers
    assert (cache.hits, len(cache)) == (1, 1)
    fresh = mlx_decision.load(folder, prefix_cache_gb=0, check_memory=False)
    fresh = fresh.decide(LONG_STATE, second).answers
    for qid in second:
        assert np.allclose(as_list(kept[qid]), as_list(fresh[qid]), atol=1e-5), qid
    # Short prefixes are part of every row, not kept.
    model.decide(STATE, QUESTIONS)
    assert len(cache) == 1


def test_prefix_cache_off(folder):
    model = mlx_decision.load(folder, prefix_cache_gb=0, check_memory=False)
    assert model.backend.prefix_cache is None
    with pytest.raises(ValueError, match="prefix_cache_gb"):
        mlx_decision.load(folder, prefix_cache_gb=-1)


def test_images_inside_the_state_are_read_as_text(model):
    """Text only: nothing is lifted out of the state, so nothing is refused."""
    picture = "data:image/png;base64," + "iVBORw0KGgo" + "A" * 80
    state = {"note": "see the photo", "photo": picture}
    encoded, _, _ = model.backend.encode(
        parse_request({"state": state, "questions": {"refund": QUESTIONS["refund"]}})
    )
    text = model.backend.tokenizer.decode(encoded[0].input_ids, skip_special_tokens=False)
    assert "[image" not in text
    assert 0 < model.decide(state, {"refund": QUESTIONS["refund"]}).answers["refund"].noul < 1
