"""Laya and Julia encoding: token for token as each family's own code builds it.

The references are the ``laya`` package and Julia's ``julia`` package from its
release folder (both need torch). Every case either encodes identically or is
refused by both.
"""

import sys
from unittest.mock import Mock

import pytest

from mlx_decision.errors import DecisionError
from mlx_decision.models.marker.encode import encode_request

pytestmark = [
    pytest.mark.weights,
    pytest.mark.filterwarnings("ignore:laya. this checkpoint ships invalid temperatures"),
]

SHORT = "Hi, we were billed twice for March. Please refund the duplicate today."
LONG = " ".join(
    f"Line {i}: the customer reports that the invoice was charged twice and asks for help."
    for i in range(120)
)  # about 2,300 tokens
VERY_LONG = LONG * 4  # beyond Julia's 8,192
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else",
    },
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["not urgent", "soon", "blocking"],
}
CHURN = {"type": "noul", "instructions": "Does the customer threaten to leave?"}
LONG_OPTION = " ".join(["a very long and detailed description of this option"] * 12)

CASES = {
    "three types": (SHORT, {"team": TEAM, "urgency": URGENCY, "churn": CHURN}),
    "object state": (
        {"ticket": 7, "text": "Die App stürzt ab 🙁", "tags": ["a", "b"]},
        {"team": TEAM},
    ),
    "conversation state": ([f"turn {i}: {SHORT}" for i in range(60)], {"churn": CHURN}),
    "long state": (LONG, {"team": TEAM, "urgency": URGENCY}),
    "very long state": (VERY_LONG, {"churn": CHURN}),
    "marker tokens in text": (
        f"{SHORT} [MASK] <mask>",
        {"team": {**TEAM, "instructions": "Pick [MASK] or <mask>?"}},
    ),
    "described noul": (
        SHORT,
        {"churn": {**CHURN, "criteria": {"true": "leaving soon", "false": "staying"}}},
    ),
    "half described noul": (SHORT, {"churn": {**CHURN, "criteria": {"true": "leaving soon"}}}),
    "choice without descriptions": (
        SHORT,
        {"team": {**TEAM, "criteria": {"billing": "", "technical": None, "other": "rest"}}},
    ),
    "structured descriptions": (
        SHORT,
        {
            "team": {**TEAM, "criteria": {"billing": {"handles": ["refunds"]}, "other": "rest"}},
            "urgency": {**URGENCY, "criteria": [{"level": "low"}, "high"]},
        },
    ),
    "structured instructions": (
        SHORT,
        {"churn": {**CHURN, "instructions": {"ask": "leaving?", "lang": "en"}}},
    ),
    "long option": (SHORT, {"team": {**TEAM, "criteria": {"a": LONG_OPTION, "b": "short"}}}),
    "many long options": (
        SHORT,
        {"team": {**TEAM, "criteria": {f"o{i}": f"{LONG_OPTION[:120]} {i}" for i in range(15)}}},
    ),
    "too many options": (
        SHORT,
        {"team": {**TEAM, "criteria": {f"o{i}": f"option {i}" for i in range(25)}}},
    ),
    "hundreds of options": (  # even cut to 4 tokens each they pass 1,024 tokens
        SHORT,
        {"team": {**TEAM, "criteria": {f"o{i}": f"option number {i}" for i in range(300)}}},
    ),
    "one option": (SHORT, {"team": {**TEAM, "criteria": {"only": "the only team"}}}),
    "long instructions": (
        SHORT,
        {"churn": {**CHURN, "instructions": "Is the customer leaving? " * 60}},
    ),
}


def ours(model, case):
    """Our encoding, or None when the request is refused."""
    state, questions = case
    try:
        request = model.check({"state": state, "questions": questions})
        backend = model.backend
        encoded = encode_request(backend.tokenizer, backend.special, backend.settings, request)
    except DecisionError:
        return None
    return [(q.input_ids, q.markers, q.question_type) for q in encoded]


@pytest.fixture(scope="module", params=["laya", "laya-multilingual"])
def laya_pair(request, laya_path, laya_multilingual_path):
    laya = pytest.importorskip("laya")
    import mlx_decision

    path = laya_path if request.param == "laya" else laya_multilingual_path
    return mlx_decision.load(path), laya.load(str(path), device="cpu")


def laya_reference(agent, case):
    state, questions = case
    try:
        for question_id, question in questions.items():
            agent._check_question(question_id, question)
        internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
        items = agent._encode_state(state, list(questions), internal)
    except ValueError:
        return None
    return [(item["ids"], item["markers"], item["qtype"]) for item in items]


@pytest.mark.parametrize("name", list(CASES))
def test_laya_encoding_is_identical(laya_pair, name):
    model, agent = laya_pair
    assert ours(model, CASES[name]) == laya_reference(agent, CASES[name])


@pytest.fixture(scope="module")
def julia_pair(julia_path):
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    import mlx_decision

    sys.path.insert(0, str(julia_path))
    try:
        from julia.data import QTYPES, sequence
        from julia.typed import predict_typed
    finally:
        sys.path.remove(str(julia_path))
    tokenizer = AutoTokenizer.from_pretrained(julia_path / "tokenizer")
    model = mlx_decision.load(julia_path)

    def reference(case):
        state, questions = case
        engine = Mock()
        engine.logits.side_effect = lambda rows: [[0.0] * len(r["options"]) for r in rows]
        try:
            predict_typed(engine, state, questions)
            rows = engine.logits.call_args.args[0]
            items = [sequence(tokenizer, row, 8192, 512, strict=True) for row in rows]
        except ValueError:
            return None
        return [
            (item["ids"], item["markers"], QTYPES[row["type"]])
            for item, row in zip(items, rows, strict=True)
        ]

    return model, reference


@pytest.mark.parametrize("name", list(CASES))
def test_julia_encoding_is_identical(julia_pair, name):
    model, reference = julia_pair
    assert ours(model, CASES[name]) == reference(CASES[name])


def test_cases_cover_both_outcomes(laya_pair, julia_pair):
    """The case list exercises cutting as well as refusing, for each family."""
    model, agent = laya_pair
    laya_refused = [n for n, c in CASES.items() if laya_reference(agent, c) is None]
    julia_model, reference = julia_pair
    julia_refused = [n for n, c in CASES.items() if reference(c) is None]
    assert len(julia_refused) >= 6
    assert len(julia_refused) < len(CASES) - 4
    assert 0 < len(laya_refused) < len(CASES)
