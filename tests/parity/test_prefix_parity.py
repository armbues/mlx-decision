"""Prefix reuse on clef-flash, over the parity set.

The parity test's outputs are computed with reuse on (the default), so their
match with the reference covers a first request. Each request is then asked
again after another question on the same state: both continue from the kept
prefix, and the request must give exactly the same numbers as the first time.
"""

import pytest
from test_parity import CASE_IDS, REFERENCE

pytestmark = pytest.mark.weights


def test_reuse_is_on_for_the_parity_outputs(clef):
    assert clef.backend.prefix_cache is not None


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_a_kept_prefix_gives_the_same_answers(parity_runs, case_id):
    first, again = parity_runs
    output, reused = again[case_id]
    assert output.probabilities == first[case_id].probabilities
    assert output.input_tokens == first[case_id].input_tokens
    if not REFERENCE[case_id]["truncated"]:
        # A state that is not cut is the same prefix whatever the questions.
        assert reused, "the kept prefix was not reused"


def test_most_requests_are_reused(parity_runs):
    reused = sum(hit for _, hit in parity_runs[1].values())
    assert reused >= len(CASE_IDS) - sum(r["truncated"] for r in REFERENCE.values())
