"""Fixtures shared by the parity tests."""

import pytest
from test_parity import CASE_IDS, request

from mlx_decision.types import parse_request

WARM_UP = {"warm_up": {"type": "noul", "instructions": "Is this a test?"}}


@pytest.fixture(scope="session")
def parity_runs(clef):
    """Backend outputs for every request of the parity set, computed once.

    ``first``: the request as it comes (reuse on, so its prefix is computed
    and kept). ``again``: the same request after a question on the same state,
    continuing from the kept prefix, and whether both were cache hits.
    """
    backend = clef.backend
    first, again = {}, {}
    for case_id in CASE_IDS:
        original = request(case_id)
        first[case_id] = backend.score(original)
        body = {"state": original.state, "questions": WARM_UP}
        if original.images:
            body["images"] = original.images
        hits = backend.prefix_cache.hits
        backend.score(parse_request(body))
        output = backend.score(original)
        again[case_id] = (output, backend.prefix_cache.hits - hits == 2)
    return first, again


@pytest.fixture(scope="session")
def outputs(parity_runs):
    return parity_runs[0]
