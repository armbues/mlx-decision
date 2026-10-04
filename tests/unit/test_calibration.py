"""The built-in calibration requests."""

import json

from mlx_decision.calibration import calibration_requests
from mlx_decision.types import parse_request


def test_requests_are_valid_and_cover_all_question_types():
    requests = [parse_request(request) for request in calibration_requests()]
    assert len(requests) >= 8
    types = {q.type for r in requests for q in r.questions.values()}
    assert types == {"noul", "choice", "score"}
    assert {type(r.state).__name__ for r in requests} >= {"str", "dict", "list"}


def test_requests_are_deterministic():
    assert json.dumps(calibration_requests()) == json.dumps(calibration_requests())


def test_no_overlap_with_the_parity_set():
    from pathlib import Path

    parity = (Path(__file__).parents[1] / "parity" / "requests.json").read_text()
    for request in calibration_requests():
        state = request["state"]
        sample = state if isinstance(state, str) else json.dumps(state)
        assert sample[:60] not in parity
