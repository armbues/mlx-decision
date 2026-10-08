"""A backend with fixed, made-up probabilities and no weights.

It is a family of its own, found through the marker file like a converted
model, so it runs through ``load``, ``run`` and ``server`` unchanged.
"""

import json
from pathlib import Path

from mlx_decision.backend import BackendOutput, Capabilities
from mlx_decision.registry import MARKER_FILE, register_family
from mlx_decision.types import Choice, Noul, Request

FAMILY = "fake"


class FakeBackend:
    """Gives the first option the most weight: probabilities fall as n, n-1, ..., 1.

    A noul answers 0.75. ``fixed`` overrides the probabilities of a question
    id. The input length is the number of words in the state, which is cut to
    ``max_input_tokens``, if given.
    """

    def __init__(
        self,
        name: str = "fake",
        fixed: dict[str, dict[str, float]] | None = None,
        max_input_tokens: int | None = None,
        **capabilities,
    ):
        self.name = name
        self.fixed = fixed or {}
        self.capabilities = Capabilities(max_input_tokens=max_input_tokens, **capabilities)

    def score(self, request: Request) -> BackendOutput:
        state = request.state if isinstance(request.state, str) else json.dumps(request.state)
        words = len(state.split())
        limit = self.capabilities.max_input_tokens
        truncated = limit is not None and words > limit
        return BackendOutput(
            probabilities={
                question_id: self.fixed.get(question_id) or _falling(question)
                for question_id, question in request.questions.items()
            },
            input_tokens=min(words, limit) if truncated else words,
            truncated=truncated,
        )


def _falling(question) -> dict[str, float]:
    if isinstance(question, Noul):
        return {"true": 0.75, "false": 0.25}
    if isinstance(question, Choice):
        options = [str(key) for key in question.criteria]
    else:
        options = [str(index) for index in range(len(question.criteria))]
    n = len(options)
    total = n * (n + 1) / 2
    return {option: (n - index) / total for index, option in enumerate(options)}


def load(path: Path, **options) -> FakeBackend:
    config = json.loads((path / MARKER_FILE).read_text())
    config.pop("family")
    return FakeBackend(**{**config, **options})


def write_model(folder: Path, **config) -> Path:
    """Write a fake model folder that ``mlx_decision.load`` recognises."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MARKER_FILE).write_text(json.dumps({"family": FAMILY, **config}))
    return folder


def write_sized_model(folder: Path, weights: int) -> Path:
    """A fake model whose memory check counts ``weights`` bytes (plus 10%)."""
    import mlx.core as mx

    write_model(folder)
    mx.save_safetensors(str(folder / "model.safetensors"), {"w": mx.zeros(weights, mx.uint8)})
    return folder


register_family(FAMILY, detect=lambda path: False, loader="fake_backend:load")
