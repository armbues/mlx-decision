"""The d1-3B backend: LFM2 text model, with the prompt and readout of ``prompt`` and ``readout``.

Each question is its own prompt: the state (``prefix``) and the question
(``suffix``). With one question the whole prompt is tokenized at once; with
several, the prefix is tokenized once and each question's own text after it,
which gives the release's token ids in both cases. The answer is read at the
prompt's last token from the logits of the tokens that stand for the
options (rows of the tied output embeddings, in float32): each option's best
token, then a softmax over the options. The logits of the rest of the
vocabulary would shift every option's score by the same amount, so they are
not computed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.lfm2.load import load_text_model
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Choice, Noul, Request
from .prompt import prefix, suffix
from .readout import option_token_ids, probabilities

# The model's context (the release enforces no limit of its own).
MAX_INPUT_TOKENS = 32768
# All option codes up to 255 options are single tokens; the model card gives
# score questions 2 to 10 levels.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
REQUIRED_FILES = ("config.json", "tokenizer.json")


@dataclass
class Encoded:
    question_id: str
    question: dict[str, Any]  # as the prompt code takes it
    input_ids: list[int]
    option_ids: list[str]  # the answer's keys, in option order


def question_dict(question_id: str, question) -> dict[str, Any]:
    """A request's question as the prompt code takes it; missing instructions
    become the question id, as for Laya and Julia."""
    data = question.model_dump()
    if data.get("instructions") is None:
        data["instructions"] = question_id
    return data


def option_keys(question) -> list[str]:
    if isinstance(question, Noul):
        return ["true", "false"]
    if isinstance(question, Choice):
        return list(question.criteria)
    return [str(i) for i in range(len(question.criteria))]


class D1Backend:
    def __init__(self, name: str, model, tokenizer: Tokenizer, max_input_tokens: int):
        """``model`` None gives a backend that only encodes."""
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            max_choice_options=MAX_CHOICE_OPTIONS,
            max_score_levels=MAX_SCORE_LEVELS,
            truncates_input=False,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode(self, request: Request) -> list[Encoded]:
        """Every question's token ids, checked against the input limit."""
        head = prefix(request.state)
        shared = self._encode_text(head) if len(request.questions) > 1 else None
        limit = self.capabilities.max_input_tokens
        encoded = []
        for question_id, question in request.questions.items():
            data = question_dict(question_id, question)
            tail = suffix(data, self.tokenizer)
            if shared is None:
                ids = self._encode_text(head + tail)
            else:
                ids = shared + self._encode_text(tail)
            if len(ids) > limit:
                raise DecisionError(
                    f"the question with its state needs {len(ids)} tokens; the limit is {limit}",
                    param=f"questions.{question_id}",
                )
            encoded.append(Encoded(question_id, data, ids, option_keys(question)))
        return encoded

    def answer(self, question: Encoded) -> list[float]:
        """The question's option probabilities, from one pass over its prompt."""
        hidden = self.model(mx.array([question.input_ids]))[0, -1].astype(mx.float32)
        groups = option_token_ids(question.question, self.tokenizer)
        tokens = sorted({token for group in groups for token in group})
        rows = self.model.output_embedding_rows(mx.array(tokens)).astype(mx.float32)
        logits = rows @ hidden
        return probabilities(
            question.question, self.tokenizer, dict(zip(tokens, logits.tolist(), strict=True))
        )

    def score(self, request: Request) -> BackendOutput:
        encoded = self.encode(request)
        answers = {}
        for question in encoded:
            values = self.answer(question)
            answers[question.question_id] = dict(zip(question.option_ids, values, strict=True))
        return BackendOutput(
            probabilities=answers,
            input_tokens=sum(len(q.input_ids) for q in encoded),
        )


def load(path: Path, max_input_tokens: int | None = None) -> D1Backend:
    """Load d1-3B from a release folder (text only: the vision tower is not read).

    ``max_input_tokens`` lowers the limit per question's prompt (default and
    maximum 32,768, the model's context).
    """
    if max_input_tokens is None:
        max_input_tokens = MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= MAX_INPUT_TOKENS:
        raise ValueError(
            f"max_input_tokens must be between 1 and {MAX_INPUT_TOKENS} (the model's context)"
        )
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    model = load_text_model(path)
    backend = D1Backend(path.resolve().name, model, tokenizer, max_input_tokens)
    backend.precision = str(model.model.embed_tokens.weight.dtype).rsplit(".", 1)[-1]
    return backend
