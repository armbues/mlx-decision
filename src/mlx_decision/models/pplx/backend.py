"""The joint backend of pplx-decider v1.2: one pass per request.

The letter-readout backend with the joint request format: a request's
state and questions share one sequence (``joint.render``), full attention
is non-causal over all of it, and each question's answer is read at its
marker, the hidden state there times the readout. A request over the
input limit is answered in chunks (its questions halved until each part
fits, every part with the whole state and images); a question that does
not fit alone is refused. Nothing is cut.
"""

from dataclasses import dataclass

import mlx.core as mx

from ...backend import BackendOutput
from ...errors import DecisionError
from ...types import Request
from ..letter import LetterBackend
from ..qwen import count_image_tokens, image_features, open_images, prepare
from ..readout import Features, Readout, pick, read_out
from .joint import MARKER, TooLong, plan, render
from .prompt import options


@dataclass
class Chunk:
    """One sequence: some of a request's questions, in request order."""

    question_ids: list[str]
    option_ids: list[list[str]]
    input_ids: list[int]
    markers: list[int]


class JointBackend(LetterBackend):
    def __init__(self, *args, escapes: list[str], **kwargs):
        """``escapes``: the tokens user text may not contain as they are."""
        super().__init__(*args, **kwargs)
        self.escapes = escapes
        self.marker = self.tokenizer.token_to_id(MARKER)

    def encode(self, request: Request, image_tokens: list[int] = ()) -> list[Chunk]:
        """The request's chunks (one unless it is over the limit), all encoded
        before any pass; refuses the request if a question does not fit alone.

        ``image_tokens`` holds each image's token count: its placeholder in
        the prompt becomes that many image pad tokens, as the processor does.
        """
        questions = request.questions

        def encode(ids: list[str]) -> Chunk:
            text = render(
                request.state,
                [questions[q] for q in ids],
                self.codes,
                self.escapes,
                len(image_tokens),
            )
            tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
            if image_tokens:
                tokens = self._expand_images(tokens, image_tokens, request.state, "state")
            markers = [i for i, token in enumerate(tokens) if token == self.marker]
            if len(markers) != len(ids):
                raise ValueError("the readout markers do not match the questions")
            return Chunk(ids, [options(questions[q])[0] for q in ids], tokens, markers)

        for question_id, question in questions.items():
            count = len(options(question)[0])
            if count > len(self.codes):
                raise DecisionError(
                    f"this model answers questions with at most {len(self.codes)} options",
                    param=f"questions.{question_id}.criteria",
                )
        limit = self.capabilities.max_input_tokens
        try:
            parts = plan(list(questions), encode, lambda chunk: len(chunk.input_ids), limit)
        except TooLong as error:
            raise DecisionError(
                f"the question with its state needs {error.tokens} tokens; "
                f"the limit is {limit} and input is not cut",
                param=f"questions.{error.question_id}",
            ) from None
        return [chunk for _, chunk in parts]

    def passes(self, request: Request) -> tuple[list[Chunk], list[mx.array]]:
        """Every chunk's encoding and its hidden states at the markers ([questions, hidden])."""
        # Image sizes come from the file headers, so the input limit is checked
        # before any image is decoded or cut into patches.
        opened, config = [], None
        if request.images:
            config = self.image_config()
            opened = open_images(request.images)
        chunks = self.encode(
            request, [count_image_tokens(image, index, config) for index, image in opened]
        )
        images = [prepare(image, index, config) for index, image in opened]
        features = image_features(self.load_vision(), images) if images else None
        hidden = []
        for chunk in chunks:
            states = self.hidden_states(chunk.input_ids, images, features)
            at_markers = states[mx.array(chunk.markers)].astype(mx.float32)
            mx.eval(at_markers)
            hidden.append(at_markers)
        return chunks, hidden

    def _questions(self, request: Request):
        """(question id, option ids, hidden state) of every question, in request order."""
        chunks, hidden = self.passes(request)
        found = {}
        for chunk, states in zip(chunks, hidden, strict=True):
            for row, (question_id, option_ids) in enumerate(
                zip(chunk.question_ids, chunk.option_ids, strict=True)
            ):
                found[question_id] = (option_ids, states[row])
        tokens = sum(len(chunk.input_ids) for chunk in chunks)
        return [(q, *found[q]) for q in request.questions], tokens

    def features(self, request: Request) -> dict[str, Features]:
        """What each question's readout reads, before the readout."""
        questions, _ = self._questions(request)
        return {q: self.question_features(option_ids, state) for q, option_ids, state in questions}

    def score(
        self,
        request: Request,
        readouts: dict[str, Readout] | None = None,
        finish: bool = True,
    ) -> BackendOutput:
        """``readouts`` replace the readout rows of the questions (by id) with the
        same options; ``finish`` has no step to leave out here."""
        questions, tokens = self._questions(request)
        probabilities = {}
        for question_id, option_ids, state in questions:
            features = self.question_features(option_ids, state)
            values = read_out(features, pick(readouts, question_id, option_ids))
            probabilities[question_id] = dict(zip(option_ids, values, strict=True))
        return BackendOutput(probabilities=probabilities, input_tokens=tokens)
