"""The d1-3B backend: LFM2 text model, with the prompt and readout of ``prompt`` and ``readout``.

Each question is its own prompt: the state (``prefix``) and the question
(``suffix``). With one question the whole prompt is tokenized at once; with
several, the prefix is tokenized once and each question's own text after it,
which gives the release's token ids in both cases.

The questions of a request run together, as rows of one right-padded batch
(the model is causal, so padding at the end changes no real token). Either
every row holds its whole prompt, or the prefix is computed once (or taken
from the prefix cache) and the rows hold only the questions' own tokens,
continuing from copies of its cache. The second costs one more pass, worth
it once the prefix would otherwise be repeated ``SHARE_TOKENS`` times or
more (measured on d1-3B: the two are level at about 190 repeated tokens). A
single question takes the second way only to keep a long prefix
(``KEEP_PREFIX`` tokens) for later requests. Rows go in batches of at most
``BATCH_TOKENS`` tokens, prefix included.

A state too long for the input limit with the request's longest question is
cut: its first and last tokens are kept around a ``CUT_MARK`` line, and the
answer is marked truncated. The rendering of the state stays with
``prefix``: the state's text is what ``prefix`` adds to an empty state, and
the cut text goes back in as a string state, which ``prefix`` keeps as it
is.

The answer is read at each prompt's last token from the logits of the tokens
that stand for the options (rows of the tied output embeddings, in float32):
each option's best token, then a softmax over the options (``readout``'s
rule, computed by the readout the families share). The logits of the
rest of the vocabulary would shift every option's score by the same amount,
so they are not computed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.lfm2.load import load_text_model
from ...backbones.qwen3_5.cache import ConcatenateKVCache
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Choice, Noul, Request
from ..prefix import DEFAULT_PREFIX_CACHE_GB, PrefixCache, prefix_key
from ..readout import Features, Readout, pick, read_out
from .prompt import prefix, suffix
from .readout import option_token_ids

# The model's context (the release enforces no limit of its own).
MAX_INPUT_TOKENS = 32768
# All option codes up to 255 options are single tokens; the model card gives
# score questions 2 to 10 levels.
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
REQUIRED_FILES = ("config.json", "tokenizer.json")
# Repeated prefix tokens ((questions - 1) x prefix) from which computing the
# prefix once beats one more pass.
SHARE_TOKENS = 192
# Prefix tokens from which a single question's prefix is computed apart, to
# be kept for later requests (with the prefix cache on).
KEEP_PREFIX = 512
# Tokens per batch of rows, each row counted with its prefix: bounds the
# copies of a long prefix's cache, one per row.
BATCH_TOKENS = 16384
# Where a cut state's head and tail meet, and the tokens left for it and for
# how the cut text tokenizes at its edges.
CUT_MARK = "\n...\n"
CUT_MARGIN = 8


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


def batched(cache: list, rows: int) -> list:
    """Copies of a one-sequence cache for ``rows`` sequences that continue it."""
    out = []
    for layer in cache:
        copy = layer.fork()
        if isinstance(copy, ConcatenateKVCache):
            copy.keys = mx.broadcast_to(copy.keys, (rows, *copy.keys.shape[1:]))
            copy.values = mx.broadcast_to(copy.values, (rows, *copy.values.shape[1:]))
        else:
            copy.cache = [
                None if a is None else mx.broadcast_to(a, (rows, *a.shape[1:])) for a in copy.cache
            ]
        out.append(copy)
    return out


def state_text(state) -> str | None:
    """The state as the prompt shows it: what ``prefix`` adds to an empty state's prefix."""
    if state is None:
        return None
    head = prefix(None)
    tail = prefix("")[len(head) :]
    full = prefix(state)
    if not (full.startswith(head) and full.endswith(tail)):
        return None
    return full[len(head) : len(full) - len(tail)]


class D1Backend:
    def __init__(
        self,
        name: str,
        model,
        tokenizer: Tokenizer,
        max_input_tokens: int,
        prefix_cache_gb: float = 0.0,
    ):
        """``model`` None gives a backend that only encodes; ``prefix_cache_gb``
        0 keeps no prefixes between requests."""
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_cache = (
            PrefixCache(round(prefix_cache_gb * 1e9)) if prefix_cache_gb > 0 else None
        )
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            max_choice_options=MAX_CHOICE_OPTIONS,
            max_score_levels=MAX_SCORE_LEVELS,
            truncates_input=True,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def cut(self, state, room: int) -> str | None:
        """``state`` cut to about ``room`` tokens around ``CUT_MARK``, or None if it fits
        (or cannot be cut)."""
        text = state_text(state)
        if text is None or room <= 0:
            return None
        ids = self._encode_text(text)
        if len(ids) <= room:
            return None
        head = room // 2
        decode = self.tokenizer.decode
        return (
            decode(ids[:head], skip_special_tokens=False)
            + CUT_MARK
            + decode(ids[len(ids) - (room - head) :], skip_special_tokens=False)
        )

    def encode(self, request: Request) -> tuple[list[Encoded], int, bool]:
        """Every question's token ids, checked against the input limit; how many of
        them are the shared prefix (0 if a prompt does not start with its tokens);
        and whether the state was cut to fit."""
        questions = {
            question_id: question_dict(question_id, question)
            for question_id, question in request.questions.items()
        }
        tails = {qid: suffix(data, self.tokenizer) for qid, data in questions.items()}
        tail_ids = {qid: self._encode_text(tail) for qid, tail in tails.items()}
        limit = self.capabilities.max_input_tokens
        head = prefix(request.state)
        shared = self._encode_text(head)
        longest = max(len(ids) for ids in tail_ids.values())
        cut = False
        if len(shared) + longest > limit:
            room = limit - len(self._encode_text(prefix(""))) - longest - CUT_MARGIN
            if (text := self.cut(request.state, room)) is not None:
                head, cut = prefix(text), True
                shared = self._encode_text(head)
        several = len(questions) > 1
        encoded = []
        for question_id, question in request.questions.items():
            if several:
                ids = shared + tail_ids[question_id]
            else:
                ids = self._encode_text(head + tails[question_id])
            if len(ids) > limit:
                after = " after cutting the state" if cut else ""
                raise DecisionError(
                    f"the question with its state needs {len(ids)} tokens{after}; "
                    f"the limit is {limit}",
                    param=f"questions.{question_id}",
                )
            encoded.append(Encoded(question_id, questions[question_id], ids, option_keys(question)))
        if not several and encoded[0].input_ids != shared + tail_ids[encoded[0].question_id]:
            # The whole prompt tokenizes differently across the prefix's end.
            return encoded, 0, cut
        return encoded, len(shared), cut

    def prefix_state(self, ids: list[int]) -> list:
        """The model's cache after ``ids``: kept from an earlier request, or computed."""
        key = None
        if self.prefix_cache is not None:
            key = prefix_key(ids)
            if (entry := self.prefix_cache.get(key)) is not None:
                return entry.cache
        cache = self.model.make_cache()
        self.model(mx.array([ids]), cache=cache)
        mx.eval([c.state for c in cache])
        if key is not None:
            self.prefix_cache.put(key, cache)
        return cache

    def last_hidden(self, rows: list[list[int]], cache: list | None, offset: int) -> list:
        """The final hidden state at each row's last token (float32), the rows
        continuing ``cache`` (after ``offset`` tokens) if given."""
        out = []
        start = 0
        while start < len(rows):
            # As many rows as fit the budget, prefix included; at least one.
            end = start + 1
            longest = len(rows[start])
            while end < len(rows):
                widest = max(longest, len(rows[end]))
                if (end + 1 - start) * (offset + widest) > BATCH_TOKENS:
                    break
                longest = widest
                end += 1
            chunk = rows[start:end]
            padded = mx.array([row + [0] * (longest - len(row)) for row in chunk])
            state = batched(cache, len(chunk)) if cache is not None else None
            hidden = self.model(padded, cache=state)
            ends = mx.array([len(row) - 1 for row in chunk])
            last = hidden[mx.arange(len(chunk)), ends].astype(mx.float32)
            mx.eval(last)
            out += [last[i] for i in range(len(chunk))]
            start = end
        return out

    def question_features(
        self, question: Encoded, hidden: mx.array, truncated: bool = False
    ) -> Features:
        """The rows of the tokens that stand for the question's options, each
        option's among them, and the hidden state at its last token."""
        groups = option_token_ids(question.question, self.tokenizer)
        tokens = sorted({token for group in groups for token in group})
        rows = self.model.output_embedding_rows(mx.array(tokens)).astype(mx.float32)
        indices = [[tokens.index(token) for token in group] for group in groups]
        return Features(question.option_ids, hidden, rows, indices, 1.0, truncated)

    def passes(self, request: Request) -> tuple[list[Encoded], bool, list[mx.array]]:
        """Every question's encoding and last hidden state, and whether the state was cut."""
        encoded, shared, cut = self.encode(request)
        ids = [question.input_ids for question in encoded]
        repeated = (len(encoded) - 1) * shared
        keep = self.prefix_cache is not None and shared >= KEEP_PREFIX
        if shared and (repeated >= SHARE_TOKENS or keep):
            cache = self.prefix_state(ids[0][:shared])
            hidden = self.last_hidden([row[shared:] for row in ids], cache, shared)
        else:
            hidden = self.last_hidden(ids, None, 0)
        return encoded, cut, hidden

    def features(self, request: Request) -> dict[str, Features]:
        """What each question's readout reads, before the readout."""
        encoded, cut, hidden = self.passes(request)
        return {
            q.question_id: self.question_features(q, last, cut)
            for q, last in zip(encoded, hidden, strict=True)
        }

    def finish(self, question, probabilities: dict[str, float]) -> dict[str, float]:
        """Nothing follows the readout."""
        return probabilities

    def score(
        self,
        request: Request,
        readouts: dict[str, Readout] | None = None,
        finish: bool = True,
    ) -> BackendOutput:
        """``readouts`` replace the option tokens' rows of the questions (by id)
        with the same options; ``finish`` has no step to leave out here."""
        encoded, cut, hidden = self.passes(request)
        answers = {}
        for question, last in zip(encoded, hidden, strict=True):
            features = self.question_features(question, last)
            readout = pick(readouts, question.question_id, question.option_ids)
            values = read_out(features, readout)
            answers[question.question_id] = dict(zip(question.option_ids, values, strict=True))
        return BackendOutput(
            probabilities=answers,
            input_tokens=sum(len(q.input_ids) for q in encoded),
            truncated=cut,
        )


def load(
    path: Path,
    max_input_tokens: int | None = None,
    prefix_cache_gb: float = DEFAULT_PREFIX_CACHE_GB,
) -> D1Backend:
    """Load d1-3B from a release folder (text only: the vision tower is not read).

    ``max_input_tokens`` lowers the limit per question's prompt (default and
    maximum 32,768, the model's context); a state too long for it with the
    request's longest question is cut around a "..." line, keeping its start
    and end, and only questions too long on their own are refused.
    ``prefix_cache_gb`` is the memory
    for kept prefixes (the computed state of a long request state), so a
    request with a state seen recently and other questions computes only its
    questions; 0 turns keeping off (the questions of one request still share
    their prefix).
    """
    if prefix_cache_gb < 0:
        raise ValueError("prefix_cache_gb must be 0 (off) or more")
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
    backend = D1Backend(
        path.resolve().name, model, tokenizer, max_input_tokens, prefix_cache_gb=prefix_cache_gb
    )
    backend.precision = str(model.model.embed_tokens.weight.dtype).rsplit(".", 1)[-1]
    return backend
