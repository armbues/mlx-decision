"""The readout shared by the families that answer from the last hidden state.

H2O, d1 and pplx read a question's answer from the hidden state at its last
token: one row per scored token (H2O: the option labels' rows of the tied
embeddings; d1: the rows of the tokens that stand for each option; pplx: its
readout's rows), the best token's logit per option, divided by a temperature
and normalized. ``Features`` hold what goes in, so a readout can be fitted
without running the model again, and a fitted ``Readout`` (rows and a bias
per option) can stand in for the release's rows:

    p = softmax(max over each option's rows (rows @ hidden) / temperature + bias)
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Features:
    """What a question's readout reads: ``hidden`` (float32, ``[width]``),
    ``rows`` (float32, ``[tokens, width]``), each option's rows in ``groups``
    (indices into ``rows``) and the family's temperature for the question."""

    option_ids: list[str]
    hidden: mx.array
    rows: mx.array
    groups: list[list[int]]
    temperature: float


@dataclass
class Readout:
    """Fitted rows (laid out like the features' rows) and one bias per option,
    used for a question with the same options."""

    option_ids: list[str]
    rows: mx.array
    bias: mx.array


def singletons(count: int) -> list[list[int]]:
    """Groups of one row per option, in order."""
    return [[index] for index in range(count)]


def option_logits(features: Features, readout: Readout | None = None) -> mx.array:
    """Each option's logit, temperature and bias applied (bias only with ``readout``)."""
    rows, bias = features.rows, None
    if readout is not None:
        if readout.option_ids != features.option_ids:
            raise ValueError("the readout is for other options")
        if readout.rows.shape != rows.shape or readout.bias.shape != (len(features.groups),):
            raise ValueError(
                f"the readout's rows {readout.rows.shape} and bias {readout.bias.shape} do "
                f"not fit {rows.shape} and {len(features.groups)} options"
            )
        rows, bias = readout.rows, readout.bias
    logits = rows @ features.hidden
    if features.groups != singletons(len(features.groups)):
        # Each group padded with its own first row, which leaves its maximum alone.
        width = max(len(group) for group in features.groups)
        index = [group + [group[0]] * (width - len(group)) for group in features.groups]
        logits = logits[mx.array(index)].max(axis=1)
    logits = logits / features.temperature
    return logits if bias is None else logits + bias


def read_out(features: Features, readout: Readout | None = None) -> list[float]:
    """The options' probabilities, from the release's rows or ``readout``'s."""
    values = mx.softmax(option_logits(features, readout))
    mx.eval(values)
    return values.tolist()


def pick(readouts: dict[str, Readout] | None, question_id: str, option_ids: list[str]):
    """The readout for a question, if there is one for its options."""
    readout = (readouts or {}).get(question_id)
    return readout if readout is not None and readout.option_ids == option_ids else None
