# Written from a behaviour specification and the observed outputs of the
# d1-3B release (LiquidAI/d1-3B); no code from the release was read or used.
"""Reading d1-3B's answer from the next-token log-probabilities."""

import math

from .prompt import instructions, option_codes, single_token


def _forms(words: tuple[str, ...], tokenizer) -> list[int]:
    ids = []
    for word in words:
        token = single_token(word, tokenizer)
        if token is not None and token not in ids:
            ids.append(token)
    return ids


def option_token_ids(question: dict, tokenizer) -> list[list[int]]:
    """The token ids each option is scored by, one group per option (noul: yes, no)."""
    instructions(question)
    kind = question.get("type")
    if kind == "noul":
        return [_forms(("yes", "Yes", "YES"), tokenizer), _forms(("no", "No", "NO"), tokenizer)]
    if kind == "score":
        groups = []
        for i in range(len(question["criteria"])):
            token = single_token(str(i), tokenizer)
            if token is None:
                raise ValueError(f"score level {i} is not a single token")
            groups.append([token])
        return groups
    if kind == "choice":
        groups = []
        for code, token in option_codes(question, tokenizer):
            spaced = single_token(" " + code, tokenizer)
            groups.append([token] if spaced is None or spaced == token else [token, spaced])
        return groups
    raise ValueError(f"unknown question type: {kind!r}")


def probabilities(question: dict, tokenizer, logprobs) -> list[float]:
    """Softmax over each option's best log-probability."""
    groups = option_token_ids(question, tokenizer)
    scores = [max(float(logprobs[i]) for i in group) for group in groups]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]
