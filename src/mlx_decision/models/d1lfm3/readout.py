# Written from SPEC.md and the observed reference data in fixtures/; no
# code from the d1-2B release was read or used.
# Copied unchanged but for the import of ``prompt``, made relative.
"""Which tokens score each option of d1-2B's questions, and the answer's probabilities."""

import numpy as np

from .prompt import check_question, option_codes, single_token

YES_FORMS = ("yes", "Yes", "YES")
NO_FORMS = ("no", "No", "NO")


def _single_tokens(forms, tokenizer) -> list[int]:
    """The ids of the forms that are single tokens, without duplicates, in order."""
    ids = []
    for form in forms:
        token = single_token(form, tokenizer)
        if token is not None and token not in ids:
            ids.append(token)
    return ids


def option_token_ids(question: dict, tokenizer) -> list[list[int]]:
    """Per option, the token ids it is scored by (noul: yes, then no)."""
    check_question(question)
    kind = question["type"]
    if kind == "noul":
        return [_single_tokens(YES_FORMS, tokenizer), _single_tokens(NO_FORMS, tokenizer)]
    if kind == "score":
        return [[single_token(str(i), tokenizer)] for i in range(len(question["criteria"]))]
    groups = []
    for code in option_codes(question, tokenizer):
        group = [single_token(code, tokenizer)]
        spaced = single_token(" " + code, tokenizer)
        if spaced is not None and spaced != group[0]:
            group.append(spaced)
        groups.append(group)
    return groups


def probabilities(question: dict, tokenizer, logprobs, temperature: float) -> list[float]:
    """The answer: a softmax over each option's best log-probability, divided by the temperature."""
    scores = np.array(
        [max(float(logprobs[token]) for token in group) for group in option_token_ids(question, tokenizer)],
        dtype=np.float64,
    )
    scores /= temperature
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()).tolist()
