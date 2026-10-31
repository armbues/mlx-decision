# Copyright the OneJev authors.
#
# Ported from OneJev (https://github.com/OmniJev/OneJev, commit
# 81ce62f1597c91e46767d4d02d6ac2e18534fe94), file qev/prompt.py (the answer
# codes), under the Apache License 2.0. See LICENSES/onejev-Apache-2.0.txt.
"""The prompt Base Labs' models were trained on (OneJev's ``qev-labels-v2``).

An option's answer code is ``A`` to ``Z``, then two-letter codes, 256 in
all; each is one token of the models' tokenizer.
"""

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Two-letter codes the tokenizer splits, left out of the codes.
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ"}
_NOT_ONE_TOKEN |= {"GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
CODES: tuple[str, ...] = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)]
)
