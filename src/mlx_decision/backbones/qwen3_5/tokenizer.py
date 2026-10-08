# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
#
# Ported from transformers (https://github.com/huggingface/transformers) 5.18.0,
# file models/qwen3_5/tokenization_qwen3_5.py (the pre-tokenizer of
# Qwen3_5Tokenizer), under the Apache License 2.0. See
# LICENSES/transformers-Apache-2.0.txt.
#
# Modified for mlx-decision: applied to a tokenizer loaded from
# ``tokenizer.json`` with the `tokenizers` library.
"""Load a Qwen3.5 tokenizer as transformers does."""

import json
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers

# Keeps combining marks (\p{M}) with their letters. The releases' tokenizer.json
# still has the older pattern without them; transformers replaces it with this
# one, so Hindi, Bengali, Thai or vowelled Arabic split into other tokens.
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501


def load_tokenizer(path: str | Path) -> Tokenizer:
    """``tokenizer.json`` of a Qwen3.5 folder with transformers' pre-tokenizer.

    Only a Qwen-style pre-tokenizer (a split by pattern, then byte level) is
    replaced; any other (the word-level tokenizers of the test models) is
    kept as it is.
    """
    tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    stored = json.loads(tokenizer.to_str()).get("pre_tokenizer") or {}
    kinds = [step.get("type") for step in stored.get("pretokenizers", [])]
    if stored.get("type") != "Sequence" or kinds != ["Split", "ByteLevel"]:
        return tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    return tokenizer
