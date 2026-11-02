"""Write a Base Labs model as an MLX folder, with the backbone optionally quantized.

The LM head is not written: its answer codes' rows go, in float32, into
``answer_rows.safetensors`` with their token ids (the readout reads only
those), taken from the tensor loading reads them from (``head_source``).
``readout.json`` and the tokenizer are copied as released; the
``modeling_overrides`` stay in ``config.json``. See ``qwen_convert`` for the
backbone and the vision tower, kept as stored so images need no new
conversion later.
"""

from collections.abc import Iterable, Iterator
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import source_shards
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ..qwen_convert import convert_qwen
from .model import (
    EMBEDDINGS,
    HEAD,
    ROWS_FILE,
    _header,
    code_token_ids,
    head_source,
    read_rows,
    ties_head,
)

COPIED = (
    "readout.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "LICENSE",
    "README.md",
)


def _load_for_sweep(path: Path):
    from .model import load

    # No prefix reuse: the sweep answers the same requests with other weights.
    return load(path, prefix_cache_gb=0)


def keeping_rows(
    shards: Iterable[Path], ids: list[int], into: dict, tied: bool = False
) -> Iterator[Path]:
    """Hand over ``shards``; once the one with the LM head is read, keep its
    answer-code rows, and with ``tied`` the embeddings' too (taken when the
    converter asks for the next shard, so before a fetching generator
    deletes the file)."""
    names = (HEAD, EMBEDDINGS) if tied else (HEAD,)
    for shard in shards:
        yield shard
        header = _header(shard)[0]
        for name in names:
            if name in header:
                into[name] = read_rows(shard, name, ids)


def convert(
    path: Path,
    output: Path,
    source: str,
    shards: Iterable[Path] | None = None,
    target_bits: float | None = None,
    **options,
) -> None:
    """Convert (see ``convert_qwen``); ``target_bits`` switches to mixed precision."""
    ids = code_token_ids(load_tokenizer(path))
    kept: dict = {}
    if target_bits is None:
        files = source_shards(path) if shards is None else shards
        streamed = keeping_rows(files, ids, kept, ties_head(path))
    else:
        streamed = shards  # mixed precision reads the folder (and refuses shards)
    convert_qwen(
        "baselabs",
        COPIED,
        _load_for_sweep,
        path,
        output,
        source,
        target_bits=target_bits,
        shards=streamed,
        lm_head=False,
        **options,
    )
    # As head_source: a stored head first, the embeddings of a tied folder without one.
    rows = kept.get(HEAD, kept.get(EMBEDDINGS))
    if rows is None:
        rows = read_rows(*head_source(path), ids)
    mx.save_safetensors(
        str(output / ROWS_FILE),
        {"rows": rows, "token_ids": mx.array(ids, dtype=mx.int32)},
        metadata={"format": "mlx"},
    )
