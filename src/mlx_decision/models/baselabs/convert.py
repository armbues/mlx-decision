"""Write a Base Labs model as an MLX folder, with the backbone optionally quantized.

The LM head is not written: its answer codes' rows go, in float32, into
``answer_rows.safetensors`` with their token ids (the readout reads only
those). ``readout.json`` and the tokenizer are copied as released; the
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
from .model import HEAD, ROWS_FILE, _header, code_token_ids, head_file, read_rows

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


def keeping_rows(shards: Iterable[Path], ids: list[int], into: dict) -> Iterator[Path]:
    """Hand over ``shards``; once the one with the LM head is read, keep its
    answer-code rows (taken when the converter asks for the next shard, so
    before a fetching generator deletes the file)."""
    for shard in shards:
        yield shard
        if HEAD in _header(shard)[0]:
            into["rows"] = read_rows(shard, HEAD, ids)


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
        streamed = keeping_rows(source_shards(path) if shards is None else shards, ids, kept)
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
    rows = kept.get("rows")
    if rows is None:
        rows = read_rows(head_file(path), HEAD, ids)
    mx.save_safetensors(
        str(output / ROWS_FILE),
        {"rows": rows, "token_ids": mx.array(ids, dtype=mx.int32)},
        metadata={"format": "mlx"},
    )
