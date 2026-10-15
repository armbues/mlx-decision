"""Write a pplx model as an MLX folder (see ``letter_convert``)."""

from pathlib import Path

from .. import letter_convert


def convert(path: Path, output: Path, source: str, **options) -> None:
    letter_convert.convert(path, output, source, "pplx", **options)
