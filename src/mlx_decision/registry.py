"""Model families: how a model folder is recognised and loaded."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .backend import Backend

MARKER_FILE = "mlx_decision.json"


@dataclass(frozen=True)
class Family:
    name: str
    detect: Callable[[Path], bool]
    loader: str  # "module:function"; imported only when the family is used
    converter: str | None = None  # same form; writes an MLX copy of a model


_FAMILIES: dict[str, Family] = {}


def register_family(
    name: str, detect: Callable[[Path], bool], loader: str, converter: str | None = None
) -> None:
    _FAMILIES[name] = Family(name, detect, loader, converter)


def resolve(reference: str) -> Callable:
    """Import ``module:function``."""
    module, function = reference.split(":")
    return getattr(import_module(module), function)


def detect_family(path: Path) -> Family:
    marker = path / MARKER_FILE
    if marker.exists():
        name = json.loads(marker.read_text()).get("family")
        if name not in _FAMILIES:
            raise ValueError(f"{path}: unknown model family {name!r}")
        return _FAMILIES[name]
    for family in _FAMILIES.values():
        if family.detect(path):
            return family
    known = ", ".join(sorted(_FAMILIES)) or "none"
    raise ValueError(f"{path}: not a supported decision model (known families: {known})")


def load_backend(path: Path, **options) -> Backend:
    return resolve(detect_family(path).loader)(path, **options)


register_family(
    "clef",
    detect=lambda path: (path / "joint_head_config.json").exists(),
    loader="mlx_decision.models.clef.model:load",
    converter="mlx_decision.models.clef.convert:convert",
)
