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


_FAMILIES: dict[str, Family] = {}


def register_family(name: str, detect: Callable[[Path], bool], loader: str) -> None:
    _FAMILIES[name] = Family(name, detect, loader)


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
    family = detect_family(path)
    module, function = family.loader.split(":")
    return getattr(import_module(module), function)(path, **options)


register_family(
    "clef",
    detect=lambda path: (path / "joint_head_config.json").exists(),
    loader="mlx_decision.models.clef.model:load",
)
