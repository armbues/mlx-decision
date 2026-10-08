"""Model families: how a model folder is recognised and loaded."""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .backend import Backend

MARKER_FILE = "mlx_decision.json"


@dataclass(frozen=True)
class KnownModel:
    """A model on the Hugging Face Hub that a family loads, offered by ``download``."""

    repo_id: str
    description: str
    size: str  # approximate download size, for the menu


@dataclass(frozen=True)
class Family:
    name: str
    detect: Callable[[Path], bool]
    loader: str  # "module:function"; imported only when the family is used
    # Same form; writes an MLX copy of a model: (path, output, source, bits=,
    # shards=, revision=, ...), shards being the weight files read in turn.
    converter: str | None = None
    known_models: tuple[KnownModel, ...] = ()
    # Files a download fetches (Hub patterns); None: the whole repository.
    files: tuple[str, ...] | None = None
    # "module:function" (path, load options) -> [(weight file, load dtype or None)],
    # for the memory check; None: every top-level *.safetensors, as stored.
    weight_files: str | None = None
    # (tensor name, shape) -> whether the converter quantizes it, for size
    # estimates before converting; None: every tensor.
    quantizable: Callable[[str, list[int]], bool] | None = None


_FAMILIES: dict[str, Family] = {}


def register_family(
    name: str,
    detect: Callable[[Path], bool],
    loader: str,
    converter: str | None = None,
    known_models: tuple[KnownModel, ...] = (),
    files: tuple[str, ...] | None = None,
    weight_files: str | None = None,
    quantizable: Callable[[str, list[int]], bool] | None = None,
) -> None:
    _FAMILIES[name] = Family(
        name, detect, loader, converter, known_models, files, weight_files, quantizable
    )


def known_models() -> list[KnownModel]:
    return [model for family in _FAMILIES.values() for model in family.known_models]


def get_family(name: str) -> Family:
    return _FAMILIES[name]


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


def loader_parameters(family: Family) -> dict[str, inspect.Parameter] | None:
    """The loader's parameters by name; None when it takes any keyword."""
    parameters = inspect.signature(resolve(family.loader)).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return dict(parameters)


def accepted_options(family: Family, options: dict) -> dict:
    """The options the family's loader takes."""
    parameters = loader_parameters(family)
    if parameters is None:
        return dict(options)
    return {name: value for name, value in options.items() if name in parameters}


def check_options(family: Family, options: dict) -> None:
    """Fail unless the family's loader takes every option."""
    unknown = [name for name in options if name not in accepted_options(family, options)]
    if unknown:
        raise ValueError(f"{family.name} models do not take {', '.join(unknown)}")


def load_backend(path: Path, **options) -> Backend:
    family = detect_family(path)
    check_options(family, options)
    backend = resolve(family.loader)(path, **options)
    backend.family = family.name
    return backend


register_family(
    "clef",
    detect=lambda path: (path / "joint_head_config.json").exists(),
    loader="mlx_decision.models.clef.model:load",
    converter="mlx_decision.models.clef.convert:convert",
    known_models=(
        KnownModel(
            "Cloudflare/clef-flash",
            "Clef flash: Cloudflare's 9B decision model (Qwen3.5), text and images",
            "19 GB",
        ),
        KnownModel(
            "Cloudflare/clef",
            "Clef: Cloudflare's 27B decision model (Qwen3.5), text and images; "
            "below 96 GB of memory it runs quantized (offered after picking it)",
            "55 GB",
        ),
    ),
    # Weights, head, tokenizer and configs; not the reference code or chat template.
    files=("*.json", "model*.safetensors", "joint_head.safetensors", "LICENSE*", "README.md"),
    weight_files="mlx_decision.models.clef.model:weight_files",
    # The text backbone's matrices (named as released or as MLX saves them);
    # the vision tower and the decision head stay as they are.
    quantizable=lambda name, shape: (
        len(shape) == 2
        and name.startswith(("model.language_model.", "lm_head.", "language_model."))
    ),
)

# The encoder and tokenizer folders plus the files at the top; the Laya repo
# also holds other checkpoints in sub-folders, Julia's its PyTorch code.
_MARKER_FILES = ("model.safetensors", "encoder/*", "tokenizer/*", "README.md", "LICENSE*")

register_family(
    "laya",
    detect=lambda path: (path / "rl_agent_config.json").exists(),
    loader="mlx_decision.models.marker.model:load_laya",
    known_models=(
        KnownModel(
            "convaiinnovations/laya",
            "Laya: English decision model (ModernBERT-large, 421M), calibrated",
            "843 MB",
        ),
        KnownModel(
            "convaiinnovations/laya-multilingual",
            "Laya multilingual: 100+ languages (mmBERT-base, 322M)",
            "678 MB",
        ),
        KnownModel(
            "convaiinnovations/laya-typed-decisions",
            "Laya fine-tuned for typed-decision workflows (ModernBERT-large)",
            "846 MB",
        ),
    ),
    files=(*_MARKER_FILES, "rl_agent_config.json"),
    weight_files="mlx_decision.models.marker.model:weight_files",
)

register_family(
    "julia",
    detect=lambda path: (path / "julia_config.json").exists(),
    loader="mlx_decision.models.marker.model:load_julia",
    known_models=(
        KnownModel(
            "SupersonicLabs/Julia-1",
            "Julia 1: multilingual decision model (mmBERT-small, 144M), 2-20 options",
            "577 MB",
        ),
    ),
    files=(*_MARKER_FILES, "config.json", "julia_config.json", "inference-policy.json"),
    weight_files="mlx_decision.models.marker.model:weight_files",
)
