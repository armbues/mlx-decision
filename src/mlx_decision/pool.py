"""Several models behind one server: which models there are and which are loaded.

``discover`` turns model references (folders, folders of models, repo ids)
into named models without loading anything. ``ModelPool`` loads them on
demand and keeps as many as fit a memory budget, unloading the least
recently used first. Sizes are the memory check's estimates, not measured
memory, so what gets unloaded does not depend on timing.
"""

import gc
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import memory
from .errors import DecisionError
from .hub import is_repo_id, resolve_model_path
from .memory import check_fits, required_bytes
from .model import DecisionModel, load
from .registry import Family, accepted_options, detect_family, loader_parameters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A model a pool can load: its name, folder, family and load options."""

    name: str
    path: Path
    family: Family
    options: dict = field(default_factory=dict)


def _log_skip(path: Path, reason: str) -> None:
    logger.info("skipped %s", reason)


def discover(
    references: Iterable[str | Path],
    options: dict | None = None,
    on_skip: Callable[[Path, str], None] = _log_skip,
) -> list[ModelSpec]:
    """The models that ``references`` name, in order; nothing is loaded.

    A reference is a model folder or repo id (one model, named as ``load``
    names it), or a folder of models: each subfolder a family loads is a
    model named after the subfolder, in alphabetical order; the others are
    passed to ``on_skip`` with the reason (logged by default). ``options``
    go to every model whose family takes them. Raises ``ValueError`` for a
    folder without models, two models with the same name and an option no
    model takes; ``ModelNotFoundError`` for a reference that is not found.
    """
    found: list[tuple[str, Path, Family]] = []
    for reference in references:
        path = resolve_model_path(reference)
        try:
            family = detect_family(path)
        except ValueError as error:
            if is_repo_id(reference):
                raise
            models = _scan(path, on_skip)
            if not models:
                raise ValueError(f"{error}, and none of its subfolders is one") from None
            found += models
            continue
        name = str(reference).rstrip("/").rsplit("/", 1)[-1] if is_repo_id(reference) else None
        found.append((name or path.resolve().name, path, family))

    paths: dict[str, Path] = {}
    for name, path, _ in found:
        if name in paths:
            raise ValueError(f"two models are named {name}: {paths[name]} and {path}")
        paths[name] = path

    options = options or {}
    families = {family.name: family for _, _, family in found}
    taken = {key for family in families.values() for key in accepted_options(family, options)}
    unused = [key for key in options if key not in taken]
    if unused:
        raise ValueError(f"{', '.join(families)} models do not take {', '.join(unused)}")
    return [
        ModelSpec(name, path, family, accepted_options(family, options))
        for name, path, family in found
    ]


def _scan(folder: Path, on_skip: Callable[[Path, str], None]) -> list[tuple[str, Path, Family]]:
    models = []
    subfolders = [p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for path in sorted(subfolders, key=lambda p: (p.name.casefold(), p.name)):
        try:
            models.append((path.name, path, detect_family(path)))
        except ValueError as error:
            on_skip(path, str(error))
    return models


def prefix_cache_bytes(spec: ModelSpec) -> int:
    """The memory a model may keep for prefixes: its option, else its loader's default."""
    size = spec.options.get("prefix_cache_gb")
    if size is None:
        parameter = (loader_parameters(spec.family) or {}).get("prefix_cache_gb")
        size = parameter.default if parameter is not None else 0
    return round((size or 0) * 1e9)


def _load(spec: ModelSpec) -> DecisionModel:
    return load(spec.path, check_memory=False, **spec.options)


class ModelPool:
    """Models loaded on demand, as many as fit ``budget`` bytes.

    ``default`` names the model for requests that name none or an unknown
    one; without it, a pool of one model uses that model and a pool of
    several has no default. ``budget`` defaults to the GPU's recommended
    working set. A model counts as the memory check's estimate plus its
    prefix cache limit. With ``check_memory=False`` a model larger than the
    budget is loaded anyway (alone). Not thread-safe: use it from the
    thread that answers requests.
    """

    def __init__(
        self,
        specs: Iterable[ModelSpec],
        default: str | None = None,
        budget: int | None = None,
        load: Callable[[ModelSpec], DecisionModel] = _load,
        check_memory: bool = True,
    ):
        self.specs = {spec.name: spec for spec in specs}
        if default is not None and default not in self.specs:
            raise ValueError(f"no model is named {default} (models: {', '.join(self.specs)})")
        if default is None and len(self.specs) == 1:
            default = next(iter(self.specs))
        self.default = default
        self.budget = budget
        self.check_memory = check_memory
        self._load = load
        self._loaded: OrderedDict[str, DecisionModel] = OrderedDict()
        self._sizes: dict[str, int] = {}
        # A copy for other threads (the server's /health), replaced after each change.
        self._loaded_names: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def loaded(self) -> list[str]:
        """The loaded models, least recently used first; safe to call from any thread."""
        return list(self._loaded_names)

    def resolve(self, name: str | None) -> str:
        """The model that answers a request naming ``name``."""
        if name in self.specs:
            return name
        if self.default is None:
            raise DecisionError(
                f"name one of the models: {', '.join(self.specs)} (this server has no "
                f"default model)",
                param="model",
            )
        return self.default

    def size(self, name: str) -> int:
        """The bytes a model is taken to need.

        Raises ``ModelTooLargeError`` when it alone exceeds the budget (unless
        the memory check is off).
        """
        if name not in self._sizes:
            spec = self.specs[name]
            extra = prefix_cache_bytes(spec)
            if self.check_memory:
                size = check_fits(
                    spec.path,
                    spec.options,
                    self.budget,
                    name=name,
                    extra=extra,
                    source=str(spec.path),
                )
            else:
                size = required_bytes(spec.path, spec.options) + extra
            self._sizes[name] = size
        return self._sizes[name]

    def get(self, name: str | None) -> DecisionModel:
        """The model for ``name`` (see ``resolve``), loaded if need be."""
        name = self.resolve(name)
        if name in self._loaded:
            self._loaded.move_to_end(name)
            self._loaded_names = tuple(self._loaded)
            return self._loaded[name]
        size = self.size(name)
        limit = self._limit()
        unloaded = False
        while self._loaded and sum(self._sizes[n] for n in self._loaded) + size > limit:
            old, _ = self._loaded.popitem(last=False)
            logger.info("unloaded %s to make room for %s", old, name)
            self._loaded_names = tuple(self._loaded)
            unloaded = True
        if unloaded:
            _release_memory()
        logger.info("loading %s ...", name)
        model = self._load(self.specs[name])
        model.backend.name = name
        self._loaded[name] = model
        self._loaded_names = tuple(self._loaded)
        return model

    def _limit(self) -> int:
        return self.budget if self.budget is not None else memory.working_set()


def _release_memory() -> None:
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()
