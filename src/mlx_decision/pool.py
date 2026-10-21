"""Several models behind one server: which models there are and which are loaded.

``discover`` turns model references (folders, folders of models, repo ids)
into named models without loading anything. ``ModelPool`` loads them on
demand and keeps as many as fit a memory budget, unloading the least
recently used first. Sizes are the memory check's estimates, not measured
memory, so what gets unloaded does not depend on timing.

A fitted model that leaves its base's weights alone (calibration, readout)
shares the loaded base with the base itself and with other such fits: the
base is loaded and counted once, each fit adds its tensors, and unloading
the base unloads them all.
"""

import gc
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from . import fitted, memory
from .errors import DecisionError
from .hub import is_repo_id, resolve_model_path
from .memory import check_fits, required_bytes
from .model import DecisionModel, load
from .registry import (
    FITTED,
    Family,
    accepted_options,
    detect_family,
    families,
    loader_parameters,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """A model a pool can load: its name, folder, family and load options.

    A fitted model's family is its base's (whose loader takes the options);
    ``fit`` is its folder, read and checked. ``base`` is the model it shares
    when it leaves the base's weights alone.
    """

    name: str
    path: Path
    family: Family
    options: dict = field(default_factory=dict)
    fit: fitted.FittedFolder | None = None
    base: "ModelSpec | None" = None


def _log_skip(path: Path, reason: str) -> None:
    logger.info("skipped %s", reason)


def discover(
    references: Iterable[str | Path],
    options: dict | None = None,
    on_skip: Callable[[Path, str], None] = _log_skip,
    strict_options: bool = True,
) -> list[ModelSpec]:
    """The models that ``references`` name, in order; nothing is loaded.

    A reference is a model folder or repo id (one model, named as ``load``
    names it), or a folder of models: each subfolder a family loads is a
    model named after the subfolder, in alphabetical order; the others are
    passed to ``on_skip`` with the reason (logged by default). ``options``
    go to every model whose family takes them. Raises ``ValueError`` for a
    folder without models, two models with the same name and (with
    ``strict_options``) an option no model takes; ``ModelNotFoundError`` for a
    reference that is not found.
    """
    found: list[tuple[str, Path, Family, fitted.FittedFolder | None]] = []
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
        found.append((model_name(reference), path, *_identify(path, family)))

    paths: dict[str, Path] = {}
    for name, path, _, _ in found:
        if name in paths:
            raise ValueError(f"two models are named {name}: {paths[name]} and {path}")
        paths[name] = path

    options = options or {}
    names = {family.name for _, _, family, _ in found}
    used = [family for family in families() if family.name in names]
    taken = {key for family in used for key in accepted_options(family, options)}
    unused = [key for key in options if key not in taken]
    if unused and strict_options:
        listed = ", ".join(family.name for family in used)
        raise ValueError(f"{listed} models do not take {', '.join(unused)}")
    specs = []
    for name, path, family, fit in found:
        family_options = accepted_options(family, options)
        base = None
        if fit is not None and fit.shares_base:
            base = ModelSpec(fit.base_name, fit.base, family, family_options)
        specs.append(ModelSpec(name, path, family, family_options, fit, base))
    return specs


def model_name(reference: str | Path) -> str:
    """The name ``discover`` gives the model a folder or repo id names (resolving nothing)."""
    if is_repo_id(reference):
        return str(reference).rstrip("/").rsplit("/", 1)[-1]
    return Path(reference).expanduser().resolve().name


def find_reference(references: Iterable[str], name: str) -> str | None:
    """The reference among ``references`` for the model ``name``, as ``discover``
    would name it: a reference equal to ``name``, a model named so, or a subfolder
    named so of a folder of models. Nothing is downloaded or loaded."""
    for reference in references:
        path = Path(reference).expanduser()
        if is_repo_id(reference) or not path.is_dir():
            if name in (reference, model_name(reference)):
                return reference
            continue
        try:
            detect_family(path)
        except ValueError:
            subfolder = path / name
            if subfolder.is_dir() and not name.startswith(".") and "/" not in name:
                return str(subfolder)
            continue
        if name in (reference, model_name(reference)):
            return reference
    return None


def _identify(path: Path, family: Family) -> tuple[Family, fitted.FittedFolder | None]:
    """A model's family, and for a fitted model its folder with the base checked
    (``BaseModelError``) and the base's family."""
    if family is not FITTED:
        return family, None
    fit = fitted.open_folder(path)
    return fit.family, fit


def identify(path: Path) -> tuple[Family, bool]:
    """The family that loads the model at ``path`` (a fitted model's base) and
    whether it is fitted; ``ValueError`` when no family loads it."""
    family, fit = _identify(path, detect_family(path))
    return family, fit is not None


def folder_models(folder: Path) -> list[str]:
    """The names of the models in a folder of models (subfolders a family loads),
    in ``discover``'s order; nothing is loaded."""
    try:
        return [name for name, *_ in _scan(folder, lambda path, reason: None)]
    except OSError:
        return []


def _scan(
    folder: Path, on_skip: Callable[[Path, str], None]
) -> list[tuple[str, Path, Family, fitted.FittedFolder | None]]:
    models = []
    subfolders = [p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for path in sorted(subfolders, key=lambda p: (p.name.casefold(), p.name)):
        try:
            models.append((path.name, path, *_identify(path, detect_family(path))))
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


def _key(spec: ModelSpec) -> tuple[str, str]:
    """What makes two loads of a model the same: its folder and its load options."""
    return str(spec.path.resolve()), json.dumps(spec.options, sort_keys=True, default=str)


@dataclass
class _Group:
    """A loaded model and the fitted models that share it, counted together."""

    root: ModelSpec
    model: DecisionModel
    size: int
    # Named models answering through it, with the bytes each adds.
    members: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.size + sum(self.members.values())


class ModelPool:
    """Models loaded on demand, as many as fit ``budget`` bytes.

    ``default`` names the model for requests that name none or an unknown
    one; without it, a pool of one model uses that model and a pool of
    several has no default. ``budget`` defaults to the GPU's recommended
    working set. A model counts as the memory check's estimate plus its
    prefix cache limit. A fitted model that shares its base counts only
    its own tensors once the base is loaded; the base is loaded for it if
    need be and unloaded with everything that shares it. With
    ``check_memory=False`` a model larger than the budget is loaded anyway
    (alone). ``on_change``, when set, is called with no arguments after
    models were loaded or unloaded and after ``add``, on the thread that
    made the change; ``on_use`` likewise after a loaded model was used
    (``last_used`` changed). Not thread-safe: use it from the thread that
    answers requests (``loaded``, ``groups``, ``last_used``, ``used_bytes``
    and ``names`` excepted).
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
        self.on_change: Callable[[], None] | None = None
        self._roots = {name: self._root_of(spec) for name, spec in self.specs.items()}
        self._keys = {name: _key(root) for name, root in self._roots.items()}
        self._groups: OrderedDict[tuple[str, str], _Group] = OrderedDict()
        self._loaded: OrderedDict[str, DecisionModel] = OrderedDict()
        self._sizes: dict[str, int] = {}
        # Models kept loaded by name -> who keeps them (a batch): never unloaded
        # to make room, and not by ``unload``.
        self._pinned: dict[str, str] = {}
        # Copies for other threads (the server's /health), replaced after each change.
        self._loaded_names: tuple[str, ...] = ()
        self._used_bytes = 0
        self._groups_view: tuple[dict, ...] = ()
        self._used_at: dict[str, float] = {}
        self.on_use: Callable[[], None] | None = None

    def _root_of(self, spec: ModelSpec) -> ModelSpec:
        """What a model loads: itself, or the base it shares (a listed model where
        there is one, so the base answers under its own name too)."""
        if spec.base is None:
            return spec
        key = _key(spec.base)
        listed = (s for s in reversed(self.specs.values()) if s.base is None and _key(s) == key)
        return next(listed, spec.base)

    @property
    def names(self) -> list[str]:
        return list(self.specs)

    def add(self, spec: ModelSpec) -> None:
        """A model to load on demand from now on (made while the pool is in use).

        It does not change the default model. Raises ``ValueError`` when a
        model of that name is already in the pool.
        """
        if spec.name in self.specs:
            raise ValueError(f"a model is already named {spec.name}: {self.specs[spec.name].path}")
        root = self._root_of(spec)
        self._roots[spec.name] = root
        self._keys[spec.name] = _key(root)
        # A new dict, so other threads reading the names never see one change.
        self.specs = {**self.specs, spec.name: spec}
        self._changed()

    def remove(self, name: str) -> None:
        """Take a model out of the pool (its folder is gone). Raises ``KeyError``
        for a name the pool does not have and ``ValueError`` for the default
        model or one that is loaded (in use: it stays until unloaded)."""
        if name not in self.specs:
            raise KeyError(name)
        if name == self.default:
            raise ValueError(f"{name} is the default model")
        in_memory = any(
            name in group.members or group.root.name == name for group in self._groups.values()
        )
        if name in self._loaded or in_memory:
            raise ValueError(f"{name} is loaded")
        self.specs = {key: spec for key, spec in self.specs.items() if key != name}
        for table in (self._roots, self._keys, self._sizes, self._used_at):
            table.pop(name, None)
        self._changed()

    def unload_all(self) -> list[str]:
        """Unload every model and free their memory; returns the names unloaded."""
        names = list(self._loaded)
        if not names:
            return names
        self._groups.clear()
        self._loaded.clear()
        self._used_at.clear()
        logger.info("unloaded %s", ", ".join(names))
        _release_memory()
        self._changed()
        return names

    def pin(self, name: str, holder: str) -> None:
        """Keep ``name`` (with the weights it shares) loaded for ``holder`` until
        ``unpin``: a model that needs its memory is refused with ``Busy``."""
        self._pinned[name] = holder

    def unpin(self, name: str) -> None:
        self._pinned.pop(name, None)

    def pinned_by(self, name: str) -> str | None:
        """Who keeps ``name``'s weights loaded (it or a model sharing them), or None."""
        key = self._keys.get(name)
        pins = tuple(self._pinned.items())  # changed on the event loop, read on the worker
        return next((holder for pinned, holder in pins if self._keys.get(pinned) == key), None)

    def unload(self, name: str) -> list[str]:
        """Unload ``name`` with everything that shares its weights (a base and
        the fitted models on it) and free their memory; returns the names
        unloaded, none when it is not loaded. Raises ``KeyError`` for a name
        the pool does not have, ``Busy`` for one ``pin`` keeps loaded."""
        if name not in self.specs:
            raise KeyError(name)
        if name not in self._loaded:
            return []
        holder = self.pinned_by(name)
        if holder is not None:
            from .gpu import Busy

            raise Busy(holder)
        names = self._drop(self._keys[name], "")
        _release_memory()
        self._changed()
        return names

    def _snapshot(self) -> None:
        """The copies other threads read, replaced after each change."""
        self._loaded_names = tuple(self._loaded)
        self._used_bytes = sum(group.total for group in self._groups.values())
        self._groups_view = tuple(
            {"root": group.root.name, "members": list(group.members), "bytes": group.total}
            for group in self._groups.values()
        )

    def _changed(self) -> None:
        self._snapshot()
        if self.on_change is not None:
            try:
                self.on_change()
            except Exception:
                logger.exception("the pool's change callback failed")

    def loaded(self) -> list[str]:
        """The loaded models, least recently used first; safe to call from any thread."""
        return list(self._loaded_names)

    def groups(self) -> list[dict]:
        """The loaded models by the weights they share, least recently used first:
        ``{"root", "members", "bytes"}`` each, ``root`` the model whose weights
        the group loaded (a member, or a base no request named), ``bytes`` what
        the group counts against the budget. Safe to call from any thread."""
        return [dict(group, members=list(group["members"])) for group in self._groups_view]

    def last_used(self) -> dict[str, float]:
        """When each loaded model was last loaded or used (``time.time()``);
        safe to call from any thread."""
        used = dict(self._used_at)
        return {name: used[name] for name in self._loaded_names if name in used}

    def footprint(self, name: str) -> tuple[int, int, str]:
        """``(alone, added, root)``: the bytes ``name`` needs loaded on its own
        (with its base), the bytes it adds when the weights it shares (``root``'s)
        are loaded already, and that root's name. Raises ``ModelTooLargeError``
        as ``size`` does."""
        spec, root = self.specs[name], self._roots[name]
        alone = self.size(name)
        added = alone if spec is root else max(alone - self._root_size(root), 0)
        return alone, added, root.name

    def footprint_group(self, name: str) -> str:
        """The name of the model whose weights ``name`` loads (itself or its base)."""
        return self._roots[name].name

    def used_bytes(self) -> int:
        """What the loaded models count against the budget (the estimates, not
        measured memory); safe to call from any thread."""
        return self._used_bytes

    def limit(self) -> int:
        """The budget in bytes: ``budget``, else the GPU's recommended working set."""
        return self.budget if self.budget is not None else memory.working_set()

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
        """The bytes a model is taken to need on its own (a fitted model: with its base).

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

    def _root_size(self, root: ModelSpec) -> int:
        """The bytes of a model that others share, listed or not."""
        if self.specs.get(root.name) is root:
            return self.size(root.name)
        return required_bytes(root.path, root.options) + prefix_cache_bytes(root)

    def get(self, name: str | None) -> DecisionModel:
        """The model for ``name`` (see ``resolve``), loaded if need be."""
        name = self.resolve(name)
        if name in self._loaded:
            self._touch(name)
            if self.on_use is not None:
                try:
                    self.on_use()
                except Exception:
                    logger.exception("the pool's use callback failed")
            return self._loaded[name]
        spec, root, key = self.specs[name], self._roots[name], self._keys[name]
        size = self.size(name)
        group = self._groups.get(key)
        root_size = group.size if group is not None else self._root_size(root)
        own = 0 if spec is root else max(size - root_size, 0)
        self._make_room(own if group is not None else root_size + own, name, keep=key)
        if group is not None:
            base = group.model
            logger.info("loading %s (sharing the loaded %s) ...", name, root.name)
        else:
            logger.info(
                "loading %s ...", name if spec is root else f"{name} (with its base {root.name})"
            )
            base = self._load(root)
            base.backend.name = root.name
        model = base if spec is root else DecisionModel(fitted.attach(spec.fit, base.backend, name))
        if group is None:
            group = self._groups[key] = _Group(root, base, root_size)
        group.members[name] = own
        self._loaded[name] = model
        self._touch(name)
        self._changed()
        return model

    def _touch(self, name: str) -> None:
        self._loaded.move_to_end(name)
        self._groups.move_to_end(self._keys[name])
        self._used_at[name] = time.time()
        self._snapshot()

    def _make_room(self, needed: int, name: str, keep: tuple[str, str]) -> None:
        """Unload the least recently used models (with all that share their base)
        until ``needed`` more bytes fit; never the group ``keep`` nor a pinned
        one: ``Busy`` naming who pinned it when only that would make room."""
        limit = self.limit()
        unloaded = False
        pins = tuple(self._pinned.items())  # changed on the event loop, read on the worker
        pinned = {self._keys[n]: who for n, who in pins if n in self._keys}
        kept = [key for key in self._groups if key == keep or key in pinned]
        blocking = [pinned[key] for key in kept if key in pinned and key != keep]
        if blocking and sum(self._groups[key].total for key in kept) + needed > limit:
            from .gpu import Busy

            raise Busy(blocking[0])  # before unloading anything for nothing
        while sum(group.total for group in self._groups.values()) + needed > limit:
            old = next((key for key in self._groups if key != keep and key not in pinned), None)
            if old is None:
                break
            self._drop(old, f"to make room for {name}")
            unloaded = True
        if unloaded:
            _release_memory()
            self._changed()

    def _drop(self, key: tuple[str, str], why: str) -> list[str]:
        """Forget the group ``key`` (its base and every model sharing it); returns
        their names. The caller releases the memory and reports the change."""
        # Delete without binding the model, so no local keeps it (and its
        # weights) alive through the release and the next load.
        members, root = list(self._groups[key].members), self._groups[key].root.name
        del self._groups[key]
        for member in members:
            del self._loaded[member]
        with_base = "" if root in members else f" with their base {root}"
        logger.info("unloaded %s%s%s", ", ".join(members), with_base, f" {why}" if why else "")
        for member in members:
            self._used_at.pop(member, None)
        self._snapshot()
        return members


def _release_memory() -> None:
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()
