"""The Studio's settings: the config file's seven keys, the values in use and
changes made from the Settings page (CFG-4).

A change is written to the file (``config.write_key``: one line, the rest
kept), the file is read again, and the running Studio takes what it can:
``models_dir`` (where downloads go, scanned again), ``default_model``,
``memory_budget_gb`` and models added to ``models`` (a rescan). ``host``,
``port``, ``workspace``, models taken out of ``models``, and any key a
flag set for this run take effect at the next start.
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import HF_CACHE, Config, ConfigError, write_key
from ..config import load as load_config
from ..gpu import Gpu
from .catalog import Catalog
from .rescan import Rescans
from .workspace import Workspace

logger = logging.getLogger(__name__)

# The config file's keys, in the order the page shows them.
KEYS = ("models_dir", "models", "default_model", "memory_budget_gb", "host", "port", "workspace")
# The keys the running Studio takes at once (unless a flag set them).
LIVE = ("models_dir", "models", "default_model", "memory_budget_gb")
# Shown, not changed from the page: the workspace moves with --workspace or
# the config file edited by hand.
READ_ONLY = ("workspace",)
FLAGS = {
    "models": "-m",
    "default_model": "--default-model",
    "memory_budget_gb": "--memory-budget",
    "host": "--host",
    "port": "--port",
    "workspace": "--workspace",
}


class Settings:
    """``config`` is the file as last read; ``sources`` where each key's value
    comes from ("flag", "config", "default"; None when not known);
    ``on_change(key)`` is called after a change (on the loop)."""

    def __init__(
        self,
        config: Config,
        gpu: Gpu,
        catalog: Catalog,
        files: Workspace,
        host: str,
        port: int,
        rescans: Rescans,
        sources: dict[str, str] | None = None,
        on_change: Callable[[str], None] = lambda key: None,
        api_key: bool = False,
    ):
        self.config = config
        self.gpu = gpu
        self.catalog = catalog
        self.files = files
        self.host, self.port = host, port
        self.rescans = rescans
        self.sources = dict(sources or {})
        self.on_change = on_change
        self.api_key = api_key  # whether this run needs one (never the key)
        self.lock = asyncio.Lock()

    def used(self) -> dict[str, dict[str, Any]]:
        """Per key: the ``value`` the Studio uses and its ``source``."""
        pool = self.gpu.pool
        config = self.config
        values = {
            "models_dir": str(config.models_dir) if config.models_dir is not None else None,
            "models": self.named_models(),
            "default_model": pool.default,
            "memory_budget_gb": round(pool.limit() / 1e9, 1),
            "host": self.host,
            "port": self.port,
            "workspace": str(self.files.root),
        }
        used = {
            key: {"value": value, "source": self.sources.get(key)} for key, value in values.items()
        }
        used["models_dir"]["source"] = "config" if config.models_dir is not None else "default"
        return used

    def named_models(self) -> list[str]:
        """What ``models`` adds: the catalog's references without ``models_dir``,
        which the Studio scans as its own key."""
        folder = self.config.models_dir
        if not isinstance(folder, Path):
            return list(self.catalog.references)
        folder = folder.expanduser().resolve()
        kept = []
        for reference in self.catalog.references:
            try:
                same = Path(reference).expanduser().resolve() == folder
            except OSError:
                same = False
            if not same:
                kept.append(reference)
        return kept

    def listed(self) -> list[dict[str, Any]]:
        """Each key as the page lists it: ``key``, ``value`` (in use),
        ``source``, ``in_file`` (what the file says; None when nothing),
        ``live`` (whether a change takes effect now) and ``editable``
        (whether the page changes it)."""
        used = self.used()
        return [
            {
                "key": key,
                **used[key],
                "in_file": file_value(self.config, key),
                "live": key in LIVE and self.sources.get(key) != "flag",
                "editable": key not in READ_ONLY,
            }
            for key in KEYS
        ]

    async def change(self, key: str, value: Any) -> dict[str, Any]:
        """Write ``key`` (None: take it out of the file), read the file again and
        apply what the running Studio can. Returns ``{line, applies, note}``:
        ``applies`` "now" or "next start". Raises ``ConfigError``."""
        if key not in KEYS:
            raise ConfigError(f"{key}: not a key of the config file")
        if key in READ_ONLY:
            raise ConfigError(
                f"{key} is not changed from the Studio: pass --workspace, or edit the config file"
            )
        async with self.lock:
            path = self.config.path
            line = await asyncio.to_thread(write_key, key, value, path)
            updated = await asyncio.to_thread(load_config, path)
            self.config = updated
            if self.sources.get(key) != "flag":
                self.sources[key] = "config" if file_value(updated, key) is not None else "default"
            applies, note = await self._apply(key, updated)
        logger.info("config %s: %s (%s)", key, line or "taken out", applies)
        self.on_change(key)
        return {"line": line, "applies": applies, "note": note}

    async def _apply(self, key: str, config: Config) -> tuple[str, str | None]:
        if self.sources.get(key) == "flag":
            return "next start", f"{FLAGS[key]} sets it for this run"
        if key not in LIVE:
            return "next start", None
        pool = self.gpu.pool
        if key == "memory_budget_gb":
            budget = round(config.memory_budget_gb * 1e9) if config.memory_budget_gb else None
            await self.gpu.run(setattr, pool, "budget", budget, check=False)
            return "now", "models loaded now stay until a load needs the room"
        if key == "default_model":
            name = config.default_name(list(pool.specs))
            if config.default_model is not None and name is None:
                return (
                    "next start",
                    f"no model the Studio lists now is named {config.default_model}",
                )
            if name is None and len(pool.specs) == 1:
                name = next(iter(pool.specs))  # a pool of one uses it
            await self.gpu.run(setattr, pool, "default", name, check=False)
            return "now", None
        if key == "models_dir":
            folder = config.models_dir if isinstance(config.models_dir, Path) else None
            if self.sources.get("models") != "flag":
                self.catalog.models_dir = folder.expanduser() if folder is not None else None
            await self.rescans.run()
            where = (
                "the Hugging Face cache" if config.models_dir in (None, HF_CACHE) else str(folder)
            )
            return "now", f"downloads go to {where}"
        # models: the references a start without -m would use
        before = set(self.catalog.references)
        references = config.model_references()
        if self.catalog.models_dir is not None:  # models_dir is scanned on its own
            references = [
                ref for ref in references if Path(ref).expanduser() != self.catalog.models_dir
            ]
        self.catalog.references = references
        await self.rescans.run()
        if before - set(references):
            return "now", "models taken out leave the list at the next start"
        return "now", None


def file_value(config: Config, key: str) -> Any:
    """What the file sets for ``key``, as JSON (None when it sets nothing)."""
    value = getattr(config, key)
    if value is None or value == ():
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def setting_sources(config: Config, flags: dict[str, Any]) -> dict[str, str]:
    """Where each key's value comes from: "flag" when its flag was given, else
    "config" when the file sets it, else "default"."""
    sources = {}
    for key in KEYS:
        if flags.get(key) not in (None, [], ()):
            sources[key] = "flag"
        elif file_value(config, key) is not None:
            sources[key] = "config"
        else:
            sources[key] = "default"
    return sources
