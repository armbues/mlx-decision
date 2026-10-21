"""What the Studio is serving: address, version, workspace, config, models.

``GET /status`` gives what the page shows in its sidebar and Get Code
sheet; ``GET /models`` the models a run can pick, with what each takes
(from its files, before loading), for the live check's hints, and what
the Models page shows (``catalog.model_facts``, the memory a load takes).
"""

import asyncio
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ... import __version__
from ...backend import Capabilities
from ...config import Config
from ...gpu import Gpu
from ..catalog import cli_arguments
from ..security import WILDCARDS
from ..workspace import Workspace
from .models import not_convertible
from .run import Models


def address(host: str, port: int) -> str:
    """The Studio's address as a browser on this Mac reaches it."""
    if host in WILDCARDS:
        host = "127.0.0.1"
    elif ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def config_values(config: Config) -> dict[str, Any]:
    """The keys the config file sets, as JSON values."""
    values = {}
    for key in (
        "models",
        "default_model",
        "memory_budget_gb",
        "host",
        "port",
        "workspace",
        "models_dir",
    ):
        value = getattr(config, key)
        if value is None or value == ():
            continue
        values[key] = (
            str(value) if isinstance(value, Path) else list(value) if key == "models" else value
        )
    return values


def cli_path() -> str:
    """The ``mlx-decision`` command of the environment the Studio runs in (an MCP
    client needs its full path: it does not search the shell's PATH)."""
    beside = Path(sys.executable).parent / "mlx-decision"
    if beside.is_file():
        return str(beside)
    return shutil.which("mlx-decision") or "mlx-decision"


def capabilities_info(capabilities: Capabilities) -> dict[str, Any]:
    """The request limits of ``DecisionModel.info()``, from static capabilities,
    and the image limits the Playground's drop area shows."""
    return {
        "max_input_tokens": capabilities.max_input_tokens,
        "truncates_input": capabilities.truncates_input,
        "question_types": sorted(capabilities.question_types),
        "max_choice_options": capabilities.max_choice_options,
        "max_score_levels": capabilities.max_score_levels,
        "question_tokens": capabilities.question_tokens,
        "requires_instructions": capabilities.requires_instructions,
        "supports_images": capabilities.supports_images,
        "max_image_pixels": capabilities.max_image_pixels,
        "max_images": capabilities.max_images,
    }


def status_router(
    gpu: Gpu,
    files: Workspace,
    models: Models,
    config: Callable[[], Config],
    host: str,
    port: int,
    api_key: str | None,
    status: Callable[[], dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/studio/api")
    pool = gpu.pool
    home = address(host, port)

    @router.get("/status")
    async def get_status():
        current = await asyncio.to_thread(status)
        return {
            "version": __version__,
            "address": home,
            "api": {"url": f"{home}/v1/systemone", "key_required": bool(api_key)},
            "workspace": str(files.root),
            "sets_folder": str(files.root / "sets"),
            "cli_path": cli_path(),
            **current,
            "models": pool.names,
            "default_model": pool.default,
            "config": {
                "path": str(config().path),
                "exists": config().path.is_file(),
                "values": config_values(config()),
            },
        }

    @router.get("/models")
    async def list_models():
        def describe() -> list[dict[str, Any]]:
            current = config()
            loaded = set(pool.loaded())
            listed = []
            for name, spec in pool.specs.items():
                entry: dict[str, Any] = {
                    "name": name,
                    "family": spec.family.name,
                    "fitted": spec.fit is not None,
                    "base": spec.fit.base_name if spec.fit is not None else None,
                    "default": name == pool.default,
                    "loaded": name in loaded,
                    "convert_reason": not_convertible(spec),  # why Convert does not offer it
                }
                try:
                    entry["capabilities"] = capabilities_info(models.capabilities(name))
                    entry["error"] = None
                except (ValueError, OSError) as error:
                    entry["capabilities"] = None
                    entry["error"] = str(error)
                try:
                    entry.update(models.facts(name))
                    entry["cli"] = cli_arguments(spec, current, entry.get("repo"))
                    entry["memory"] = models.memory(name)
                except (ValueError, OSError) as error:
                    entry["memory"] = None
                    entry["error"] = entry["error"] or str(error)
                try:  # measured here (PG-9); None before the first answer
                    entry["speed"] = models.speed(name)
                except (ValueError, OSError):
                    entry["speed"] = None
                entry["convertible"] = entry["convert_reason"] is None
                listed.append(entry)
            return listed

        return {"models": await asyncio.to_thread(describe)}

    return router
