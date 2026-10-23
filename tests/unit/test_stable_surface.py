"""The stable surface, as docs/STABILITY.md promises it until 2.0, against a snapshot.

What is snapshotted: the names ``mlx_decision`` exports, the signatures of
``load`` (and each family's loader options) and ``DecisionModel``'s stable
methods, the fields and methods of the exported types, the errors' bases,
arguments and attributes, the stable commands' flags (read from the typer
app) and the renamed ones kept hidden, ``--version``, the ``--events``
envelope and its final stages, the config file's keys, the environment
variables, the file names and format numbers, the server's routes, headers,
error body and the keys of ``/health`` and ``/v1/models``, and MCP's tools,
``decide``'s schema (without its descriptions) and result keys.

Anything in the snapshot that is gone or changed breaks the promise: the test
fails naming it. Anything new is an addition, and a changed default is not
breaking either: the test fails until the snapshot is updated, so both are
seen too. To update it after checking
the change against docs/STABILITY.md:

    STABLE_SURFACE_UPDATE=1 pytest tests/unit/test_stable_surface.py
"""

import dataclasses
import inspect
import json
import os
from pathlib import Path
from typing import Any

import click
import pytest
import typer

import mlx_decision

SNAPSHOT = Path(__file__).with_name("stable_surface.json")
COMMANDS = ("run", "chat", "server", "mcp", "download", "convert", "benchmark", "web")
METHODS = ("info", "decide", "decide_request", "check")
UPDATE = "STABLE_SURFACE_UPDATE=1 pytest tests/unit/test_stable_surface.py"


def plain(value: Any) -> Any:
    """A default as JSON data (its type's name when it is not)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(plain(v) == v for v in value):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return f"<{type(value).__name__}>"


def signature(function) -> dict[str, Any]:
    """Each parameter's kind, position (positional ones) and default."""
    found = {}
    position = 0
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "self":
            continue
        entry: dict[str, Any] = {"kind": parameter.kind.name.lower()}
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            entry["position"] = position
            position += 1
        if parameter.default is not parameter.empty:
            entry["default"] = plain(parameter.default)
        found[name] = entry
    return found


def members(cls) -> dict[str, str]:
    """The public methods and properties a class defines (its own and its
    package bases', not pydantic's or the standard library's)."""
    found = {}
    for klass in cls.__mro__:
        if not klass.__module__.startswith("mlx_decision"):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in found:
                continue
            if isinstance(value, property):
                found[name] = "property"
            elif callable(value) or isinstance(value, (classmethod, staticmethod)):
                found[name] = "method"
    return found


def fields(cls) -> dict[str, dict[str, Any]]:
    if hasattr(cls, "model_fields"):  # pydantic
        return {
            name: {"required": info.is_required()}
            | ({} if info.is_required() else {"default": plain(info.get_default())})
            for name, info in cls.model_fields.items()
            if not info.exclude
        } | {
            name: {"required": False, "default": plain(info.default), "on_the_wire": False}
            for name, info in cls.model_fields.items()
            if info.exclude
        }
    if dataclasses.is_dataclass(cls):
        missing = dataclasses.MISSING
        return {
            f.name: {"required": f.default is missing and f.default_factory is missing}
            for f in dataclasses.fields(cls)
        }
    return {}


def kind(param: click.Parameter) -> str:
    if param.param_type_name == "argument":
        return "argument"
    return "flag" if getattr(param, "is_flag", False) else "option"


def label(param: click.Parameter) -> str:
    """How people know a parameter: an option by its long name, an argument by its
    metavar (not by the Python name behind it, which may change)."""
    if param.param_type_name == "argument":
        return param.human_readable_name
    return max(param.opts, key=lambda name: (name.startswith("--"), len(name)))


def renamed_flags(command: click.Command) -> dict[str, Any]:
    """The old names a command keeps hidden (they have no help), by name."""
    return {
        label(param): {name: True for name in param.opts}
        for param in command.params
        if getattr(param, "hidden", False) and not getattr(param, "help", None)
    }


def flags(command: click.Command) -> dict[str, Any]:
    found = {}
    for param in command.params:
        if getattr(param, "hidden", False):
            continue
        entry: dict[str, Any] = {
            "kind": kind(param),
            "names": {name: True for name in [*param.opts, *param.secondary_opts]},
            "type": param.type.name,
            "multiple": bool(param.multiple),
            "required": bool(param.required),
        }
        if getattr(param, "envvar", None):
            entry["envvar"] = param.envvar
        if isinstance(param.type, click.Choice):
            entry["choices"] = {str(choice): True for choice in param.type.choices}
        if not callable(param.default):
            entry["default"] = plain(param.default)
        found[label(param)] = entry
    return found


def without_descriptions(value: Any) -> Any:
    """A JSON schema without its words, which may change: its shape only."""
    if isinstance(value, dict):
        return {
            key: without_descriptions(inner)
            for key, inner in value.items()
            if key not in ("description", "title", "examples")
        }
    if isinstance(value, list):
        return [without_descriptions(inner) for inner in value]
    return value


def family_options() -> dict[str, Any]:
    """Each family's loader options (the package's families; not the tests' fake)."""
    from mlx_decision.registry import families, resolve

    found = {}
    for family in families():
        if not family.loader.startswith("mlx_decision"):
            continue
        options = signature(resolve(family.loader))
        options.pop("path", None)  # the folder, not an option
        options.pop("fit_weights", None)  # a fitted model's, set by the package
        found[family.name] = options
    return found


def error_attributes() -> dict[str, Any]:
    """The attributes an instance of each error has."""
    made = {
        "DecisionError": mlx_decision.DecisionError("m", param="p", code="c"),
        "ModelTooLargeError": mlx_decision.ModelTooLargeError("m", 2, 1),
        "DataError": mlx_decision.DataError("m", 1),
        "PlatformError": mlx_decision.PlatformError("m"),
    }
    return {name: {key: True for key in vars(error)} for name, error in made.items()}


def mcp_surface() -> dict[str, Any]:
    """MCP's tools, ``decide``'s schema without descriptions, and its result's keys."""
    pytest.importorskip("mcp")
    from concurrent.futures import ThreadPoolExecutor

    import anyio
    from mcp import Client

    from fake_backend import FakeBackend
    from mlx_decision.mcp_server import create_server

    model = mlx_decision.DecisionModel(FakeBackend())
    arguments = {"state": "x", "questions": {"q": {"type": "noul"}}}

    async def main():
        with ThreadPoolExecutor(max_workers=1) as worker:
            async with Client(create_server(model, worker, False)) as client:
                listed = await client.list_tools()
                answered = await client.call_tool("decide", arguments)
                return listed, answered

    listed, answered = anyio.run(main)
    tools = {tool.name: tool for tool in listed.tools}
    return {
        "tools": {name: True for name in tools},
        "decide_schema": without_descriptions(tools["decide"].input_schema),
        "decide_result": {key: True for key in answered.structured_content},
    }


def wire() -> dict[str, Any]:
    """The server's headers, error body and the keys of what it answers."""
    import tempfile

    from fastapi.testclient import TestClient

    from fake_backend import write_model
    from mlx_decision.pool import ModelPool, discover
    from mlx_decision.server import TRUNCATED_HEADER, create_app, error_response

    with tempfile.TemporaryDirectory() as folder:
        model = write_model(Path(folder) / "fake")
        with TestClient(create_app(ModelPool(discover([model]), budget=10**9))) as client:
            health = client.get("/health").json()
            listed = client.get("/v1/models").json()
    error = json.loads(error_response(422, "m", "invalid_request_error", "p", "c").body)
    return {
        "headers": {TRUNCATED_HEADER: True, "Retry-After": True},
        "error_body": {key: {inner: True for inner in error[key]} for key in error},
        "health": {key: True for key in health},
        "models": {key: True for key in listed},
        "model": {key: True for key in listed["models"][0]},
    }


def routes() -> dict[str, Any]:
    from mlx_decision.pool import ModelPool
    from mlx_decision.server import create_app

    paths = create_app(ModelPool([])).openapi()["paths"]
    return {path: {method.upper(): True for method in methods} for path, methods in paths.items()}


def surface() -> dict[str, Any]:
    from mlx_decision.cli import app
    from mlx_decision.config import _READERS, FILE_NAME, HOME_VARIABLE
    from mlx_decision.events import FINAL_STAGES, Event
    from mlx_decision.fitted import FINGERPRINT_ALGORITHM, FIT_FILE
    from mlx_decision.fitted import FORMAT as FIT_FORMAT
    from mlx_decision.registry import MARKER_FILE
    from mlx_decision.studio.workspace import WORKSPACE_FILE, WORKSPACE_FORMAT
    from mlx_decision.watchdog import WATCH_VARIABLE

    exported = {name: getattr(mlx_decision, name) for name in mlx_decision.__all__}
    types, errors = {}, {}
    for name, value in exported.items():
        if isinstance(value, type) and issubclass(value, BaseException):
            errors[name] = {
                "bases": {base.__name__: True for base in value.__mro__[1:-2]},
                "arguments": signature(value.__init__),
            }
        elif isinstance(value, type) and name != "DecisionModel":
            types[name] = {"fields": fields(value), "members": members(value)}
    model = mlx_decision.DecisionModel
    group = typer.main.get_command(app)
    commands = {name: flags(group.commands[name]) for name in COMMANDS}
    from_flags = {
        entry["envvar"] for command in commands.values() for entry in command.values()
        if "envvar" in entry
    }  # fmt: skip
    return {
        "exports": {name: True for name in mlx_decision.__all__},
        "load": signature(mlx_decision.load),
        "load_options": family_options(),
        "DecisionModel": {
            "name": "property" if isinstance(model.name, property) else "missing",
            **{method: signature(getattr(model, method)) for method in METHODS},
        },
        "types": types,
        "errors": errors,
        "error_attributes": error_attributes(),
        "commands": commands,
        "renamed_flags": {
            name: aliases for name in COMMANDS if (aliases := renamed_flags(group.commands[name]))
        },
        "top_level": {label(param): True for param in group.params},
        "environment": {
            name: True for name in sorted({HOME_VARIABLE, WATCH_VARIABLE, *from_flags})
        },
        "events": {
            "fields": {f.name: True for f in dataclasses.fields(Event)},
            "final_stages": {stage: True for stage in FINAL_STAGES},
        },
        "config_keys": {key: True for key in _READERS},
        "formats": {
            "fitted_model": FIT_FORMAT,
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "workspace": WORKSPACE_FORMAT,
        },
        "files": {name: True for name in (MARKER_FILE, FIT_FILE, FILE_NAME, WORKSPACE_FILE)},
        "routes": routes(),
        "wire": wire(),
        "mcp": mcp_surface(),
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        found = {}
        for key, inner in value.items():
            found.update(flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
        return found
    return {prefix: value}


def differences(was: dict, now: dict) -> tuple[list[str], list[str], list[str]]:
    """What breaks (gone or changed), what is new and which defaults changed, as
    dotted paths."""
    old, new = flatten(was), flatten(now)
    gone = [_top_gone(path, new) for path in old if path not in new and not _grown(path, new)]
    broken = [f"{path}: gone" for path in dict.fromkeys(gone)]
    changed = [path for path in old if path in new and new[path] != old[path]]
    shown = {path: f"{path}: {old[path]!r} -> {new[path]!r}" for path in changed}
    broken += [line for path, line in shown.items() if not path.endswith(".default")]
    defaults = [line for path, line in shown.items() if path.endswith(".default")]
    added = [_top_gone(path, old) for path in new if path not in old and not _grown(path, old)]
    return broken, list(dict.fromkeys(added)), defaults


def _top_gone(path: str, new: dict) -> str:
    """The shortest part of ``path`` of which nothing is in ``new`` (a whole flag,
    not each of its details); with the two swapped, the shortest part that is new."""
    parts = path.split(".")
    for end in range(1, len(parts) + 1):
        prefix = ".".join(parts[:end])
        if prefix not in new and not _grown(prefix, new):
            return prefix
    return path


def _grown(path: str, other: dict) -> bool:
    """Whether ``path`` was an empty mapping that now has entries (or the reverse)."""
    return any(key.startswith(f"{path}.") for key in other)


def test_the_stable_surface_matches_the_snapshot():
    now = json.loads(json.dumps(surface()))
    if os.environ.get("STABLE_SURFACE_UPDATE"):
        SNAPSHOT.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"snapshot written: {SNAPSHOT.name}")
    broken, added, defaults = differences(json.loads(SNAPSHOT.read_text()), now)
    message = []
    if broken:
        message.append(
            "The stable surface changed, which docs/STABILITY.md promises not to do "
            "before 2.0 (keep the old name or shape beside the new one, or move the "
            "change to 2.0):\n  " + "\n  ".join(broken)
        )
    if added:
        message.append(
            "Additions to the stable surface (fine in a minor release; check them "
            f"against docs/STABILITY.md, then update the snapshot: {UPDATE}):\n  "
            + "\n  ".join(added)
        )
    if defaults:
        message.append(
            "Defaults changed (not breaking by the page, but say so in the CHANGELOG; "
            f"then update the snapshot: {UPDATE}):\n  " + "\n  ".join(defaults)
        )
    assert not message, "\n\n".join(message)


def test_a_removed_flag_and_a_renamed_field_are_breaking():
    was = {"commands": {"run": {"json": {"kind": "flag"}}}, "types": {"Usage": {"input_tokens": 1}}}
    now = {"commands": {"run": {}}, "types": {"Usage": {"tokens_in": 1}}}
    broken, added, defaults = differences(was, now)
    assert broken == ["commands.run.json: gone", "types.Usage.input_tokens: gone"]
    assert added == ["types.Usage.tokens_in"] and defaults == []
    assert differences(now, now) == ([], [], [])


def test_a_dropped_alias_and_mcp_key_are_breaking_and_a_default_is_not():
    was = {
        "renamed_flags": {"benchmark": {"--out": {"--out": True}}},
        "mcp": {"decide_result": {"truncated": True}},
        "commands": {"server": {"--port": {"default": None}}},
    }
    now = {
        "renamed_flags": {},
        "mcp": {"decide_result": {}},
        "commands": {"server": {"--port": {"default": 8000}}},
    }
    broken, added, defaults = differences(was, now)
    assert broken == ["renamed_flags.benchmark: gone", "mcp.decide_result.truncated: gone"]
    assert defaults == ["commands.server.--port.default: None -> 8000"] and added == []


def test_the_stability_page_names_everything_the_snapshot_holds():
    page = (Path(__file__).parents[2] / "docs" / "STABILITY.md").read_text()
    snapshot = json.loads(SNAPSHOT.read_text())
    named = [
        *snapshot["exports"],
        *snapshot["DecisionModel"],
        *snapshot["commands"],
        *snapshot["events"]["fields"],
        *snapshot["events"]["final_stages"],
        *snapshot["config_keys"],
        *snapshot["environment"],
        *(path.lstrip("/").removeprefix("v1/") for path in snapshot["routes"]),
    ]
    missing = [name for name in named if f"`{name}" not in page and f"{name}`" not in page]
    assert not missing, f"docs/STABILITY.md does not name: {', '.join(missing)}"
