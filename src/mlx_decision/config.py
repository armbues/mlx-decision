"""The config file: defaults for the commands' flags.

``~/.mlx-decision/config.toml`` (``MLX_DECISION_HOME`` moves the folder)
holds six keys. A flag on the command line wins over the file, the file
over the built-in default. The file is optional and only ever read.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOME_VARIABLE = "MLX_DECISION_HOME"
FILE_NAME = "config.toml"


class ConfigError(ValueError):
    """The config file cannot be read, or a value has the wrong type."""


def home() -> Path:
    """The folder of the config file and the Studio's workspace."""
    if folder := os.environ.get(HOME_VARIABLE):
        return Path(folder).expanduser()
    return Path.home() / ".mlx-decision"


@dataclass(frozen=True)
class Config:
    """The file's values; None (or no models) where the file sets none.

    ``models`` are references as ``-m`` takes them; ``default_model`` is a
    name among them, as ``--default-model`` takes it, or a reference of its
    own. Paths start with ``/``, ``~`` or ``./`` (relative to the file's
    folder). ``warnings`` name the keys the file has that are not read.
    """

    path: Path
    models: tuple[str, ...] = ()
    default_model: str | None = None
    memory_budget_gb: float | None = None
    host: str | None = None
    port: int | None = None
    workspace: Path | None = None
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def resolve(self, flag: Any, key: str, default: Any = None) -> Any:
        """The flag's value if given, else the file's, else ``default``."""
        if flag is not None:
            return flag
        value = getattr(self, key)
        return default if value is None or value == () else value

    def default_reference(self) -> str | None:
        """What a one-model command loads without ``-m``: the reference in ``models``
        that names ``default_model``, else ``default_model`` taken as a reference.

        Raises ``ConfigError`` for a bare name that is none of the models.
        """
        from .pool import find_reference

        name = self.default_model
        if name is None:
            return None
        if (reference := find_reference(self.models, name)) is not None:
            return reference
        if self.models and "/" not in name and not name.startswith(("~", ".")):
            raise ConfigError(
                f"{self.path}: default_model names none of the models in models: {name}"
            )
        return name

    def model_references(self) -> list[str]:
        """What a many-model command serves without ``-m``: ``models``, plus
        ``default_model`` when it is a reference of its own."""
        from .pool import find_reference

        references = list(self.models)
        reference = self.default_reference()
        if reference is not None and find_reference(references, self.default_model) is None:
            references.append(reference)
        return references

    def default_name(self, names: list[str]) -> str | None:
        """``default_model`` as one of ``names`` (the models in use), else None."""
        from .pool import model_name

        name = self.default_model
        if name is None or name in names:
            return name
        return model_name(name) if model_name(name) in names else None


def load(path: Path | None = None) -> Config:
    """Read the config file (``home()/config.toml`` by default).

    A missing file gives a config without values. Raises ``ConfigError``
    naming the file (and the key) for a file that is not valid TOML and for
    a value of the wrong type.
    """
    path = path if path is not None else home() / FILE_NAME
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        return Config(path)
    except OSError as error:
        raise ConfigError(f"{path}: cannot read it: {error.strerror or error}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from None

    values: dict[str, Any] = {}
    warnings = []
    for key, value in data.items():
        read = _READERS.get(key)
        if read is None:
            hint = _HINTS.get(key, "it is not read")
            warnings.append(f"{path}: unknown key {key!r} ({hint})")
            continue
        try:
            values[key] = read(value, path.parent)
        except _WrongValue as error:
            raise ConfigError(f"{path}: {key} {error}") from None
    return Config(path, **values, warnings=tuple(warnings))


class _WrongValue(Exception):
    pass


def _local(text: str, folder: Path) -> str:
    """``~`` expanded; a path starting with ``./`` or ``../`` taken from ``folder``."""
    if text.startswith("~"):
        return str(Path(text).expanduser())
    if text.startswith(("./", "../")) or text in (".", ".."):
        return str(folder / text)
    return text


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise _WrongValue(f"must be a string, not {_kind(value)}")
    if not value.strip():
        raise _WrongValue("must not be empty")
    return value.strip()


def _models(value: Any, folder: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _WrongValue(
            f'must be a list of models (folders or repo ids, as -m), like ["~/Models/laya"], '
            f"not {_kind(value)}"
        )
    try:
        return tuple(_local(_text(item), folder) for item in value)
    except _WrongValue as error:
        raise _WrongValue(f"entries {error}") from None


def _default_model(value: Any, folder: Path) -> str:
    return _local(_text(value), folder)


def _memory_budget(value: Any, folder: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _WrongValue(f"must be a number of GB, not {_kind(value)}")
    if value < 0:
        raise _WrongValue("must not be negative")
    return float(value)


def _host(value: Any, folder: Path) -> str:
    return _text(value)


def _port(value: Any, folder: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _WrongValue(f"must be a whole number, not {_kind(value)}")
    if not 0 < value < 65536:
        raise _WrongValue("must be between 1 and 65535")
    return value


def _workspace(value: Any, folder: Path) -> Path:
    path = Path(_text(value)).expanduser()
    return path if path.is_absolute() else folder / path


def _kind(value: Any) -> str:
    """A TOML value's type in the file's words."""
    names = {
        bool: "true or false",
        int: "a whole number",
        float: "a number",
        str: "a string",
        list: "a list",
        dict: "a table",
    }
    return names.get(type(value), "a date or time")


_READERS = {
    "models": _models,
    "default_model": _default_model,
    "memory_budget_gb": _memory_budget,
    "host": _host,
    "port": _port,
    "workspace": _workspace,
}

_HINTS = {
    "api_key": "the API key is not read from the file: pass --api-key or set MLX_DECISION_API_KEY",
}
