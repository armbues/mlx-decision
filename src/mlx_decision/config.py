"""The config file: defaults for the commands' flags.

``~/.mlx-decision/config.toml`` (``MLX_DECISION_HOME`` moves the folder)
holds seven keys. A flag on the command line wins over the file, the file
over the built-in default. The file is optional and only read, except
that ``set_models_dir`` adds that key when the user chooses a folder.

``models_dir`` is the folder where models live (or ``"hf-cache"``): its
models are served where ``-m`` is absent, as ``models`` are, and ``-m
NAME`` finds them by name.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HOME_VARIABLE = "MLX_DECISION_HOME"
FILE_NAME = "config.toml"
# ``models_dir``'s value for the Hugging Face cache.
HF_CACHE = "hf-cache"


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
    folder). ``models_dir`` is a folder of models or ``HF_CACHE``.
    ``warnings`` name the keys the file has that are not read and a
    ``models_dir`` that does not exist.
    """

    path: Path
    models: tuple[str, ...] = ()
    default_model: str | None = None
    memory_budget_gb: float | None = None
    host: str | None = None
    port: int | None = None
    workspace: Path | None = None
    models_dir: Path | str | None = None
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def resolve(self, flag: Any, key: str, default: Any = None) -> Any:
        """The flag's value if given, else the file's, else ``default``."""
        if flag is not None:
            return flag
        value = getattr(self, key)
        return default if value is None or value == () else value

    @property
    def models_folder(self) -> Path | None:
        """``models_dir`` when it is a folder that exists (not the cache)."""
        folder = self.models_dir
        return folder if isinstance(folder, Path) and folder.is_dir() else None

    def lookup(self, name: str) -> str | None:
        """The model ``name`` names among ``models_dir``'s models and ``models``: a
        subfolder of ``models_dir``, else a reference in ``models`` (``find_reference``).
        None for anything but a bare name and for a name not found."""
        from .pool import find_reference

        if not is_bare_name(name):
            return None
        folder = self.models_folder
        if folder is not None and (reference := find_reference([str(folder)], name)):
            return reference
        return find_reference(self.models, name)

    def model_names(self) -> list[str]:
        """The names ``lookup`` knows, in order (without loading or downloading)."""
        from .pool import folder_models, model_name

        names = folder_models(self.models_folder) if self.models_folder else []
        for reference in self.models:
            path = Path(reference).expanduser()
            listed = folder_models(path) if path.is_dir() else []
            names += listed or [model_name(reference)]
        return list(dict.fromkeys(names))

    def default_reference(self) -> str | None:
        """What a one-model command loads without ``-m``: the model ``default_model``
        names (``lookup``), else ``default_model`` taken as a reference.

        Raises ``ConfigError`` for a bare name that is none of the models.
        """
        name = self.default_model
        if name is None:
            return None
        if (reference := self.lookup(name)) is not None:
            return reference
        if (self.models or self.models_folder) and is_bare_name(name):
            raise ConfigError(
                f"{self.path}: default_model names none of the models in {self.places()}: {name}"
            )
        return name

    def places(self) -> str:
        """The keys names are looked up in, for messages: ``models_dir or models``."""
        keys = [key for key in ("models_dir", "models") if getattr(self, key)]
        return " or ".join(keys) or "models_dir or models"

    def model_references(self) -> list[str]:
        """What a many-model command serves without ``-m``: ``models_dir`` when it
        has models, ``models``, and ``default_model`` when it is a reference of its
        own."""
        from .pool import folder_models

        references = list(self.models)
        reference = self.default_reference()
        if reference is not None and self.lookup(self.default_model) is None:
            references.append(reference)
        folder = self.models_folder
        if folder is not None and folder_models(folder):
            # What is models_dir or one of its models is listed by it.
            inside = {folder.resolve()} | {path.resolve() for path in folder.iterdir()}
            references = [str(folder)] + [
                reference
                for reference in references
                if Path(reference).expanduser().resolve() not in inside
            ]
        return references

    def default_name(self, names: list[str]) -> str | None:
        """``default_model`` as one of ``names`` (the models in use), else None."""
        from .pool import model_name

        name = self.default_model
        if name is None or name in names:
            return name
        return model_name(name) if model_name(name) in names else None


def is_bare_name(text: str) -> bool:
    """Whether ``text`` is a model's name rather than a path or a repo id."""
    return bool(text) and "/" not in text and not text.startswith(("~", "."))


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
    folder = values.get("models_dir")
    if isinstance(folder, Path) and not folder.exists():
        warnings.append(f"{path}: models_dir {folder} does not exist (no models listed from it)")
    return Config(path, **values, warnings=tuple(warnings))


def set_models_dir(value: Path | None, path: Path | None = None) -> str:
    """Add ``models_dir`` to the config file: ``value``, or the Hugging Face cache
    for None. Returns the line added.

    The one write to the file: it is created if missing; otherwise the line
    goes before the first table (or at the end) and every other byte is kept.
    A folder under the home folder is written with ``~``. Raises
    ``ConfigError`` when the file already has the key, is not valid TOML, or
    cannot be written.
    """
    import json
    import re

    path = path if path is not None else home() / FILE_NAME
    if value is None:
        text = HF_CACHE
    else:
        folder = Path(value).expanduser().absolute()
        try:
            text = "~/" + str(folder.relative_to(Path.home()))
        except ValueError:
            text = str(folder)
    line = f"models_dir = {json.dumps(text, ensure_ascii=False)}"
    try:
        old = path.read_bytes().decode() if path.exists() else ""
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"{path}: cannot read it: {error}") from None
    try:
        if "models_dir" in tomllib.loads(old):
            raise ConfigError(f"{path}: already sets models_dir")
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from None
    table = re.search(r"^[ \t]*\[", old, re.MULTILINE)
    if table is not None:
        new = old[: table.start()] + line + "\n\n" + old[table.start() :]
    else:
        new = old + ("\n" if old and not old.endswith("\n") else "") + line + "\n"
    try:
        added = tomllib.loads(new).get("models_dir")
    except tomllib.TOMLDecodeError:
        added = None
    if added != text:  # what looked like a table header was inside a value
        raise ConfigError(f"{path}: cannot add models_dir to it; add {line} yourself")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(new.encode())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise ConfigError(f"{path}: cannot write it: {error.strerror or error}") from None
    return line


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


def _models_dir(value: Any, folder: Path) -> Path | str:
    text = _text(value)
    if text == HF_CACHE:
        return HF_CACHE
    path = Path(text).expanduser()
    path = path if path.is_absolute() else folder / path
    if path.exists() and not path.is_dir():
        raise _WrongValue(f'must be a folder (or "{HF_CACHE}"), not a file: {path}')
    return path


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
    "models_dir": _models_dir,
}

_HINTS = {
    "api_key": "the API key is not read from the file: pass --api-key or set MLX_DECISION_API_KEY",
}
