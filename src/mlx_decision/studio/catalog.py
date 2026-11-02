"""The models the Studio serves: named ones, its own, and the Hugging Face cache's.

``web`` serves the models ``-m`` or the config file names (as ``server``
does), plus the models the Studio made in the workspace's ``models/`` and
the known families' complete downloads in the Hugging Face cache. Nothing
is loaded or downloaded to list them. When names clash the named model
keeps its name: a workspace model of the same name is left out, a cached
one is listed by its repo id.
"""

import dataclasses
import json
import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path

from ..hub import cached_models, snapshot_revision
from ..memory import weights_summary
from ..picker import folder_size
from ..pool import ModelSpec, discover, folder_models, model_name
from ..registry import MARKER_FILE


def studio_models(
    references: Iterable[str],
    workspace_models: Path | None,
    options: dict | None = None,
    include_cache: bool = True,
    on_skip: Callable[[Path, str], None] = lambda path, reason: None,
) -> list[ModelSpec]:
    """The models for the Studio's pool, named ones first.

    ``references`` are what ``-m`` or the config file gives (errors in them
    are raised, as ``discover`` raises them); the workspace's and the cache's
    models are added where they can be read, the others passed to
    ``on_skip``. ``options`` go to every model whose family takes them; only
    the named models must take each one.
    """
    references = list(references)
    specs = discover(references, options, on_skip) if references else []
    names = {spec.name for spec in specs}
    paths = {spec.path.resolve() for spec in specs}

    def add(spec: ModelSpec, name: str) -> None:
        specs.append(spec if spec.name == name else dataclasses.replace(spec, name=name))
        names.add(name)
        paths.add(spec.path.resolve())

    if workspace_models is not None and folder_models(workspace_models):
        for spec in discover([workspace_models], options, on_skip, strict_options=False):
            if spec.path.resolve() in paths:
                continue
            if spec.name in names:
                on_skip(spec.path, f"{spec.path}: a model named {spec.name} is listed already")
                continue
            add(spec, spec.name)

    if include_cache:
        for repo_id, snapshot in cached_models():
            if snapshot.resolve() in paths:
                continue
            try:
                [spec] = discover([snapshot], options, on_skip, strict_options=False)
            except (ValueError, OSError) as error:
                on_skip(snapshot, f"{repo_id}: {error}")
                continue
            name = model_name(repo_id)
            add(spec, repo_id if name in names else name)
    return specs


@dataclasses.dataclass
class Catalog:
    """Where the Studio's models come from (``studio_models``' arguments), kept
    so it can look again after a download, convert or fit.

    ``models_dir`` is the config's folder of models when it serves them (no
    ``-m``): scanned once it holds a model, so the first download into an
    empty folder is found too. ``downloads`` are the folders the Studio
    downloaded, which ``references`` may hold too (``add_download``).
    """

    references: list[str] = dataclasses.field(default_factory=list)
    workspace_models: Path | None = None
    options: dict = dataclasses.field(default_factory=dict)
    include_cache: bool = True
    models_dir: Path | None = None
    downloads: list[str] = dataclasses.field(default_factory=list)

    def scan(self, on_skip: Callable[[Path, str], None] = lambda path, reason: None):
        """The models there are now. A named model that cannot be read any more
        is passed to ``on_skip`` and left out; the others are still listed."""
        references = self._references()
        try:
            return studio_models(
                references, self.workspace_models, self.options, self.include_cache, on_skip
            )
        except (ValueError, OSError):
            readable = []
            for reference in references:
                try:
                    discover([reference], self.options, on_skip)
                except (ValueError, OSError) as error:
                    on_skip(Path(reference), f"{reference}: {error}")
                    continue
                readable.append(reference)
            return studio_models(
                readable, self.workspace_models, self.options, self.include_cache, on_skip
            )

    def add_download(self, folder: Path) -> None:
        """A model the Studio downloaded into ``folder``: listed from now on,
        also when ``models_dir`` is not scanned (``web -m ...``)."""
        folder = Path(folder)
        within = self.models_dir is not None
        scanned = within and folder.parent.resolve() == self.models_dir.resolve()
        if str(folder) not in self.downloads:
            self.downloads.append(str(folder))
        if not scanned and str(folder) not in self.references:
            self.references.append(str(folder))

    def named(self) -> list[str]:
        """The references named with ``-m`` or in the config's ``models``: all
        but the Studio's downloads."""
        downloads = set(self.downloads)
        return [reference for reference in self.references if reference not in downloads]

    def _references(self) -> list[str]:
        """``references``, and ``models_dir`` last when it has models and is not
        among them."""
        references = list(self.references)
        folder = self.models_dir
        if folder is None or not folder_models(folder):
            return references
        listed = set()
        for reference in references:
            try:
                listed.add(Path(reference).expanduser().resolve())
            except OSError:
                continue
        if folder.resolve() not in listed:
            references.append(str(folder))
        return references


def model_facts(spec: ModelSpec, workspace_models: Path | None) -> dict:
    """What the Models page says about a model, read from its files (nothing loaded).

    ``maker`` and ``family_title`` from the family's listing; ``parameters``
    and ``precision`` from the weights it loads (a fitted model: its base's);
    ``place``: "cache" (with ``repo`` and ``revision``), "studio" (the
    workspace's ``models/``) or "folder" (the Models route tells "models",
    a folder in ``models_dir``, from it); ``link``, whether the folder is a
    link; ``converted_from``, the repo a converted copy was made from;
    ``disk_bytes``, the folder's size; ``licence_note``, the family's note
    on terms of use.
    """
    family = spec.family
    weights = spec.fit.base if spec.fit is not None else spec.path
    try:
        parameters, precision = weights_summary(weights, spec.options)
    except (OSError, ValueError, KeyError):
        parameters, precision = None, None
    path = spec.path
    cached = snapshot_revision(path)
    if cached is not None:
        place = "cache"
    elif workspace_models is not None and path.resolve().is_relative_to(workspace_models.resolve()):
        place = "studio"
    else:
        place = "folder"
    try:
        marker = json.loads((path / MARKER_FILE).read_text())
        converted_from = marker.get("source") if isinstance(marker, dict) else None
    except (OSError, ValueError):
        converted_from = None
    return {
        "family_title": family.listing.title if family.listing else family.name,
        "maker": family.listing.maker if family.listing else None,
        "parameters": parameters,
        "precision": precision,
        "path": str(path),
        "place": place,
        "link": path.is_symlink(),
        "repo": cached[0] if cached else None,
        "revision": cached[1] if cached else None,
        "converted_from": converted_from if isinstance(converted_from, str) else None,
        "disk_bytes": folder_size(path),
        "licence_note": family.licence,
    }


def repo_size(snapshot: Path) -> int:
    """The bytes a Hugging Face cache repo holds, every revision: the files in
    its folder, links not followed (a snapshot's links count once, as their
    blobs; payloads in the cache's shared store are left out, since another
    repo may hold them)."""
    total = 0
    for folder, _, files in os.walk(Path(snapshot).resolve().parents[1]):
        for name in files:
            try:
                info = os.lstat(os.path.join(folder, name))
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


# Load options as the CLI's flags take them.
OPTION_FLAGS = {
    "max_input_tokens": "--max-input-tokens",
    "prefix_cache_gb": "--prefix-cache",
    "noul_floor": "--noul-floor",
}


def option_arguments(options: dict) -> list[str]:
    """The flags that give a model these load options (``load_options`` reversed)."""
    arguments = []
    if "max_image_pixels" in options:
        pixels = options["max_image_pixels"]
        arguments += ["--max-image-mp", "0" if not pixels else f"{pixels / 2**20:g}"]
    for key, flag in OPTION_FLAGS.items():
        if options.get(key) is not None:
            arguments += [flag, f"{options[key]:g}"]
    return arguments


def cli_arguments(spec: ModelSpec, config, repo: str | None) -> dict[str, list[str]]:
    """What a command in a terminal passes to get this model, as it is served.

    ``short``: ``-m`` with the name when the config file names it (as
    ``run -m NAME`` looks it up), else as ``full``; ``full``: ``-m`` with the
    repo id of a model in the Hugging Face cache, else its folder (for an MCP
    client's config, which reads no config file of ours). Both end with the
    load options as flags.
    """
    full = repo if repo is not None and spec.fit is None else str(spec.path)
    short = full
    try:
        found = config.lookup(spec.name)
    except (ValueError, OSError):
        found = None
    if found is not None and (
        found == repo
        or (
            Path(found).expanduser().exists()
            and Path(found).expanduser().resolve() == spec.path.resolve()
        )
    ):
        short = spec.name
    flags = option_arguments(spec.options)
    return {"short": ["-m", short, *flags], "full": ["-m", full, *flags]}
