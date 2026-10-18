"""The models the Studio serves: named ones, its own, and the Hugging Face cache's.

``web`` serves the models ``-m`` or the config file names (as ``server``
does), plus the models the Studio made in the workspace's ``models/`` and
the known families' complete downloads in the Hugging Face cache. Nothing
is loaded or downloaded to list them. When names clash the named model
keeps its name: a workspace model of the same name is left out, a cached
one is listed by its repo id.
"""

import dataclasses
from collections.abc import Callable, Iterable
from pathlib import Path

from ..hub import cached_models
from ..pool import ModelSpec, discover, folder_models, model_name


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
