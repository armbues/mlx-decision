"""The Studio's workspace: a folder of plain files the CLI reads as they are.

Layout (``README.md`` in the folder says the same for its readers)::

    config.toml                  the config file (only read here)
    sets/<name>.json             question sets: request bodies for ``run -q``
    history/<name>.jsonl         one line per run of a set
    history/images/<sha256>.<ext>  the images of those runs, each stored once
    data/<name>/                 datasets: dataset.json and one file per split
    data/<name>.jsonl            or one file of training data (``training fit --data``)
    batches/<name>/              a dataset's rows answered by a model: batch.json,
                                 questions.json, rows.jsonl, results.jsonl
    models/<name>/               converted and fitted models (``-m``)
    jobs/<id>/                   one folder per job
    speed/<fingerprint>.jsonl    measured request times per model

Names are ``[A-Za-z0-9._-]``, at most 80 characters, not starting with a dot;
every path is resolved and must stay inside the workspace, so a name or a
symlink cannot reach a file elsewhere. Whole files are written to a hidden
temporary file beside them and renamed into place. A set is saved only over
the version it was loaded from (its SHA-256), so a set changed on disk in
the meantime is not overwritten.

The methods block on file work: the Studio calls them off the event loop.
``on_change``, when set, is called after each change to a set or a history
(on the thread that made it) with what changed: ``folder`` (``sets`` or
``history``), ``name``, ``action`` and details (``old_name`` for a rename,
``id`` for a run added).
"""

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Config, home
from ..data.dataset import DATASET_FILE, DATASET_FORMAT, check_format, open_dataset
from ..errors import DecisionError
from ..images import check_data_urls

FOLDERS = ("sets", "history", "data", "batches", "models", "jobs", "speed")
NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}")
SET_SUFFIX = ".json"
DATA_SUFFIX = ".jsonl"
HISTORY_SUFFIX = ".jsonl"
SPEED_SUFFIX = ".jsonl"
IMAGES = "images"
JOB_FILE = "job.json"
# Speed records kept per model; the file is trimmed back to this once it is
# a tenth longer, so not every request rewrites it.
SPEED_KEEP = 1000
SPEED_SLACK = SPEED_KEEP // 10
# Hidden leftovers of an interrupted write, download, convert or fit (a
# fit's stored hidden states, a quantized download's full-size files, a
# folder moved aside to be removed).
STALE_SUFFIXES = (".tmp", ".partial", ".states", ".download", ".removing")
INTERRUPTED = "interrupted"
DATA_URL = re.compile(r"data:image/([A-Za-z0-9.+-]+)((?:;[^;,]*)*?);base64,(.*)", re.S | re.I)
EXTENSIONS = {"jpeg": "jpg", "svg+xml": "svg"}

logger = logging.getLogger(__name__)

README = """\
# mlx-decision workspace

The Studio (`mlx-decision web`) keeps its files here. Each one is a plain
file in a format the command line reads as it is. The Studio rewrites this
README when it starts.

- `config.toml`: defaults for the commands' flags; read by `run`, `chat`,
  `server`, `mcp`, `web`, `training` and `benchmark`. It lives in
  `~/.mlx-decision` (or `MLX_DECISION_HOME`), so it is not here when
  `--workspace` or the file's `workspace` moved the workspace.
- `sets/`: question sets. Each file is a request body (`state`,
  `questions`, `images` as data URLs):
  `mlx-decision run -q sets/NAME.json -m MODEL` answers it as the Studio does.
- `history/`: the Playground's runs, one JSON-lines file per set (a
  dataset's questions tried in the Playground: `history/datasets/`), one line
  per run (request, model, answers, timings). Images are stored once in
  `history/images/`, named by their SHA-256, and the lines refer to them.
  Clearing a set's history in the Studio deletes its file.
- `data/`: datasets, each a folder (`dataset.json` with its questions and
  what each label value means, and one JSON-lines file per split) or a
  single JSON-lines file: `mlx-decision training fit -m MODEL --data data/NAME`.
- `batches/`: a dataset's rows answered by a model, one folder per batch:
  `batch.json` (what it ran, how far it got), `questions.json`, `rows.jsonl`
  and `results.jsonl` (one line per row answered).
  `mlx-decision run -m MODEL -q batches/NAME/questions.json --states
  batches/NAME/rows.jsonl --json` answers the same rows.
- `models/`: models the Studio converted or fitted, each a model folder:
  `mlx-decision run -m models/NAME`.
- `jobs/`: one folder per job (download, convert, fit, eval): `job.json`
  (the command, times, outcome), `events.jsonl` and `stderr.log`.
- `speed/`: how long requests took on this Mac, per model (the latest
  1,000 each); the Studio's Models page shows the median.
"""


class WorkspaceError(Exception):
    """A request the workspace cannot carry out; ``status`` is the HTTP status
    the Studio's API answers with."""

    status = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFound(WorkspaceError):
    status = 404


class Conflict(WorkspaceError):
    """The file changed on disk since it was loaded, or the name is taken.
    ``current`` is the SHA-256 of what is there now (None when nothing is)."""

    status = 409

    def __init__(self, message: str, current: str | None = None):
        super().__init__(message)
        self.current = current


def check_name(name: Any, what: str = "name") -> str:
    """``name`` if it can name a file in the workspace, else ``WorkspaceError``."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise WorkspaceError(
            f"{what} must be 1 to 80 letters, digits, '.', '_' or '-', "
            f"not starting with '.': {name!r}"
        )
    return name


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> bytes:
    """A set's file: indented JSON, UTF-8, a newline at the end."""
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def line(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def now() -> str:
    """The time for a history or speed line: ISO 8601 in UTC, to the millisecond."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def root_folder(config: Config | None = None, flag: Path | None = None) -> Path:
    """The workspace: ``--workspace``, else the config's ``workspace``, else ``home()``."""
    if flag is not None:
        return Path(flag).expanduser()
    if config is not None and config.workspace is not None:
        return config.workspace
    return home()


class Workspace:
    """The files under ``root``. Writes are serialised by one lock, so the tabs of
    one Studio cannot interleave them."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.lock = threading.Lock()
        self.on_change: Callable[[dict[str, str]], None] | None = None

    def changed(self, folder: str, name: str, action: str, **details: Any) -> None:
        if self.on_change is not None:
            try:
                self.on_change({"folder": folder, "name": name, "action": action, **details})
            except Exception:
                logger.exception("the workspace's change callback failed")

    # Layout and paths

    def open(self) -> list[Path]:
        """Create the layout and the README, then clean up after a previous run:
        returns what ``clean_up`` removed."""
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in FOLDERS:
            (self.root / folder).mkdir(exist_ok=True)
        (self.root / "history" / IMAGES).mkdir(exist_ok=True)
        readme = self.root / "README.md"
        if not readme.is_file() or readme.read_text(encoding="utf-8") != README:
            self.write(readme, README.encode("utf-8"))
        return self.clean_up()

    def path(self, folder: str, name: str, suffix: str = "", what: str = "name") -> Path:
        """``folder/<name><suffix>``, checked to be inside the workspace."""
        check_name(name, what)
        return self.inside(self.root / folder / f"{name}{suffix}")

    def inside(self, path: Path) -> Path:
        """``path`` resolved (symlinks followed), refused unless under the workspace."""
        resolved = path.resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise WorkspaceError(f"not inside the workspace: {path}")
        return resolved

    def write(self, path: Path, data: bytes) -> None:
        """Write ``path`` whole: a hidden temporary file beside it, renamed over it."""
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            with open(temporary, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def clean_up(self) -> list[Path]:
        """Remove hidden ``.tmp`` and ``.partial`` leftovers from the workspace's
        folders (not inside models) and mark jobs that never ended as interrupted.
        Returns the paths removed and the job files changed."""
        changed = []
        places = [self.root, *(self.root / f for f in FOLDERS), self.root / "history" / IMAGES]
        for place in places:
            for path in _entries(place):
                if path.name.startswith(".") and path.name.endswith(STALE_SUFFIXES):
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                    changed.append(path)
        for folder in _entries(self.root / "jobs"):
            job_file = folder / JOB_FILE
            if folder.name.startswith(".") or not job_file.is_file():
                continue
            try:
                job = json.loads(job_file.read_bytes())
            except (OSError, ValueError):
                continue
            if isinstance(job, dict) and not job.get("outcome"):
                job["outcome"] = INTERRUPTED
                job["state"] = "ended"
                job.setdefault("ended", None)
                self.write(job_file, encode(job))
                changed.append(job_file)
        return changed

    # Datasets: a dataset folder data/NAME/, or a plain file data/NAME.jsonl

    def data_path(self, name: str) -> Path:
        """Where a plain data file of ``name`` is (or goes)."""
        return self.path("data", name, DATA_SUFFIX, "a dataset's name")

    def dataset_folder(self, name: str) -> Path:
        """Where a dataset folder of ``name`` is (or goes)."""
        return self.path("data", name, what="a dataset's name")

    def data_taken(self, name: str) -> bool:
        """Whether a dataset of ``name`` is there (a folder or a file)."""
        return self.dataset_folder(name).exists() or self.data_path(name).exists()

    def dataset_location(self, name: str) -> Path:
        """The folder (else the plain file) of dataset ``name``; ``NotFound``."""
        folder = self.dataset_folder(name)
        if (folder / DATASET_FILE).is_file():
            return folder
        path = self.data_path(name)
        if path.is_file():
            return path
        raise NotFound(f"no dataset named {name!r}")

    def list_data(self) -> list[dict[str, Any]]:
        """The datasets, by name: each as ``dataset_summary`` gives it."""
        found = []
        for path in _entries(self.root / "data"):
            if path.name.startswith("."):
                continue
            if path.is_dir() and (path / DATASET_FILE).is_file() and NAME.fullmatch(path.name):
                found.append(dataset_summary(path))
            elif path.is_file() and path.name.endswith(DATA_SUFFIX):
                name = path.name[: -len(DATA_SUFFIX)]
                if NAME.fullmatch(name) and not (path.parent / name / DATASET_FILE).is_file():
                    found.append(dataset_summary(path))
        return sorted(found, key=lambda item: (item["name"].casefold(), item["name"]))

    def save_data(self, name: str, data: bytes) -> Path:
        """Write a new plain data file (checked by the caller); ``Conflict``
        when a dataset of that name is there."""
        path = self.data_path(name)
        with self.lock:
            if self.data_taken(name):
                raise Conflict(f"a dataset named {name} exists already")
            self.write(path, data)
        self.changed("data", name, "saved")
        return path

    def delete_data(self, name: str) -> None:
        """Delete a dataset (its folder, or its file); ``NotFound`` when there is none."""
        with self.lock:
            path = self.dataset_location(name)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            history = self.history_path(f"{DATASET_DOCUMENT}{name}")
            if history.exists():
                history.unlink()
                self._prune_images()
        self.changed("data", name, "deleted")

    def save_dataset_questions(
        self,
        name: str,
        questions: dict[str, Any] | None,
        answers: dict[str, Any],
        loaded: str | None = None,
    ) -> str:
        """Write a dataset folder's questions and answers into its
        ``dataset.json`` (checked by the caller), the rest as it is; returns
        the file's new SHA-256. With ``loaded`` (the SHA-256 it was read with),
        ``Conflict`` when it changed since."""
        with self.lock:
            folder = self.dataset_location(name)
            if not folder.is_dir():
                raise WorkspaceError(f"{name} is a plain file: its lines hold their own questions")
            path = folder / DATASET_FILE
            if loaded is not None and (current := _hash_of(path)) != loaded:
                raise Conflict(
                    f"{name}'s {DATASET_FILE} changed on disk since it was loaded", current
                )
            body = json.loads(path.read_text(encoding="utf-8"))
            check_format(path, body)
            ordered: dict[str, Any] = {"format": body.get("format", DATASET_FORMAT)}
            if questions:
                ordered["questions"] = questions
            if answers:
                ordered["answers"] = answers
            ordered.update(
                {
                    key: value
                    for key, value in body.items()
                    if key not in ("format", "questions", "answers")
                }
            )
            data = (json.dumps(ordered, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            self.write(path, data)
        self.changed("data", name, "saved")
        return sha256(data)

    # Sets

    def set_path(self, name: str) -> Path:
        return self.path("sets", name, SET_SUFFIX, "a set's name")

    def history_path(self, name: str) -> Path:
        """A Playground document's history: a set's ``history/NAME.jsonl``; a
        dataset opened in the Playground (``data:NAME``) has
        ``history/datasets/NAME.jsonl``."""
        dataset = dataset_document(name)
        if dataset is not None:
            folder = self.root / "history" / DATASET_HISTORY
            folder.mkdir(exist_ok=True)
            return self.inside(folder / f"{dataset}{HISTORY_SUFFIX}")
        return self.path("history", name, HISTORY_SUFFIX, "a set's name")

    def document_exists(self, name: str) -> bool:
        """Whether a Playground document is there: a set, or a dataset folder."""
        dataset = dataset_document(name)
        if dataset is not None:
            return (self.dataset_folder(dataset) / DATASET_FILE).is_file()
        return self.set_path(name).is_file()

    def list_sets(self) -> list[dict[str, Any]]:
        """The sets, by name (case ignored): name, size in bytes, modification time."""
        sets = []
        for path in _entries(self.root / "sets"):
            name = path.name.removesuffix(SET_SUFFIX)
            if path.suffix != SET_SUFFIX or not NAME.fullmatch(name) or not path.is_file():
                continue
            stat = path.stat()
            sets.append({"name": name, "bytes": stat.st_size, "modified": stat.st_mtime})
        return sorted(sets, key=lambda entry: (entry["name"].lower(), entry["name"]))

    def load_set(self, name: str) -> tuple[Any, str]:
        """The set's body and the SHA-256 of its file, to save over it later."""
        path = self.set_path(name)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise NotFound(f"no set named {name!r}") from None
        try:
            body = json.loads(data)
        except ValueError as error:
            raise WorkspaceError(f"{path.name} is not valid JSON: {error}") from None
        if not isinstance(body, dict):
            # Opened as an empty set, it would be saved over (a set is a JSON object).
            raise WorkspaceError(
                f"{path.name} is not a question set: a set is a JSON object with questions"
            )
        return body, sha256(data)

    def save_set(self, name: str, body: Any, loaded: str | None = None) -> str:
        """Write the set; returns the SHA-256 of the file written.

        ``loaded`` is the SHA-256 the set was loaded with: the file must still
        be that version. Without it the set is new and the name must be free.
        Either way ``Conflict`` otherwise, with the hash of what is there.
        """
        path = self.set_path(name)
        check_body(body)
        data = encode(body)
        with self.lock:
            current = _hash_of(path)
            if loaded is None and current is not None:
                raise Conflict(f"a set named {name!r} exists already", current)
            if loaded is not None and current != loaded:
                what = "was deleted" if current is None else "changed on disk"
                raise Conflict(f"the set {name!r} {what} since it was loaded", current)
            self.write(path, data)
        self.changed("sets", name, "saved")
        return sha256(data)

    def save_new_set(self, name: str, body: Any, tries: int = 100) -> tuple[str, str]:
        """Save a new set under ``name`` or, when taken, the first free
        ``name-2``, ``name-3``, ...; returns the name and the file's SHA-256."""
        for attempt in range(1, tries + 1):
            candidate = name if attempt == 1 else f"{name[:76]}-{attempt}"
            try:
                return candidate, self.save_set(candidate, body)
            except Conflict:
                continue
        raise Conflict(f"no free name for {name!r}")

    def rename_set(self, name: str, new_name: str) -> None:
        """Rename the set and its history; the new name must be free."""
        source, target = self.set_path(name), self.set_path(new_name)
        history, new_history = self.history_path(name), self.history_path(new_name)
        if new_name == name:
            return
        with self.lock:
            if not source.is_file():
                raise NotFound(f"no set named {name!r}")
            # A change of case only names the same file on a case-insensitive disk.
            same = target.exists() and os.path.samefile(source, target)
            if not same and (target.exists() or new_history.exists()):
                raise Conflict(f"a set named {new_name!r} exists already", _hash_of(target))
            os.rename(source, target)
            if history.exists():
                os.rename(history, new_history)
        self.changed("sets", new_name, "renamed", old_name=name)

    def delete_set(self, name: str) -> None:
        """Delete the set and its history (and images no other history uses)."""
        path, history = self.set_path(name), self.history_path(name)
        with self.lock:
            if not path.is_file():
                raise NotFound(f"no set named {name!r}")
            path.unlink()
            history.unlink(missing_ok=True)
            self._prune_images()
        self.changed("sets", name, "deleted")

    # History

    def append_history(self, name: str, run: dict[str, Any]) -> dict[str, Any] | None:
        """Add a run to the set's history and return the line as stored; None
        when the set does not exist (deleted while the run was answered), so no
        history is left without its set.

        ``run`` holds what the Studio keeps of it (``model``, ``fingerprint``,
        ``request``, ``response``, timings); ``id`` and ``time`` are added. The
        request's data-URL images go to ``history/images/`` and the line holds
        ``{"sha256": ..., "type": ...}`` in their place.
        """
        path = self.history_path(name)
        entry = {"id": uuid.uuid4().hex, "time": now(), **run}
        with self.lock:
            if not self.document_exists(name):
                return None
            if isinstance(request := entry.get("request"), dict):
                entry["request"] = self._store_images(request)
            data = line(entry)
            with open(path, "ab") as handle:
                if handle.tell() and not _ends_with_newline(path):
                    data = b"\n" + data  # after a line cut short by a crash
                handle.write(data)
        self.changed("history", name, "added", id=entry["id"])
        return entry

    def read_history(self, name: str) -> list[dict[str, Any]]:
        """The set's runs, oldest first (lines that do not parse are skipped)."""
        return [entry for entry in _read_lines(self.history_path(name)) if isinstance(entry, dict)]

    def clear_history(self, name: str) -> None:
        path = self.history_path(name)
        with self.lock:
            path.unlink(missing_ok=True)
            self._prune_images()
        self.changed("history", name, "cleared")

    def delete_run(self, name: str, run_id: str) -> list[str]:
        """Delete one entry of the set's history: the run ``run_id`` and, when it
        belongs to a Run All or a Compare, the other runs of that group (the
        History lists them as one). Images no history uses any more go too.
        Returns the deleted runs' ids; raises ``NotFound``."""
        path = self.history_path(name)
        with self.lock:
            try:
                raws = path.read_bytes().splitlines()
            except FileNotFoundError:
                raws = []
            parsed = []
            for raw in raws:
                try:
                    entry = json.loads(raw)
                except ValueError:
                    entry = None
                parsed.append(entry if isinstance(entry, dict) else None)
            target = next((entry for entry in parsed if entry and entry.get("id") == run_id), None)
            if target is None:
                raise NotFound(f"no run {run_id!r} in the history of {name!r}")
            group = next(
                ((key, target[key]) for key in ("batch", "compare") if target.get(key)), None
            )

            def goes(entry: dict | None) -> bool:
                if entry is None:
                    return False
                if group is not None:
                    return entry.get(group[0]) == group[1]
                return entry.get("id") == run_id

            deleted = [entry["id"] for entry in parsed if goes(entry)]
            kept = [
                raw
                for raw, entry in zip(raws, parsed, strict=True)
                if raw.strip() and not goes(entry)
            ]
            if kept:
                self.write(path, b"".join(raw + b"\n" for raw in kept))
            else:
                path.unlink(missing_ok=True)
            self._prune_images()
        self.changed("history", name, "deleted", ids=deleted)
        return deleted

    def clear_all_history(self) -> list[str]:
        """Delete every history (sets' and datasets') and every stored image;
        returns the documents whose history went (``data:NAME`` for a
        dataset's)."""
        cleared = []
        with self.lock:
            places = [
                (self.root / "history", ""),
                (self.root / "history" / DATASET_HISTORY, DATASET_DOCUMENT),
            ]
            for folder, prefix in places:
                for path in _entries(folder):
                    if (
                        path.is_file()
                        and path.suffix == HISTORY_SUFFIX
                        and not path.name.startswith(".")
                    ):
                        path.unlink(missing_ok=True)
                        cleared.append(prefix + path.name.removesuffix(HISTORY_SUFFIX))
            for path in _entries(self.root / "history" / IMAGES):
                if path.is_file():
                    path.unlink(missing_ok=True)
        for name in cleared:
            self.changed("history", name, "cleared")
        return cleared

    def clear_speed(self) -> int:
        """Delete every speed record; returns how many files went."""
        count = 0
        with self.lock:
            for path in _entries(self.root / "speed"):
                if path.is_file():
                    path.unlink(missing_ok=True)
                    count += 1
        return count

    def restore_images(self, request: dict[str, Any]) -> dict[str, Any]:
        """A history line's request with its images as data URLs again; an image
        whose file is gone stays a reference."""
        images = request.get("images")
        if not isinstance(images, list):
            return request
        restored = []
        for image in images:
            path = self._image_path(image)
            if path is not None and path.is_file():
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                image = f"data:{image['type']};base64,{encoded}"
            restored.append(image)
        return {**request, "images": restored}

    def image_file(self, digest: str) -> Path | None:
        """The stored image with this SHA-256, if there is one."""
        if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
            return None
        return next(iter((self.root / "history" / IMAGES).glob(f"{digest}.*")), None)

    def _store_images(self, request: dict[str, Any]) -> dict[str, Any]:
        images = request.get("images")
        if not isinstance(images, list):
            return request
        stored = []
        for image in images:
            match = DATA_URL.fullmatch(image) if isinstance(image, str) else None
            raw = _decode_base64(match.group(3)) if match else None
            if raw is None:
                stored.append(image)
                continue
            subtype = match.group(1).lower()
            digest = sha256(raw)
            extension = EXTENSIONS.get(subtype, re.sub(r"[^a-z0-9]", "", subtype) or "bin")
            path = self.root / "history" / IMAGES / f"{digest}.{extension}"
            if not path.is_file():
                self.write(path, raw)
            stored.append({"sha256": digest, "type": f"image/{subtype}"})
        return {**request, "images": stored}

    def _image_path(self, image: Any) -> Path | None:
        if not isinstance(image, dict) or not isinstance(image.get("type"), str):
            return None
        return self.image_file(image.get("sha256"))

    def _prune_images(self) -> None:
        """Delete stored images that no history line refers to (under the lock)."""
        used = set()
        histories = [
            *_entries(self.root / "history"),
            *_entries(self.root / "history" / DATASET_HISTORY),
        ]
        for path in histories:
            if path.suffix != HISTORY_SUFFIX or path.name.startswith(".") or not path.is_file():
                continue
            for entry in _read_lines(path):
                request = entry.get("request") if isinstance(entry, dict) else None
                images = request.get("images") if isinstance(request, dict) else None
                for image in images if isinstance(images, list) else ():
                    if isinstance(image, dict) and isinstance(image.get("sha256"), str):
                        used.add(image["sha256"])
        for path in _entries(self.root / "history" / IMAGES):
            if not path.name.startswith(".") and path.stem not in used:
                path.unlink(missing_ok=True)

    # Speed

    def speed_path(self, fingerprint: str) -> Path:
        return self.path("speed", fingerprint, SPEED_SUFFIX, "a model's fingerprint")

    def record_speed(self, fingerprint: str, record: dict[str, Any]) -> None:
        """Add one answered request's record (``chip``, ``latency_ms``,
        ``input_tokens``, ...; ``time`` is added) to the model's file, keeping the
        latest ``SPEED_KEEP``."""
        path = self.speed_path(fingerprint)
        data = line({"time": now(), **record})
        with self.lock:
            with open(path, "ab") as handle:
                if handle.tell() and not _ends_with_newline(path):
                    data = b"\n" + data
                handle.write(data)
            lines = path.read_bytes().splitlines(keepends=True)
            if len(lines) > SPEED_KEEP + SPEED_SLACK:
                self.write(path, b"".join(lines[-SPEED_KEEP:]))

    def read_speed(self, fingerprint: str, chip: str | None = None) -> list[dict[str, Any]]:
        """The model's latest ``SPEED_KEEP`` records, oldest first, on ``chip``
        when given."""
        records = [
            entry
            for entry in _read_lines(self.speed_path(fingerprint))
            if isinstance(entry, dict) and (chip is None or entry.get("chip") == chip)
        ]
        return records[-SPEED_KEEP:]


def check_body(body: Any) -> None:
    """A set must be a JSON object (a request body) whose images are data URLs;
    whether the model accepts it is the live check's business, so a set being
    edited can be saved with problems still open."""
    if not isinstance(body, dict):
        raise WorkspaceError("a set must be a JSON object: a request body")
    examples = body.get("examples", [])
    if not isinstance(examples, list):
        raise WorkspaceError("a set's examples must be a list")
    # The set's own images, then each example's (``examples.N.images.M``).
    parts = [("", body)] + [
        (f"examples.{index}.", example)
        for index, example in enumerate(examples)
        if isinstance(example, dict)
    ]
    for where, part in parts:
        if "images" in part and not isinstance(part["images"], list):
            raise WorkspaceError(f"{where}images: must be a list of data URLs")
        try:
            check_data_urls(part)
        except DecisionError as error:
            raise WorkspaceError(f"{where}{error.param}: {error.message}") from None


# A dataset opened in the Playground goes by "data:NAME" where a set goes by
# its name (no set's name holds a colon); its history is in history/datasets/.
DATASET_DOCUMENT = "data:"
DATASET_HISTORY = "datasets"


def dataset_document(name: str) -> str | None:
    """The dataset a Playground document's name stands for (checked), or None
    for a set."""
    if not isinstance(name, str) or not name.startswith(DATASET_DOCUMENT):
        return None
    return check_name(name[len(DATASET_DOCUMENT) :], "a dataset's name")


def dataset_summary(path: Path) -> dict[str, Any]:
    """A dataset as the list shows it: ``name``, ``kind`` (``folder`` or
    ``file``), ``bytes`` (all its files), ``modified``, ``splits`` (each
    ``name``, ``role``, ``examples``, the source ``split``, ``sample``,
    ``part``), ``questions`` (their ids), ``missing`` (what a fit still
    needs), ``source``, ``hides`` (a plain file of the folder's name, which
    the list cannot show beside it) and, when it does not read, ``error``."""
    folder = path.is_dir()
    files = [p for p in path.iterdir() if p.is_file()] if folder else [path]
    stats = [p.stat() for p in files]
    newest = max((s.st_mtime for s in stats), default=path.stat().st_mtime)
    summary: dict[str, Any] = {
        "name": path.name if folder else path.name[: -len(DATA_SUFFIX)],
        "kind": "folder" if folder else "file",
        "bytes": sum(s.st_size for s in stats),
        "modified": datetime.fromtimestamp(newest, UTC).isoformat(timespec="milliseconds"),
        "splits": [],
        "questions": [],
        "missing": [],
        "source": None,
    }
    try:
        dataset, _ = open_dataset(path)
        summary["missing"] = dataset.missing()
    except (DecisionError, OSError, ValueError, TypeError, AttributeError) as error:
        return summary | {"error": str(error)}
    if folder and (path.parent / f"{path.name}{DATA_SUFFIX}").is_file():
        summary["hides"] = f"{path.name}{DATA_SUFFIX}"  # a plain file of its name, not listed
    summary["splits"] = [
        {
            "name": name,
            "role": meta.get("role"),
            "examples": meta.get("examples"),
            "split": meta.get("split"),
            "sample": meta.get("sample"),
            "part": meta.get("part"),
        }
        for name, meta in dataset.splits.items()
    ]
    # The questions: dataset.json's, else those the lines ask (a plain file's).
    asked = dataset.questions or dataset.first_line().get("questions")
    summary["questions"] = list(asked) if isinstance(asked, dict) else []
    summary["source"] = dataset.source
    return summary


def _entries(folder: Path) -> Iterator[Path]:
    with suppress(FileNotFoundError, NotADirectoryError):
        yield from sorted(folder.iterdir())


def _hash_of(path: Path) -> str | None:
    try:
        return sha256(path.read_bytes())
    except FileNotFoundError:
        return None


def _ends_with_newline(path: Path) -> bool:
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def _read_lines(path: Path) -> list[Any]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    entries = []
    for raw in data.splitlines():
        with suppress(ValueError):
            entries.append(json.loads(raw))
    return entries


def _decode_base64(text: str) -> bytes | None:
    try:
        return base64.b64decode(re.sub(r"\s+", "", text), validate=True)
    except (binascii.Error, ValueError):
        return None
