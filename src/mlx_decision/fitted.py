"""Fitted models: a small folder that points at its base model and changes its answers.

A fitted model is what training writes: ``mlx_decision.json`` names the base
(a Hugging Face repo id at a commit, or a folder) with a fingerprint of it,
and the fit (its kinds, the questions it was fitted on with their options,
metrics, seed, data hash, package version); ``fit.safetensors`` holds the
trained tensors; ``training.json`` the full report. The base's weights are
not copied. Loading resolves the base, refuses it if it changed since the
fit, loads it and wraps it in a ``FittedBackend`` named after the folder.

Per question (same id, type and options as fitted): a readout's rows and
biases replace the release's (families with readout hooks), then a
calibration, ``log p / temperature + bias``, renormalized; then the family's
own last steps (H2O's yes/no floor). Other questions are answered by the
base, with a warning once per question id. Weight kinds (a trained head, a
fine-tuned encoder) load into the base and apply to every question.
"""

import hashlib
import json
import logging
import math
import shutil
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import BackendOutput, Readable
from .hub import ModelNotFoundError, is_repo_id, resolve_model_path, snapshot_revision
from .registry import FITTED, MARKER_FILE, Family, detect_family, load_backend
from .types import Choice, Noul, Request, parse_request

FORMAT = 1
FIT_FILE = "fit.safetensors"
REPORT_FILE = "training.json"
# Kinds that change answers per question, and kinds whose tensors load into the base.
QUESTION_KINDS = ("calibrate", "readout")
WEIGHT_KINDS = ("head", "finetune")

log = logging.getLogger(__name__)


class BaseModelError(ValueError):
    """A fitted model's base that is missing or no longer the one it was fitted on."""


def option_ids(question) -> list[str]:
    """A question's option ids, as answers name them."""
    if isinstance(question, Noul):
        return ["true", "false"]
    if isinstance(question, Choice):
        return list(question.criteria)
    return [str(level) for level in range(len(question.criteria))]


@dataclass(frozen=True)
class FitQuestion:
    """A question a fit applies to: the same id, type and options."""

    id: str
    type: str
    options: list[str]

    @classmethod
    def of(cls, question_id: str, question) -> "FitQuestion":
        return cls(question_id, question.type, option_ids(question))


@dataclass
class Calibration:
    temperature: float
    bias: list[float] | None = None  # one per option, or none


@dataclass
class Fit:
    """What a fit changes. ``calibrations`` and ``readouts`` are indexed like
    ``questions``; ``meta`` holds the rest of the marker's ``fit`` block."""

    kinds: list[str]
    questions: list[FitQuestion] = field(default_factory=list)
    calibrations: dict[int, Calibration] = field(default_factory=dict)
    readouts: dict[int, Any] = field(default_factory=dict)  # index -> Readout
    meta: dict[str, Any] = field(default_factory=dict)

    def index(self, question_id: str, question) -> int | None:
        """The fitted question this one is, or None."""
        wanted = FitQuestion.of(question_id, question)
        return next((i for i, q in enumerate(self.questions) if q == wanted), None)

    def add(self, question: FitQuestion) -> int:
        """The index of ``question`` among the fitted ones, added if new."""
        if question not in self.questions:
            self.questions.append(question)
        return self.questions.index(question)

    def tensors(self) -> dict:
        """The question kinds' tensors as ``fit.safetensors`` stores them."""
        import mlx.core as mx

        tensors = {}
        for i, calibration in self.calibrations.items():
            tensors[f"calibration.{i}.temperature"] = mx.array(calibration.temperature)
            if calibration.bias is not None:
                tensors[f"calibration.{i}.bias"] = mx.array(calibration.bias)
        for i, readout in self.readouts.items():
            tensors[f"readout.{i}.rows"] = readout.rows
            tensors[f"readout.{i}.bias"] = readout.bias
        return tensors


def calibrate(probabilities: Mapping[str, float], calibration: Calibration) -> dict[str, float]:
    """``softmax(log p / temperature + bias)`` over the options, in float64."""
    keys = list(probabilities)
    bias = calibration.bias or [0.0] * len(keys)
    logits = [
        (math.log(p) / calibration.temperature if p > 0 else -math.inf) + b
        for p, b in zip(probabilities.values(), bias, strict=True)
    ]
    top = max(logits)
    weights = [math.exp(z - top) for z in logits]
    total = sum(weights)
    return {key: w / total for key, w in zip(keys, weights, strict=True)}


class FittedBackend:
    """A base backend with a fit applied; capabilities, family and precision are the base's."""

    def __init__(self, name: str, base, fit: Fit, base_reference: dict, path: Path | None = None):
        if fit.readouts and not isinstance(base, Readable):
            raise ValueError(f"{name}: a readout fit needs a family that reads out (H2O, d1, pplx)")
        self.name = name
        self.base = base
        self.fit = fit
        self.base_reference = base_reference
        self.path = path
        self.capabilities = base.capabilities
        self.precision = getattr(base, "precision", None)
        self.warned: set[str] = set()

    @property
    def family(self) -> str | None:
        return getattr(self.base, "family", None)

    def fitted_info(self) -> dict[str, Any]:
        """What ``info()`` adds: the base and the fit."""
        return {
            "base": self.base_reference,
            "fit": {
                "kinds": self.fit.kinds,
                "questions": [q.id for q in self.fit.questions],
                **{k: v for k, v in self.fit.meta.items() if k in ("metrics", "seed")},
            },
        }

    def readouts(self, questions: Mapping[str, Any]) -> dict[str, Any]:
        """The fit's readouts for ``questions``, by question id (fitted ones only)."""
        readouts = {}
        for question_id, question in questions.items():
            i = self.fit.index(question_id, question)
            if i is not None and i in self.fit.readouts:
                readouts[question_id] = self.fit.readouts[i]
        return readouts

    def score(self, request: Request) -> BackendOutput:
        if isinstance(self.base, Readable):
            readouts = self.readouts(request.questions)
            output = self.base.score(request, readouts=readouts or None, finish=False)
        else:
            output = self.base.score(request)
        probabilities = self.adjust(request.questions, output.probabilities)
        for message in self.unfitted(request.questions):
            log.warning(message)
        return BackendOutput(probabilities, output.input_tokens, output.truncated)

    def adjust(
        self, questions: Mapping[str, Any], probabilities: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """The base's probabilities, before its last steps (with this fit's readouts),
        through the fit's calibrations and then those steps (H2O's yes/no floor)."""
        result = {}
        for question_id, values in probabilities.items():
            question = questions[question_id]
            i = self.fit.index(question_id, question)
            if i is not None and i in self.fit.calibrations:
                options = self.fit.questions[i].options
                values = calibrate({o: values[o] for o in options}, self.fit.calibrations[i])
            if isinstance(self.base, Readable):
                values = self.base.finish(question, dict(values))
            result[question_id] = dict(values)
        return result

    def unfitted(self, questions: Mapping[str, Any]) -> list[str]:
        """A warning for each question the fit does not cover (when it has question
        kinds), once per question id: reported ids are not reported again."""
        if not any(kind in QUESTION_KINDS for kind in self.fit.kinds):
            return []
        new = [
            question_id
            for question_id, question in questions.items()
            if question_id not in self.warned and self.fit.index(question_id, question) is None
        ]
        self.warned.update(new)
        return [
            f"{self.name}: question {question_id!r} is not one it was fitted on (same id and "
            "options); the base answers it"
            for question_id in new
        ]


def unfitted_warnings(model, body: Mapping[str, Any]) -> list[str]:
    """``FittedBackend.unfitted`` for the questions of a request body, so that ``run``
    and ``chat`` warn before answering; empty for a model that is not fitted."""
    backend = model.backend
    if not isinstance(backend, FittedBackend) or not body.get("questions"):
        return []
    return backend.unfitted(parse_request({"state": "", "questions": body["questions"]}).questions)


def fingerprint(path: Path) -> str:
    """SHA-256 over a model folder's JSON files, its weight files' names and sizes
    and their safetensors headers (no weight bytes are read)."""
    sha = hashlib.sha256()
    files = sorted(
        file
        for pattern in ("*.json", "*.safetensors")
        for file in path.rglob(pattern)
        if not any(part.startswith(".") for part in file.relative_to(path).parts)
    )
    for file in files:
        name = file.relative_to(path).as_posix()
        sha.update(f"{name}\0".encode())
        if file.suffix == ".json":
            sha.update(file.read_bytes())
        else:
            with open(file, "rb") as handle:
                raw = handle.read(8)
                (length,) = struct.unpack("<Q", raw) if len(raw) == 8 else (0,)
                header = handle.read(length)
            sha.update(f"{file.stat().st_size}\0".encode() + header)
        sha.update(b"\0")
    return sha.hexdigest()


def base_reference(model: str | Path, path: Path) -> dict[str, str]:
    """How a fitted model names its base: a repo id at its commit when the base
    is a Hub cache snapshot (loaded by id or by its path), else its folder."""
    cached = snapshot_revision(path)
    if cached is not None:
        return {"repo_id": cached[0], "revision": cached[1]}
    if is_repo_id(model):
        return {"repo_id": str(model)}
    return {"path": str(Path(path).expanduser().absolute())}


def describe_base(reference: Mapping[str, str]) -> str:
    if "repo_id" in reference:
        revision = reference.get("revision")
        return reference["repo_id"] + (f" at {revision[:12]}" if revision else "")
    return reference["path"]


def is_fitted(path: Path) -> bool:
    """Whether a folder holds a fitted model (its marker names a base)."""
    marker = path / MARKER_FILE
    try:
        return marker.is_file() and "base" in json.loads(marker.read_text())
    except (OSError, ValueError):
        return False


def read_marker(path: Path) -> dict[str, Any]:
    marker = json.loads((path / MARKER_FILE).read_text())
    if marker.get("format") != FORMAT or not isinstance(marker.get("base"), dict):
        raise ValueError(f"{path}: not a fitted model this version reads (format {FORMAT})")
    return marker


def resolve_base(path: Path, marker: Mapping[str, Any]) -> Path:
    """The base's folder, or ``BaseModelError`` naming it."""
    reference = marker["base"]
    name = path.name
    if "path" in reference:
        base = Path(reference["path"])
        if not base.is_dir():
            raise BaseModelError(
                f"{name}: its base model {base} is missing; it was fitted on that folder "
                "(put the base back there, or fit again)"
            )
        return base
    try:
        return resolve_model_path(reference["repo_id"], reference.get("revision"))
    except ModelNotFoundError as error:
        raise BaseModelError(
            f"{name}: its base model {describe_base(reference)} cannot be found ({error})"
        ) from None


def check_base(path: Path, marker: Mapping[str, Any], base: Path) -> None:
    if fingerprint(base) != marker.get("fingerprint"):
        raise BaseModelError(
            f"{path.name}: its base model {describe_base(marker['base'])} has changed since "
            "the fit (other weights or configuration); restore the base it was fitted on, "
            "or fit again"
        )


def read_fit(path: Path, marker: Mapping[str, Any]) -> Fit:
    """The fit's kinds, questions and question-kind tensors."""
    import mlx.core as mx

    from .models.readout import Readout

    block = dict(marker.get("fit") or {})
    questions = [
        FitQuestion(q["id"], q["type"], [str(o) for o in q["options"]])
        for q in block.pop("questions", [])
    ]
    fit = Fit(list(block.pop("kinds", [])), questions, meta=block)
    tensors = mx.load(str(path / FIT_FILE)) if (path / FIT_FILE).exists() else {}
    for key, value in tensors.items():
        part, _, rest = key.partition(".")
        if part not in ("calibration", "readout"):
            continue  # weight kinds' tensors, read by the family's loader
        index, _, name = rest.partition(".")
        i = int(index)
        if not 0 <= i < len(questions):
            raise ValueError(f"{path}: {FIT_FILE} holds {key} for no fitted question")
        if part == "calibration":
            calibration = fit.calibrations.setdefault(i, Calibration(1.0))
            if name == "temperature":
                calibration.temperature = float(value.item())
            else:
                calibration.bias = [float(b) for b in value.tolist()]
        elif name in ("rows", "bias"):
            fit.readouts.setdefault(i, {})[name] = value.astype(mx.float32)
    for i, parts in list(fit.readouts.items()):
        fit.readouts[i] = Readout(questions[i].options, parts["rows"], parts["bias"])
    return fit


def weight_files(path: Path, options: dict) -> list[tuple[Path, str | None]]:
    """The base's weights plus the fit's tensors, for the memory check."""
    from .memory import weight_files as base_weights

    base = resolve_base(path, read_marker(path))
    files = list(base_weights(base, options))
    if (path / FIT_FILE).exists():
        files.append((path / FIT_FILE, None))
    return files


@dataclass(frozen=True)
class FittedFolder:
    """A fitted model folder whose base was found and checked; nothing loaded."""

    path: Path
    marker: dict[str, Any]
    base: Path
    family: Family  # the base's

    @property
    def kinds(self) -> list[str]:
        return list((self.marker.get("fit") or {}).get("kinds", []))

    @property
    def shares_base(self) -> bool:
        """Whether the fit leaves the base's weights alone, so a loaded base can be shared."""
        return not any(kind in WEIGHT_KINDS for kind in self.kinds)

    @property
    def base_name(self) -> str:
        """How the base is named when loaded for this fit: its repo or its folder."""
        reference = self.marker["base"]
        if "repo_id" in reference:
            return reference["repo_id"].rstrip("/").rsplit("/", 1)[-1]
        return self.base.name


def open_folder(path: Path) -> FittedFolder:
    """Read a fitted model folder and resolve and check its base (``BaseModelError``)."""
    marker = read_marker(path)
    base = resolve_base(path, marker)
    check_base(path, marker, base)
    family = detect_family(base)
    if family is FITTED:
        raise BaseModelError(f"{path.name}: its base model {base} is itself a fitted model")
    return FittedFolder(path, marker, base, family)


def model_family(path: Path) -> Family:
    """The family whose loader options a model takes: a fitted model's base's."""
    family = detect_family(path)
    return open_folder(path).family if family is FITTED else family


def attach(folder: FittedFolder, base, name: str | None = None) -> FittedBackend:
    """The fit applied to its loaded base backend (which may serve other models too)."""
    fit = read_fit(folder.path, folder.marker)
    name = name or folder.path.resolve().name
    return FittedBackend(name, base, fit, folder.marker["base"], folder.path)


def load(path: Path, **options) -> FittedBackend:
    """A fitted model: its base resolved, checked and loaded (``options`` go to the
    base's loader), with the fit applied."""
    folder = open_folder(path)
    if not folder.shares_base:
        options = {**options, "fit_weights": path / FIT_FILE}
    return attach(folder, load_backend(folder.base, **options))


def write(
    folder: Path,
    *,
    family: str,
    base: Mapping[str, str],
    base_fingerprint: str,
    fit: Fit,
    tensors: Mapping[str, Any] | None = None,
    report: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a fitted model folder: into ``.<name>.partial`` beside it, renamed when
    complete, so an interrupted write leaves no fitted model behind. ``tensors``
    are the weight kinds' tensors (stored next to the question kinds')."""
    import mlx.core as mx

    folder = Path(folder)
    if folder.exists() and not overwrite:
        raise FileExistsError(f"{folder} exists (pass overwrite to replace it)")
    folder.parent.mkdir(parents=True, exist_ok=True)
    partial = folder.parent / f".{folder.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    try:
        marker = {
            "format": FORMAT,
            "family": family,
            "base": dict(base),
            "fingerprint": base_fingerprint,
            "fit": {
                "kinds": fit.kinds,
                "questions": [
                    {"id": q.id, "type": q.type, "options": q.options} for q in fit.questions
                ],
                **fit.meta,
            },
        }
        (partial / MARKER_FILE).write_text(json.dumps(marker, indent=2, ensure_ascii=False))
        all_tensors = {**fit.tensors(), **(tensors or {})}
        if all_tensors:
            mx.save_safetensors(str(partial / FIT_FILE), all_tensors)
        if report is not None:
            (partial / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        if folder.exists():
            old = folder.parent / f".{folder.name}.old"
            shutil.rmtree(old, ignore_errors=True)
            folder.rename(old)
            partial.rename(folder)
            shutil.rmtree(old, ignore_errors=True)
        else:
            partial.rename(folder)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return folder
