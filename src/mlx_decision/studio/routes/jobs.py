"""Jobs: start a download, conversion, fit or evaluation, follow it, stop it.

The page names what a job works on, never a path: a model by its name in
the Studio's list, a dataset by its name in the workspace's ``data/`` (and
a split of it by its name),
a new model by the name it gets in ``models/``, a download by its repo id.
The command line is built here from those. A job that needs the GPU while
another one has it is refused (409, naming the running one).

``POST /studio/api/jobs`` answers at once (202): a GPU job may still wait
for the request on the worker. Its progress comes over the stream (``job``
and ``event`` messages); ``GET /studio/api/jobs/{id}`` gives a job with its
events so far, for a tab that opens one later.
"""

import asyncio
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response

from ...config import Config
from ...data.dataset import ROLES
from ...data.presets import PRESETS
from ...data.presets import SPLITS as PRESETS_SPLITS
from ...download import recorded_repo
from ...fitted import REPORT_FILE
from ...flags import join_list, join_pairs
from ...gpu import Gpu
from ...training.plan import CALIBRATIONS
from ...training.plan import KINDS as FIT_KINDS
from ...watchdog import remove
from ..catalog import option_arguments
from ..checks import training_data_warnings
from ..jobs import Job, JobBlocked, JobConflict, Jobs
from ..removal import ModelRemoval, RemovalError
from ..security import on_this_mac
from ..workspace import Workspace, WorkspaceError, check_name
from . import ApiError, field, json_body
from .picks import pick_split
from .sets import reveal

KINDS = ("download", "convert", "fit", "eval", "data")
DATA_SUFFIX = ".jsonl"
# A dataset's config or split name.
PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
MAX_SPLITS = 10  # splits one download takes
MAX_NAME_CHARS = 200  # a value's name from the dataset card
# A label column has at most this many classes (the preview lists no more).
MAX_CLASSES = 50
MAX_STATE_COLUMNS = 20  # the object shape's columns
# A multiple-choice dataset's columns, and the data command's flag for each.
CHOICE_ROLES = ("question", "choices", "answer", "context")
CHOICE_FLAGS = {
    "question": "--question-column",
    "choices": "--choices",
    "answer": "--answer",
    "context": "--context",
}
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
BITS = (8, 4)
# What fit and eval take from a model's load options (the others do not apply).
TRAINING_OPTIONS = ("max_input_tokens",)
METHODS = ("auto", *FIT_KINDS)
# The share kept aside for the report, as the fit wizard asks it (a fit with
# none has nothing to report on).
HOLDOUT_MIN, HOLDOUT_MAX = 0.05, 0.9


class Plan:
    """A job to start: the command's arguments, what it writes, its title and
    ``details``, the fields it was started with (names, not paths)."""

    def __init__(
        self,
        argv: list[str],
        *,
        holds_gpu: bool,
        title: str,
        details: dict[str, Any],
        output: Path | None = None,
        inputs: dict[str, bytes] | None = None,
        also: list[Path] | None = None,
        undo: Callable[[], None] | None = None,
    ):
        self.argv, self.holds_gpu, self.title, self.output = argv, holds_gpu, title, output
        self.details = details
        self.inputs = inputs  # files for the job's folder (FOLDER in argv)
        self.also = also or []  # more files it writes
        self.undo = undo  # takes back what planning saved, when the job does not start


def jobs_router(
    jobs: Jobs,
    gpu: Gpu,
    files: Workspace,
    config: Callable[[], Config],
    max_body_bytes: int,
    removal: ModelRemoval | None = None,
) -> APIRouter:
    """``config`` returns the config file as it is now (where downloads go);
    ``removal`` deletes a fit's model with its entry."""
    router = APIRouter(prefix="/studio/api/jobs")
    pool = gpu.pool

    def model(body: dict, key: str = "model", required: bool = True) -> str | None:
        name = field(body, key, str, required)
        if name is not None and name not in pool.specs:
            raise ApiError(404, f"no model is named {name}", param=key)
        return name

    def model_arguments(name: str) -> list[str]:
        spec = pool.specs[name]
        options = {key: spec.options[key] for key in TRAINING_OPTIONS if key in spec.options}
        return ["-m", str(spec.path), *option_arguments(options)]

    def named(body: dict, key: str, what: str) -> str:
        try:
            return check_name(field(body, key, str), what)
        except WorkspaceError as error:
            raise ApiError(400, error.message, param=key) from None

    def new_model(body: dict, default: str) -> tuple[str, Path]:
        """The name of a model a job writes into ``models/``: free on disk, in the
        list and among the jobs writing."""
        name = named(body, "name", "the model's name") if body.get("name") is not None else default
        try:
            path = files.path("models", name, what="the model's name")
        except WorkspaceError as error:
            raise ApiError(400, error.message, param="name") from None
        # Case ignored: on macOS's disks Fit and fit are one folder.
        if path.exists() or name.casefold() in {other.casefold() for other in pool.specs}:
            raise ApiError(409, f"a model named {name} exists already", param="name")
        writing(path, "name")
        return name, path

    def writing(path: Path, key: str) -> None:
        title = jobs.writer(path)
        if title is not None:
            raise ApiError(409, f"{title} is writing {path.name}", param=key)

    def bits(body: dict, default: int | None) -> int | None:
        value = field(body, "bits", int, required=False)
        if value is None:
            return default
        if value not in BITS:
            raise ApiError(400, "bits must be 8 or 4", param="bits")
        return value

    # One plan per kind

    def download(body: dict) -> Plan:
        repo = field(body, "repo", str)
        if not REPO_ID.fullmatch(repo) or ".." in repo:
            raise ApiError(400, f"not a Hugging Face repo id: {repo!r}", param="repo")
        quantized = bits(body, None)
        argv = ["download", repo]
        output = None
        models_dir = config().models_dir
        if isinstance(models_dir, Path):  # else the CLI puts it into the Hub's cache
            output = models_dir.expanduser() / repo.rsplit("/", 1)[-1]
            argv += ["--local-dir", str(output)]
        for job in jobs.jobs.values():
            if job.state != "ended" and job.kind == "download" and job.argv[1] == repo:
                raise ApiError(409, f"{repo} is being downloaded already", param="repo")
        if output is not None:
            writing(output, "repo")
            held = recorded_repo(output)
            if held is not None and held != repo:
                raise ApiError(409, f"{output.name} in models_dir holds {held}", param="repo")
        if quantized is not None:
            argv += ["--bits", str(quantized)]
        title = f"Download {repo}" + (f" ({quantized}-bit)" if quantized else "")
        details = {"repo": repo, "bits": quantized}
        # A quantized download converts each weight file as it arrives, so it
        # holds the GPU from the start, as a conversion does; full size does not.
        quantizing = quantized is not None
        return Plan(argv, holds_gpu=quantizing, title=title, details=details, output=output)

    def convert(body: dict) -> Plan:
        source = model(body)
        quantized = bits(body, 8)
        name, output = new_model(body, f"{short(source)}-q{quantized}")
        argv = ["convert", "-m", str(pool.specs[source].path), "-o", str(output)]
        argv += ["--bits", str(quantized)]
        details = {"model": source, "bits": quantized, "name": name}
        return Plan(argv, holds_gpu=True, title=f"Convert {name}", details=details, output=output)

    def fit(body: dict) -> Plan:
        base = model(body)
        data = pick_split(files, body, "data", "train", "split")
        data_name = data.name
        name, output = new_model(body, f"{short(base)}-{data_name}")
        argv = ["training", "fit", *model_arguments(base), "--data", str(data.path)]
        argv += ["-o", str(output)]
        method = field(body, "method", str, required=False)
        if method is not None:
            if method not in METHODS:
                raise ApiError(400, f"method must be one of {', '.join(METHODS)}", param="method")
            argv += ["--method", method]
        epochs = field(body, "epochs", int, required=False)
        if epochs is not None:
            if epochs < 1:
                raise ApiError(400, "epochs must be 1 or more", param="epochs")
            argv += ["--epochs", str(epochs)]
        report_name = report_split = None
        if body.get("report") is not None:  # the report on another split or dataset
            report = pick_split(files, body, "report", "report")
            report_name, report_split = report.name, report.split
            if report.path == data.path:
                raise ApiError(400, "report on another split than the one to train on", "report")
            argv += ["--validation", str(report.path)]
        holdout = field(body, "holdout", (int, float), required=False)
        if holdout is not None:
            if report_name is not None:
                raise ApiError(400, "a report file or a holdout share, not both", "holdout")
            if not HOLDOUT_MIN <= holdout <= HOLDOUT_MAX:
                raise ApiError(400, "holdout must be from 0.05 to 0.9", param="holdout")
            argv += ["--holdout", f"{holdout:g}"]
        seed = field(body, "seed", int, required=False)
        if seed is not None:
            argv += ["--seed", str(seed)]
        early_stop = field(body, "early_stop", bool, required=False)
        if early_stop is False:
            argv.append("--no-early-stop")
        head_only = field(body, "head_only", bool, required=False)
        if head_only:
            if method != "finetune":
                raise ApiError(400, "head_only goes with the finetune method", param="head_only")
            argv.append("--head-only")
        learning_rate = field(body, "learning_rate", (int, float), required=False)
        rank = field(body, "rank", int, required=False)
        if (learning_rate is not None or rank is not None) and method != "lora":
            raise ApiError(400, "learning_rate and rank go with the lora method", param="method")
        if learning_rate is not None:
            if not 0 < learning_rate < 1:
                raise ApiError(400, "learning_rate must be between 0 and 1", "learning_rate")
            argv += ["--learning-rate", f"{learning_rate:g}"]
        if rank is not None:
            if not 1 <= rank <= 256:
                raise ApiError(400, "rank must be from 1 to 256", param="rank")
            argv += ["--rank", str(rank)]
        calibration = field(body, "calibration", str, required=False)
        if calibration is not None:
            if calibration not in CALIBRATIONS:
                listed = ", ".join(CALIBRATIONS)
                raise ApiError(400, f"calibration must be one of {listed}", param="calibration")
            if calibration == "none" and method == "calibrate":
                raise ApiError(
                    400, "a calibrate fit needs a calibration other than none", param="calibration"
                )
            if calibration != "auto":
                argv += ["--calibration", calibration]
        details = {
            "model": base,
            "data": data_name,
            "split": data.split,
            "report": report_name,
            "report_split": report_split,
            "name": name,
            "method": method,
            "epochs": epochs,
            "holdout": holdout,
            "seed": seed,
            "early_stop": early_stop,
            "head_only": head_only,
            "calibration": calibration,
            "learning_rate": learning_rate,
            "rank": rank,
        }
        return Plan(argv, holds_gpu=True, title=f"Fit {name}", details=details, output=output)

    def evaluate(body: dict) -> Plan:
        name = model(body)
        data = pick_split(files, body, "data", "evaluate", "split")
        data_name = data.name
        compare = model(body, "compare", required=False)
        if compare == name:
            raise ApiError(400, "compare with another model", param="compare")
        argv = ["training", "eval", *model_arguments(name), "--data", str(data.path)]
        title = f"Evaluate {name}"
        if compare is not None:
            argv += ["--compare", str(pool.specs[compare].path)]
            title += f" against {compare}"
        # Kept with the job, so its report says them too.
        shown = f"{data_name} ({data.split})" if data.dataset.folder else data_name
        warnings = training_data_warnings(pool, [name, compare], data.path, shown)
        details = {"model": name, "data": data_name, "split": data.split, "compare": compare,
                   "warnings": warnings}  # fmt: skip
        return Plan(argv, holds_gpu=True, title=title, details=details)

    def get_splits(body: dict, known: dict[str, int] | None) -> list[dict[str, Any]]:
        """``splits``: each ``split`` once (one of ``known`` when given), its
        ``role``, or ``parts`` (each a ``role`` and its ``share``, adding up
        to 1 at most) when it is cut; a role once across them; an optional
        ``sample`` (up to the split's rows: ``known``'s, or the entry's
        ``rows`` when it has them)."""
        listed = body.get("splits")
        if not isinstance(listed, list) or not listed:
            raise ApiError(400, "pick at least one split", param="splits")
        if len(listed) > MAX_SPLITS:
            raise ApiError(400, f"at most {MAX_SPLITS} splits", param="splits")
        result: list[dict[str, Any]] = []
        roles: list[str] = []
        for item in listed:
            if not isinstance(item, dict):
                raise ApiError(400, "each split is an object: {split, role, sample}", "splits")
            split = field(item, "split", str)
            if not PART.fullmatch(split) or (known is not None and split not in known):
                have = f": {', '.join(known)}" if known else ""
                raise ApiError(400, f"no split {split!r}{have}", param="splits")
            if any(entry["split"] == split for entry in result):
                raise ApiError(400, f"{split} is listed twice", param="splits")
            parts = item.get("parts")
            if parts is not None:
                if not isinstance(parts, list) or not 1 <= len(parts) <= len(ROLES):
                    raise ApiError(400, f"{split}: parts is a list of 1 to 3", param="splits")
                spec = []
                for part in parts:
                    if not isinstance(part, dict):
                        raise ApiError(400, "each part is {role, share}", param="splits")
                    share = field(part, "share", (int, float))
                    if not 0 < share <= 1:
                        raise ApiError(400, f"{split}: a share is above 0, at most 1", "splits")
                    spec.append((field(part, "role", str), float(share)))
                if sum(share for _, share in spec) > 1 + 1e-9:
                    raise ApiError(400, f"{split}: the shares add up to more than 1", "splits")
            else:
                spec = [(field(item, "role", str), None)]
            for role, _ in spec:
                if role not in ROLES:
                    raise ApiError(400, f"a role is one of {', '.join(ROLES)}", param="splits")
                if role in roles:
                    raise ApiError(400, f"one split per role: two are {role}", param="splits")
                roles.append(role)
            rows = known[split] if known is not None else field(item, "rows", int, False)
            sample = field(item, "sample", int, required=False)
            if sample is not None and (sample < 1 or (rows and sample > rows)):
                most = f" to {rows}" if rows else " or more"
                raise ApiError(400, f"{split}: a sample is 1{most} rows", param="splits")
            base = sample or rows
            if parts is not None and base:
                for role, share in spec:
                    if round(share * base) < 1:
                        raise ApiError(
                            400, f"{split}: {share:.0%} of {base} rows is no row ({role})",
                            param="splits",
                        )  # fmt: skip
            entry = {"split": split, "role": spec[0][0] if parts is None else None,
                     "parts": None if parts is None else [
                         {"role": role, "share": share} for role, share in spec],
                     "sample": sample}  # fmt: skip
            result.append(entry)
        return result

    def get_folder(body: dict, default: str) -> tuple[str, Path]:
        """The dataset's name (``name``, else ``default``) and its folder in
        ``data/``: free (no dataset of that name, none a job writes)."""
        name = named(body, "name", "a dataset's name") if body.get("name") is not None else default
        try:
            folder = files.dataset_folder(name)
            taken = files.data_taken(name)
        except WorkspaceError as error:
            raise ApiError(400, error.message, param="name") from None
        if taken:
            raise ApiError(409, f"a dataset named {name} exists already", param="name")
        writing(folder, "name")
        writing(files.data_path(name), "name")
        return name, folder

    def get_plan(argv: list[str], body: dict, title: str, name: str, folder: Path,
                 splits: list[dict], details: dict) -> Plan:  # fmt: skip
        """``data`` with each split, its sample, the seed and the roles (or
        parts), into the dataset folder."""
        seed = field(body, "seed", int, required=False)
        for entry in splits:
            argv += ["--split", entry["split"]]
        sampled = any(entry["sample"] is not None for entry in splits)
        if sampled:
            for entry in splits:
                argv += ["--sample", str(entry["sample"] or 0)]
        cut_any = any(entry["parts"] for entry in splits)
        if sampled or cut_any:
            argv += ["--seed", str(seed or 0)]
        roles = ",".join(
            "+".join(f"{p['role']}:{p['share']:g}" for p in entry["parts"])
            if entry["parts"]
            else entry["role"]
            for entry in splits
        )
        argv += ["--roles", roles, "-o", str(folder)]
        shown = [
            entry["split"] + (f" ({entry['sample']} rows)" if entry["sample"] else "")
            for entry in splits
        ]
        details = {**details, "splits": splits, "seed": seed if sampled or cut_any else None,
                   "name": name}  # fmt: skip
        return Plan(argv, holds_gpu=False, title=f"Download {title} {', '.join(shown)}",
                    details=details, output=folder)  # fmt: skip

    def dataset(body: dict) -> Plan:
        """A preset's splits, or any Hugging Face dataset's (its rows without a
        question), as one dataset folder in ``data/`` (DS-5)."""
        if body.get("source") is not None:
            return any_dataset(body)
        name = field(body, "preset", str)
        if name not in PRESETS:
            raise ApiError(400, f"preset must be one of {', '.join(PRESETS)}", param="preset")
        splits = get_splits(body, PRESETS_SPLITS[name])
        base, folder = get_folder(body, name)
        argv = ["data", "--preset", name]
        return get_plan(argv, body, name, base, folder, splits, {"preset": name})

    def any_dataset(body: dict) -> Plan:
        """``hf:ORG/NAME``'s splits, by ``shape``: "text" (default; ``text`` the
        state's column), "object" (``columns``, the state's columns),
        "choice" (``question``, ``choices``, ``answer``, optional ``context``
        columns and the question's ``instructions``) or "image" (``image``, the
        picture's column, and an optional ``text`` as the state; the pictures
        go into the folder's images/). Text, object and image rows keep
        the ``label`` column's values (answered by questions written later;
        ``value_names`` the values' names, from the dataset card); choice rows
        ask the question with their own choices."""
        source = field(body, "source", str).strip()
        repo = source.removeprefix("hf:")
        if not source.startswith("hf:") or not REPO_ID.fullmatch(repo) or ".." in repo:
            raise ApiError(400, "give a Hugging Face dataset as hf:ORG/NAME", param="source")
        config = field(body, "config", str, required=False)
        if config is not None and not PART.fullmatch(config):
            raise ApiError(400, f"not a dataset's config: {config!r}", param="config")
        splits = get_splits(body, None)
        shape = field(body, "shape", str, required=False) or "text"
        if shape not in ("text", "object", "choice", "image"):
            raise ApiError(400, "shape must be text, object, choice or image", param="shape")

        def column(key: str, required: bool = True) -> str | None:
            value = field(body, key, str, required=required)
            if value is not None and (not value.strip() or len(value) > 200):
                raise ApiError(400, f"name the {key} column", param=key)
            return value

        argv = ["data", source]
        if config is not None:
            argv += ["--subset", config]
        details: dict[str, Any] = {"source": source, "config": config, "shape": shape}
        if shape == "choice":
            roles = {role: column(role, role != "context") for role in CHOICE_ROLES}
            named = [name for name in roles.values() if name is not None]
            if len(set(named)) < len(named):
                raise ApiError(400, "each role takes a column of its own", param="answer")
            instructions = field(body, "instructions", str)
            if not instructions.strip() or len(instructions) > 2000:
                raise ApiError(400, "write the question's instructions", param="instructions")
            argv += ["--shape", "choice"]
            for role, name in roles.items():
                if name is not None:
                    argv += [CHOICE_FLAGS[role], name]
            argv += ["--question", instructions.strip()]
            details |= {**roles, "instructions": instructions.strip()}
        else:
            label = column("label")
            if shape == "object":
                state = body.get("columns")
                if (
                    not isinstance(state, list)
                    or not 0 < len(state) <= MAX_STATE_COLUMNS
                    or not all(
                        isinstance(name, str) and name.strip() and len(name) <= 200
                        for name in state
                    )
                    or len(set(state)) < len(state)
                ):
                    raise ApiError(400, "pick the state's columns, each once", param="columns")
                if label in state:
                    raise ApiError(
                        400, "the label is not one of the state's columns", param="label"
                    )
                argv += ["--shape", "object", "--columns", join_list(state), "--label", label]
                details |= {"columns": state, "label": label}
            elif shape == "image":
                image = column("image")
                text = column("text", required=False)
                if label in (image, text) or (text is not None and text == image):
                    raise ApiError(400, "the image, text and label are different columns", "label")
                argv += ["--shape", "image", "--image", image, "--label", label]
                if text is not None:
                    argv += ["--text", text]
                details |= {"image": image, "text": text, "label": label}
            else:
                text = column("text")
                if text == label:
                    raise ApiError(
                        400, "the state and the label are different columns", param="label"
                    )
                argv += ["--shape", "text", "--text", text, "--label", label]
                details |= {"text": text, "label": label}
            names = body.get("value_names")
            if names is not None:
                ok = isinstance(names, dict) and len(names) <= MAX_CLASSES and all(
                    isinstance(k, str) and isinstance(v, str) and 0 < len(v) <= MAX_NAME_CHARS
                    and "=" not in k and k == k.strip() and k
                    for k, v in names.items()
                )  # fmt: skip
                if not ok:
                    raise ApiError(
                        400, "value_names maps each value to a short name", "value_names"
                    )
                if names:
                    argv += ["--value-names", join_pairs(names)]
        base, folder = get_folder(body, repo.rsplit("/", 1)[-1])
        return get_plan(argv, body, repo, base, folder, splits, details)

    plans: dict[str, Callable[[dict], Plan]] = {
        "data": dataset,
        "download": download,
        "convert": convert,
        "fit": fit,
        "eval": evaluate,
    }

    # Routes

    @router.get("")
    async def list_jobs():
        """``{jobs}``: every job's summary, newest first (also those of earlier runs)."""
        return {"jobs": jobs.list()}

    @router.post("", status_code=202)
    async def start(request: Request):
        """``{kind, ...}`` -> ``{job}``. Kinds and their fields:
        ``download``: ``repo``, ``bits`` (8 or 4; default full size);
        ``convert``: ``model``, ``bits`` (default 8), ``name``;
        ``fit``: ``model``, ``data`` (a dataset) and ``split`` (default its train
        split), ``name``, ``method``, ``epochs``, ``holdout`` or ``report``
        (a dataset, with ``report_split``, default its report split, the
        report is scored on), ``seed``, ``early_stop``, ``head_only``
        (finetune); ``eval``: ``model``, ``data`` and ``split`` (default its
        evaluate split), ``compare``; ``data``: ``preset`` (or ``source``
        with ``config``, ``text``, ``label``, ``value_names``), ``splits``
        (each ``split`` with its ``role``: train, report or evaluate, or its
        ``parts`` as ``[{role, share}]``; a role once; ``sample``), ``seed``,
        ``name`` (the dataset folder's in ``data/``; default the preset's or
        dataset's). ``name`` is the new model's in ``models/`` (default as
        the CLI names it)."""
        body = await json_body(request, max_body_bytes)
        kind = field(body, "kind", str)
        if kind not in plans:
            raise ApiError(400, f"kind must be one of {', '.join(KINDS)}", param="kind")
        plan = plans[kind](body)
        try:
            job = jobs.submit(
                kind,
                plan.argv,
                holds_gpu=plan.holds_gpu,
                output=plan.output,
                title=plan.title,
                details=plan.details,
                inputs=plan.inputs,
                also=plan.also,
            )
        except (JobConflict, JobBlocked, RuntimeError) as error:
            if plan.undo is not None:
                plan.undo()
            if isinstance(error, JobConflict):
                raise conflict(error) from None
            if isinstance(error, JobBlocked):
                raise ApiError(409, str(error), code="batch_running") from None
            raise ApiError(503, str(error), kind="api_error") from None
        return {"job": job.summary()}

    @router.get("/{job_id}")
    async def one(job_id: str):
        """``{job, events}``: the summary and the events so far."""
        try:
            return {"job": jobs.get(job_id), "events": jobs.events(job_id)}
        except (KeyError, WorkspaceError):
            raise ApiError(404, f"no job {job_id}") from None

    def summary_of(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except (KeyError, WorkspaceError):
            raise ApiError(404, f"no job {job_id}") from None

    @router.get("/{job_id}/report")
    async def report(job_id: str):
        """``{path, report}``: what a fit that ended done wrote, its folder and
        the report in it (``training.json``); for an evaluation that ended
        done ``{result}``, its last event's results (both models'). 404 for
        another job, one that did not end done, or a folder that is gone."""
        job = summary_of(job_id)
        if job["kind"] == "eval" and job.get("outcome") == "done":
            events = await asyncio.to_thread(jobs.events, job_id)
            done = next((e for e in reversed(events) if e.get("stage") == "done"), None)
            if done is None or not isinstance(done.get("metrics"), dict):
                raise ApiError(404, f"{job['title']} has no report")
            return {"path": None, "result": done["metrics"]}
        if job["kind"] != "fit" or job.get("outcome") != "done" or not job.get("output"):
            raise ApiError(404, f"{job['title']} has no report")
        folder = Path(job["output"])
        if job.get("replaced"):
            raise ApiError(
                404,
                f"The fitted model {folder.name} was replaced by a later fit; this fit's "
                "report went with it.",
                code="model_gone",
            )
        try:
            text = await asyncio.to_thread((folder / REPORT_FILE).read_text)
            written = json.loads(text)
        except (OSError, ValueError):
            if not folder.exists():
                raise ApiError(
                    404,
                    f"The fitted model {folder.name} was deleted; its report went with it.",
                    code="model_gone",
                ) from None
            raise ApiError(404, f"{folder.name} has no report ({REPORT_FILE})") from None
        return {"path": str(folder), "report": written}

    @router.post("/{job_id}/reveal", status_code=204)
    async def reveal_output(job_id: str, request: Request):
        """Show what the job wrote in Finder (only for a browser on this Mac)."""
        if not on_this_mac(request):
            raise ApiError(
                403, "Finder opens only for a browser on this Mac", kind="permission_error"
            )
        job = summary_of(job_id)
        output = Path(job["output"]) if job.get("output") else None
        if output is None or not output.exists():
            raise ApiError(404, f"{job['title']} left nothing to show")
        if sys.platform == "darwin":
            await asyncio.to_thread(reveal, output)
        return Response(status_code=204)

    @router.delete("/{job_id}")
    async def delete(job_id: str, model: bool = False):
        """``{job, model}``: an ended job's entry deleted (its folder in
        ``jobs/``); with ``?model=true`` a fit's model too, first (``model`` its
        name, else None). 409 for a job that has not ended, or a model that
        cannot go now (``ModelRemoval``); then nothing is deleted."""
        job = summary_of(job_id)
        if job.get("state") != "ended":
            raise ApiError(409, f"{job['title']} has not ended: stop it first")
        deleted = None
        output = Path(job["output"]) if job.get("output") else None
        mine = not job.get("replaced")  # a later fit of the same name is not this one's
        if model and mine and job["kind"] == "fit" and output is not None and removal is not None:
            name = removal.name_of(output)
            try:
                if name is not None:
                    await removal.delete(name)
                    deleted = name
                elif removal.in_workspace(output) and output.exists():
                    await asyncio.to_thread(remove, output)  # written, not listed yet
                    deleted = output.name
            except RemovalError as error:
                raise ApiError(error.status, str(error), param="model") from None
        try:
            jobs.delete(job_id)
        except (KeyError, WorkspaceError):
            raise ApiError(404, f"no job {job_id}") from None
        except ValueError as error:
            raise ApiError(409, str(error)) from None
        return {"job": job_id, "model": deleted}

    @router.post("/{job_id}/stop")
    async def stop(job_id: str):
        """``{job}``: Stop pressed. It answers at once; the outcome follows on
        the stream. A job that has ended stays as it is."""
        if job_id in jobs.jobs:
            return {"job": (await jobs.stop(job_id)).summary()}
        try:
            return {"job": jobs.get(job_id)}  # one from an earlier run: it has ended
        except (KeyError, WorkspaceError):
            raise ApiError(404, f"no job {job_id}") from None

    return router


def short(name: str) -> str:
    """A model's name without the owner of a repo id (as the CLI names outputs)."""
    return name.rsplit("/", 1)[-1]


def conflict(error: JobConflict) -> ApiError:
    return ApiError(409, str(error), code="gpu_job_running")


class JobMessages:
    """A job's changes as stream messages, for ``Jobs.on_change``.

    ``job`` carries the job's summary on each change; ``event`` the job's
    id, the event's place among its events (``index``) and the event. Pure
    progress (a step without metrics or a message: a fit scores one example
    per event) is thinned to one per ``interval`` per job, plus the last of
    each stage; a held one goes out when ``interval`` has passed, unless a
    later one went out first. Every event stays in the job's file and in
    ``GET /studio/api/jobs/{id}``, so a tab that loads them and takes the
    stream's from there (by ``index``) misses none. Called on the loop.
    """

    def __init__(self, publish: Callable[[str, dict[str, Any]], Any], interval: float = 0.25):
        self.publish = publish
        self.interval = interval
        self.jobs: dict[str, _Sent] = {}

    def __call__(self, job: Job, event: dict[str, Any] | None) -> None:
        sent = self.jobs.setdefault(job.id, _Sent())
        if event is None:
            if job.state == "ended":
                sent.cancel()
                self.jobs.pop(job.id, None)
            self.publish("job", {"job": job.summary()})
            return
        message = {"job": job.id, "index": len(job.events) - 1, "event": event}
        loop = asyncio.get_running_loop()
        now = loop.time()
        if (
            _progress(event)
            and event.get("stage") == sent.stage
            and now - sent.time < self.interval
            and not _last_step(event)
        ):
            sent.cancel()  # a newer step replaces the one held
            wait = sent.time + self.interval - now
            sent.timer = loop.call_later(wait, self._flush, sent, message)
            return
        self._send(sent, message, now)

    def _flush(self, sent: "_Sent", message: dict[str, Any]) -> None:
        sent.timer = None
        self._send(sent, message, asyncio.get_running_loop().time())

    def _send(self, sent: "_Sent", message: dict[str, Any], now: float) -> None:
        sent.cancel()
        sent.time, sent.stage = now, message["event"].get("stage")
        self.publish("event", message)


class _Sent:
    """What went out last for a job, and the step held back."""

    def __init__(self):
        self.time = float("-inf")
        self.stage: str | None = None
        self.timer: asyncio.TimerHandle | None = None

    def cancel(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None


def _progress(event: dict[str, Any]) -> bool:
    return "step" in event and not event.get("metrics") and not event.get("message")


def _last_step(event: dict[str, Any]) -> bool:
    total = event.get("total")
    return isinstance(total, int) and isinstance(event.get("step"), int) and event["step"] >= total
