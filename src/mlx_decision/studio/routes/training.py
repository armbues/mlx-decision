"""The fit form's plan: what a fit of a model on a dataset would do (TRS-1).

``POST /studio/api/fit/plan`` ``{model, data, split, holdout, seed, report,
report_split}`` reads the data (a dataset's split, by default its train split)
and finds the base as ``training fit``'s check stage does (no model loads)
and answers what the fit wizard's method menu shows for the same data: the
methods the family takes (``auto``'s pick first), each with the kinds it
runs, its peak memory and disk while it runs (estimates, as
``training.plan.estimate``) and whether that fits this Mac's GPU working
set (the fit's own check); plus the splits (to train on, to tune, kept
aside for the report), the label counts, the data's warnings and the
fitted model's default name. With ``report`` (a dataset and its
``report_split``, by default its report split; the same dataset's other
split too) the report is scored on that split instead of a share of
``data``: refused when
it labels other questions (ids, types, options), and a warning when
examples of it have a state that is also in ``data``. The fit itself is a job (``POST
/studio/api/jobs`` with ``kind`` "fit").
"""

import asyncio
from typing import Any

from fastapi import APIRouter, Request

from ...data.dataset import read_for
from ...errors import DecisionError
from ...gpu import Gpu
from ...memory import working_set
from ...training import lora as lora_defaults
from ...training import plan as kinds_plan
from ...training.fitting import label_counts, prepare, report_mismatch, shared_states
from ...training.wizard import DEFAULT_HOLDOUT, EPOCH_KINDS
from ..checks import training_data_warnings
from ..workspace import Workspace
from . import ApiError, field, json_body
from .jobs import HOLDOUT_MAX, HOLDOUT_MIN, TRAINING_OPTIONS, short
from .picks import pick_split

# What each method gives the user (the fit form's cards; D204).
DESCRIPTIONS = {
    "head": "Learns your labels on top of the model; the model itself stays. The usual choice.",
    "readout": "Learns your labels from what the model reads, a question at a time. Quick.",
    "finetune": "Changes the model itself to your labels: the most it can learn, and the most "
    "memory and time.",
    "finetune-head": "Learns your labels in the model's last part only: less memory than a "
    "full fine-tune.",
    "calibrate": "Only makes the confidence match how often it is right; the answers stay "
    "the same.",
    "lora": "Trains small adapters inside the model with its answer rows, where the methods "
    "above leave the model as it is. Takes minutes, not seconds.",
}


def training_router(gpu: Gpu, files: Workspace, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api")
    pool = gpu.pool

    @router.post("/eval/plan")
    async def eval_plan(request: Request):
        """``{model, data, split, compare}`` -> ``{model, compare, data, split,
        examples, questions, warnings}``: the dataset's split read (by default
        its evaluate split; its labelled questions, each with its type) and a
        warning for a fitted model that was fitted on this very data (TRS-4).
        404 for an unknown model, dataset or split, 422 for data an evaluation
        cannot take."""
        body = await json_body(request, max_body_bytes)
        names = []
        for key in ("model", "compare"):
            name = field(body, key, str, required=key == "model")
            if name is not None and name not in pool.specs:
                raise ApiError(404, f"no model is named {name}", param=key)
            names.append(name)
        model_name, compare = names
        if compare == model_name:
            raise ApiError(400, "compare with another model", param="compare")
        picked = pick_split(files, body, "data", "evaluate", "split")
        data_name, data = picked.name, picked.path
        shown = f"{data_name} ({picked.split})" if picked.dataset.folder else data_name

        def plan() -> dict[str, Any]:
            examples = read_for(data, "evaluate").examples
            questions = {}
            for example in examples:
                for question_id in example.labels:
                    questions.setdefault(
                        question_id, example.request["questions"][question_id].get("type")
                    )
            warnings = training_data_warnings(pool, names, data, shown)
            return {"examples": len(examples), "questions": questions, "warnings": warnings}

        try:
            planned = await asyncio.to_thread(plan)
        except (DecisionError, ValueError, OSError) as error:
            raise ApiError(422, str(error), param="data") from None
        return {
            "model": model_name,
            "compare": compare,
            "data": data_name,
            "split": picked.split,
            **planned,
        }

    @router.post("/fit/plan")
    async def fit_plan(request: Request):
        """``{model, data, holdout, seed, report}`` -> what the form shows (see
        the module). 404 for an unknown model or data file, 422 for data or a
        model a fit cannot take, or a report file that does not match."""
        body = await json_body(request, max_body_bytes)
        name = field(body, "model", str)
        if name not in pool.specs:
            raise ApiError(404, f"no model is named {name}", param="model")
        picked = pick_split(files, body, "data", "train", "split")
        data_name, data = picked.name, picked.path
        report_name, report, report_split = None, None, None
        if body.get("report") is not None:
            other = pick_split(files, body, "report", "report")
            report_name, report, report_split = other.name, other.path, other.split
            if report == data:
                raise ApiError(400, "report on another split than the one to train on", "report")
            if other.dataset.folder:
                report_name = f"{other.name} ({other.split})"
        holdout = field(body, "holdout", (int, float), required=False)
        holdout = DEFAULT_HOLDOUT if holdout is None else float(holdout)
        if not HOLDOUT_MIN <= holdout <= HOLDOUT_MAX:
            raise ApiError(400, "holdout must be from 0.05 to 0.9", param="holdout")
        seed = field(body, "seed", int, required=False) or 0
        spec = pool.specs[name]
        options = {key: spec.options[key] for key in TRAINING_OPTIONS if key in spec.options}

        def plan() -> dict[str, Any]:
            prepared = prepare(
                spec.path,
                data,
                validation=report,
                holdout=holdout,
                seed=seed,
                load_options=options,
            )
            if prepared.held is not None:
                why = report_mismatch(
                    prepared.source.examples, prepared.held.examples, data_name, report_name
                )
                if why:
                    raise ReportMismatch(why)
            planned = describe(prepared, working_set())
            if prepared.held is not None:
                shared = shared_states(prepared.source.examples, prepared.held.examples)
                if shared:
                    planned["warnings"].append(
                        f"{shared} of {len(prepared.held.examples)} examples in {report_name} "
                        f"have a state that is also in {data_name}: the report would score "
                        "examples the fit learns from"
                    )
            return planned

        try:
            planned = await asyncio.to_thread(plan)
        except ReportMismatch as error:
            raise ApiError(422, str(error), param="report") from None
        except (DecisionError, ValueError, OSError) as error:
            raise ApiError(422, str(error), param="data") from None
        return {
            "model": name,
            "data": data_name,
            "split": picked.split,
            "report": body.get("report"),
            "report_split": report_split,
            "name": f"{short(name)}-{data_name}",
            **planned,
        }

    return router


class ReportMismatch(ValueError):
    """A report file that labels other questions than the data to train on."""


def describe(prepared, limit: int) -> dict[str, Any]:
    """The plan as the form shows it (``prepared``: ``fitting.prepare``'s)."""
    family = prepared.family
    main = kinds_plan.MAIN_KIND.get(family)
    methods = []
    lora = None
    for method, head_only in kinds_plan.methods(family):
        kinds = kinds_plan.resolve(family, method)
        estimate = prepared.estimate(kinds, head_only)
        if method == "lora":  # offered beside the form's methods, not as one
            lora = {
                "kinds": kinds,
                "description": DESCRIPTIONS["lora"],
                "epochs": kinds_plan.EPOCHS["lora"],
                "learning_rate": lora_defaults.LEARNING_RATE,
                "rank": lora_defaults.RANK,
                "peak_bytes": estimate.peak_bytes,
                "fits": estimate.peak_bytes <= limit,
            }
            continue
        methods.append(
            {
                "method": method,
                "head_only": head_only,
                "auto": method == main and not head_only,
                "kinds": kinds,
                "epochs": kinds[0] in EPOCH_KINDS,
                "description": DESCRIPTIONS["finetune-head" if head_only else method],
                "peak_bytes": estimate.peak_bytes,
                "disk_bytes": estimate.disk_bytes,
                "fits": estimate.peak_bytes <= limit,
            }
        )
    splits = prepared.splits
    return {
        "family": family,
        "base": prepared.origin.name,
        "methods": methods,
        "lora": lora,
        "splits": {
            "examples": len(prepared.source.examples),
            "fit": len(splits.fit),
            "dev": len(splits.dev),
            "holdout": len(splits.holdout),
        },
        "labels": label_counts(splits.fit + splits.dev),
        "warnings": list(prepared.warnings),
        "working_set": limit,
    }
