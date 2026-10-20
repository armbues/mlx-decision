"""The fit form's plan: what a fit of a model on a data file would do (TRS-1).

``POST /studio/api/fit/plan`` ``{model, data, holdout, seed, report}`` reads the data
and finds the base as ``training fit``'s check stage does (no model loads)
and answers what the fit wizard's method menu shows for the same data: the
methods the family takes (``auto``'s pick first), each with the kinds it
runs, its peak memory and disk while it runs (estimates, as
``training.plan.estimate``) and whether that fits this Mac's GPU working
set (the fit's own check); plus the splits (to train on, to tune, kept
aside for the report), the label counts, the data's warnings and the
fitted model's default name. With ``report`` (another data file) the
report is scored on that file instead of a share of ``data``: refused when
it labels other questions (ids, types, options), and a warning when
examples of it have a state that is also in ``data``. The fit itself is a job (``POST
/studio/api/jobs`` with ``kind`` "fit").
"""

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from ...errors import DecisionError
from ...gpu import Gpu
from ...memory import working_set
from ...training import plan as kinds_plan
from ...training.fitting import label_counts, prepare, report_mismatch, shared_states
from ...training.wizard import DEFAULT_HOLDOUT, EPOCH_KINDS
from ..workspace import Workspace, WorkspaceError
from . import ApiError, field, json_body
from .jobs import HOLDOUT_MAX, HOLDOUT_MIN, TRAINING_OPTIONS, short

DESCRIPTIONS = {
    "head": "Trains Clef's joint-schema head on your labels, then calibrates.",
    "readout": "Fits a readout on the model's features per question, then calibrates.",
    "finetune": "Fine-tunes the encoder and its marker head, then calibrates.",
    "finetune-head": "Trains the marker head alone, the encoder as it is: less memory. "
    "Then calibrates.",
    "calibrate": "Adjusts how confident the answers are; the answers themselves stay.",
}


def training_router(gpu: Gpu, files: Workspace, max_body_bytes: int) -> APIRouter:
    router = APIRouter(prefix="/studio/api/fit")
    pool = gpu.pool

    @router.post("/plan")
    async def fit_plan(request: Request):
        """``{model, data, holdout, seed, report}`` -> what the form shows (see
        the module). 404 for an unknown model or data file, 422 for data or a
        model a fit cannot take, or a report file that does not match."""
        body = await json_body(request, max_body_bytes)
        name = field(body, "model", str)
        if name not in pool.specs:
            raise ApiError(404, f"no model is named {name}", param="model")
        data_name, data = data_file(files, body, "data")
        report_name, report = None, None
        if body.get("report") is not None:
            report_name, report = data_file(files, body, "report")
            if report_name == data_name:
                raise ApiError(400, "report on another file than the one to train on", "report")
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
            "report": report_name,
            "name": f"{short(name)}-{data_name}",
            **planned,
        }

    return router


class ReportMismatch(ValueError):
    """A report file that labels other questions than the data to train on."""


def data_file(files: Workspace, body: dict, key: str) -> tuple[str, Path]:
    """The data file ``body[key]`` names: 400 for a name that is not one, 404
    when there is none."""
    name = field(body, key, str)
    try:
        path = files.data_path(name)
    except WorkspaceError as error:
        raise ApiError(400, error.message, param=key) from None
    if not path.is_file():
        raise ApiError(404, f"no data file is named {name}", param=key)
    return name, path


def describe(prepared, limit: int) -> dict[str, Any]:
    """The plan as the form shows it (``prepared``: ``fitting.prepare``'s)."""
    family = prepared.family
    main = kinds_plan.MAIN_KIND.get(family)
    methods = []
    for method, head_only in kinds_plan.methods(family):
        kinds = kinds_plan.resolve(family, method)
        estimate = prepared.estimate(kinds, head_only)
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
