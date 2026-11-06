"""A stand-in for the CLI in job tests: ``python fake_job.py [OPTIONS] --events``.

It behaves like ``training fit --events``: event lines on stdout, a first
SIGINT stops it at the next step (``{"stage": "cancelled"}``, exit 130), a
second one at once, and with a pipe on stdin the watchdog stops it when the
pipe closes. With ``-o PATH`` it writes ``.NAME.partial`` beside ``PATH``
while it works and renames it at the end, as the real commands do. Options
it does not know are ignored (the job's own arguments).

    --steps N      work events (default 10)
    --every S      seconds between them (default 0.05)
    --end HOW      done (default), error, crash (exit 1 after a line on
                   stderr, no last event), quiet (exit 0, no last event)
    --deaf         ignore SIGINT (only SIGKILL ends it early)
    --stubborn     ignore the first SIGINT, stop at the second
    --hold S       after the last event, wait S seconds before exiting
    --fake-model   write a fake model (``fake_backend``'s family) at -o
    --fit-events   with ``training fit`` (and -m, -o) in the arguments, act
                   as it does: a fit's events, the train stage's steps
                   with a loss for the epoch kinds, then a fitted model
                   over -m at -o with its ``training.json``; with
                   ``training eval``, each model's load and scoring, then
                   both models' results; on a mix (--data in a folder whose
                   dataset.json has ``mix``) both say each source too
    (with ``data`` in the arguments, an -o ending in .jsonl is written as a
    file, a training example as ``data`` writes one; another -o as a dataset
    folder: ``dataset.json`` and a file per --split, or per part as --roles
    cuts one; a preset's with its question and answers map, another
    dataset's without questions)
"""

import json
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_decision.registry import MARKER_FILE  # noqa: E402
from mlx_decision.watchdog import watch_stdin  # noqa: E402


def option(name: str, default: str | None = None) -> str | None:
    args = sys.argv[1:]
    return args[args.index(name) + 1] if name in args else default


def options(name: str) -> list[str]:
    args = sys.argv[1:]
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == name]


KINDS = {"head": ["head", "calibrate"], "readout": ["readout", "calibrate"],
         "finetune": ["finetune", "calibrate"], "calibrate": ["calibrate"]}  # fmt: skip
TEAM_LABELS = {"billing": 12, "tech": 30, "sales": 18}
BEFORE = {"n": 12, "refused": 0, "truncated": 0, "accuracy": 0.5, "macro_f1": 0.4167,
          "ece": 0.21}  # fmt: skip
AFTER = {"n": 12, "refused": 0, "truncated": 0, "accuracy": 0.8333, "macro_f1": 0.8,
         "ece": 0.0612}  # fmt: skip


def fit_script(steps: int, output: Path) -> list[dict]:
    """The events of a ``training fit --events`` run, in order: ``steps`` train
    steps (split over the epochs, a loss each, for the epoch kinds)."""
    kinds = KINDS[option("--method", "calibrate")]
    trainer = kinds[0] if kinds[0] != "calibrate" else None
    base = Path(option("-m", "base")).name
    epochs = int(option("--epochs", "3"))
    events = [
        {"stage": "check", "message": "reading the data"},
        {"stage": "check", "message": f"loading {base}", "metrics": {
            "examples": 60, "holdout": 12, "fit": 43, "dev": 5, "labels": {"team": TEAM_LABELS},
            "kinds": kinds, "estimate": {"peak_gb": 0.1, "disk_gb": 0.0}}},
    ]  # fmt: skip
    events += [{"stage": "before", "step": i, "total": 12, "eta_s": 12 - i} for i in range(1, 13)]
    events.append({"stage": "before", "metrics": {"team": BEFORE}})
    if trainer != "finetune":
        events += [{"stage": "features", "step": i, "total": 48, "eta_s": (48 - i) / 4}
                   for i in range(1, 49)]  # fmt: skip
    if trainer in ("head", "finetune"):
        if trainer == "finetune":
            events.append({"stage": "train", "message": "preparing the training examples"})
        per = max(1, steps // epochs)
        step = 0
        for epoch in range(1, epochs + 1):
            for _ in range(per):
                step += 1
                loss = round(1.2 / (1 + step / per) + 0.02 * (step % 3), 4)
                left = per * epochs - step
                events.append(
                    {"stage": "train", "step": step, "total": per * epochs,
                     "metrics": {"loss": loss}, "eta_s": left}
                )  # fmt: skip
            dev = [0.9, 0.62, 0.7][min(epoch, 3) - 1]
            row = {"epoch": epoch, "loss": round(1.0 / epoch, 4), "dev_loss": dev, "seconds": 1.0}
            message = f"epoch {epoch}: loss {row['loss']}, dev loss {dev}"
            events.append({"stage": "train", "metrics": row, "message": message})
    elif trainer == "readout":
        events += [{"stage": "train", "step": i, "total": steps, "eta_s": steps - i}
                   for i in range(1, steps + 1)]  # fmt: skip
        events.append({"stage": "train", "message": "readout for 'team'",
                       "metrics": {"question": "team", "steps": 40}})  # fmt: skip
    if trainer == "finetune":
        events += [{"stage": "calibrate", "step": i, "total": 5} for i in range(1, 6)]
    events.append({"stage": "calibrate", "message": "fitting the calibration"})
    if trainer == "finetune":
        events += [{"stage": "after", "step": i, "total": 12} for i in range(1, 13)]
    questions = {"team": {"before": BEFORE, "after": AFTER}}
    events.append({"stage": "after", "metrics": questions})
    events.append({"stage": "write", "message": f"writing {output}"})
    return events


def mix_sources() -> list[tuple[str, bool]]:
    """The mix --data names, as (source name, held out); empty for other data."""
    data = option("--data")
    if not data:
        return []
    folder = Path(data) if Path(data).is_dir() else Path(data).parent
    try:
        body = json.loads((folder / "dataset.json").read_text())
    except (OSError, ValueError):
        return []
    sources = (body.get("mix") or {}).get("sources") or []
    return [(Path(s["path"]).name.removesuffix(".jsonl"), bool(s.get("held_out"))) for s in sources]


def eval_script(steps: int) -> list[dict]:
    """The events of a ``training eval --events`` run: each model (-m, then
    --compare) loaded and scored on ``steps`` examples, then the results."""
    names = [Path(option("-m")).name] + (
        [Path(option("--compare")).name] if option("--compare") else []
    )
    events = []
    results = []
    for index, name in enumerate(names):
        events.append({"stage": "load", "message": f"loading {name}"})
        events += [{"stage": "score", "step": i, "total": steps, "eta_s": (steps - i) * 0.1}
                   for i in range(1, steps + 1)]  # fmt: skip
        right = 18 - 4 * index  # the first model does better
        results.append({
            "name": name, "model": name, "seconds": 2.5,
            "questions": {"team": {
                "type": "choice", "n": 20, "refused": 0, "truncated": 0,
                "accuracy": right / 20, "macro_f1": round(right / 20 - 0.05, 4),
                "ece": round(0.04 + 0.05 * index, 4),
                "labels": ["billing", "tech", "sales"],
                "confusion": {"billing": {"billing": 5, "tech": 1, "sales": 0},
                              "tech": {"billing": 1, "tech": right - 8, "sales": 1},
                              "sales": {"billing": 0, "tech": 0, "sales": 3}},
            }},
        })  # fmt: skip
        sources = {
            source: {"team": {"n": 10, "refused": 0, "truncated": 0,
                              "accuracy": (right - 2 * place) / 20, "macro_f1": 0.5,
                              "ece": round(0.05 + 0.02 * place, 4)}}
            for place, (source, _) in enumerate(mix_sources())
        }  # fmt: skip
        if sources:
            results[-1]["sources"] = sources
    data = {"file": option("--data"), "sha256": "0" * 64, "examples": 20}
    events.append({"stage": "done", "metrics": {"data": data, "models": results}})
    return events


def write_fit(output: Path) -> None:
    """A fitted model over -m at ``output``, with a report as a fit writes it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_backend import write_fitted

    kinds = KINDS[option("--method", "calibrate")]
    write_fitted(output, Path(option("-m")), tuple(kinds))
    training = None
    if kinds[0] in ("head", "finetune"):
        epochs = int(option("--epochs", "3"))
        rows = [{"epoch": e, "loss": round(1.0 / e, 4)} for e in range(1, epochs + 1)]
        training = {"epochs": rows, "best_epoch": min(2, epochs), "stopped_early": False}
    report = {
        "base": {"name": Path(option("-m")).name},
        "options": {"method": option("--method"), "kinds": kinds,
                    "seed": int(option("--seed", "0"))},
        "data": {"file": option("--data"), "examples": 60, "holdout": 12, "fit": 43, "dev": 5},
        "questions": {"team": {"before": BEFORE, "after": AFTER}},
        "training": training,
        "warnings": [],
        "seconds": {"total": 4.2},
    }  # fmt: skip
    sources = {
        source: {"held_out": held,
                 "questions": {"team": {"before": BEFORE, "after": BEFORE if held else AFTER}}}
        for source, held in mix_sources()
    }  # fmt: skip
    if sources:
        report["sources"] = sources
    (output / "training.json").write_text(json.dumps(report, indent=2))


PART_NAMES = {"train": "train", "report": "validation", "evaluate": "test"}


def write_dataset_folder(folder: Path) -> None:
    """A dataset folder as ``data -o FOLDER`` writes it: two rows per split
    file, the label's values yes and no; a preset's with its question."""
    splits = options("--split") or ["train"]
    roles = option("--roles").split(",") if option("--roles") else [None] * len(splits)
    files = {}
    for split, role in zip(splits, roles, strict=True):
        if role and "+" in role or (role and ":" in role):
            for part in role.split("+"):
                name, share = part.split(":")
                files[PART_NAMES[name]] = {"role": name, "split": split, "part": float(share)}
        else:
            files[split] = {"role": role, "split": split}
    preset = option("--preset")
    body = {}
    if preset:
        body["questions"] = {"q": {"type": "noul", "instructions": "Is it so?"}}
        mapped = {"yes": True, "no": False}
        body["answers"] = {"q": {"column": "label", "map": mapped, "left_out": []}}
    counts = {"rows": len(files)}
    body["columns"] = {"label": {"values": {"yes": dict(counts), "no": dict(counts)}}}
    source = f"hf:fake/{preset}" if preset else sys.argv[sys.argv.index("data") + 1]
    body["splits"] = {name: {**meta, "rows": 100, "examples": 2} for name, meta in files.items()}
    body["source"] = {"source": source, "preset": preset, "revision": "0123456789abcdef"}
    body["created"] = "2026-10-20T12:00:00+00:00"
    folder.mkdir(parents=True)
    for name in files:
        rows = [{"id": f"{name}/{i}", "state": f"row {i} of {name}", "values": {"label": value}}
                for i, value in enumerate(("yes", "no"))]  # fmt: skip
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (folder / "dataset.json").write_text(json.dumps(body, indent=2))


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}), flush=True)


def main() -> int:
    steps = int(option("--steps", "10"))
    every = float(option("--every", "0.05"))
    end = option("--end", "done")
    hold = float(option("--hold", "0"))
    output = Path(option("-o")) if option("-o") else None
    making_data = "data" in sys.argv and output is not None
    partial = output.parent / f".{output.name}.partial" if output else None
    cancel = threading.Event()
    interrupts = []

    def on_sigint(signum, frame) -> None:
        interrupts.append(time.monotonic())
        if "--deaf" in sys.argv or ("--stubborn" in sys.argv and len(interrupts) == 1):
            print(f"ignored SIGINT {len(interrupts)}", file=sys.stderr, flush=True)
            return
        if cancel.is_set():
            raise KeyboardInterrupt
        cancel.set()

    signal.signal(signal.SIGINT, on_sigint)
    with watch_stdin(cancel, [partial] if partial else []):
        emit("check", message="starting")
        if making_data and output.suffix == ".jsonl":
            example = {"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
            partial.write_text(json.dumps(example) + "\n")
        elif making_data:
            write_dataset_folder(partial)
        elif partial:
            partial.mkdir(parents=True)
            (partial / "weights").write_text("half")
            if "--fake-model" in sys.argv:
                (partial / MARKER_FILE).write_text(json.dumps({"family": "fake"}))
        acting = "--fit-events" in sys.argv and "training" in sys.argv and option("-m") is not None
        fitting = acting and "fit" in sys.argv and output is not None
        evaluating = acting and "eval" in sys.argv
        if fitting:
            script = fit_script(steps, output)
        elif evaluating:
            script = eval_script(steps)
        else:
            script = [{"stage": "work", "step": i, "total": steps} for i in range(1, steps + 1)]
        try:
            for event in script:
                if "step" in event or not (fitting or evaluating):
                    time.sleep(every)
                if cancel.is_set():
                    raise KeyboardInterrupt
                emit(**event)
        except KeyboardInterrupt:
            if partial and partial.is_file():
                partial.unlink()
            elif partial:
                import shutil

                shutil.rmtree(partial, ignore_errors=True)
            emit("cancelled", message="nothing was written")
            return 130
        if end == "crash":
            print("Traceback (most recent call last):\nRuntimeError: it broke", file=sys.stderr)
            return 1
        if end == "quiet":
            return 0
        if end == "error":
            emit("error", message="the data has no labels")
            return 1
        if partial and fitting:
            import shutil

            shutil.rmtree(partial, ignore_errors=True)
            write_fit(output)
        elif partial:
            partial.rename(output)
        if not evaluating:  # its done event, with the results, was the script's last
            emit("done", metrics={"path": str(output) if output else None}, message="finished")
    if hold:
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # as the real CLI after its run
        time.sleep(hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
