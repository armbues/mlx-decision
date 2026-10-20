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
                   both models' results
    (an -o ending in .jsonl is written as a file: a training example, as
    ``data`` writes one; with {split} in it, one file per --split, and with
    --record each with a record as ``data --record`` writes it, --roles
    naming their roles)
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


def get_files(pattern: Path) -> list[tuple[Path, str | None]]:
    """The files of a ``data`` run: one per --split with {split} in -o."""
    if "{split}" not in pattern.name:
        return [(pattern, option("--split"))]
    return [(pattern.with_name(pattern.name.replace("{split}", s)), s) for s in options("--split")]


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
    (output / "training.json").write_text(json.dumps(report, indent=2))


def write_records(files: list[tuple[Path, str | None]]) -> None:
    """Records beside the files, as ``data --record`` writes them."""
    from mlx_decision.data.make import file_name, get_name, partial_path, record_path

    roles = option("--roles").split(",") if option("--roles") else [None] * len(files)
    listed = [{"name": file_name(path), "split": split, "role": role}
              for (path, split), role in zip(files, roles, strict=True)]  # fmt: skip
    samples = options("--sample") or [None] * len(files)
    if len(samples) == 1:
        samples *= len(files)
    preset = option("--preset")
    source = f"hf:fake/{preset}" if preset else sys.argv[sys.argv.index("data") + 1]
    for (path, split), role, sample in zip(files, roles, samples, strict=True):
        record = {
            "source": source, "preset": preset, "config": option("--config"),
            "revision": "0123456789abcdef", "split": split, "rows": 100,
            "sample": int(sample) if sample and sample != "0" else None, "limit": None,
            "seed": int(option("--seed", "0")) if sample and sample != "0" else None,
            "examples": 1, "labels": {"q": {"true": 1}}, "question": None, "label_map": None,
            "role": role, "get": {"name": get_name(option("-o")), "files": listed},
            "created": "2026-10-20T12:00:00+00:00",
        }  # fmt: skip
        partial_path(record_path(path)).write_text(json.dumps(record))


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}), flush=True)


def main() -> int:
    steps = int(option("--steps", "10"))
    every = float(option("--every", "0.05"))
    end = option("--end", "done")
    hold = float(option("--hold", "0"))
    output = Path(option("-o")) if option("-o") else None
    files = get_files(output) if output and output.suffix == ".jsonl" else []
    if files:
        output = files[0][0]
    partial = output.parent / f".{output.name}.partial" if output else None
    more = [path.parent / f".{path.name}.partial" for path, _ in files[1:]]
    if files and "--record" in sys.argv:
        from mlx_decision.data.make import partial_path, record_path

        more += [partial_path(record_path(path)) for path, _ in files]
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
    with watch_stdin(cancel, ([partial] if partial else []) + more):
        emit("check", message="starting")
        if partial and output.suffix == ".jsonl":
            example = {"state": "x", "questions": {"q": {"type": "noul"}}, "labels": {"q": True}}
            for path, _ in files:
                (path.parent / f".{path.name}.partial").write_text(json.dumps(example) + "\n")
            if "--record" in sys.argv:
                write_records(files)
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
            for leftover in more:
                leftover.unlink(missing_ok=True)
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
        for leftover in more:  # the other files, then the records
            leftover.rename(leftover.with_name(leftover.name[1:].removesuffix(".partial")))
        if not evaluating:  # its done event, with the results, was the script's last
            emit("done", metrics={"path": str(output) if output else None}, message="finished")
    if hold:
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # as the real CLI after its run
        time.sleep(hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
