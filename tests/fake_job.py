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
        try:
            for step in range(1, steps + 1):
                time.sleep(every)
                if cancel.is_set():
                    raise KeyboardInterrupt
                emit("work", step=step, total=steps)
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
        if partial:
            partial.rename(output)
        for leftover in more:  # the other files, then the records
            leftover.rename(leftover.with_name(leftover.name[1:].removesuffix(".partial")))
        emit("done", metrics={"path": str(output) if output else None}, message="finished")
    if hold:
        signal.signal(signal.SIGINT, signal.SIG_DFL)  # as the real CLI after its run
        time.sleep(hold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
