"""Fitted models on the coverage suite: fit on a dataset's train split, score its test split.

usage:
  python scripts/training_benchmark.py fit --model MODEL --preset NAME[,NAME...] --out DIR
      --datasets DIR [--method auto] [--sample N] [--split PRESET=SPLIT] [--seed S] [--epochs E]
      [--no-early-stop] [--name NAME] [--estimate] [--skip-scoring]
  python scripts/training_benchmark.py score DIR --datasets DIR
  python scripts/training_benchmark.py report DIR [DIR ...] --base COVERAGE_DIR

``fit`` builds training examples from the presets' train splits (the
``data`` command's presets, read from the Hugging Face Hub at the pinned
commits through the HF cache; ``--split PRESET=SPLIT`` takes another split
of a preset, e.g. ANLI-R3's dev split; ``--sample N`` draws N rows of each
with ``--seed``), fits ``MODEL`` with ``mlx_decision.training.fit`` (its own
20% holdout reports before and after), writes the fitted model to
``DIR/model`` and the run to ``DIR/run.json`` (the fit's report, the time,
the estimate, the GPU's peak memory), then scores the fitted model with
``accuracy_benchmark.py local`` on the presets' test splits (``--datasets``
as there) in a process of its own, into ``DIR/coverage.json``.
``--estimate`` prints the estimate and stops before any model loads.

``report`` prints a Markdown table: per run and dataset, the base's and the
fitted model's accuracy, macro-F1 and ECE on the test split (the base's
from ``COVERAGE_DIR/<base name>.json``, a coverage run of the base), the
fit's time and peak memory.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_SPLITS = {"anli-r3": "train_r3"}  # the others are called "train"


def train_examples(presets: list[str], sample: int | None, seed: int, splits: dict[str, str]):
    from mlx_decision.data.make import make

    examples = []
    for name in presets:
        split = splits.get(name) or TRAIN_SPLITS.get(name, "train")
        made = make(None, preset=name, split=split, sample=sample, seed=seed)
        examples += made.examples
        print(f"{name}: {len(made.examples):,} training examples", file=sys.stderr, flush=True)
    return examples


def fit_run(args) -> None:
    import mlx.core as mx

    from mlx_decision.data.presets import PRESETS
    from mlx_decision.training import fit
    from mlx_decision.training.fitting import prepare
    from mlx_decision.training.plan import resolve

    presets = [name.strip() for name in args.preset.split(",")]
    unknown = [name for name in presets if name not in PRESETS]
    if unknown:
        raise SystemExit(f"no preset {', '.join(unknown)} (the presets: {', '.join(PRESETS)})")
    out = Path(args.out)
    splits = dict(part.split("=", 1) for part in args.split) if args.split else {}
    examples = train_examples(presets, args.sample, args.seed, splits)
    prepared = prepare(args.model, examples, seed=args.seed)
    kinds = resolve(prepared.family, args.method)
    estimate = prepared.estimate(kinds, args.epochs, False)
    plan = ", then ".join(kinds)
    print(
        f"{prepared.origin.name}: {plan}; estimate {json.dumps(estimate.as_json())}",
        file=sys.stderr,
        flush=True,
    )
    if args.estimate:
        return
    out.mkdir(parents=True, exist_ok=True)
    stages: dict[str, float] = {}

    def on_event(event) -> None:
        if event.message:
            print(f"{event.stage}: {event.message}", file=sys.stderr, flush=True)
        stages[event.stage] = time.perf_counter()

    mx.reset_peak_memory()
    started = time.perf_counter()
    result = fit(
        args.model,
        examples,
        output=out / "model",
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
        early_stop=not args.no_early_stop,
        on_event=on_event,
        overwrite=True,
    )
    run = {
        "name": args.name or out.name,
        "model": args.model,
        "base": prepared.origin.name,
        "presets": presets,
        "sample": args.sample,
        "splits": {name: splits.get(name) or TRAIN_SPLITS.get(name, "train") for name in presets},
        "seed": args.seed,
        "method": args.method,
        "epochs": args.epochs,
        "early_stop": not args.no_early_stop,
        "kinds": result.kinds,
        "examples": len(examples),
        "seconds": round(time.perf_counter() - started, 1),
        "estimate": estimate.as_json(),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "holdout": result.metrics,
        "training": result.report.get("training"),
        "calibrations": result.report.get("calibrations"),
        "warnings": result.report.get("warnings"),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"fitted in {run['seconds']} s, peak {run['peak_gb']} GB", file=sys.stderr, flush=True)
    if not args.skip_scoring:
        score(out, args.datasets)


def score(out: Path, datasets: Path) -> None:
    """The run's fitted model on its presets' test splits, in a process of its own."""
    from mlx_decision.data.presets import PRESETS

    run = json.loads((out / "run.json").read_text())
    keys = ",".join(PRESETS[name].key for name in run["presets"])
    command = [
        sys.executable, str(HERE / "accuracy_benchmark.py"), "local", "--datasets",
        str(datasets), "--model", str(out / "model"), "--benchmarks", keys,
        "--out", str(out / "coverage.json"),
    ]  # fmt: skip
    subprocess.run(command, check=True)


def summarize(results: dict) -> dict[str, dict]:
    """A coverage run's metrics per benchmark."""
    from mlx_decision.metrics import metrics

    return {name: metrics(records) for name, records in results["benchmarks"].items() if records}


def report(args) -> None:
    sys.path.insert(0, str(HERE))
    from accuracy_benchmark import cell

    from mlx_decision.data.presets import PRESETS

    lines = [
        "| run | dataset | base | fitted | fit time | peak GB |",
        "|---|---|---|---|---|---|",
    ]
    for folder in map(Path, args.runs):
        run = json.loads((folder / "run.json").read_text())
        fitted = summarize(json.loads((folder / "coverage.json").read_text()))
        base_file = Path(args.base) / f"{run['base']}.json"
        base = summarize(json.loads(base_file.read_text())) if base_file.exists() else {}
        for name in run["presets"]:
            key = PRESETS[name].key
            before = cell(base[key]) if key in base else "(no base run)"
            lines.append(
                f"| {run['name']} | {key} | {before} | {cell(fitted[key])} | "
                f"{run['seconds'] / 60:.1f} min | {run['peak_gb']} |"
            )
    print("accuracy / macro-F1 / ECE (%), test split\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="fit a model and score it")
    fit_parser.add_argument("--model", required=True, help="model folder or Hub repo id")
    fit_parser.add_argument("--preset", required=True, help="preset(s), comma-separated")
    fit_parser.add_argument("--out", required=True, help="folder for the run")
    fit_parser.add_argument(
        "--datasets", type=Path, required=True, help="test splits, as accuracy_benchmark reads them"
    )
    fit_parser.add_argument("--method", default="auto")
    fit_parser.add_argument("--sample", type=int, help="rows drawn from each train split")
    fit_parser.add_argument(
        "--split",
        action="append",
        metavar="PRESET=SPLIT",
        help="fit a preset on another split (e.g. anli-r3=dev_r3); repeatable",
    )
    fit_parser.add_argument("--seed", type=int, default=0)
    fit_parser.add_argument("--epochs", type=int, default=3, help="head and finetune kinds")
    fit_parser.add_argument("--no-early-stop", action="store_true", help="train every epoch")
    fit_parser.add_argument("--name", help="the run's name in the report (default: the folder)")
    fit_parser.add_argument("--estimate", action="store_true", help="print the estimate only")
    fit_parser.add_argument(
        "--skip-scoring", action="store_true", help="fit only; score later with `score`"
    )
    score_parser = commands.add_parser("score", help="score a fitted run on its test splits")
    score_parser.add_argument("run", type=Path, help="a run folder written by fit")
    score_parser.add_argument("--datasets", type=Path, required=True)
    report_parser = commands.add_parser("report", help="the Markdown table of runs")
    report_parser.add_argument("runs", nargs="+", help="run folders")
    report_parser.add_argument("--base", required=True, help="folder of the bases' coverage runs")
    args = parser.parse_args()
    if args.command == "fit":
        fit_run(args)
    elif args.command == "score":
        score(args.run, args.datasets)
    else:
        report(args)


if __name__ == "__main__":
    main()
