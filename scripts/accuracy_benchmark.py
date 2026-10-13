"""Accuracy of local models on public classification benchmarks.

usage:
  python scripts/accuracy_benchmark.py local --datasets DIR --model MODEL --out FILE
      [--suite coverage|classic] [--benchmarks NAME,...] [--samples N]
  python scripts/accuracy_benchmark.py report RESULTS [RESULTS ...]

Two suites of benchmarks:

- ``coverage`` (default): TREC-6 (answer type, ``choice``), TweetEval
  Offensive (``noul``), ANLI round 3 (``choice``), OpenBookQA (``choice``)
  and SST-5 (sentiment, ``score``): complete evaluation splits, 5,270
  examples, each checked against the pinned Hub commit it was saved from
  (row count and a digest of the rows), so results stay comparable as the
  datasets change on the Hub.
- ``classic``: AG News, DAIR Emotion, ANLI r1-r3, BANKING77 and MMLU,
  500 examples of each, sampled with a fixed seed.

``--samples N`` answers a fixed-seed sample of each benchmark instead (of
the whole split, or of 500). ``--benchmarks`` picks some of the suite.

DIR holds the splits saved with the ``datasets`` library (not a
dependency of this package; ``pip install datasets``) as ``DIR/<org>/<name>``:

  from datasets import load_dataset
  # coverage suite, at the commits the script checks
  load_dataset("facebook/anli", revision="8e4813d81f46d313dac7892e1c28076917cfcdf9"
               ).save_to_disk("DIR/facebook/anli")  # all splits, also for classic
  load_dataset("cardiffnlp/tweet_eval", "offensive", split="test",
               revision="b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
               ).save_to_disk("DIR/cardiffnlp/tweet_eval")
  load_dataset("CogComp/trec", split="test",  # the Hub's Parquet conversion
               revision="65752bf53af25bc935a0dce92fb5b6c930728450"
               ).save_to_disk("DIR/CogComp/trec")
  load_dataset("allenai/openbookqa", "main", split="test",
               revision="388097ea7776314e93a529163e0fea805b8a6454"
               ).save_to_disk("DIR/allenai/openbookqa")
  load_dataset("SetFit/sst5", split="test",
               revision="e51bdcd8cd3a30da231967c1a249ba59361279a3"
               ).save_to_disk("DIR/SetFit/sst5")
  # classic suite
  load_dataset("fancyzhx/ag_news", split="test").save_to_disk("DIR/fancyzhx/ag_news")
  load_dataset("dair-ai/emotion", split="test").save_to_disk("DIR/dair-ai/emotion")
  load_dataset("mteb/banking77", split="test").save_to_disk("DIR/mteb/banking77")
  load_dataset("cais/mmlu", "all", split="test").save_to_disk("DIR/cais/mmlu")

The coverage suite's questions, option descriptions, label maps and pinned
commits are the ``data`` command's presets (``mlx_decision.data.presets``),
the classic suite's are written here; results are therefore comparable
between models run with this script, and only roughly with published
numbers. Every example has a stable id. ``local`` writes FILE as
it goes (after each benchmark, every 100 examples and when interrupted)
and resumes from it: examples already answered are not asked again.
``report`` prints a Markdown table of one or more result files: accuracy,
macro-F1 and ECE (expected calibration error of the top probability) in
percent, for score questions also the mean absolute error of the expected
level, and how many examples were refused or truncated (``--json``: the
numbers).
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from mlx_decision.data.presets import NLI, PRESETS, Preset, Source, digest, examples
from mlx_decision.metrics import metrics, record

SEED = 1234

AG_NEWS = {
    "world": "World news, politics and international affairs",
    "sports": "Sports",
    "business": "Business, companies and the economy",
    "scitech": "Science and technology",
}
EMOTION = {
    "sadness": "Sad, unhappy, hurt or lonely",
    "joy": "Happy, content or cheerful",
    "love": "Loving, affectionate or caring",
    "anger": "Angry, irritated or resentful",
    "fear": "Afraid, anxious or nervous",
    "surprise": "Surprised, amazed or shocked",
}


def load(datasets_dir: Path, name: str):
    from datasets import load_from_disk

    return load_from_disk(str(datasets_dir / name))


def load_source(datasets_dir: Path, source: Source) -> list[dict]:
    """The rows of a saved split; exits if they are not the pinned ones."""
    from datasets import DatasetDict

    data = load(datasets_dir, source.repo)
    if isinstance(data, DatasetDict):
        data = data[source.split]
    rows = list(data)
    if len(rows) != source.rows or digest(rows) != source.digest:
        raise SystemExit(
            f"{datasets_dir / source.repo}: not {source.repo} {source.split} at "
            f"{source.revision[:12]} ({len(rows)} rows, digest {digest(rows)}; expected "
            f"{source.rows}, {source.digest}); see --help for how to save it"
        )
    return rows


def select(examples: list[dict], count: int | None) -> list[dict]:
    """A fixed-seed sample of ``count`` examples, or all of them."""
    if count is None or count >= len(examples):
        return examples
    # The same picks as sampling the rows themselves: only the length matters.
    return [examples[i] for i in random.Random(SEED).sample(range(len(examples)), count)]


def ag_news(datasets_dir: Path) -> list[dict]:
    data = load(datasets_dir, "fancyzhx/ag_news")
    labels = list(AG_NEWS)
    return [
        {
            "id": f"ag_news:{i}",
            "state": row["text"],
            "questions": {
                "topic": {
                    "type": "choice",
                    "instructions": "What is the main topic of this news article?",
                    "criteria": AG_NEWS,
                }
            },
            "gold": labels[row["label"]],
        }
        for i, row in enumerate(data)
    ]


def emotion(datasets_dir: Path) -> list[dict]:
    data = load(datasets_dir, "dair-ai/emotion")
    labels = data.features["label"].names
    return [
        {
            "id": f"emotion:{i}",
            "state": row["text"],
            "questions": {
                "emotion": {
                    "type": "choice",
                    "instructions": "Which emotion does the writer express?",
                    "criteria": EMOTION,
                }
            },
            "gold": labels[row["label"]],
        }
        for i, row in enumerate(data)
    ]


def anli(datasets_dir: Path) -> list[dict]:
    data = load(datasets_dir, "facebook/anli")
    labels = data["test_r1"].features["label"].names
    rows = [
        (f"{split}/{i}", row)
        for split in ("test_r1", "test_r2", "test_r3")
        for i, row in enumerate(data[split])
    ]
    return [
        {
            "id": f"anli:{key}",
            "state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
            "questions": {
                "relation": {
                    "type": "choice",
                    "instructions": "How does the premise relate to the hypothesis?",
                    "criteria": NLI,
                }
            },
            "gold": labels[row["label"]],
        }
        for key, row in rows
    ]


def banking77(datasets_dir: Path) -> list[dict]:
    data = load(datasets_dir, "mteb/banking77")
    rows = list(data)
    intents = sorted({row["label_text"] for row in rows})
    criteria = {intent: intent.replace("_", " ") for intent in intents}
    return [
        {
            "id": f"banking77:{i}",
            "state": row["text"],
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which banking customer-support intent does this message have?",
                    "criteria": criteria,
                }
            },
            "gold": row["label_text"],
        }
        for i, row in enumerate(rows)
    ]


def mmlu(datasets_dir: Path) -> list[dict]:
    data = load(datasets_dir, "cais/mmlu")
    letters = ["A", "B", "C", "D"]
    return [
        {
            "id": f"mmlu:{i}",
            "state": row["question"],
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Which option answers this "
                    f"{row['subject'].replace('_', ' ')} question correctly?",
                    "criteria": dict(zip(letters, row["choices"], strict=True)),
                }
            },
            "gold": letters[row["answer"]],
        }
        for i, row in enumerate(data)
    ]


def coverage(preset: Preset) -> Callable[[Path], list[dict]]:
    """The builder of a preset's evaluation split, checked against its pinned commit."""
    return lambda datasets_dir: examples(preset, load_source(datasets_dir, preset.source))


@dataclass(frozen=True)
class Benchmark:
    make: Callable[[Path], list[dict]]  # every example of the split, in order
    source: Source | None = None  # pinned split (coverage suite)


SUITES: dict[str, dict[str, Benchmark]] = {
    "coverage": {
        preset.key: Benchmark(coverage(preset), preset.source) for preset in PRESETS.values()
    },
    "classic": {
        "ag_news": Benchmark(ag_news),
        "emotion": Benchmark(emotion),
        "anli": Benchmark(anli),
        "banking77": Benchmark(banking77),
        "mmlu": Benchmark(mmlu),
    },
}
DEFAULT_SAMPLES = {"coverage": None, "classic": 500}


def build(
    datasets_dir: Path,
    count: int | None = None,
    suite: str = "classic",
    names: list[str] | None = None,
) -> dict[str, list[dict]]:
    """The examples of a suite's benchmarks (``count`` None: the suite's default)."""
    if count is None:
        count = DEFAULT_SAMPLES[suite]
    benchmarks = SUITES[suite]
    return {
        name: select(benchmarks[name].make(datasets_dir), count) for name in (names or benchmarks)
    }


def wire(example: dict) -> dict:
    return {"state": example["state"], "questions": example["questions"]}


def write(path: Path, results: dict) -> None:
    results["updated"] = datetime.now().isoformat(timespec="seconds")
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(results))
    partial.replace(path)


def start_results(args, model, names: list[str]) -> dict:
    """The results so far in ``args.out`` (to resume), or a new results object."""
    import mlx_decision

    info = model.info()
    key = {"suite": args.suite, "samples": args.samples, "model": model.name}
    if args.out.exists():
        results = json.loads(args.out.read_text())
        found = {k: results.get(k) for k in key}
        if found != key:
            raise SystemExit(f"{args.out} holds other results ({found}); use another --out")
        print(f"resuming {args.out}", file=sys.stderr, flush=True)
    else:
        results = {
            "format": 1,
            **key,
            "seed": SEED,
            "family": info["family"],
            "precision": info["precision"],
            "mlx_decision": mlx_decision.__version__,
            "date": datetime.now().isoformat(timespec="seconds"),
            "datasets": {},
            "benchmarks": {},
            "not_applicable": {},
        }
    for name in names:
        source = SUITES[args.suite][name].source
        results["datasets"][name] = asdict(source) if source else None
    return results


def answer_all(model, name: str, examples: list[dict], results: dict, out: Path) -> None:
    """Ask every example not answered yet; refusals are records too."""
    from mlx_decision.errors import DecisionError

    records = results["benchmarks"].setdefault(name, [])
    answered = {r["id"] for r in records}
    todo = [e for e in examples if e["id"] not in answered]
    start = time.perf_counter()
    for count, example in enumerate(todo, 1):
        tick = time.perf_counter()
        try:
            result = model.decide_request(wire(example))
        except DecisionError as error:
            # Refused (e.g. an option too long for strict encoding): counted as wrong,
            # left out of the calibration error.
            records.append(
                {
                    "id": example["id"],
                    "gold": example["gold"],
                    "predicted": None,
                    "refused": str(error),
                }
            )
        else:
            answer = next(iter(result.to_wire()["answers"].values()))
            records.append(
                {
                    "id": example["id"],
                    **record(answer, example["gold"]),
                    "seconds": round(time.perf_counter() - tick, 4),
                    "truncated": result.truncated,
                }
            )
        if count % 100 == 0:
            write(out, results)
    write(out, results)
    print(
        f"{name:10s} {len(todo):5d} answered, {len(answered):5d} resumed, "
        f"{time.perf_counter() - start:7.1f}s",
        file=sys.stderr,
        flush=True,
    )


def local(args) -> None:
    import mlx_decision
    from mlx_decision.errors import DecisionError

    names = benchmark_names(args)
    model = mlx_decision.load(args.model)
    results = start_results(args, model, names)
    try:
        for name, examples in build(args.datasets, args.samples, args.suite, names).items():
            try:
                model.check(wire(examples[0]))
            except DecisionError as error:
                # e.g. more options than the model answers (BANKING77's 77 for Julia)
                results["not_applicable"][name] = str(error)
                print(f"{name:10s} not applicable: {error}", file=sys.stderr, flush=True)
                continue
            answer_all(model, name, examples, results, args.out)
    finally:
        write(args.out, results)


def cell(m: dict) -> str:
    text = f"{100 * m['accuracy']:.1f} / {100 * m['macro_f1']:.1f} / {100 * m['ece']:.1f}"
    if "mae" in m:
        text += f" / MAE {m['mae']:.2f}"
    count = [str(m["n"])]
    if m["refused"]:
        count.append(f"{m['refused']} refused")
    if m["truncated"]:
        count.append(f"{m['truncated']} truncated")
    return f"{text} ({', '.join(count)})"


def report(args) -> None:
    runs = [json.loads(Path(path).read_text()) for path in args.results]
    names = [name for suite in SUITES.values() for name in suite]
    names = [n for n in names if any(n in run["benchmarks"] for run in runs)]
    table = [
        {
            "model": run["model"],
            "file": path,
            "benchmarks": {
                name: "n/a"
                if name in run.get("not_applicable", {})
                else metrics(run["benchmarks"][name])
                if run["benchmarks"].get(name)
                else None
                for name in names
            },
        }
        for path, run in zip(args.results, runs, strict=True)
    ]
    if args.json:
        print(json.dumps(table, indent=2))
        return
    lines = [
        "| Benchmark | "
        + " | ".join(f"{run['model']}: acc / macro-F1 / ECE (n)" for run in runs)
        + " |",
        "|---|" + "---|" * len(runs),
    ]
    for name in names:
        cells = []
        for column in table:
            m = column["benchmarks"][name]
            cells.append(m if m == "n/a" else cell(m) if m else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(lines))


def add_dataset_options(parser) -> None:
    parser.add_argument("--datasets", type=Path, required=True, help="folder of saved datasets")
    parser.add_argument("--suite", choices=list(SUITES), default="coverage", help="benchmark suite")
    parser.add_argument(
        "--benchmarks", help="comma-separated benchmarks of the suite (default: all)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="a fixed-seed sample of each benchmark (default: coverage all, classic 500)",
    )


def benchmark_names(args) -> list[str]:
    suite = SUITES[args.suite]
    if not args.benchmarks:
        return list(suite)
    names = [name.strip() for name in args.benchmarks.split(",")]
    unknown = [name for name in names if name not in suite]
    if unknown:
        raise SystemExit(
            f"not in the {args.suite} suite: {', '.join(unknown)} (has {', '.join(suite)})"
        )
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    local_parser = commands.add_parser("local", help="answer the benchmarks with a local model")
    add_dataset_options(local_parser)
    local_parser.add_argument("--model", required=True, help="model folder or Hub repo id")
    local_parser.add_argument("--out", type=Path, required=True, help="results file (JSON)")
    report_parser = commands.add_parser("report", help="Markdown table of result files")
    report_parser.add_argument("results", nargs="+", help="result files from local")
    report_parser.add_argument("--json", action="store_true", help="the metrics as JSON")
    args = parser.parse_args()
    try:
        {"local": local, "report": report}[args.command](args)
    except KeyboardInterrupt:
        raise SystemExit("interrupted; the results so far are saved, run again to resume") from None


if __name__ == "__main__":
    main()
