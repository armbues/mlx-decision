"""Compare models (e.g. quantized copies) on the parity set; print a Markdown report.

usage: python scripts/quantization_report.py BASELINE MODEL [MODEL ...] [--out FILE]

Each model runs in its own process, one after another, so only one is in
memory at a time. Every model answers the parity set (after one warm-up
request); its probabilities are compared with the baseline's (the first
model) and with the PyTorch reference fixture.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
MARGIN = 0.04


def run_model(model: str, out: Path) -> None:
    """Worker: answer the parity set with one model and write the numbers."""
    import mlx.core as mx

    import mlx_decision
    from mlx_decision.types import parse_request

    cases = json.loads((PARITY / "requests.json").read_text())
    start = time.perf_counter()
    loaded = mlx_decision.load(model)
    load_seconds = time.perf_counter() - start
    active_after_load = mx.get_active_memory()
    requests = {
        c["id"]: parse_request({"state": c["state"], "questions": c["questions"]}) for c in cases
    }
    loaded.backend.score(requests[cases[0]["id"]])  # warm-up
    mx.reset_peak_memory()
    probabilities, seconds = {}, {}
    for case_id, request in requests.items():
        start = time.perf_counter()
        output = loaded.backend.score(request)
        seconds[case_id] = time.perf_counter() - start
        probabilities[case_id] = output.probabilities
    out.write_text(
        json.dumps(
            {
                "load_seconds": load_seconds,
                "active_gb": active_after_load / 2**30,
                "peak_gb": mx.get_peak_memory() / 2**30,
                "seconds": seconds,
                "probabilities": probabilities,
            }
        )
    )


def folder_gb(model: str) -> float:
    path = Path(model)
    return sum(f.stat().st_size for f in path.resolve().rglob("*") if f.is_file()) / 2**30


def compare(ours: dict, theirs: dict, reference: dict) -> dict:
    """Differences of ``ours`` to ``theirs`` over the parity set.

    ``reference`` supplies the option order and the questions.
    """
    differences, flips, decided_flips = [], 0, 0
    for case_id, entry in reference.items():
        for question in entry["questions"]:
            options = question["option_ids"]
            a = [ours[case_id][question["id"]][o] for o in options]
            b = [theirs[case_id][question["id"]][o] for o in options]
            differences += [abs(x - y) for x, y in zip(a, b, strict=True)]
            if a.index(max(a)) != b.index(max(b)):
                flips += 1
                first, second = sorted(b, reverse=True)[:2]
                decided_flips += first - second > MARGIN
    return {
        "max": max(differences),
        "mean": statistics.mean(differences),
        "flips": flips,
        "decided_flips": decided_flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="model folders; the first is the baseline")
    parser.add_argument("--out", type=Path, help="also write the report here")
    parser.add_argument("--worker", nargs=2, metavar=("MODEL", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_model(args.worker[0], Path(args.worker[1]))
        return
    if not args.models:
        parser.error("give at least one model")

    reference = json.loads((PARITY / "reference.json").read_text())["results"]
    torch_probabilities = {
        case_id: {
            q["id"]: dict(zip(q["option_ids"], q["probabilities"], strict=True))
            for q in entry["questions"]
        }
        for case_id, entry in reference.items()
    }
    results = {}
    with tempfile.TemporaryDirectory() as scratch:
        for model in args.models:
            out = Path(scratch) / "result.json"
            print(f"running {model} ...", file=sys.stderr, flush=True)
            command = [sys.executable, __file__, "--worker", model, str(out)]
            subprocess.run(command, check=True)
            results[model] = json.loads(out.read_text())

    baseline = args.models[0]
    questions = sum(len(entry["questions"]) for entry in reference.values())
    lines = [
        f"Parity set: {len(reference)} requests, {questions} questions. Baseline: "
        f"`{Path(baseline).name}`. Changed top answers counted over all questions; "
        f"in brackets those where the baseline's top-two margin exceeds {MARGIN}.",
        "",
        "| Model | Disk GB | Load s | Memory GB (loaded / peak) | Parity set s | "
        "Median s | vs baseline: max / mean diff | changed top | "
        "vs PyTorch: max / mean diff | changed top |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, result in results.items():
        seconds = list(result["seconds"].values())
        vs_base = compare(result["probabilities"], results[baseline]["probabilities"], reference)
        vs_torch = compare(result["probabilities"], torch_probabilities, reference)
        lines.append(
            f"| `{Path(model).name}` | {folder_gb(model):.1f} | {result['load_seconds']:.1f} "
            f"| {result['active_gb']:.1f} / {result['peak_gb']:.1f} | {sum(seconds):.1f} "
            f"| {statistics.median(seconds):.3f} "
            f"| {vs_base['max']:.4f} / {vs_base['mean']:.4f} "
            f"| {vs_base['flips']} ({vs_base['decided_flips']}) "
            f"| {vs_torch['max']:.4f} / {vs_torch['mean']:.4f} "
            f"| {vs_torch['flips']} ({vs_torch['decided_flips']}) |"
        )
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report)


if __name__ == "__main__":
    main()
