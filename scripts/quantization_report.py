"""Compare models (e.g. quantized copies) on the parity set; print a Markdown report.

usage: python scripts/quantization_report.py BASELINE MODEL [MODEL ...] [--out FILE]
                                            [--cache DIR [--rerun]] [--reference FILE]

Each model runs in its own process, one after another, so only one is in
memory at a time. Every model answers the parity set (after one warm-up
request); its probabilities are compared with the baseline's (the first
model) and with the PyTorch reference fixture. With ``--cache`` each model's
raw results are kept in ``DIR/<model folder name>.json`` and reused while
the parity set is unchanged, so adding a model only runs that model
(``--rerun`` ignores the cache). ``--reference`` takes another reference
fixture, such as ``tests/parity/reference-clef.json`` from a larger Mac; when
it also holds mlx-decision's bf16 results, those are the baseline (the first
row), so only the quantized copies need to run here. H2O-Lightning:
``--reference tests/parity/h2o/reference-bf16.json`` (its parity set, images
in states lifted out; answers compared before the yes/no floor). d3:
``--reference tests/parity/d3/reference-bf16.json`` (its parity set; the
questions the release refuses left out; the input limit raised to the
backbone's context for the 40k request).
"""

import argparse
import hashlib
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
# The request files of each parity set: Clef's, H2O-Lightning's and d3's.
SETS = {
    "clef": ["requests.json"],
    "h2o": ["requests.json", "marker/requests.json", "h2o/requests.json"],
    "d3": ["requests.json", "marker/requests.json", "d3/requests.json"],
}


def requests_digest(kind: str = "clef") -> str:
    """Identifies the parity set a cached result was computed on."""
    digest = hashlib.sha256()
    for name in SETS[kind]:
        digest.update((PARITY / name).read_bytes())
    return digest.hexdigest()[:16] if kind == "clef" else f"{kind}-{digest.hexdigest()[:16]}"


def h2o_reference(fixture: dict) -> dict:
    """H2O-Lightning's reference in the shape of Clef's: questions with option
    ids and probabilities, before the floor; refused requests and questions left out."""
    reference = {}
    for case_id, entry in fixture["results"].items():
        if "refused" in entry:
            continue
        questions = []
        for question_id, result in entry.items():
            if "refused" in result:
                continue
            answer = result["answer_no_floor"]
            if answer["type"] == "noul":
                probabilities = {"true": answer["noul"], "false": 1 - answer["noul"]}
            else:
                probabilities = answer["probabilities"]
            questions.append(
                {
                    "id": question_id,
                    "option_ids": list(probabilities),
                    "probabilities": list(probabilities.values()),
                }
            )
        reference[case_id] = {"questions": questions}
    return reference


def d3_reference(fixture: dict) -> dict:
    """d3's reference in the shape of Clef's: each question's option ids (as the
    answers name them) and probabilities; refused requests and questions left out."""
    from mlx_decision.models.d3.prompt import options
    from mlx_decision.types import parse_request

    cases = {}
    for name in SETS["d3"]:
        cases |= {case["id"]: case for case in json.loads((PARITY / name).read_text())}
    reference = {}
    for case_id, entry in fixture["results"].items():
        if "refused" in entry:
            continue
        case = cases[case_id]
        request = parse_request({"state": case["state"], "questions": case["questions"]})
        questions = [
            {
                "id": question_id,
                "option_ids": options(request.questions[question_id])[0],
                "probabilities": result["probabilities"],
            }
            for question_id, result in entry.items()
            if "refused" not in result
        ]
        if questions:
            reference[case_id] = {"questions": questions}
    return reference


def answerable(loaded, request):
    """The request without the questions the model refuses, or None if none is left."""
    from mlx_decision.errors import DecisionError

    kept = {}
    for question_id, question in request.questions.items():
        try:
            loaded.backend.encode(request.model_copy(update={"questions": {question_id: question}}))
        except DecisionError:
            continue
        kept[question_id] = question
    return request.model_copy(update={"questions": kept}) if kept else None


def run_model(model: str, out: Path, kind: str = "clef") -> None:
    """Worker: answer the parity set with one model and write the numbers."""
    import mlx.core as mx

    import mlx_decision
    from mlx_decision.benchmark import forget_prefixes
    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request

    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    cases = []
    for name in SETS[kind]:
        cases += json.loads((PARITY / name).read_text())
    from mlx_decision.hub import resolve_model_path
    from mlx_decision.registry import detect_family

    family = detect_family(resolve_model_path(model)).name
    # Clef: no image cap beyond the processor's own, as the reference.
    # H2O-Lightning: the answers before the floor, as compared.
    # d3: the 40k request, past the default limit.
    options = {
        "clef": {"max_image_pixels": None},
        "h2o": {"noul_floor": 0},
        "d3": {"max_input_tokens": 262144},
    }.get(family, {})
    start = time.perf_counter()
    loaded = mlx_decision.load(model, **options)
    load_seconds = time.perf_counter() - start
    active_after_load = mx.get_active_memory()
    # Image requests carry their images, as when the reference was computed.
    bodies = {
        c["id"]: {
            "state": c["state"],
            "questions": c["questions"],
            **({"images": [open_image(s) for s in c["images"]]} if c.get("images") else {}),
        }
        for c in cases
    }
    if kind == "clef":
        requests = {case_id: parse_request(body) for case_id, body in bodies.items()}
    elif kind == "d3":
        requests = {}
        for case_id, body in bodies.items():
            try:
                request = answerable(loaded, parse_request(body))
            except DecisionError:
                continue
            if request is not None:
                requests[case_id] = request
    else:
        # As served: images in states lifted out; what the model refuses, left out.
        requests = {}
        for case_id, body in bodies.items():
            try:
                request = loaded.check(body)
                loaded.backend.open_images(request)
            except DecisionError:
                continue
            requests[case_id] = request
    loaded.backend.score(requests[cases[0]["id"]])  # warm-up
    mx.reset_peak_memory()
    probabilities, seconds = {}, {}
    for case_id, request in requests.items():
        forget_prefixes(loaded)  # time every request as a new one
        start = time.perf_counter()
        output = loaded.backend.score(request)
        seconds[case_id] = time.perf_counter() - start
        probabilities[case_id] = output.probabilities
    out.write_text(
        json.dumps(
            {
                "model": Path(model).name,
                "requests": requests_digest(kind),
                "disk_gb": folder_gb(model),
                "load_seconds": load_seconds,
                "active_gb": active_after_load / 2**30,
                "peak_gb": mx.get_peak_memory() / 2**30,
                "seconds": seconds,
                "probabilities": probabilities,
            }
        )
    )


def folder_gb(model: str) -> float:
    from mlx_decision.hub import resolve_model_path

    path = resolve_model_path(model).resolve()
    return sum(f.resolve().stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30


def baseline_from_file(fixture: dict) -> dict:
    """mlx-decision's results stored in a reference file, in a worker's shape."""
    results = fixture["mlx"]["results"]
    return {
        "model": "Clef bf16 (larger Mac)",
        "requests": requests_digest(),
        "disk_gb": None,
        "load_seconds": fixture["mlx"]["meta"].get("load_s"),
        "active_gb": None,
        "peak_gb": max(entry["peak_gb"] for entry in results.values()),
        "seconds": {case_id: entry["seconds"] for case_id, entry in results.items()},
        "probabilities": {case_id: entry["probabilities"] for case_id, entry in results.items()},
    }


def number(value, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


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
    parser.add_argument("--cache", type=Path, help="keep per-model results here and reuse them")
    parser.add_argument("--rerun", action="store_true", help="run every model again")
    parser.add_argument(
        "--reference",
        type=Path,
        default=PARITY / "reference.json",
        help="reference fixture (default: clef-flash's); one with mlx results gives the baseline",
    )
    parser.add_argument(
        "--worker", nargs=3, metavar=("MODEL", "OUT", "SET"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        run_model(args.worker[0], Path(args.worker[1]), args.worker[2])
        return
    fixture = json.loads(args.reference.read_text())
    repo_id = fixture.get("meta", {}).get("repo_id", "")
    kind = {"h2oai/h2o-lightning-4b": "h2o"}.get(repo_id, "d3" if "/d3" in repo_id else "clef")
    if kind == "h2o":
        fixture = {"results": h2o_reference(fixture)}
    elif kind == "d3":
        fixture = {"results": d3_reference(fixture)}
    if (
        "requests_sha256" in fixture
        and fixture["requests_sha256"]
        != hashlib.sha256((PARITY / "requests.json").read_bytes()).hexdigest()
    ):
        parser.error(f"{args.reference} was computed on another parity set")
    stored = "mlx" in fixture
    if not args.models and not stored:
        parser.error("give at least one model")
    reference = fixture.get("reference", fixture)["results"]
    torch_probabilities = {
        case_id: {
            q["id"]: dict(zip(q["option_ids"], q["probabilities"], strict=True))
            for q in entry["questions"]
        }
        for case_id, entry in reference.items()
    }
    results = {}
    if stored:
        results["Clef bf16 (larger Mac)"] = baseline_from_file(fixture)
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for model in args.models:
            cached = args.cache / f"{Path(model).name}.json" if args.cache else None
            if (
                cached
                and cached.exists()
                and not args.rerun
                and json.loads(cached.read_text()).get("requests") == requests_digest(kind)
            ):
                print(f"cached {model}", file=sys.stderr, flush=True)
                results[model] = json.loads(cached.read_text())
                continue
            out = cached or Path(scratch) / "result.json"
            print(f"running {model} ...", file=sys.stderr, flush=True)
            command = [sys.executable, __file__, "--worker", model, str(out), kind]
            subprocess.run(command, check=True)
            results[model] = json.loads(out.read_text())

    baseline = next(iter(results))
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
        disk_gb = result.get("disk_gb") if model in args.models else None
        if model in args.models and disk_gb is None:
            disk_gb = folder_gb(model)
        lines.append(
            f"| `{Path(model).name}` | {number(disk_gb, 1)} | {number(result['load_seconds'], 1)} "
            f"| {number(result['active_gb'], 1)} / {result['peak_gb']:.1f} | {sum(seconds):.1f} "
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
