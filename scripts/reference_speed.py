"""Compare the speed of mlx-decision with Cloudflare's PyTorch reference on the same requests.

usage:
  python scripts/reference_speed.py torch --model PATH --out reference.json
  python scripts/reference_speed.py mlx --model PATH --out mlx.json
  python scripts/reference_speed.py report reference.json mlx.json [mlx-8bit.json ...]

Run each side in its own process, one at a time: each loads a 9B model.
The ``torch`` side needs torch, transformers, safetensors, Pillow and
torchvision (not dependencies of the package) and builds the model the way
``make_parity_reference.py`` does: bf16 on MPS, as released. The ``mlx``
side needs the ``images`` extra.

Both sides answer the same requests, built from the model's tokenizer: the
text grid of ``mlx-decision benchmark`` (state lengths x numbers of
questions) and a short state with a photo at three sizes. Each request is
timed end to end, from the request to the probabilities (input encoding and
image preprocessing included), once as a warm-up and then ``--repeats``
times; the median counts. ``report`` prints a Markdown table with the
speedup over the reference and the largest probability difference to it.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "parity"))

LENGTHS = (250, 1000, 4000, 15000)
QUESTION_COUNTS = (1, 5, 20)
IMAGE_SIZES = ((640, 480), (1024, 768), (1920, 1080))
IMAGE_FILE = "lake.jpg"
IMAGE_STATE = "A customer sent this photo with their support request about a booking."


def make_cases(model_path: Path) -> list[dict]:
    """The requests, identical on both sides: text grid first, then images."""
    from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer
    from mlx_decision.benchmark import make_questions, make_state

    tokenizer = load_tokenizer(model_path)
    rate = len(tokenizer.encode(make_state(1000)).ids) / 1000
    cases = []
    for length in LENGTHS:
        state = make_state(round(length / rate))
        for count in QUESTION_COUNTS:
            cases.append(
                {
                    "id": f"text-{length}-q{count}",
                    "state_tokens": length,
                    "questions": make_questions(count),
                    "state": state,
                }
            )
    for width, height in IMAGE_SIZES:
        cases.append(
            {
                "id": f"image-{width}x{height}",
                "image": [width, height],
                "questions": make_questions(3),
                "state": IMAGE_STATE,
            }
        )
    return cases


def open_case_image(case: dict):
    from parity_images import open_image

    return open_image({"file": IMAGE_FILE, "size": case["image"]})


def time_runs(answer, repeats: int) -> tuple[list[float], tuple]:
    """Warm-up plus ``repeats`` timed calls of ``answer()``; returns times and the last output."""
    output = answer()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = answer()
        times.append(time.perf_counter() - start)
    return times, output


def machine() -> str:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        memory = int(
            subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        )
        return f"{chip}, {memory // 2**30} GB"
    except (OSError, ValueError):
        return platform.machine()


def run_torch(args) -> dict:
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_parity_reference import PeakMemory, load_reference

    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    start = time.perf_counter()
    model, processor = load_reference(args.model, "mps")
    load_s = time.perf_counter() - start
    from joint_schema_model import collate_records, encode_record

    device = torch.device("mps")
    results = []
    for case in make_cases(args.model):
        record = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
        if "image" in case:
            record["images"] = [open_case_image(case)]

        def answer(record=record):
            encoded = encode_record(processor.tokenizer, record, processor=processor)
            batch = collate_records([encoded], processor.tokenizer.pad_token_id, device)
            with torch.inference_mode():
                logits = model(batch)[0]
            # Moving to the CPU waits for the GPU.
            probabilities = {
                question.question_id: dict(
                    zip(question.option_ids, values.float().cpu().softmax(-1).tolist(), strict=True)
                )
                for question, values in zip(encoded.questions, logits, strict=True)
            }
            return len(encoded.input_ids), probabilities

        with PeakMemory(torch) as memory:
            times, (tokens, probabilities) = time_runs(answer, args.repeats)
        results.append(summarise(case, times, tokens, probabilities, memory.peak_gb))
        torch.mps.empty_cache()
    meta = {
        "backend": "torch",
        "label": args.label or "PyTorch reference (MPS, bf16)",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    return {"meta": meta, "load_s": round(load_s, 2), "cases": results}


def run_mlx(args) -> dict:
    import mlx.core as mx

    import mlx_decision

    start = time.perf_counter()
    model = mlx_decision.load(args.model)
    load_s = time.perf_counter() - start
    results = []
    for case in make_cases(args.model):
        images = [open_case_image(case)] if "image" in case else None

        def answer(case=case, images=images):
            result = model.decide(case["state"], case["questions"], images=images)
            probabilities = {
                key: {"true": answer.noul} if answer.type == "noul" else answer.probabilities
                for key, answer in result.answers.items()
            }
            return result.usage.input_tokens, probabilities

        mx.reset_peak_memory()
        times, (tokens, probabilities) = time_runs(answer, args.repeats)
        peak_gb = round(mx.get_peak_memory() / 2**30, 1)
        results.append(summarise(case, times, tokens, probabilities, peak_gb))
    meta = {
        "backend": "mlx",
        "label": args.label or f"mlx-decision ({args.model.name})",
        "mlx": mx.__version__,
        "mlx_decision": mlx_decision.__version__,
    }
    return {"meta": meta, "load_s": round(load_s, 2), "cases": results}


def summarise(case, times, tokens, probabilities, peak_gb) -> dict:
    row = {
        "id": case["id"],
        "questions": len(case["questions"]),
        "image": case.get("image"),
        "input_tokens": tokens,
        "median_s": round(statistics.median(times), 3),
        "times": [round(t, 3) for t in times],
        "peak_gb": peak_gb,
        "probabilities": probabilities,
    }
    print(
        f"{case['id']:20s} tokens={tokens:6d} median={row['median_s']:7.3f}s peak={peak_gb} GB",
        file=sys.stderr,
        flush=True,
    )
    return row


def max_difference(reference: dict, ours: dict) -> float:
    """Largest difference of any probability; a noul answer carries only ``true``."""
    return max(
        abs(value - reference[question][option])
        for question, options in ours.items()
        for option, value in options.items()
    )


def report(paths: list[Path]) -> str:
    runs = [json.loads(path.read_text()) for path in paths]
    reference, others = runs[0], runs[1:]
    if reference["meta"]["backend"] != "torch":
        raise SystemExit("the first file must be the torch run")
    header = ["Input tokens", "Questions", "Image", reference["meta"]["label"]]
    header += [run["meta"]["label"] for run in others]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    worst = [0.0] * len(others)
    for index, base in enumerate(reference["cases"]):
        cells = [
            f"{base['input_tokens']:,}",
            str(base["questions"]),
            "x".join(map(str, base["image"])) if base["image"] else "-",
            f"{base['median_s']:.2f} s",
        ]
        for column, run in enumerate(others):
            case = run["cases"][index]
            if case["id"] != base["id"] or case["input_tokens"] != base["input_tokens"]:
                raise SystemExit(f"{run['meta']['label']}: {case['id']} does not match")
            speedup = base["median_s"] / case["median_s"]
            cells.append(f"{case['median_s']:.2f} s ({speedup:.1f}x)")
            difference = max_difference(base["probabilities"], case["probabilities"])
            worst[column] = max(worst[column], difference)
        lines.append("| " + " | ".join(cells) + " |")

    def peak(run):
        return f"{max(case['peak_gb'] for case in run['cases']):.1f} GB"

    lines.append("| Peak memory | | | " + " | ".join(peak(run) for run in runs) + " |")
    lines.append("| Load | | | " + " | ".join(f"{run['load_s']:.1f} s" for run in runs) + " |")
    lines.append("")
    meta = reference["meta"]
    lines.append(
        f"{runs[0].get('machine', '')}; median of {len(reference['cases'][0]['times'])} "
        f"runs after a warm-up; torch {meta['torch']}, transformers {meta['transformers']}."
    )
    for run, difference in zip(others, worst, strict=True):
        lines.append(
            f"{run['meta']['label']}: largest probability difference to the reference "
            f"{difference:.3f}."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("torch", "mlx"):
        command = commands.add_parser(name, help=f"time the {name} side")
        command.add_argument("--model", type=Path, required=True, help="local model folder")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--repeats", type=int, default=3)
        command.add_argument("--label", help="column heading in the report")
        if name == "torch":
            command.add_argument(
                "--memory-fraction",
                type=float,
                default=0.9,
                help="cap on MPS memory, as a fraction of the recommended maximum",
            )
    command = commands.add_parser("report", help="print the comparison table")
    command.add_argument("runs", type=Path, nargs="+", help="the torch run first")
    args = parser.parse_args()

    if args.command == "report":
        print(report(args.runs))
        return
    run = run_torch(args) if args.command == "torch" else run_mlx(args)
    run["machine"] = machine()
    run["date"] = date.today().isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(run, indent=1) + "\n")


if __name__ == "__main__":
    main()
