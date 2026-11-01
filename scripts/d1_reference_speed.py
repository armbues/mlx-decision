"""Compare the speed of mlx-decision with the d1 releases' own PyTorch code on the same requests.

usage:
  python scripts/d1_reference_speed.py torch --model PATH --out reference.json
  python scripts/d1_reference_speed.py mlx --model PATH --out mlx.json
  python scripts/d1_reference_speed.py report reference.json mlx.json

Run each side in its own process. The ``torch`` side needs torch and
transformers (not dependencies of the package) and runs the release's own
code from the model folder (``trust_remote_code``), as its model card shows:
``system_one`` in bfloat16 on MPS (not compiled). The ``mlx`` side loads the
model with its defaults (bfloat16; d1-3B's prefix cache on but cleared before
every request).

Both sides answer the model card's cases (one question, three questions in one
request, a long state: 3,400 tokens for d1-3B; for d1-2B 5,000 tokens and a
384 x 384 picture) and the text grid of ``mlx-decision benchmark`` (250 to
15,000 tokens x 1, 5 and 20 questions). Each request is
timed end to end, from the request to the probabilities, once as a warm-up
and then ``--repeats`` times; the median counts. ``report`` prints a Markdown
table with the speedup and the largest probability difference.
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
# The picture of d1-2B's card case, scaled to 384 x 384 on both sides.
PICTURE = ROOT / "tests" / "parity" / "images" / "dog.jpg"
QUESTION_COUNTS = (1, 5, 20)
LENGTHS = (250, 1000, 4000, 15000)
TICKET = "I was charged twice this month, please refund one of them."
CARD_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}


def is_lfm3(path: Path) -> bool:
    """d1-2B (an LFM3-VL release) rather than d1-3B."""
    return json.loads((path / "config.json").read_text()).get("model_type") == "lfm3_vl"


def make_cases(path: Path) -> list[dict]:
    """The requests, identical on both sides."""
    from tokenizers import Tokenizer

    from mlx_decision.benchmark import make_questions, make_state

    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    rate = len(tokenizer.encode(make_state(1000)).ids) / 1000
    one = {"refund": CARD_QUESTIONS["refund"]}
    long = 5000 if is_lfm3(path) else 3400
    cases = [
        {"id": "card: one question", "state": TICKET, "questions": one},
        {"id": "card: three questions", "state": TICKET, "questions": CARD_QUESTIONS},
        {
            "id": f"card: {long:,}-token state",
            "state": make_state(round(long / rate)),
            "questions": one,
        },
    ]
    if is_lfm3(path):
        from PIL import Image

        picture = Image.open(PICTURE).convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)
        question = {"animal": {"type": "noul", "instructions": "Is there an animal?"}}
        cases.append(
            {
                "id": "card: 384 px picture",
                "state": None,
                "questions": question,
                "images": [picture],
            }  # fmt: skip
        )
    for length in LENGTHS:
        state = make_state(round(length / rate))
        for count in QUESTION_COUNTS:
            cases.append(
                {"id": f"{length} x {count}", "state": state, "questions": make_questions(count)}
            )
    return cases


def time_runs(answer, repeats: int):
    result = answer()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = answer()
        times.append(time.perf_counter() - start)
    return times, result


def machine() -> str:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        chip = platform.machine()
    return f"{chip}, macOS {platform.mac_ver()[0]}"


def probabilities_of(answers: dict) -> dict:
    """Option probabilities per question, a noul as [false, true]."""
    out = {}
    for qid, answer in answers.items():
        if answer["type"] == "noul":
            out[qid] = [1 - answer["noul"], answer["noul"]]
        else:
            out[qid] = [float(v) for v in answer["probabilities"].values()]
    return out


def run_torch(args) -> dict:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to("mps")

    def run(case):
        # Only d1-2B's cases have pictures (d1-3B's system_one takes none).
        media = {"media": case["images"]} if case.get("images") else {}
        result = model.system_one(case["state"], case["questions"], **media)["answers"]
        torch.mps.synchronize()
        return result

    name = args.model.resolve().name
    return measure(args, run, f"{name}'s code, torch {torch.__version__} (MPS, bf16)")


def run_mlx(args) -> dict:
    import mlx_decision

    model = mlx_decision.load(args.model)
    cache = getattr(model.backend, "prefix_cache", None)

    def run(case):
        if cache is not None:
            cache.clear()  # every request as a new one
        result = model.decide(case["state"], case["questions"], images=case.get("images"))
        return {qid: a.model_dump() for qid, a in result.answers.items()}

    return measure(args, run, f"mlx-decision {mlx_decision.__version__} (bf16)")


def measure(args, run, label: str) -> dict:
    rows = []
    for case in make_cases(args.model.resolve()):
        times, answers = time_runs(lambda case=case: run(case), args.repeats)
        rows.append(
            {
                "id": case["id"],
                "median_s": statistics.median(times),
                "probabilities": probabilities_of(answers),
            }
        )
        print(f"{case['id']}: {rows[-1]['median_s'] * 1000:.1f} ms", flush=True)
    return {
        "label": args.label or label,
        "model": args.model.resolve().name,
        "machine": machine(),
        "date": date.today().isoformat(),
        "repeats": args.repeats,
        "rows": rows,
    }


def report(paths: list[Path]) -> str:
    runs = [json.loads(p.read_text()) for p in paths]
    reference, ours = runs[0], runs[1:]
    lines = [
        f"{reference['model']}, {reference['machine']}, {reference['date']}; "
        f"median of {reference['repeats']} runs",
        "",
        "| request | " + " | ".join(r["label"] for r in runs) + " | speedup | max dp |",
        "|---|" + "---:|" * (len(runs) + 2),
    ]
    for index, row in enumerate(reference["rows"]):
        others = [run["rows"][index] for run in ours]
        times = [row["median_s"], *(o["median_s"] for o in others)]
        diff = max(
            abs(a - b)
            for o in others
            for qid, values in row["probabilities"].items()
            for a, b in zip(values, o["probabilities"][qid], strict=True)
        )
        cells = " | ".join(f"{t * 1000:.0f} ms" for t in times)
        speedup = row["median_s"] / others[0]["median_s"]
        lines.append(f"| {row['id']} | {cells} | {speedup:.1f}x | {diff:.3f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("torch", "mlx"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True, help="local model folder")
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--repeats", type=int, default=3)
        command.add_argument("--label", help="column heading in the report")
    command = commands.add_parser("report")
    command.add_argument("runs", type=Path, nargs="+", help="the torch run first")
    args = parser.parse_args()
    if args.command == "report":
        print(report(args.runs))
        return
    result = (run_torch if args.command == "torch" else run_mlx)(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
