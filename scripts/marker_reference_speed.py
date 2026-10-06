"""Compare the speed of mlx-decision with Laya's and Julia's own PyTorch code on the same requests.

usage:
  python scripts/marker_reference_speed.py torch --model PATH --out reference.json
  python scripts/marker_reference_speed.py mlx --model PATH --out mlx.json
  python scripts/marker_reference_speed.py report reference.json mlx.json

Run each side in its own process. The ``torch`` side needs torch and
transformers (not dependencies of the package) and the family's code: Laya's
from the ``laya`` package, Julia's from its release folder (put on the import
path). Both run on MPS as their code runs there: Laya with ``laya.load``
(float32, float16 autocast from 5 questions per request), Julia with its
``TransformerEngine`` (float32; it autocasts only on CUDA). The ``mlx`` side
loads the model with its defaults (float16).

Both sides answer the text grid of ``mlx-decision benchmark`` for the model
(state lengths that fit its input limit x 1, 5 and 20 questions). Each
request is timed end to end, from the request to the probabilities, once as
a warm-up and then ``--repeats`` times; the median counts. ``report`` prints
a Markdown table with the speedup and the largest probability difference.
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

QUESTION_COUNTS = (1, 5, 20)


def family_of(path: Path) -> str:
    if (path / "rl_agent_config.json").exists():
        return "laya"
    if (path / "julia_config.json").exists():
        return "julia"
    sys.exit(f"{path}: neither a Laya nor a Julia release")


def input_limit(path: Path) -> int:
    if family_of(path) == "laya":
        return json.loads((path / "rl_agent_config.json").read_text()).get("max_len", 512)
    policy = path / "inference-policy.json"
    return json.loads(policy.read_text()).get("max_length", 8192) if policy.exists() else 8192


def make_cases(path: Path) -> list[dict]:
    """The requests, identical on both sides."""
    from tokenizers import Tokenizer

    from mlx_decision.benchmark import default_lengths, make_questions, make_state

    tokenizer = Tokenizer.from_file(str(path / "tokenizer" / "tokenizer.json"))
    rate = len(tokenizer.encode(make_state(1000)).ids) / 1000
    return [
        {
            "id": f"{length}x{count}",
            "state": make_state(round(length / rate)),
            "questions": make_questions(count),
        }
        for length in default_lengths(input_limit(path))
        for count in QUESTION_COUNTS
    ]


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
            out[qid] = list(answer["probabilities"].values())
    return out


def run_torch(args) -> dict:
    import torch

    path = args.model.resolve()
    if family_of(path) == "laya":
        import warnings

        import laya

        warnings.filterwarnings("ignore", message="laya: this checkpoint ships")
        agent = laya.load(str(path), device="mps")

        def answer(case):
            return agent.system_one(case["state"], case["questions"])["answers"]

        label = f"laya {laya.__version__}, torch {torch.__version__} (MPS)"
    else:
        sys.path.insert(0, str(path))
        from julia.inference import TransformerEngine

        engine = TransformerEngine(path, device="mps", head_length=512, memory_map=False)

        def answer(case):
            answers = engine.predict(state=case["state"], questions=case["questions"])["answers"]
            for a in answers.values():
                if a["type"] == "noul":
                    a["noul"] = a["probabilities"]["true"]
            return answers

        label = f"Julia's code, torch {torch.__version__} (MPS, float32)"

    def run(case):
        result = answer(case)
        torch.mps.synchronize()
        return result

    return measure(args, run, label)


def run_mlx(args) -> dict:
    import mlx_decision

    model = mlx_decision.load(args.model)

    def run(case):
        result = model.decide(case["state"], case["questions"])
        return {qid: a.model_dump() for qid, a in result.answers.items()}

    return measure(args, run, f"mlx-decision {mlx_decision.__version__} (float16)")


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
        "| state x questions | " + " | ".join(r["label"] for r in runs) + " | speedup | max dp |",
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
    result = run_torch(args) if args.command == "torch" else run_mlx(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
