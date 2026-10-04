"""Accuracy on established classification benchmarks, for local models and Jev.

usage:
  python scripts/accuracy_benchmark.py estimate --datasets DIR
  python scripts/accuracy_benchmark.py local --datasets DIR --model MODEL --out FILE
  python scripts/accuracy_benchmark.py jev --datasets DIR --out FILE
  python scripts/accuracy_benchmark.py report RESULTS [RESULTS ...]

DIR holds the test splits saved with ``datasets.save_to_disk`` as
``<org>/<name>``: fancyzhx/ag_news, dair-ai/emotion, facebook/anli,
mteb/banking77, cais/mmlu. Each benchmark is sampled to ``--samples``
examples (default 500) with a fixed seed. Questions and option descriptions
are written here; results are therefore comparable between the models run
with this script, and only roughly with published numbers.

``jev`` needs TYPESAFE_BASE_URL and TYPESAFE_API_KEY, costs credits, and
resumes: examples already in FILE are not sent again.
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 1234
PRICE_PER_MILLION = 0.042
JEV_TOKEN_FACTOR = 1.2  # Jev counts about this many more input tokens than Clef

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
NLI = {
    "entailment": "The premise shows that the hypothesis is true",
    "neutral": "The premise neither proves nor contradicts the hypothesis",
    "contradiction": "The premise shows that the hypothesis is false",
}


def load(datasets_dir: Path, name: str):
    from datasets import load_from_disk

    return load_from_disk(str(datasets_dir / name))


def sample(rows: list, count: int) -> list:
    rng = random.Random(SEED)
    return rng.sample(rows, min(count, len(rows)))


def ag_news(datasets_dir: Path, count: int) -> list[dict]:
    data = load(datasets_dir, "fancyzhx/ag_news")
    labels = list(AG_NEWS)
    rows = sample(list(data), count)
    return [
        {
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
        for row in rows
    ]


def emotion(datasets_dir: Path, count: int) -> list[dict]:
    data = load(datasets_dir, "dair-ai/emotion")
    labels = data.features["label"].names
    return [
        {
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
        for row in sample(list(data), count)
    ]


def anli(datasets_dir: Path, count: int) -> list[dict]:
    data = load(datasets_dir, "facebook/anli")
    labels = data["test_r1"].features["label"].names
    rows = [row for split in ("test_r1", "test_r2", "test_r3") for row in data[split]]
    return [
        {
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
        for row in sample(rows, count)
    ]


def banking77(datasets_dir: Path, count: int) -> list[dict]:
    data = load(datasets_dir, "mteb/banking77")
    rows = list(data)
    intents = sorted({row["label_text"] for row in rows})
    criteria = {intent: intent.replace("_", " ") for intent in intents}
    return [
        {
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
        for row in sample(rows, count)
    ]


def mmlu(datasets_dir: Path, count: int) -> list[dict]:
    data = load(datasets_dir, "cais/mmlu")
    letters = ["A", "B", "C", "D"]
    return [
        {
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
        for row in sample(list(data), count)
    ]


BENCHMARKS = {
    "ag_news": ag_news,
    "emotion": emotion,
    "anli": anli,
    "banking77": banking77,
    "mmlu": mmlu,
}


def build(datasets_dir: Path, count: int) -> dict[str, list[dict]]:
    return {name: make(datasets_dir, count) for name, make in BENCHMARKS.items()}


def wire(example: dict) -> dict:
    return {"state": example["state"], "questions": example["questions"]}


def record(answer: dict, gold: str) -> dict:
    probabilities = answer["probabilities"]
    return {
        "gold": gold,
        "predicted": answer["choice"],
        "p_max": max(probabilities.values()),
        "confidence": answer["confidence"],
        "probabilities": probabilities,
    }


def estimate(args) -> None:
    from tokenizers import Tokenizer

    from mlx_decision.models.clef.encode import encode_request
    from mlx_decision.types import parse_request

    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer)))
    total = 0
    for name, examples in build(args.datasets, args.samples).items():
        tokens = [
            len(encode_request(tokenizer, parse_request(wire(e))).input_ids) for e in examples
        ]
        total += sum(tokens)
        print(
            f"{name:10s} {len(examples):5d} examples {sum(tokens):9d} tokens "
            f"(mean {statistics.mean(tokens):.0f})"
        )
    jev_tokens = total * JEV_TOKEN_FACTOR
    print(
        f"total {total} tokens; Jev about {jev_tokens:.0f} tokens, "
        f"${jev_tokens * PRICE_PER_MILLION / 1e6:.3f}; Clef about {total / 1700 / 60:.0f} min"
    )


def local(args) -> None:
    import mlx_decision

    model = mlx_decision.load(args.model)
    results = {"model": model.name, "benchmarks": {}}
    for name, examples in build(args.datasets, args.samples).items():
        start = time.perf_counter()
        records = []
        for example in examples:
            answer = model.decide_request(wire(example)).to_wire()["answers"]
            records.append(record(next(iter(answer.values())), example["gold"]))
        results["benchmarks"][name] = records
        print(f"{name:10s} {time.perf_counter() - start:6.1f}s", file=sys.stderr, flush=True)
    args.out.write_text(json.dumps(results))


def jev(args) -> None:
    import httpx

    base_url = os.environ["TYPESAFE_BASE_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    done = json.loads(args.out.read_text()) if args.out.exists() else {"model": "jev"}
    done.setdefault("benchmarks", {})
    raw = done.setdefault("raw", {})
    lock = threading.Lock()
    usage = {"tokens": 0, "errors": 0}

    def send(client, key, example):
        body = {"model": "jev-latest", **wire(example)}
        for attempt in range(3):
            response = client.post(f"{base_url}/v1/systemone", json=body, headers=headers)
            if response.status_code not in (429, 500, 502, 503):
                break
            time.sleep(2 * (attempt + 1))
        with lock:
            raw[key] = {"status": response.status_code, "body": response.json()}
            if response.status_code == 200:
                usage["tokens"] += response.json()["usage"]["input_tokens"]
            else:
                usage["errors"] += 1

    with httpx.Client(timeout=120) as client, ThreadPoolExecutor(args.concurrency) as pool:
        for name, examples in build(args.datasets, args.samples).items():
            todo = [(f"{name}/{i}", e) for i, e in enumerate(examples) if f"{name}/{i}" not in raw]
            list(pool.map(lambda item: send(client, *item), todo))
            records = []
            for i, example in enumerate(examples):
                response = raw.get(f"{name}/{i}")
                if response and response["status"] == 200:
                    answer = next(iter(response["body"]["answers"].values()))
                    records.append(record(answer, example["gold"]))
            done["benchmarks"][name] = records
            args.out.write_text(json.dumps(done))
            print(
                f"{name:10s} sent {len(todo)}, answered {len(records)}/{len(examples)}",
                file=sys.stderr,
                flush=True,
            )
    print(f"Jev input tokens this run: {usage['tokens']}, errors: {usage['errors']}")


def macro_f1(records: list[dict]) -> float:
    labels = {r["gold"] for r in records}
    scores = []
    for label in labels:
        tp = sum(r["predicted"] == label and r["gold"] == label for r in records)
        fp = sum(r["predicted"] == label and r["gold"] != label for r in records)
        fn = sum(r["predicted"] != label and r["gold"] == label for r in records)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return statistics.mean(scores)


def ece(records: list[dict], bins: int = 10) -> float:
    """Expected calibration error of the top probability."""
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        group = [r for r in records if low < r["p_max"] <= high or (b == 0 and r["p_max"] == 0)]
        if group:
            accuracy = sum(r["predicted"] == r["gold"] for r in group) / len(group)
            mean_p = statistics.mean(r["p_max"] for r in group)
            total += len(group) / len(records) * abs(accuracy - mean_p)
    return total


def report(args) -> None:
    runs = [json.loads(Path(path).read_text()) for path in args.results]
    lines = [
        "| Benchmark | "
        + " | ".join(f"{run['model']}: acc / macro-F1 / ECE (n)" for run in runs)
        + " |",
        "|---|" + "---|" * len(runs),
    ]
    for name in BENCHMARKS:
        cells = []
        for run in runs:
            records = run["benchmarks"].get(name, [])
            if not records:
                cells.append("-")
                continue
            accuracy = sum(r["predicted"] == r["gold"] for r in records) / len(records)
            cells.append(
                f"{100 * accuracy:.1f} / {100 * macro_f1(records):.1f} / "
                f"{100 * ece(records):.1f} ({len(records)})"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "local", "jev"):
        sub = commands.add_parser(name)
        sub.add_argument("--datasets", type=Path, required=True)
        sub.add_argument("--samples", type=int, default=500)
        if name == "estimate":
            sub.add_argument("--tokenizer", required=True, help="a model's tokenizer.json")
        if name == "local":
            sub.add_argument("--model", required=True)
        if name in ("local", "jev"):
            sub.add_argument("--out", type=Path, required=True)
        if name == "jev":
            sub.add_argument("--concurrency", type=int, default=4)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    {"estimate": estimate, "local": local, "jev": jev, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
