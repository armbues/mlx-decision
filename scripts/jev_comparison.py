"""Compare local models with the hosted Jev service on the parity set.

usage:
  python scripts/jev_comparison.py fetch --out FILE
      Send every parity-set request to Jev once (needs TYPESAFE_BASE_URL and
      TYPESAFE_API_KEY in the environment; costs credits) and save the raw
      responses. Refuses to overwrite FILE.
  python scripts/jev_comparison.py compare --jev FILE MODEL [MODEL ...]
      Answer the same requests with each local model and print, per question
      type, how often it agrees with Jev.

Jev and Clef are different models; agreement is information, not a test.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ROOT / "tests" / "parity" / "requests.json"


def fetch(out: Path) -> None:
    import httpx

    if out.exists():
        raise SystemExit(f"{out} exists; not sending the requests again")
    base_url = os.environ["TYPESAFE_BASE_URL"].rstrip("/")
    headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}
    results = {}
    input_tokens = 0
    with httpx.Client(timeout=120) as client:
        for case in json.loads(REQUESTS.read_text()):
            body = {"model": "jev-latest", "state": case["state"], "questions": case["questions"]}
            start = time.perf_counter()
            response = client.post(f"{base_url}/v1/systemone", json=body, headers=headers)
            seconds = time.perf_counter() - start
            try:
                data = response.json()
            except ValueError:
                data = response.text
            results[case["id"]] = {"status": response.status_code, "seconds": seconds, "body": data}
            if response.status_code == 200:
                input_tokens += data["usage"]["input_tokens"]
                note = f"{data['usage']['input_tokens']} tokens"
            else:
                error = data.get("error", {}) if isinstance(data, dict) else {}
                note = f"{error.get('param')}: {error.get('message', data)}"
            print(f"{case['id']:30s} {response.status_code} {seconds:5.2f}s {note}", flush=True)
            # Save as we go, so an interruption never makes us resend.
            out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    ok = sum(r["status"] == 200 for r in results.values())
    print(f"{ok}/{len(results)} answered, {input_tokens} input tokens billed by Jev's count")


def local_answers(model: str, cases: list[dict]) -> dict:
    import mlx.core as mx

    import mlx_decision

    loaded = mlx_decision.load(model)
    answers = {}
    for case in cases:
        result = loaded.decide(case["state"], case["questions"])
        answers[case["id"]] = result.to_wire()["answers"]
    del loaded
    mx.clear_cache()
    return answers


def agreement(jev: dict, ours: dict, cases: list[dict]) -> dict:
    """Per question type: count and agreement measures."""
    stats = {
        "choice": {"n": 0, "same": 0},
        "score": {"n": 0, "same_level": 0, "diffs": []},
        "noul": {"n": 0, "same_side": 0, "diffs": []},
    }
    for case in cases:
        response = jev.get(case["id"])
        if not response or response["status"] != 200:
            continue
        for question_id, theirs in response["body"]["answers"].items():
            mine = ours[case["id"]][question_id]
            kind = theirs["type"]
            entry = stats[kind]
            entry["n"] += 1
            if kind == "choice":
                entry["same"] += mine["choice"] == theirs["choice"]
            elif kind == "score":
                entry["same_level"] += round(mine["score"]) == round(theirs["score"])
                entry["diffs"].append(abs(mine["score"] - theirs["score"]))
            else:
                entry["same_side"] += (mine["noul"] >= 0.5) == (theirs["noul"] >= 0.5)
                entry["diffs"].append(abs(mine["noul"] - theirs["noul"]))
    return stats


def compare(jev_file: Path, models: list[str]) -> None:
    jev = json.loads(jev_file.read_text())
    cases = json.loads(REQUESTS.read_text())
    answered = [c for c in cases if jev.get(c["id"], {}).get("status") == 200]
    rejected = {c["id"]: jev[c["id"]] for c in cases if c["id"] in jev and c not in answered}
    lines = [
        f"Parity set: {len(cases)} requests; Jev answered {len(answered)}, "
        f"rejected {len(rejected)}.",
        "",
        "| Local model | Choice: same answer | Score: same rounded level / mean abs diff "
        "| Noul: same side of 0.5 / mean abs diff |",
        "|---|---|---|---|",
    ]

    def pct(part: int, whole: int) -> str:
        return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "-"

    for model in models:
        print(f"answering with {model} ...", file=sys.stderr, flush=True)
        stats = agreement(jev, local_answers(model, answered), answered)
        choice, score, noul = stats["choice"], stats["score"], stats["noul"]
        lines.append(
            f"| `{Path(model).name}` | {pct(choice['same'], choice['n'])} "
            f"| {pct(score['same_level'], score['n'])} / "
            f"{statistics.mean(score['diffs']) if score['diffs'] else 0:.2f} "
            f"| {pct(noul['same_side'], noul['n'])} / "
            f"{statistics.mean(noul['diffs']) if noul['diffs'] else 0:.2f} |"
        )
    if rejected:
        lines += ["", "Rejected by Jev:"]
        for case_id, response in rejected.items():
            body = response["body"]
            error = body.get("error", body) if isinstance(body, dict) else body
            lines.append(f"- `{case_id}` ({response['status']}): {json.dumps(error)[:160]}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("--out", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--jev", type=Path, required=True)
    compare_parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.out)
    else:
        compare(args.jev, args.models)


if __name__ == "__main__":
    main()
