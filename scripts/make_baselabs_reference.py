"""Run Base Labs' serving code on the parity set; store prompts, token ids and answers.

usage: python scripts/make_baselabs_reference.py [--onejev DIR] [--only ID,ID]
       python scripts/make_baselabs_reference.py --model PATH [--only ID,ID]

The parity set is the text requests of tests/parity/requests.json and
tests/parity/baselabs/requests.json (Base Labs' edge cases). Needs torch
and transformers (not dependencies of the package), the releases' small
files (fetched into the Hugging Face cache at the pinned revisions, no
weights) and a checkout of OneJev at the commit the releases serve with,
with their `serve.patch` applied (`--onejev`, default `_reference/onejev`;
the script checks both):

    git clone https://github.com/OmniJev/OneJev.git && cd OneJev
    git checkout 81ce62f1597c91e46767d4d02d6ac2e18534fe94
    git apply <elephant-1 snapshot>/serve.patch

The releases serve through OneJev's `qev.engine.DecisionEngine`, imported
from that checkout and run as the releases' `engine.py` does, on the CPU
in float32: it renders each question, finds the shared prefix, refuses
requests over its limits, runs the prefix once and the questions off
copies of its cache (block-bidirectional attention in the full-attention
layers) and reads the answer codes' logits from the LM head. The models
are tiny random ones in both releases' shapes (tests/tiny_models.py,
`write_baselabs`) with the releases' tokenizer and chat template.

Writes tests/parity/baselabs/reference.json with, per request: the shared
prefix's length and the request's tokens, or the release's refusal; per
question: the token count, a SHA-256 of the token ids and of the prompt
text, for questions of at most IDS_TOKENS tokens the prompt text and the
token ids, and unless the release refuses the request, per shape the
answer-code logits and the release's answer. Every request is answered
both batched (the releases' mode) and one question at a time; the largest
difference between the two, per shape, goes into the meta. Existing entries are kept,
so a run can be continued with --only.

With --model (elephant-1's release folder with its weights, at the pinned
revision) the real model runs instead, as the release's engine.py serves
it: bfloat16 on MPS, the LM head in float32, the engine's own choice of
batched or sequential branches. It answers every request within the
release's limits and writes tests/parity/baselabs/reference-bf16.json with,
per question, the token count and digests, the answer-code logits and the
release's answer (no prompts or ids), after every request.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
RELEASES = {
    "elephant": ("baselabs/elephant-1", "b3f8a93462fb1051a9de031e94a7ff380e254072"),
    "mammoth": ("baselabs/mammoth-1", "61cec44c281c693992e3d6ebeb44dd3402dfa186"),
}
ONEJEV_COMMIT = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
IDS_TOKENS = 4096  # longer questions keep digests only
SEED = 0
# The release's batched and sequential paths in float32 (its docstring: 3e-4 on Qwen3.5-4B).
FORK_TOLERANCE = 1e-4
MEMORY_FRACTION = 0.7  # of the GPU's working set, on MPS (--model)


def parity_set() -> list[dict]:
    cases = [c for c in json.loads((PARITY / "requests.json").read_text()) if not c.get("images")]
    return cases + json.loads((PARITY / "baselabs" / "requests.json").read_text())


def dump(data: dict) -> str:
    """JSON with one level per line, but lists of numbers on one line."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[[-0-9.e,\s]+\]", lambda m: re.sub(r"\s+", "", m[0]).replace(",", ", "), text)


def sha256(value) -> str:
    data = value if isinstance(value, str) else json.dumps(value)
    return hashlib.sha256(data.encode()).hexdigest()


def fetch(repo_id: str, revision: str) -> Path:
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            ignore_patterns=["*.safetensors", "assets/*"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


def check_onejev(onejev: Path, patch: Path) -> str:
    """The checkout's commit, with the releases' patch applied; its digest."""

    def git(*args) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(onejev), *args], capture_output=True, text=True)

    head = git("rev-parse", "HEAD").stdout.strip()
    if head != ONEJEV_COMMIT:
        raise SystemExit(f"{onejev} is at {head or 'no commit'}, not {ONEJEV_COMMIT}")
    if git("apply", "--reverse", "--check", str(patch)).returncode != 0:
        raise SystemExit(f"{onejev} does not have {patch} applied (git apply it)")
    return hashlib.sha256(patch.read_bytes()).hexdigest()


class Reference:
    """The release's engine on a tiny model in ``folder`` (CPU, float32), or with
    ``real`` on the release's own weights as its engine.py serves them (bfloat16
    on ``device``, the head in float32, the engine's own fork mode)."""

    def __init__(self, folder: Path, real: bool = False, device: str = "cpu"):
        from qev.engine import DecisionEngine

        self.real = real
        if real:
            # The releases' engine.py passes only the model and the device.
            self.engine = DecisionEngine(str(folder), device=device)
            return
        self.engine = DecisionEngine(str(folder), device="cpu", dtype="float32")
        if self.engine.info.fork_mode != "batched":
            raise SystemExit(f"the release's fork check failed: {self.engine.info}")

    def layout(self, request) -> tuple[list[int], list[list[int]], list]:
        """The shared prefix's ids and each question's suffix ids, as the engine
        splits them (with its check that the prefix ends at the state block)."""
        from qev.prompt import render_question

        engine = self.engine
        questions = request.questions.items()
        rendered = [render_question(qid, q, style=engine.style) for qid, q in questions]
        prefix, suffixes, _ = engine._split(request.state, rendered)
        return prefix, suffixes, rendered

    def answers(self, request) -> tuple[dict, dict, float]:
        """Per question the answer-code logits and the answer, batched, and the
        largest probability difference from one question at a time."""
        from qev.calibrate import softmax

        engine = self.engine
        response, meta = engine.decide(request, debug=True)
        diagnostics = meta["diagnostics"]
        logits = {qid: entry["slot_logits"][0] for qid, entry in diagnostics.items()}
        answers = {qid: a.model_dump(exclude_none=True) for qid, a in response.answers.items()}
        if self.real:
            return logits, answers, None
        engine.info.fork_mode = "sequential"
        try:
            _, sequential = engine.decide(request, debug=True)
        finally:
            engine.info.fork_mode = "batched"
        difference = 0.0
        for qid, entry in diagnostics.items():
            a = softmax(entry["slot_logits"][0], 1.0)
            b = softmax(sequential["diagnostics"][qid]["slot_logits"][0], 1.0)
            difference = max(difference, max(abs(x - y) for x, y in zip(a, b, strict=True)))
        return logits, answers, difference


def run_case(case: dict, references: dict[str, Reference]) -> dict:
    from pydantic import ValidationError
    from qev.schema import SystemOneRequest

    body = {"state": case["state"], "questions": case["questions"]}
    try:
        request = SystemOneRequest(**body)
    except ValidationError as error:
        # The question types are a union: the failing type's own check says why.
        errors = [e for e in error.errors() if e["type"] == "value_error"] or error.errors()
        return {"refused": f"schema: {errors[0]['msg'].removeprefix('Value error, ')}"}
    first = next(iter(references.values()))
    engine = first.engine
    try:
        prefix, suffixes, rendered = first.layout(request)
    except ValueError as error:
        return {"refused": str(error)}
    for other in references.values():
        assert other.layout(request)[:2] == (prefix, suffixes), case["id"]
    questions = {}
    for r, suffix in zip(rendered, suffixes, strict=True):
        ids = prefix + suffix
        prompt = engine._render_prompt(request.state, r.suffix)
        assert engine._encode(prompt) == ids, case["id"]
        entry = {
            "kind": r.kind,
            "options": r.n_slots,
            "tokens": len(ids),
            "ids_sha256": sha256(ids),
            "prompt_sha256": sha256(prompt),
        }
        if len(ids) <= IDS_TOKENS and not first.real:
            entry |= {"prompt": prompt, "ids": ids}
        questions[r.qid] = entry
    result = {"prefix_tokens": len(prefix), "tokens": len(prefix) + sum(map(len, suffixes))}
    differences = {}
    for shape, reference in references.items():
        try:
            logits, answers, difference = reference.answers(request)
        except ValueError as error:  # over the release's limits
            result["refused"] = str(error)
            break
        if difference is not None:
            differences[shape] = difference
        for qid, entry in questions.items():
            entry.setdefault("logits", {})[shape] = logits[qid]
            entry.setdefault("answer", {})[shape] = answers[qid]
    result["questions"] = questions
    result["_fork"] = differences
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onejev", type=Path, default=ROOT / "_reference" / "onejev")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--model", type=Path, help="run the real model in this release folder")
    parser.add_argument("--device", default="mps", help="device for --model (default: mps)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    name = "reference-bf16.json" if args.model else "reference.json"
    args.out = args.out or PARITY / "baselabs" / name
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    from tiny_models import write_baselabs

    torch.set_grad_enabled(False)
    if args.model and args.device == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    snapshots = {shape: fetch(*release) for shape, release in RELEASES.items()}
    patches = {(path / "serve.patch").read_bytes() for path in snapshots.values()}
    if len(patches) != 1:
        raise SystemExit("the releases' serve.patch files differ")
    onejev = args.onejev.resolve()
    patch_sha256 = check_onejev(onejev, snapshots["elephant"] / "serve.patch")
    sys.path.insert(0, str(onejev))
    from qev.prompt import PROMPT_VERSION

    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    results = dict(existing.get("results", {}))
    fork = dict(existing.get("meta", {}).get("fork_max_difference", {}))
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    with tempfile.TemporaryDirectory() as scratch:
        if args.model:
            references = {"elephant": Reference(args.model.resolve(), True, args.device)}
        else:
            references = {}
            for shape, snapshot in snapshots.items():
                folder = write_baselabs(Path(scratch) / shape, SEED, shape, files_from=snapshot)
                references[shape] = Reference(folder)
        engine = references["elephant"].engine
        meta = {
            "releases": {s: {"repo_id": r, "revision": v} for s, (r, v) in RELEASES.items()},
            "onejev": {"commit": ONEJEV_COMMIT, "serve_patch_sha256": patch_sha256},
            "prompt_version": PROMPT_VERSION,
            **({"model": "elephant-1 release"} if args.model else {"seed": SEED}),
            **({} if args.model else {"ids_tokens": IDS_TOKENS}),
            "max_branch_tokens": engine.max_branch_tokens,
            "max_request_tokens": engine.max_request_tokens,
            "slot_ids": engine.slot_ids,
            "device": str(engine.device),
            "dtype": engine.info.dtype,
            "head_dtype": engine.head_dtype,
            "fork_mode": engine.info.fork_mode,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

        def write() -> None:
            ordered = {c["id"]: results[c["id"]] for c in parity_set() if c["id"] in results}
            if not args.model:
                meta["fork_max_difference"] = fork
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(dump({"meta": meta, "results": ordered}) + "\n")
            return ordered

        for case in cases:
            start = time.perf_counter()
            result = run_case(case, references)
            for shape, difference in result.pop("_fork", {}).items():
                fork[shape] = max(fork.get(shape, 0.0), difference)
            results[case["id"]] = result
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
            if args.model:
                write()  # a long run can be continued with --only
        worst = max(fork.values(), default=0.0)
        if worst > FORK_TOLERANCE:
            raise SystemExit(f"batched and sequential differ by {worst:.2e}, over {FORK_TOLERANCE}")
        ordered = write()
    print(f"wrote {len(ordered)} results to {args.out}; batched vs sequential at most {worst:.2e}")


if __name__ == "__main__":
    main()
