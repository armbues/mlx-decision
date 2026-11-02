"""Run Base Labs' serving code on the parity set; store prompts, token ids and answers.

usage: python scripts/make_baselabs_reference.py [--onejev DIR] [--only ID,ID]
       python scripts/make_baselabs_reference.py --model PATH [--only ID,ID]
       python scripts/make_baselabs_reference.py --images [--only ID,ID]

The parity set is the text requests of tests/parity/requests.json and
tests/parity/baselabs/requests.json (Base Labs' edge cases). Needs torch
and transformers (not dependencies of the package), the releases' small
files (fetched into the Hugging Face cache at the pinned revisions, no
weights) and a checkout of OneJev at the commit the releases serve with,
with their `serve.patch` applied (`--onejev`, default `_reference/onejev`;
the script checks both, and that the releases' patches are the same):

    git clone https://github.com/OmniJev/OneJev.git && cd OneJev
    git checkout 81ce62f1597c91e46767d4d02d6ac2e18534fe94
    git apply <elephant-1 snapshot>/serve.patch

The releases serve through OneJev's `qev.engine.DecisionEngine`, imported
from that checkout and run as the releases' `engine.py` does, on the CPU
in float32: it renders each question, finds the shared prefix, refuses
requests over its limits, runs the prefix once and the questions off
copies of its cache (block-bidirectional attention in the full-attention
layers) and reads the answer codes' logits from the LM head. The models
are tiny random ones in the three releases' shapes (tests/tiny_models.py,
`write_baselabs`), each with its release's tokenizer and chat template.
joey-1's config ties the head to the embeddings yet stores an `lm_head`
too; the tiny joey's stored head differs from its embeddings, and the
release's transformers ties only equal tensors, so its logits come from
the stored head.

Writes tests/parity/baselabs/reference.json with, per request: the shared
prefix's length and the request's tokens, or the release's refusal; per
question: the token count, a SHA-256 of the token ids and of the prompt
text, for questions of at most IDS_TOKENS tokens the prompt text and the
token ids, and unless the release refuses the request, per shape the
answer-code logits and the release's answer. Every request is answered
both batched (the releases' mode) and one question at a time; the largest
difference between the two, per shape, goes into the meta. Existing entries are kept,
so a run can be continued with --only.

With --model (a release folder with its weights, at the pinned revision;
which release it is comes from its config.json) the real model runs
instead, as the release's engine.py serves it: bfloat16 on MPS, the LM
head in float32, the engine's own choice of batched or sequential
branches. It answers every request within the release's limits and writes
tests/parity/baselabs/reference-bf16.json (elephant-1) or
reference-<shape>-bf16.json (joey-1: reference-joey-bf16.json) with, per
question, the token count and digests, the answer-code logits and the
release's answer (no prompts or ids), after every request.

With --images the image requests run instead: the parity set's and
tests/parity/baselabs/image-requests.json (images inside the state as
{"image_file": ...}, see tests/parity/parity_images.py). The releases serve
text only, so the reference is their training-side scoring, which the
serve.patch routes through train.common.batch_scores: train.sft.collect_mm
with the multimodal bundle and the block mask, one question per row with its
whole prompt, on the three tiny models (CPU, float32). A request is mapped to
the releases' convention: images inside the state become <image:N>
placeholders in place, top-level images placeholders at the start of the
state block, before its text (numbered first). First, text requests run
through the same path and must score as the text fixture (the serving
engine). Writes tests/parity/baselabs/reference-images.json with, per
request, the mapped state and media, and per question the prompt, the
expanded token ids (up to IDS_TOKENS) and digests, the prefix length (to the
end of the state block, image tokens included), the image grids and the
logits per shape.
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
    "joey": ("baselabs/joey-1", "9efbe485efbed9598dfca5294cbbe8da00e21908"),
}
# What --model writes, per release (elephant-1's was the first).
BF16_FILES = {shape: f"reference-{shape}-bf16.json" for shape in RELEASES}
BF16_FILES["elephant"] = "reference-bf16.json"
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


def which_release(folder: Path, snapshots: dict[str, Path]) -> str:
    """The release a weights folder is: the one whose config.json it has, byte for byte."""
    config = (folder / "config.json").read_bytes()
    found = [s for s, path in snapshots.items() if (path / "config.json").read_bytes() == config]
    if len(found) != 1:
        raise SystemExit(f"{folder}: config.json is none of the pinned releases'")
    return found[0]


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


# --images: the parity set's image requests and Base Labs' own, scored by the
# releases' training-side code (D234); text requests run through the same path
# to check it against the text fixture.
IMAGE_TEXT_CHECK = ["ticket", "bl_every_type_twice", "special_tokens_in_state"]
# Largest allowed logit difference of that check (R31: 5.7e-7).
IMAGE_TEXT_TOLERANCE = 1e-4


def image_set() -> list[dict]:
    cases = [c for c in json.loads((PARITY / "requests.json").read_text()) if c.get("images")]
    return cases + json.loads((PARITY / "baselabs" / "image-requests.json").read_text())


class ImageReference:
    """The releases' training-side scoring on a tiny model in ``folder`` (CPU,
    float32): ``train.sft.collect_mm``, so ``train.common.batch_scores`` with the
    multimodal bundle and the block mask, one question per row, as
    ``train.eval_mm`` scores a checkpoint."""

    def __init__(self, folder: Path):
        from qev.readout import Readout, ReadoutConfig
        from train import common

        self.bundle = common.load_bundle(
            str(folder), device="cpu", dtype="float32", multimodal=True
        )
        config = ReadoutConfig.load(str(folder))
        if not config.noncausal or config.head:
            raise SystemExit(f"{folder}: not the releases' readout ({config})")
        self.bundle.readout = Readout.attach(
            self.bundle.model, config, slot_ids=self.bundle.slot_ids
        )

    def rows(self, state, questions: dict, media: list[dict]) -> list[dict]:
        """One row per question, its target on the first option (the scores do not use it)."""
        from qev.prompt import render_question
        from train import common

        rows = []
        for qid, question in questions.items():
            labels = render_question("q", common.build_question(question)).labels
            row = {"id": qid, "state": state, "question": question, "target": {labels[0]: 1.0}}
            rows.append(row | ({"media": media} if media else {}))
        return rows

    def layout(self, row: dict) -> dict:
        """A row's prompt, its expanded token ids, the prefix length (to the end
        of the state block, the images' tokens included) and the image grids."""
        from qev import readout as ro
        from train import common, mm

        bundle = self.bundle
        example = common.render_example(row, bundle.template_for(row), 0, seed=None, augment=False)
        _, prefix, _ = ro.token_layout(bundle.tokenizer, example.prompt, example.suffix_start, None)
        media = row.get("media") or []
        inputs, _ = mm.mm_inputs(bundle.processor, [example.prompt], [media], "cpu")
        ids = inputs["input_ids"][0].tolist()
        grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
        prefix += bundle.counter.media_extra(media)
        assert len(ids) == bundle.prompt_length(example.prompt, row), row["id"]
        return {
            "kind": example.kind,
            "options": len(example.labels),
            "prompt": example.prompt,
            "ids": ids,
            "prefix_tokens": prefix,
            "grids": grids,
        }

    def logits(self, rows: list[dict]) -> dict[str, list[float]]:
        from train.sft import collect_mm

        # One row per batch: no padding, so the scores do not depend on the batching.
        scored = collect_mm(self.bundle, rows, 1 << 20, 1 << 20, 1, "cpu")
        return {rows[s.idx]["id"]: s.logits for s in scored}


def run_images(args, snapshots: dict, meta: dict) -> None:
    """--images: write reference-images.json (see the module docstring)."""
    import torch
    import transformers
    from train import mm

    sys.path.insert(0, str(PARITY))
    from parity_images import onejev_media

    from tiny_models import write_baselabs

    cases = image_set()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only.split(","))]
    text = {c["id"]: c for c in parity_set()}
    fixture = json.loads((PARITY / "baselabs" / "reference.json").read_text())["results"]
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    results = dict(existing.get("results", {}))
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        references = {
            shape: ImageReference(write_baselabs(scratch / shape, SEED, shape, files_from=snapshot))
            for shape, snapshot in snapshots.items()
        }
        first = references["elephant"]
        worst = 0.0
        for case_id in IMAGE_TEXT_CHECK:
            case = text[case_id]
            for shape, reference in references.items():
                got = reference.logits(reference.rows(case["state"], case["questions"], []))
                for qid, values in got.items():
                    want = fixture[case_id]["questions"][qid]["logits"][shape]
                    worst = max(worst, max(abs(a - b) for a, b in zip(values, want, strict=True)))
        print(f"text requests through the image path vs the text fixture: {worst:.2e}")
        if worst > IMAGE_TEXT_TOLERANCE:
            raise SystemExit(
                f"over {IMAGE_TEXT_TOLERANCE}: the image path does not score text as serving"
            )
        for case in cases:
            start = time.perf_counter()
            folder = scratch / "media" / case["id"]
            folder.mkdir(parents=True)
            state, media = onejev_media(case, folder)
            mm.set_root(folder)
            questions = {}
            rows = first.rows(state, case["questions"], media)
            for row in rows:
                entry = first.layout(row)
                for reference in references.values():
                    assert reference.layout(row)["ids"] == entry["ids"], (case["id"], row["id"])
                ids = entry.pop("ids")
                entry |= {
                    "tokens": len(ids),
                    "ids_sha256": sha256(ids),
                    "prompt_sha256": sha256(entry["prompt"]),
                }
                if len(ids) <= IDS_TOKENS:
                    entry["ids"] = ids
                questions[row["id"]] = entry
            for shape, reference in references.items():
                for qid, values in reference.logits(rows).items():
                    questions[qid].setdefault("logits", {})[shape] = values
            results[case["id"]] = {"state": state, "media": media, "questions": questions}
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
    size = first.bundle.processor.image_processor.size
    meta = {
        **meta,
        "seed": SEED,
        "ids_tokens": IDS_TOKENS,
        "scoring": "train.sft.collect_mm (train.common.batch_scores), one question per row",
        "image_pixels": {k: size[k] for k in ("shortest_edge", "longest_edge")},
        "text_check_max_difference": worst,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    order = [c["id"] for c in image_set()]
    ordered = {i: results[i] for i in order if i in results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump({"meta": meta, "results": ordered}) + "\n")
    print(f"wrote {len(ordered)} results to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onejev", type=Path, default=ROOT / "_reference" / "onejev")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--model", type=Path, help="run the real model in this release folder")
    parser.add_argument("--device", default="mps", help="device for --model (default: mps)")
    parser.add_argument(
        "--images", action="store_true", help="the image requests (reference-images.json)"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.images and args.model:
        parser.error("--images runs the tiny models only")
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    from tiny_models import write_baselabs

    torch.set_grad_enabled(False)
    if args.model and args.device == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    snapshots = {shape: fetch(*release) for shape, release in RELEASES.items()}
    model_shape = which_release(args.model, snapshots) if args.model else None
    name = BF16_FILES[model_shape] if args.model else "reference.json"
    name = "reference-images.json" if args.images else name
    args.out = args.out or PARITY / "baselabs" / name
    patches = {(path / "serve.patch").read_bytes() for path in snapshots.values()}
    if len(patches) != 1:
        raise SystemExit("the releases' serve.patch files differ")
    onejev = args.onejev.resolve()
    patch_sha256 = check_onejev(onejev, snapshots["elephant"] / "serve.patch")
    sys.path.insert(0, str(onejev))
    from qev.prompt import PROMPT_VERSION

    if args.images:
        releases = {s: {"repo_id": r, "revision": v} for s, (r, v) in RELEASES.items()}
        onejev_meta = {"commit": ONEJEV_COMMIT, "serve_patch_sha256": patch_sha256}
        meta = {"releases": releases, "onejev": onejev_meta, "prompt_version": PROMPT_VERSION}
        run_images(args, snapshots, meta)
        return

    existing = json.loads(args.out.read_text()) if args.out.exists() else {}
    results = dict(existing.get("results", {}))
    fork = dict(existing.get("meta", {}).get("fork_max_difference", {}))
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    with tempfile.TemporaryDirectory() as scratch:
        if args.model:
            references = {model_shape: Reference(args.model.resolve(), True, args.device)}
        else:
            references = {}
            for shape, snapshot in snapshots.items():
                folder = write_baselabs(Path(scratch) / shape, SEED, shape, files_from=snapshot)
                references[shape] = Reference(folder)
        engine = next(iter(references.values())).engine
        meta = {
            "releases": {s: {"repo_id": r, "revision": v} for s, (r, v) in RELEASES.items()},
            "onejev": {"commit": ONEJEV_COMMIT, "serve_patch_sha256": patch_sha256},
            "prompt_version": PROMPT_VERSION,
            **({"model": f"{model_shape}-1 release"} if args.model else {"seed": SEED}),
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
