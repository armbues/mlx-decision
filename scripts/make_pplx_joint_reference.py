"""Run pplx-decider v1.2's own code on the parity set and store token ids and tiny-model answers.

usage: python scripts/make_pplx_joint_reference.py [--only ID,ID]
       python scripts/make_pplx_joint_reference.py --model FOLDER [--max-tokens N]

For pplx-decider-v1.2 (27B) and pplx-decider-flash-v1.2, which share the
joint request format: one sequence per request, a readout at each
question's marker. The requests are tests/parity/requests.json,
tests/parity/marker/requests.json and tests/parity/pplx/v1.2-requests.json
(joint cases: several questions, escaped tokens, request chunks). Needs
torch, torchvision, transformers and Pillow (not dependencies of the
package); both releases' small files are fetched into the Hugging Face
cache at the pinned revisions (no weights). The release's code is imported
from the flash release's `source/src` (both ship the same code).

Each request is sent whole, as the release's `Decider` takes it, parsed by
mlx-decision first (a request the wire format refuses is recorded as
refused). The release's `Encoder.plan` cuts it into chunks at the input
limit: 32,768 tokens (mlx-decision's default), or the case's
`max_input_tokens`. Two tiny random models (tests/tiny_models.py,
`write_pplx(joint=...)`) in the shapes of the two releases, with their
tokenizer, processor and configs, are run by the release's
`reference_logits` on the CPU in float32 (as its code does off CUDA), one
chunk at a time.

Writes tests/parity/pplx/reference-joint.json with, per request: the
chunks (question ids, token count, SHA-256 of the token ids, the marker
positions, image grids, and the token ids themselves for chunks of at
most 2,048 tokens), or the release's refusal; and, when every chunk is
that short, each shape's raw readout logits and probabilities per
question. Existing entries are kept, so a run can be continued with
--only.

With --model (a release folder of pplx-decider-flash-v1.2 with its
weights), the release's `Decider` runs the real weights instead, on MPS,
loaded in bf16 with its readout in bf16 (as its code runs on CUDA; off
CUDA it would load float32), on every request whose chunks at 32,768
tokens all have at most --max-tokens tokens (default 4,096: longer ones
need more attention memory than a 16 GB Mac has). Writes
tests/parity/pplx/reference-flash-bf16.json after every request, so an
interrupted run continues where it stopped.
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
RELEASES = {
    "27b": ("perplexity-ai/pplx-decider-v1.2", "cce0f3f80033e9d89d80831d5281fd72124aacce"),
    "flash": ("perplexity-ai/pplx-decider-flash-v1.2", "9ff7cf9838e59df461c276b7699d56b244d2af58"),
}
MAX_INPUT_TOKENS = 32768
TINY_TOKENS = 2048
SEED = 0


def parity_set() -> list[dict]:
    cases = json.loads((PARITY / "requests.json").read_text())
    cases += json.loads((PARITY / "marker" / "requests.json").read_text())
    return cases + json.loads((PARITY / "pplx" / "v1.2-requests.json").read_text())


def dump(data: dict) -> str:
    """JSON with one level per line, but lists of numbers on one line."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[[-0-9.e,\s]+\]", lambda m: re.sub(r"\s+", "", m[0]).replace(",", ", "), text)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def fetch(repo_id: str, revision: str) -> Path:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            ignore_patterns=["model-*.safetensors"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


def chunk_record(chunk) -> dict:
    ids = chunk.input_ids.tolist()
    record = {
        "questions": list(chunk.keys),
        "tokens": len(ids),
        "ids_sha256": ids_digest(ids),
        "markers": chunk.markers.tolist(),
    }
    if chunk.image_grid_thw is not None:
        record["image_grids"] = chunk.image_grid_thw.tolist()
    if len(ids) <= TINY_TOKENS:
        record["ids"] = ids
    return record


def real_reference(args) -> None:
    """The release's code on the real flash weights (bf16, MPS)."""
    import torch
    import transformers

    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request

    torch.set_grad_enabled(False)
    release = args.model.resolve()
    sys.path.insert(0, str(release / "source" / "src"))
    from pplx_decider import LengthError, RequestError
    from pplx_decider import model as code

    backbone_class = code.Qwen3_5Model

    class Bf16:
        @staticmethod
        def from_pretrained(path, dtype=None, **options):
            # Straight onto the GPU: loading on the CPU first would hold two
            # copies of the weights for a moment (18 GB for flash).
            return backbone_class.from_pretrained(
                path, dtype=torch.bfloat16, device_map="mps", **options
            )

    code.Qwen3_5Model = Bf16
    decider = code.Decider(release, device="mps", max_length=None)
    decider.readout.to(torch.bfloat16)
    out = args.out or PARITY / "pplx" / "reference-flash-bf16.json"
    results = json.loads(out.read_text())["results"] if out.exists() else {}
    meta = {
        "release": {"repo_id": RELEASES["flash"][0], "revision": RELEASES["flash"][1]},
        "temperature": decider.temperature,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_tokens": args.max_tokens,
        "device": "mps",
        "dtype": str(next(decider.backbone.parameters()).dtype),
        "readout_dtype": str(decider.readout.weight.dtype),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    for case in cases:
        if case["id"] in results:
            continue
        try:
            parsed = parse_request({"state": case["state"], "questions": case["questions"]})
        except DecisionError:
            continue
        request = {
            "state": case["state"],
            "questions": {qid: q.model_dump() for qid, q in parsed.questions.items()},
            "images": [open_image(spec) for spec in case.get("images", [])],
        }
        try:
            chunks = decider.encoder.plan(request, case.get("max_input_tokens", MAX_INPUT_TOKENS))
        except (LengthError, RequestError):
            continue
        if any(chunk.length > args.max_tokens for chunk in chunks):
            continue
        start = time.perf_counter()
        answers = {}
        for chunk in chunks:
            rows = decider.reference_logits([chunk])[0]
            for key, logits in zip(chunk.keys, rows, strict=True):
                values = torch.tensor(logits, dtype=torch.float32)
                probabilities = (values / decider.temperature).softmax(-1)
                answers[key] = {"logits": logits, "probabilities": probabilities.tolist()}
        torch.mps.empty_cache()
        results[case["id"]] = {
            "chunks": [
                {
                    "questions": list(c.keys),
                    "tokens": c.length,
                    "ids_sha256": ids_digest(c.input_ids.tolist()),
                }
                for c in chunks
            ],
            "answers": answers,
        }
        out.write_text(dump({"meta": meta, "results": results}) + "\n")
        print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
    print(f"{len(results)} results in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, help="flash release folder: run its real weights")
    parser.add_argument("--max-tokens", type=int, default=4096, help="with --model: longest part")
    args = parser.parse_args()
    if args.model:
        real_reference(args)
        return
    args.out = args.out or PARITY / "pplx" / "reference-joint.json"
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request
    from tiny_models import write_pplx

    torch.set_grad_enabled(False)
    releases = {shape: fetch(*RELEASES[shape]) for shape in RELEASES}
    sys.path.insert(0, str(releases["flash"] / "source" / "src"))
    from pplx_decider import Decider, LengthError, RequestError

    existing = json.loads(args.out.read_text())["results"] if args.out.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = dict(existing)
    with tempfile.TemporaryDirectory() as folder:
        deciders = {}
        for shape, release in releases.items():
            tiny = write_pplx(Path(folder) / shape, seed=SEED, files_from=release, joint=shape)
            deciders[shape] = Decider(tiny, device="cpu", max_length=None)
        encoder = deciders["flash"].encoder
        escapes = encoder._escapes
        assert escapes == deciders["27b"].encoder._escapes
        for case in cases:
            start = time.perf_counter()
            limit = case.get("max_input_tokens", MAX_INPUT_TOKENS)
            try:
                parsed = parse_request({"state": case["state"], "questions": case["questions"]})
            except DecisionError as error:
                results[case["id"]] = {"refused": str(error)}
                print(f"{case['id']}: refused by the wire format", flush=True)
                continue
            request = {
                "state": case["state"],
                "questions": {qid: q.model_dump() for qid, q in parsed.questions.items()},
                "images": [open_image(spec) for spec in case.get("images", [])],
            }
            # Both releases' chat templates differ only with thinking on.
            assert encoder.text(request) == deciders["27b"].encoder.text(request)
            try:
                chunks = encoder.plan(request, limit)
            except (LengthError, RequestError) as error:
                results[case["id"]] = {"limit": limit, "refused": str(error)}
                print(f"{case['id']}: refused by the release", flush=True)
                continue
            result = {"limit": limit, "chunks": [chunk_record(c) for c in chunks]}
            if all(c.length <= TINY_TOKENS for c in chunks):
                result["answers"] = {}
                for shape, decider in deciders.items():
                    answers = {}
                    for chunk in chunks:
                        rows = decider.reference_logits([chunk])[0]
                        for key, logits in zip(chunk.keys, rows, strict=True):
                            values = torch.tensor(logits, dtype=torch.float32)
                            probabilities = (values / decider.temperature).softmax(-1)
                            answers[key] = {
                                "logits": logits,
                                "probabilities": probabilities.tolist(),
                            }
                    result["answers"][shape] = answers
            results[case["id"]] = result
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
        decider = deciders["flash"]
        meta = {
            "releases": {
                shape: {"repo_id": repo, "revision": revision}
                for shape, (repo, revision) in RELEASES.items()
            },
            "seed": SEED,
            "temperatures": {shape: d.temperature for shape, d in deciders.items()},
            "max_input_tokens": MAX_INPUT_TOKENS,
            "tiny_tokens": TINY_TOKENS,
            "escapes": sorted(escapes),
            "marker_id": encoder.marker_id,
            "max_pixels": decider.max_pixels,
            "device": "cpu",
            "dtype": str(next(decider.backbone.parameters()).dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump({"meta": meta, "results": results}) + "\n")
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
