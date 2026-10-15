"""Run a d3 release's own runtime on the parity set and store token ids and tiny-model answers.

usage: python scripts/make_d3_reference.py [--release PATH] [--only ID,ID]

The parity set is tests/parity/requests.json (text and image requests),
tests/parity/marker/requests.json and tests/parity/d3/requests.json (d3's
edge cases). Needs torch, torchvision, transformers and Pillow (not
dependencies of the package) and the release's small files: without
--release they are fetched into the Hugging Face cache at the pinned
revision (code, configs and tokenizer, about 25 MB; no weights). The
release's runtime (`d3_runtime.py`) is imported from that folder.

A tiny random model in the release layout (tests/tiny_models.py,
`write_d3`) is written with the release's tokenizer, processors and
decision config and loaded with the release's `D3` on the CPU; the
runtime loads its backbone in bfloat16, which is loaded in float32 here
instead (rounding the weights to bfloat16 and casting back would move the
answers by about 1e-3), so they compare at float precision. Each question is prepared alone,
as one row of the release's batch, through the runtime's own `prepare`
(its rendering, tokenizer and, for images, processor). Questions are
taken as mlx-decision parses them; a request the wire format refuses is
recorded as refused, and so is a question the release refuses.

Writes tests/parity/d3/reference.json with, per question: the token count
and a SHA-256 of the token ids, the image grids, and for questions of at
most 4,096 tokens (and the 255-option request) also the token ids, the
tiny model's raw logits over the options and its probabilities. Existing
entries are kept, so a run can be continued with --only.
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
REPO_ID = "vllm-sr/d3-flash"
REVISION = "8b35c853b90e777595ec36bacae820fe08ea077a"  # tag v3.1.3
RELEASE_FILES = ["*.json", "*.py", "*.jinja", "LICENSE", "NOTICE"]
TINY_TOKENS = 4096
# Longer, but the only question with all 255 options.
TINY_ALSO = {"choice_255_options"}
SEED = 0


def parity_set() -> list[dict]:
    cases = []
    for name in ("requests.json", "marker/requests.json", "d3/requests.json"):
        cases += json.loads((PARITY / name).read_text())
    return cases


def dump(data: dict) -> str:
    """JSON with one level per line, but lists of numbers on one line."""
    text = json.dumps(data, indent=1)
    return re.sub(r"\[[-0-9.e,\s]+\]", lambda m: re.sub(r"\s+", "", m[0]).replace(",", ", "), text)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def fetch_release() -> Path:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            REPO_ID,
            revision=REVISION,
            allow_patterns=RELEASE_FILES,
            token=os.environ.get("HF_TOKEN"),
        )
    )


class Reference:
    def __init__(self, release: Path, tiny: Path):
        sys.path.insert(0, str(release))
        import d3_runtime
        import torch

        stock = d3_runtime.backbone_class

        def float32_backbone(kind):
            class Backbone(stock(kind)):
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return super().from_pretrained(*args, **{**kwargs, "dtype": torch.float32})

            return Backbone

        d3_runtime.backbone_class = float32_backbone
        self.runtime = d3_runtime.D3(tiny, device="cpu")

    def question(self, state, qid: str, question: dict, images: list, tiny: bool) -> dict:
        runtime = self.runtime
        try:
            prepared = runtime.prepare(state, {qid: question}, images or None)
        except ValueError as error:
            return {"refused": str(error)}
        if qid in prepared.errors:
            return {"refused": prepared.errors[qid]["message"]}
        count = len(prepared.questions[qid].keys)
        result = {"options": count}
        if images:
            text = prepared.texts[qid]
            encoded = runtime.processor(text=[text], images=list(images), return_tensors="pt")
            ids = encoded["input_ids"][0].tolist()
            result["image_grids"] = encoded["image_grid_thw"].tolist()
        else:
            ids = prepared.sequences[qid]
        result |= {"tokens": len(ids), "ids_sha256": ids_digest(ids)}
        if tiny or len(ids) <= TINY_TOKENS:
            if images:
                logits = runtime.image_logits([text], images, [count], len(ids))[0, :count]
            else:
                logits = runtime.logits([ids], [count])[0, :count]
            probabilities = (logits / runtime.temperature).softmax(-1)
            result |= {
                "ids": ids,
                "logits": logits.tolist(),
                "probabilities": probabilities.tolist(),
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, help="release folder (default: fetch)")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--out", type=Path, default=PARITY / "d3" / "reference.json")
    args = parser.parse_args()
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request
    from tiny_models import write_d3

    torch.set_grad_enabled(False)
    release = (args.release or fetch_release()).resolve()
    existing = json.loads(args.out.read_text())["results"] if args.out.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = dict(existing)
    with tempfile.TemporaryDirectory() as folder:
        tiny = write_d3(Path(folder), seed=SEED, files_from=release)
        reference = Reference(release, tiny)
        for case in cases:
            start = time.perf_counter()
            try:
                request = parse_request({"state": case["state"], "questions": case["questions"]})
            except DecisionError as error:
                results[case["id"]] = {"refused": str(error)}
            else:
                images = [open_image(spec) for spec in case.get("images", [])]
                results[case["id"]] = {
                    qid: reference.question(
                        case["state"],
                        qid,
                        question.model_dump(),
                        images,
                        case["id"] in TINY_ALSO,
                    )
                    for qid, question in request.questions.items()
                }
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
        runtime = reference.runtime
        meta = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "seed": SEED,
            "temperature": runtime.temperature,
            "attention_mode": runtime.attention_mode,
            "tiny_tokens": TINY_TOKENS,
            "device": "cpu",
            "dtype": str(next(runtime.backbone.parameters()).dtype),
            "readout_dtype": runtime.readout_dtype,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump({"meta": meta, "results": results}) + "\n")
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
