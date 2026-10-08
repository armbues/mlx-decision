"""Run pplx-decider's own code on the parity set and store token ids and tiny-model answers.

usage: python scripts/make_pplx_reference.py [--release PATH] [--only ID,ID]

The parity set is tests/parity/requests.json (text and image requests) plus
tests/parity/marker/requests.json. Needs torch, torchvision, transformers
and Pillow (not dependencies of the package) and the release's small files:
without --release they are fetched into the Hugging Face cache at the
pinned revision (everything but the weight shards, about 25 MB). The
release's code is imported from its `source/src`.

A tiny random model in the release layout (tests/tiny_models.py,
`write_pplx`) is written with the release's tokenizer, processor and
decision config and loaded with the release's `DecisionModel` on the CPU in
float32, as its code does there; the attention mode is the saved one
(non-causal full attention). Each question is one row, as the release
answers them. Questions are taken as mlx-decision parses them; a request
the wire format refuses is recorded as refused.

Writes tests/parity/pplx/reference.json with, per question: the token
count and a SHA-256 of the token ids (the real tokenizer), whether the
release refuses it (over 8,192 tokens, or not 1 to 255 options), the image
grids, and for questions of at most 1,024 tokens also the token ids, the
tiny model's raw logits over the options and its probabilities (also
for the 255-option request). Existing entries are kept, so a run can be
continued with --only.
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
REPO_ID = "perplexity-ai/pplx-decider-v1.1-27b"
REVISION = "3b45dead91dfa6d95aad6b95764a606fab2bf7a6"
MAX_TOKENS = 8192
TINY_TOKENS = 1024
# Longer, but the only question with all 255 options.
TINY_ALSO = {"choice_255_options"}
SEED = 0


def parity_set() -> list[dict]:
    main = json.loads((PARITY / "requests.json").read_text())
    extra = json.loads((PARITY / "marker" / "requests.json").read_text())
    return main + extra


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
            ignore_patterns=["model-*.safetensors", "source/uv.lock"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


class Reference:
    def __init__(self, release: Path, tiny: Path):
        sys.path.insert(0, str(release / "source" / "src"))
        from autojev import model

        self.model = model.DecisionModel(tiny, device="cpu")
        self.code = model

    def question(self, state, question: dict, images: list, tiny: bool) -> dict:
        row = {"state": state, "question": question, "images": images}
        count = len(self.code.options(question)[0])
        try:
            self.code.decision_messages(row, self.model.codes)
        except ValueError as error:
            return {"options": count, "refused": str(error)}
        batch = self.model.prepare([row], max_length=sys.maxsize)
        ids = batch.inputs["input_ids"][0].tolist()
        result = {"options": count, "tokens": len(ids), "ids_sha256": ids_digest(ids)}
        if "image_grid_thw" in batch.inputs:
            result["image_grids"] = batch.inputs["image_grid_thw"].tolist()
        if len(ids) > MAX_TOKENS:
            # What the release's prepare() raises at its default limit.
            result["refused"] = f"over the {MAX_TOKENS}-token limit"
        elif tiny or len(ids) <= TINY_TOKENS:
            logits = self.model(batch)[0, :count]
            probabilities = (logits / self.model.temperature).softmax(-1)
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
    parser.add_argument("--out", type=Path, default=PARITY / "pplx" / "reference.json")
    args = parser.parse_args()
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request
    from tiny_models import write_pplx

    torch.set_grad_enabled(False)
    release = (args.release or fetch_release()).resolve()
    existing = json.loads(args.out.read_text())["results"] if args.out.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = dict(existing)
    with tempfile.TemporaryDirectory() as folder:
        tiny = write_pplx(Path(folder), seed=SEED, files_from=release)
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
                        case["state"], question.model_dump(), images, case["id"] in TINY_ALSO
                    )
                    for qid, question in request.questions.items()
                }
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
        meta = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "seed": SEED,
            "temperature": reference.model.temperature,
            "attention_mode": reference.model.attention_mode,
            "max_tokens": MAX_TOKENS,
            "tiny_tokens": TINY_TOKENS,
            "device": "cpu",
            "dtype": str(next(reference.model.parameters()).dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dump({"meta": meta, "results": results}) + "\n")
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
