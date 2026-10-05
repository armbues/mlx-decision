"""Run Cloudflare's PyTorch reference on the parity set and store what it computes.

usage: python scripts/make_parity_reference.py --model PATH [--only ID,ID] [--device mps]

Needs torch, transformers and safetensors (not dependencies of the package)
and imports ``joint_schema_model.py`` from the model folder. Writes, per
request: token ids, question and option spans, whether the state was
truncated, and the probabilities per question, computed in bf16 as shipped.
Images go through the release's processor in its Pillow mode
(``use_fast=False``), which the package's preprocessing reproduces exactly;
they are opened by ``tests/parity/parity_images.py``.
Existing entries in the output are kept, so long requests can be run one at
a time with ``--only``.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"


class PeakMemory:
    """Samples MPS driver memory in the background; ``peak_gb`` after the block."""

    def __init__(self, torch):
        self.torch = torch
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.torch.mps.driver_allocated_memory())
            time.sleep(0.02)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak_gb(self) -> float:
        return round(self.peak / 2**30, 1)


def load_reference(model_path: Path, device: str):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    sys.path.insert(0, str(model_path))
    from joint_schema_model import ClefModel, JointSchemaHead

    # The release's load_release_model needs accelerate; these are its steps.
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16)
    backbone = backbone.to(device)
    backbone.config.use_cache = False
    head = JointSchemaHead(**json.loads((model_path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(model_path / "joint_head.safetensors"), strict=True)
    model = ClefModel(backbone, head.to(device=device, dtype=torch.bfloat16)).eval()
    return model, AutoProcessor.from_pretrained(model_path, use_fast=False)


def write(path: Path, meta: dict, results: dict) -> None:
    # One request per line keeps diffs readable.
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in results.items()]
    body = ",\n".join(lines)
    path.write_text(f'{{"meta": {json.dumps(meta)},\n "results": {{\n{body}\n}}}}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, default=PARITY / "requests.json")
    parser.add_argument("--out", type=Path, default=PARITY / "reference.json")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.9,
        help="cap on MPS memory, as a fraction of the recommended maximum",
    )
    args = parser.parse_args()

    import torch
    import transformers

    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    model, processor = load_reference(args.model, args.device)
    tokenizer = processor.tokenizer
    sys.path.insert(0, str(args.model))
    sys.path.insert(0, str(PARITY))
    from joint_schema_model import collate_records, encode_record
    from parity_images import open_image

    cases = json.loads(args.requests.read_text())
    if args.only:
        wanted = set(args.only.split(","))
        cases = [case for case in cases if case["id"] in wanted]
        if missing := wanted - {case["id"] for case in cases}:
            raise SystemExit(f"unknown request ids: {', '.join(sorted(missing))}")

    results = {}
    if args.out.exists():
        results = json.loads(args.out.read_text())["results"]
    if not args.only:
        # A full run drops results of requests that no longer exist.
        results = {k: v for k, v in results.items() if k in {case["id"] for case in cases}}
    meta = {
        "device": args.device,
        "dtype": "bfloat16",
        "image_processor": "pil",
        "max_length": 16384,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "joint_schema_model_sha256": hashlib.sha256(
            (args.model / "joint_schema_model.py").read_bytes()
        ).hexdigest(),
        "joint_head_sha256": hashlib.sha256(
            (args.model / "joint_head.safetensors").read_bytes()
        ).hexdigest(),
    }

    for case in cases:
        record = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
        if case.get("images"):
            record["images"] = [open_image(spec) for spec in case["images"]]
        encoded = encode_record(tokenizer, record, processor=processor)
        untruncated = encode_record(tokenizer, record, max_length=10**9, processor=processor)
        batch = collate_records([encoded], tokenizer.pad_token_id, torch.device(args.device))
        start = time.time()
        with PeakMemory(torch) as memory, torch.inference_mode():
            logits = [values.float().cpu() for values in model(batch)[0]]
        seconds = time.time() - start
        results[case["id"]] = {
            "input_ids": list(encoded.input_ids),
            "truncated": len(untruncated.input_ids) > len(encoded.input_ids),
            "questions": [
                {
                    "id": question.question_id,
                    "type": question.question_type,
                    "span": list(question.question_span),
                    "option_spans": [list(span) for span in question.option_spans],
                    "option_ids": list(question.option_ids),
                    "probabilities": values.softmax(-1).tolist(),
                }
                for question, values in zip(encoded.questions, logits, strict=True)
            ],
            "peak_gb": memory.peak_gb,
        }
        print(
            f"{case['id']:30s} tokens={len(encoded.input_ids):6d} "
            f"{seconds:6.2f}s peak={memory.peak_gb} GB",
            flush=True,
        )
        if args.device == "mps":
            torch.mps.empty_cache()
        # Save as we go: long requests take a while.
        write(args.out, meta, dict(sorted(results.items())))


if __name__ == "__main__":
    main()
