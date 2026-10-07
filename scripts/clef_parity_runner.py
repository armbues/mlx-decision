"""Clef parity on another Mac: Cloudflare's reference and mlx-decision, both in bf16.

This file is copied into a bundle by ``scripts/clef_parity_bundle.py`` and runs
there on its own, next to ``requests.json`` and ``images/``; it imports nothing
from the mlx-decision repository.

usage: python run_parity.py --model Cloudflare/clef --out clef-parity.json

Runs the reference (torch, MPS) and then mlx-decision, each in its own
process so their memory peaks do not add up, and writes one JSON file with
both results and the machine they ran on. Each stage saves after every
request and skips requests already in its file, so an interrupted run can be
started again with the same command.

Needs: torch, torchvision, transformers, safetensors, Pillow, huggingface_hub
and mlx-decision with the images extra (see README.txt).
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
FORMAT = 1


def open_image(spec):
    """As ``tests/parity/parity_images.py``: a file name, or a file resized (bicubic)."""
    from PIL import Image

    if isinstance(spec, str):
        spec = {"file": spec}
    image = Image.open(IMAGES / spec["file"]).convert("RGB")
    if "size" in spec:
        image = image.resize(tuple(spec["size"]), Image.Resampling.BICUBIC)
    return image


def model_folder(model: str) -> Path:
    """A local folder, or the whole repository (with the reference code) from the Hub."""
    if Path(model).expanduser().is_dir():
        return Path(model).expanduser()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model))


def load_cases(only: str | None) -> list[dict]:
    cases = json.loads((HERE / "requests.json").read_text())
    if only:
        wanted = set(only.split(","))
        cases = [case for case in cases if case["id"] in wanted]
        if missing := wanted - {case["id"] for case in cases}:
            raise SystemExit(f"unknown request ids: {', '.join(sorted(missing))}")
    return cases


def read_stage(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"meta": {}, "results": {}}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data) + "\n")
    temporary.replace(path)


def sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() or None


def machine() -> dict:
    return {
        "model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl("hw.memsize") or 0) or None,
        "macos": platform.mac_ver()[0] or None,
        "python": sys.version.split()[0],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def run_reference(args) -> None:
    """Cloudflare's reference in bf16 on MPS, as ``scripts/make_parity_reference.py``."""
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    folder = model_folder(args.model)
    if not (folder / "joint_schema_model.py").exists():
        raise SystemExit(
            f"{folder} has no joint_schema_model.py (Cloudflare's reference code); "
            "give the repo id instead, or add the file from the repository"
        )
    sys.path.insert(0, str(folder))
    from joint_schema_model import ClefModel, JointSchemaHead, collate_records, encode_record

    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    start = time.time()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(folder, dtype=torch.bfloat16)
    backbone = backbone.to("mps")
    backbone.config.use_cache = False
    head = JointSchemaHead(**json.loads((folder / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(folder / "joint_head.safetensors"), strict=True)
    model = ClefModel(backbone, head.to(device="mps", dtype=torch.bfloat16)).eval()
    processor = AutoProcessor.from_pretrained(folder, use_fast=False)
    tokenizer = processor.tokenizer
    load_s = round(time.time() - start, 1)

    stage = read_stage(args.out)
    stage["meta"] = {
        "device": "mps",
        "dtype": "bfloat16",
        "image_processor": "pil",
        "max_length": 16384,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "joint_schema_model_sha256": sha256(folder / "joint_schema_model.py"),
        "joint_head_sha256": sha256(folder / "joint_head.safetensors"),
        "load_s": load_s,
    }
    for case in load_cases(args.only):
        if case["id"] in stage["results"]:
            continue
        record = {"id": case["id"], "state": case["state"], "questions": case["questions"]}
        if case.get("images"):
            record["images"] = [open_image(spec) for spec in case["images"]]
        encoded = encode_record(tokenizer, record, processor=processor)
        untruncated = encode_record(tokenizer, record, max_length=10**9, processor=processor)
        batch = collate_records([encoded], tokenizer.pad_token_id, torch.device("mps"))
        start = time.time()
        with PeakMemory(torch) as memory, torch.inference_mode():
            logits = [values.float().cpu() for values in model(batch)[0]]
        seconds = round(time.time() - start, 2)
        stage["results"][case["id"]] = {
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
            "seconds": seconds,
            "peak_gb": memory.peak_gb,
        }
        print(
            f"reference {case['id']:30s} tokens={len(encoded.input_ids):6d} "
            f"{seconds:7.2f}s peak={memory.peak_gb} GB",
            flush=True,
        )
        torch.mps.empty_cache()
        write_json(args.out, stage)
    write_json(args.out, stage)


def run_mlx(args) -> None:
    """mlx-decision in bf16, without the image cap (as the reference)."""
    import mlx.core as mx

    import mlx_decision

    folder = model_folder(args.model)
    start = time.time()
    model = mlx_decision.load(folder, max_image_pixels=None)
    load_s = round(time.time() - start, 1)
    stage = read_stage(args.out)
    stage["meta"] = {
        "mlx_decision": mlx_decision.__version__,
        "mlx": getattr(mx, "__version__", None),
        "precision": model.info().get("precision"),
        "load_s": load_s,
    }
    for case in load_cases(args.only):
        if case["id"] in stage["results"]:
            continue
        body = {"state": case["state"], "questions": case["questions"]}
        if case.get("images"):
            body["images"] = [open_image(spec) for spec in case["images"]]
        mx.reset_peak_memory()
        start = time.time()
        # The backend's raw probabilities per option id, comparable with the
        # reference's; the answers built from them are the package's own.
        output = model.backend.score(model.check(body))
        seconds = round(time.time() - start, 2)
        peak = round(mx.get_peak_memory() / 2**30, 1)
        stage["results"][case["id"]] = {
            "input_tokens": output.input_tokens,
            "truncated": output.truncated,
            "probabilities": output.probabilities,
            "seconds": seconds,
            "peak_gb": peak,
        }
        print(
            f"mlx       {case['id']:30s} tokens={output.input_tokens:6d} "
            f"{seconds:7.2f}s peak={peak} GB",
            flush=True,
        )
        write_json(args.out, stage)
    write_json(args.out, stage)


def run_all(args) -> None:
    out = Path(args.out)
    stages = {}
    for stage in ("reference", "mlx"):
        part = out.with_name(f"{out.stem}.{stage}.json")
        if stage in args.skip:
            # Not run now, but results from an earlier run still go in.
            if part.exists():
                stages[stage] = read_stage(part)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), stage]
        command += ["--model", args.model, "--out", str(part)]
        if args.only:
            command += ["--only", args.only]
        if stage == "reference":
            command += ["--memory-fraction", str(args.memory_fraction)]
        print(f"== {stage}", flush=True)
        if subprocess.run(command).returncode != 0:
            raise SystemExit(f"the {stage} stage failed; run the same command again to resume")
        stages[stage] = read_stage(part)
    result = {
        "format": FORMAT,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "machine": machine(),
        "requests_sha256": sha256(HERE / "requests.json"),
        **stages,
    }
    write_json(out, result)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", nargs="?", default="all", choices=["all", "reference", "mlx"])
    parser.add_argument("--model", required=True, help="folder or Hugging Face repo id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument(
        "--skip", action="append", default=[], choices=["reference", "mlx"], help="(all only)"
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.9,
        help="cap on MPS memory for the reference, as a fraction of the recommended maximum",
    )
    args = parser.parse_args()
    {"all": run_all, "reference": run_reference, "mlx": run_mlx}[args.stage](args)


if __name__ == "__main__":
    main()
