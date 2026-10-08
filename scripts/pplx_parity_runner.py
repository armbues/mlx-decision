"""pplx-decider parity on another Mac: the release's own code and mlx-decision, both in bf16.

This file is copied into a bundle by ``scripts/pplx_parity_bundle.py`` and runs
there on its own, next to ``requests.json`` and ``images/``; it imports nothing
from the mlx-decision repository.

usage: python run_parity.py --model perplexity-ai/pplx-decider-v1.1-27b --revision SHA \
           --out pplx-parity.json

Runs the release's ``DecisionModel`` (from the repository's ``source/src``,
torch, MPS) and then mlx-decision, each in its own process so their memory
peaks do not add up, and writes one JSON file with both results and the
machine they ran on. Every question is its own pass on both sides, as the
release answers them. Each stage saves after every request and skips
requests already in its file, so an interrupted run can be started again
with the same command.

The release's code loads float32 everywhere but on CUDA; here it loads the
backbone in bf16 (as on CUDA) unless ``--reference-dtype float32``.

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
MAX_TOKENS = 8192  # the release's limit per question


def open_image(spec):
    """As ``tests/parity/parity_images.py``: a file name, or a file resized (bicubic)."""
    from PIL import Image

    if isinstance(spec, str):
        spec = {"file": spec}
    image = Image.open(IMAGES / spec["file"]).convert("RGB")
    if "size" in spec:
        image = image.resize(tuple(spec["size"]), Image.Resampling.BICUBIC)
    return image


def model_folder(model: str, revision: str | None) -> Path:
    """A local folder, or the whole repository (with the release's code) from the Hub."""
    if Path(model).expanduser().is_dir():
        return Path(model).expanduser()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision))


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
    path.parent.mkdir(parents=True, exist_ok=True)
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

    def __init__(self, torch, device: str):
        self.torch = torch
        self.active = device == "mps"
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        if self.active:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.torch.mps.driver_allocated_memory())
            time.sleep(0.02)

    def __exit__(self, *exc):
        if self.active:
            self._stop.set()
            self._thread.join()

    @property
    def peak_gb(self) -> float | None:
        return round(self.peak / 2**30, 1) if self.active else None


def parse(case: dict):
    """The request as mlx-decision parses it (defaults filled in), or the error message."""
    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request

    try:
        return parse_request({"state": case["state"], "questions": case["questions"]})
    except DecisionError as error:
        return str(error)


def run_reference(args) -> None:
    """The release's ``DecisionModel``, its backbone loaded in ``--reference-dtype``."""
    import torch
    import transformers

    folder = model_folder(args.model, args.revision)
    if not (folder / "source" / "src" / "autojev" / "model.py").exists():
        raise SystemExit(
            f"{folder} has no source/src/autojev (the release's code); "
            "give the repo id instead, or add the folder from the repository"
        )
    sys.path.insert(0, str(folder / "source" / "src"))
    from autojev import model as code

    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.reference_dtype)
    loader = code.Qwen3_5Model

    class Backbone:
        # The release picks float32 on any device but CUDA; the dtype is set here instead.
        @staticmethod
        def from_pretrained(*positional, **options):
            return loader.from_pretrained(*positional, **{**options, "dtype": dtype})

    code.Qwen3_5Model = Backbone
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    start = time.time()
    reference = code.DecisionModel(folder, device=args.device).to(dtype).eval()
    load_s = round(time.time() - start, 1)

    stage = read_stage(args.out)
    stage["meta"] = {
        "device": args.device,
        "dtype": str(next(reference.parameters()).dtype).removeprefix("torch."),
        "attention_mode": reference.attention_mode,
        "temperature": reference.temperature,
        "max_tokens": MAX_TOKENS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_py_sha256": sha256(folder / "source" / "src" / "autojev" / "model.py"),
        "readout_sha256": sha256(folder / "readout.safetensors"),
        "load_s": load_s,
    }
    for case in load_cases(args.only):
        if case["id"] in stage["results"]:
            continue
        request = parse(case)
        if isinstance(request, str):
            stage["results"][case["id"]] = {"refused": request}
            write_json(args.out, stage)
            continue
        images = [open_image(spec) for spec in case.get("images") or []]
        results, started = {}, time.time()
        for question_id, question in request.questions.items():
            row = {"state": case["state"], "question": question.model_dump(), "images": images}
            results[question_id] = reference_question(code, reference, row, args.device, torch)
        stage["results"][case["id"]] = results
        tokens = [q.get("tokens", 0) for q in results.values()]
        print(
            f"reference {case['id']:30s} questions={len(results):3d} tokens={max(tokens):6d} "
            f"{time.time() - started:7.2f}s",
            flush=True,
        )
        if args.device == "mps":
            torch.mps.empty_cache()
        write_json(args.out, stage)
    write_json(args.out, stage)


def reference_question(code, reference, row: dict, device: str, torch) -> dict:
    count = len(code.options(row["question"])[0])
    try:
        code.decision_messages(row, reference.codes)
    except ValueError as error:
        return {"options": count, "refused": str(error)}
    batch = reference.prepare([row], max_length=sys.maxsize)
    ids = batch.inputs["input_ids"][0].tolist()
    result = {
        "options": count,
        "tokens": len(ids),
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    if "image_grid_thw" in batch.inputs:
        result["image_grids"] = batch.inputs["image_grid_thw"].tolist()
    if len(ids) > MAX_TOKENS:
        # What the release's prepare() raises at its default limit.
        result["refused"] = f"over the {MAX_TOKENS}-token limit"
        return result
    start = time.time()
    with PeakMemory(torch, device) as memory:
        logits = reference(batch)[0, :count].float().cpu()
    result["logits"] = logits.tolist()
    result["probabilities"] = (logits / reference.temperature).softmax(-1).tolist()
    result["seconds"] = round(time.time() - start, 2)
    result["peak_gb"] = memory.peak_gb
    return result


def run_mlx(args) -> None:
    """mlx-decision on the release as stored (bf16), one question per request."""
    import mlx.core as mx

    import mlx_decision
    from mlx_decision.errors import DecisionError

    folder = model_folder(args.model, args.revision)
    start = time.time()
    model = mlx_decision.load(folder)
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
        request = parse(case)
        if isinstance(request, str):
            stage["results"][case["id"]] = {"refused": request}
            write_json(args.out, stage)
            continue
        images = [open_image(spec) for spec in case.get("images") or []]
        results, started = {}, time.time()
        for question_id, question in case["questions"].items():
            body = {"state": case["state"], "questions": {question_id: question}}
            if images:
                body["images"] = images
            mx.reset_peak_memory()
            begin = time.time()
            try:
                # The backend's raw probabilities per option id, comparable
                # with the reference's.
                output = model.backend.score(model.check(body))
            except DecisionError as error:
                results[question_id] = {"refused": str(error)}
                continue
            results[question_id] = {
                "tokens": output.input_tokens,
                "probabilities": output.probabilities[question_id],
                "seconds": round(time.time() - begin, 2),
                "peak_gb": round(mx.get_peak_memory() / 2**30, 1),
            }
        stage["results"][case["id"]] = results
        tokens = [q.get("tokens", 0) for q in results.values()]
        print(
            f"mlx       {case['id']:30s} questions={len(results):3d} tokens={max(tokens):6d} "
            f"{time.time() - started:7.2f}s",
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
        if args.revision:
            command += ["--revision", args.revision]
        if args.only:
            command += ["--only", args.only]
        if stage == "reference":
            command += ["--memory-fraction", str(args.memory_fraction)]
            command += ["--device", args.device, "--reference-dtype", args.reference_dtype]
        print(f"== {stage}", flush=True)
        if subprocess.run(command).returncode != 0:
            raise SystemExit(f"the {stage} stage failed; run the same command again to resume")
        stages[stage] = read_stage(part)
    result = {
        "format": FORMAT,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "revision": args.revision,
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
    parser.add_argument("--revision", help="Hub commit to download (default: the latest)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument(
        "--skip", action="append", default=[], choices=["reference", "mlx"], help="(all only)"
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"], help="for the reference")
    parser.add_argument(
        "--reference-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="the reference's weights (the release's own code uses float32 off CUDA)",
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
