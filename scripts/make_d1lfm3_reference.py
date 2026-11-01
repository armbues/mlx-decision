"""d1-2B as the release runs it: the fixtures in tests/parity/d1lfm3/.

    python scripts/make_d1lfm3_reference.py                  # reference.json (tiny model)
    python scripts/make_d1lfm3_reference.py --model PATH     # reference-bf16.json

Fetches the release's small files at a pinned revision into the Hugging Face
cache (no weights), writes the tiny random model of
``tests/tiny_models.write_d1lfm3`` with the release's tokenizer, config,
processor and calibration, puts the release's code beside it and runs that
code (its ``D1Model``, transformers, CPU, float32) on the parity sets: the
text and picture requests of ``requests.json``, the text requests of
``marker/requests.json`` and ``d1lfm3/requests.json`` (edge cases).

``reference.json``, per request: the release's error if it refuses it;
otherwise the prompt's prefix (the state, pictures' markup included) and
each question's suffix as text, each packed set (token ids, state length,
each question's token span and answer slot), each question's scored token
groups (one per option), each picture's size after the release's cap and
its patch grid with the mean of each patch, and the answers of the tiny
model as the release serves them (sets of 16). Long texts and id lists are
stored as a sha256 and a length; answers are left out above
``--max-tokens`` (16,384) per set. Also the largest difference between all
requests run in one batch and each alone (a few minutes).

``--model``: the release's weights in bfloat16 on MPS, the same requests
(answers only): ``reference-bf16.json``.

The release's code (LFM Open License v1.0) is only run here, for the
references; mlx-decision's own prompt and readout for this family were
written without it.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "parity"))

REPO = "LiquidAI/d1-2B"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"
PARITY = ROOT / "tests" / "parity"
OUT = PARITY / "d1lfm3"
# Shown whole up to these sizes, else as a digest.
TEXT_CHARS = 4000
ID_TOKENS = 4096


def fetch() -> Path:
    """The release's small files (code, configs, tokenizer) at the pinned revision."""
    from huggingface_hub import HfApi, hf_hub_download

    names = [
        name
        for name in HfApi().list_repo_files(REPO, revision=REVISION)
        if "/" not in name and not name.endswith(".safetensors") and not name.startswith(".")
    ]
    paths = [Path(hf_hub_download(REPO, name, revision=REVISION)) for name in names]
    return paths[0].parent


def digest(value) -> dict:
    data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
    size = {"chars": len(value)} if isinstance(value, str) else {"tokens": len(value)}
    return {"sha256": hashlib.sha256(data).hexdigest(), **size}


def shown(text: str):
    return text if len(text) <= TEXT_CHARS else digest(text)


def shown_ids(ids: list[int]):
    return ids if len(ids) <= ID_TOKENS else digest(ids)


def requests() -> list[dict]:
    text = json.loads((PARITY / "requests.json").read_text())
    marker = [
        r
        for r in json.loads((PARITY / "marker" / "requests.json").read_text())
        if not r.get("images")
    ]
    own = json.loads((OUT / "requests.json").read_text())
    return text + marker + own


def load(folder: Path, device: str, dtype):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(folder, trust_remote_code=True, dtype=dtype).to(device).eval()
    package = type(model).__module__.rsplit(".", 1)[0]
    return (
        model,
        model.engine,
        import_module(package + ".prompt"),
        import_module(package + ".media"),
    )


def spans(question: list) -> list[list[int]]:
    """Each question's [start, end) in a packed set, in order."""
    found: dict[int, list[int]] = {}
    for position, owner in enumerate(question):
        if owner is not None and owner >= 0:
            found.setdefault(owner, [position, position + 1])[1] = position + 1
    return [found[key] for key in sorted(found)]


def pictures(engine, media_module, media, seqs) -> list[dict]:
    """Each picture's size after the release's cap, its patch grid and patch means."""
    if not media:
        return []
    inputs = seqs[0][6]
    grids = inputs["image_grid_thw"].tolist()
    pixels = inputs["pixel_values"].float()
    out, at = [], 0
    for image, grid in zip(media, grids, strict=True):
        patches = grid[0] * grid[1] * grid[2]
        capped = media_module.cap_pixels(image, engine.max_pixels)
        means = pixels[at : at + patches].mean(-1)
        out.append(
            {
                "size": list(image.size),
                "capped": list(capped.size),
                "grid": grid,
                "patch_means": [round(float(m), 6) for m in means],
            }
        )
        at += patches
    return out


def record(engine, prompt, media_module, r, max_tokens: int, layouts: bool) -> dict:
    from parity_images import open_image

    media = [open_image(spec) for spec in r.get("images", [])]
    try:
        questions = [engine.ask(q) for q in r["questions"].values()]
        request = (r["state"], questions, media)
        seqs = engine._sequences([request])
    except ValueError as error:
        return {"refused": str(error)}
    entry = {}
    if layouts:
        prefix, _ = engine._prefix(r["state"], media_module.media_list(media))
        suffixes, _, _ = engine._suffixes(questions)
        entry.update(
            prefix=shown(prefix),
            suffixes=[shown(s) for s in suffixes],
            sets=[
                {
                    "first": first,
                    "ids": shown_ids(ids),
                    "state_length": state_len,
                    "spans": spans(question),
                    "ends": [e[0] for e in ends],
                }
                for _, first, ids, state_len, question, ends, _ in seqs
            ],
            groups=[prompt.readout_ids(engine.tokenizer, q) for q in questions],
            pictures=pictures(engine, media_module, media, seqs),
        )
    if max(len(seq[2]) for seq in seqs) <= max_tokens:
        ((probabilities, tokens),) = engine.run([request])
        entry["probabilities"] = dict(zip(r["questions"], probabilities, strict=True))
        entry["input_tokens"] = tokens
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="the release folder with its weights")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--only", help="comma-separated request ids (printed, not written)")
    args = parser.parse_args()

    import numpy as np
    import torch

    from tiny_models import write_d1lfm3

    source = fetch()
    every = requests()
    if args.only:
        keep = set(args.only.split(","))
        every = [r for r in every if r["id"] in keep]
    with tempfile.TemporaryDirectory() as tmp:
        if args.model:
            folder, device, dtype = args.model, "mps", torch.bfloat16
        else:
            folder, device, dtype = Path(tmp) / "tiny-d1lfm3", "cpu", torch.float32
            write_d1lfm3(folder, files_from=source)
            for path in source.iterdir():
                if path.suffix in (".py", ".jinja") and not (folder / path.name).exists():
                    shutil.copy(path, folder / path.name)
        model, engine, prompt, media_module = load(folder, device, dtype)
        settings = {
            "set_size": engine.set_size,
            "attention": engine.attention,
            "codes": engine.tokenizer.d1_codes,
            "temperature": engine.calibration.default,
            "lead": engine.lead,
            "max_pixels": engine.max_pixels,
        }
        result = {}
        started = time.perf_counter()
        for r in every:
            entry = record(engine, prompt, media_module, r, args.max_tokens, not args.model)
            result[r["id"]] = entry
            state = "refused" if "refused" in entry else f"{len(r['questions'])} q"
            answered = "" if "probabilities" in entry or "refused" in entry else " (not run)"
            print(f"{r['id']:36s} {state}{answered}", flush=True)
        # The same requests in one call: the release's batches against each alone.
        from parity_images import open_image

        answered = [r for r in every if "probabilities" in result[r["id"]]]
        batched = engine.run(
            [
                (
                    r["state"],
                    [engine.ask(q) for q in r["questions"].values()],
                    [open_image(spec) for spec in r.get("images", [])],
                )
                for r in answered
            ]
        )
        gap = 0.0
        for r, (probabilities, _) in zip(answered, batched, strict=True):
            alone = result[r["id"]]["probabilities"]
            for question_id, values in zip(r["questions"], probabilities, strict=True):
                gap = max(gap, float(np.abs(np.array(values) - np.array(alone[question_id])).max()))
        seconds = time.perf_counter() - started
        print(f"{len(answered)} answered, batched vs alone {gap:.2e}, {seconds:.0f} s")
    if args.only:
        print(json.dumps({key: result[key] for key in result}, indent=1)[:4000])
        return
    fixture = {
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "torch": torch.__version__,
            "device": device,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "settings": settings,
        "max_tokens": args.max_tokens,
        "batched_vs_alone": gap,
        "requests": result,
    }
    out = OUT / ("reference-bf16.json" if args.model else "reference.json")
    out.write_text(json.dumps(fixture, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
