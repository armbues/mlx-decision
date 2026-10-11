"""d1-3B as the release runs it: the fixtures in tests/parity/d1/.

    python scripts/make_d1_reference.py                     # reference.json, reference-tiny.json
    python scripts/make_d1_reference.py --model PATH        # reference-bf16.json

Fetches the release's small files at a pinned revision into the Hugging Face
cache (no weights) and runs its own code on the text requests of the parity
sets (``requests.json``, ``marker/requests.json``, ``d1/requests.json``).

``reference.json``: per question what the release would feed the model: the
prompt text, the token ids (for a request with several questions, the shared
prefix tokenized once, then the question's own text) and the token ids each
option is scored by. Requests the release refuses get its error. Long texts
and id lists are stored as a sha256 and a length (a few seconds).

``reference-tiny.json``: the answers of the tiny random model of
``tests/tiny_models.write_d1`` with the release's tokenizer, run by the
release's own model code (its ``D1Model``, transformers, CPU, float32),
each request as the release serves it (about a minute).

``--model``: the release's weights in bfloat16 on MPS, every question run
alone (one plain pass): ``reference-bf16.json`` (a few minutes).

Prompts over the model's context (32,768 tokens) are left out of the answer
fixtures: the release reads them whole, mlx-decision cuts the state. The
release's code is only run here, for the references; mlx-decision's own
prompt and readout (``models/d1``) were written without it.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
REPO_ID = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
MAX_TEXT = 4000  # characters of a prompt stored as text
MAX_IDS = 1024  # token ids stored as a list
CONTEXT = 32768  # the model's; longer prompts are cut by mlx-decision, not by the release
# The release's code, copied next to the tiny model so transformers can load it.
CODE = ("modeling_d1.py", "lfm2_vl.py", "hybrid.py", "runner.py", "prompt.py", "api.py")


def text_requests() -> list[dict]:
    cases = []
    for path in ("requests.json", "marker/requests.json", "d1/requests.json"):
        cases += [c for c in json.loads((PARITY / path).read_text()) if not c.get("images")]
    return cases


def dump(data: dict) -> str:
    """JSON with one level per line, but lists of numbers on one line."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[[-0-9.e,\s]+\]", lambda m: re.sub(r"\s+", "", m[0]).replace(",", ", "), text)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def text_entry(text: str):
    if len(text) <= MAX_TEXT:
        return text
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "length": len(text)}


def ids_entry(ids: list[int]):
    return ids if len(ids) <= MAX_IDS else {"sha256": ids_digest(ids), "length": len(ids)}


def fetch_release() -> Path:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            REPO_ID,
            revision=REVISION,
            ignore_patterns=["model.safetensors"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


def release_prompt(release: Path):
    """The release's prompt module, imported from its folder."""
    spec = importlib.util.spec_from_file_location("d1_release_prompt", release / "prompt.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tokens(release: Path) -> None:
    """``reference.json``: the release's prompts, token ids and scored tokens."""
    import transformers
    from transformers import AutoTokenizer

    p = release_prompt(release)
    tok = AutoTokenizer.from_pretrained(release)
    bos = tok.bos_token
    lead = p.default_lead("lfm2_vl")

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    results = {}
    for case in text_requests():
        try:
            questions = {qid: p.as_question(q) for qid, q in case["questions"].items()}
            prefix = p.prefix_text(tok, case["state"], bos)
            suffixes = {qid: p.suffix_text(tok, q, lead) for qid, q in questions.items()}
            option_ids = {qid: p.readout_ids(tok, q) for qid, q in questions.items()}
        except Exception as error:
            results[case["id"]] = {"error": f"{type(error).__name__}: {error}"}
            continue
        shared = encode(prefix) if len(questions) > 1 else None
        entry = {"trunk_length": None if shared is None else len(shared), "questions": {}}
        for qid in questions:
            text = prefix + suffixes[qid]
            ids = encode(text) if shared is None else shared + encode(suffixes[qid])
            entry["questions"][qid] = {
                "prompt": text_entry(text),
                "ids": ids_entry(ids),
                "option_ids": option_ids[qid],
            }
        results[case["id"]] = entry
    meta = {"repo_id": REPO_ID, "revision": REVISION, "transformers": transformers.__version__}
    out = PARITY / "d1" / "reference.json"
    out.write_text(dump({"meta": meta, "results": results}) + "\n")
    answered = sum(len(e["questions"]) for e in results.values() if "questions" in e)
    refused = sum("error" in e for e in results.values())
    print(f"{out}: {len(results)} requests, {answered} questions, {refused} refused")


def answers(folder: Path, dtype: str, device: str, alone: bool) -> dict:
    """The release's answers for the model in ``folder``: per request, per question,
    the option probabilities (as the release orders them) and the tokens read.
    ``alone``: every question in a plain pass of its own; otherwise each request
    as the release serves it (its questions over one shared state)."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(folder, trust_remote_code=True, dtype=getattr(torch, dtype))
    engine = model.to(device).engine
    p = sys.modules[type(engine).__module__.rsplit(".", 1)[0] + ".prompt"]
    reference = json.loads((PARITY / "d1" / "reference.json").read_text())["results"]
    results = {}
    for case in text_requests():
        expected = reference[case["id"]]
        if "error" in expected:
            continue
        lengths = [
            q["ids"]["length"] if isinstance(q["ids"], dict) else len(q["ids"])
            for q in expected["questions"].values()
        ]
        if max(lengths) > CONTEXT:
            results[case["id"]] = {"skipped": f"{max(lengths)} tokens, over {CONTEXT}"}
            continue
        questions = {qid: p.as_question(q) for qid, q in case["questions"].items()}
        start = time.perf_counter()
        if alone:
            done = engine.run([(case["state"], [q], ()) for q in questions.values()])
            probabilities = [d[0][0] for d in done]
        else:
            probabilities = engine.run([(case["state"], list(questions.values()), ())])[0][0]
        results[case["id"]] = {
            qid: [float(v) for v in values]
            for qid, values in zip(questions, probabilities, strict=True)
        }
        print(f"{case['id']}: {time.perf_counter() - start:.2f}s", file=sys.stderr, flush=True)
    return results


def tiny(release: Path) -> None:
    """``reference-tiny.json``: the tiny model with the release's tokenizer and code."""
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    from tiny_models import write_d1

    with tempfile.TemporaryDirectory() as temp:
        folder = write_d1(Path(temp) / "tiny-d1", files_from=release)
        for name in CODE:
            shutil.copy(release / name, folder / name)
        results = answers(folder, "float32", "cpu", alone=False)
    meta = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "model": "tests/tiny_models.write_d1(files_from=release)",
        "device": "cpu",
        "dtype": "float32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    out = PARITY / "d1" / "reference-tiny.json"
    out.write_text(dump({"meta": meta, "results": results}) + "\n")
    answered = sum("skipped" not in r for r in results.values())
    print(f"{out}: {answered} requests answered, {len(results) - answered} skipped")


def real(model: Path, device: str) -> None:
    """``reference-bf16.json``: the release's weights, every question alone."""
    import torch
    import transformers

    results = answers(model, "bfloat16", device, alone=True)
    meta = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "model": "release",
        "device": device,
        "dtype": "bfloat16",
        "questions": "alone",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    out = PARITY / "d1" / "reference-bf16.json"
    out.write_text(dump({"meta": meta, "results": results}) + "\n")
    print(f"{out}: {len(results)} requests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, help="run the real model in this release folder")
    parser.add_argument("--device", default="mps", help="device for --model (default: mps)")
    args = parser.parse_args()
    if args.model:
        real(args.model, args.device)
        return
    release = fetch_release()
    tokens(release)
    tiny(release)


if __name__ == "__main__":
    main()
