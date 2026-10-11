"""d1-3B's prompt behaviour as the release produces it: tests/parity/d1/reference.json.

    python scripts/make_d1_reference.py

Fetches the release's small files at a pinned revision into the Hugging Face
cache (no weights) and runs its own prompt code on the text requests of the
parity sets (``requests.json``, ``marker/requests.json``, ``d1/requests.json``).
Per question it records what the release would feed the model: the prompt
text, the token ids (for a request with several questions, the shared prefix
tokenized once, then the question's own text) and the token ids each option
is scored by. Requests the release refuses get its error. Long texts and id
lists are stored as a sha256 and a length.

The release's code is only run here, for the reference; mlx-decision's own
prompt and readout (``models/d1``) were written without it.
"""

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
REPO_ID = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
MAX_TEXT = 4000  # characters of a prompt stored as text
MAX_IDS = 1024  # token ids stored as a list


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


def main() -> None:
    import transformers
    from transformers import AutoTokenizer

    release = fetch_release()
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


if __name__ == "__main__":
    main()
