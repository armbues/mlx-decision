"""Run H2O-Lightning's serving code on the parity set; store token ids and tiny-model answers.

usage: python scripts/make_h2o_reference.py [--release PATH] [--only ID,ID]

The parity set is tests/parity/requests.json (text and image requests),
tests/parity/marker/requests.json and tests/parity/h2o/requests.json
(with images inside a state, lifted out by the shim's own code).
Needs torch, torchvision, transformers and Pillow (not dependencies of the
package) and the release's small files: without --release they are fetched
into the Hugging Face cache at the pinned revision (everything but the
weights and the adapter, about 30 MB).

The release serves through vLLM behind a standard-library shim
(`h2o_lightning_shim.py`), imported here from the release folder: the shim
renders each prompt, cuts long states, reads the label tokens' ids and
builds the answers. vLLM's part is done with transformers, as vLLM does it:
the folder's tokenizer, `Qwen3_5ForConditionalGeneration` on the CPU in
float32 (the release computes the logits in float32), the log-softmax over
the whole vocabulary at the answer slot, and the folder's image processor
with the shim's `max_pixels`, each image pad expanded to the image's tokens.
The model is a tiny random one in the release's layout
(tests/tiny_models.py, `write_h2o`) with the release's files. Questions go
to the shim as they appear in the request; a request the wire format
refuses is recorded as refused.

Writes tests/parity/h2o/reference.json with, per question: the token
count and a SHA-256 of the token ids, whether the state was cut, whether
the release refuses it (over
vLLM's 40,960-token limit, more than 255 options or more than 4 images),
the image grids, and for questions of at most 4,096 tokens also the token
ids, the label token ids, the tiny model's log-probabilities of the labels
and the shim's answer with and without the yes/no floor. Existing entries
are kept, so a run can be continued with --only.
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
REPO_ID = "h2oai/h2o-lightning-4b"
REVISION = "acaf0d4ea251e54de928c75ef4352670d33192d3"  # tag v1.2.3
MAX_TOKENS = 40960  # vLLM's --max-model-len in the release's serve.sh
TINY_TOKENS = 4096
SEED = 0


def parity_set() -> list[dict]:
    cases = []
    for path in ("requests.json", "marker/requests.json", "h2o/requests.json"):
        cases += json.loads((PARITY / path).read_text())
    return cases


def dump(data: dict) -> str:
    """JSON with one level per line, but lists of numbers on one line."""
    text = json.dumps(data, indent=1, ensure_ascii=False)
    return re.sub(r"\[[-0-9.e,\s]+\]", lambda m: re.sub(r"\s+", "", m[0]).replace(",", ", "), text)


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def decode_data_url(url: str):
    """An image the shim lifted out of a state (a data URL), opened with Pillow."""
    import base64
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def fetch_release() -> Path:
    import os

    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            REPO_ID,
            revision=REVISION,
            ignore_patterns=["model.safetensors", "adapter/*", "assets/*"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


class Reference:
    def __init__(self, release: Path, tiny: Path):
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        sys.path.insert(0, str(release))
        import h2o_lightning_shim as shim

        self.torch = torch
        self.shim = shim
        self.config = json.loads((tiny / "serve_config.json").read_text())
        self.contract = shim.Contract(self.config, env={})
        self.tokenizer = AutoTokenizer.from_pretrained(tiny)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            tiny, dtype=torch.float32
        ).eval()
        # What vLLM's processor does with the shim's mm_processor_kwargs.
        self.processor = Qwen2VLImageProcessor.from_pretrained(
            tiny, max_pixels=self.contract.max_pixels
        )
        self.label_ids = self.read_label_ids()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def read_label_ids(self) -> dict[str, int]:
        """The shim's start-up check: after a prompt, `" " + label` adds one token."""
        contract = self.contract
        criteria = {f"o{i}": f"d{i}" for i in range(len(contract.labels))}
        probe = {"type": "choice", "instructions": "Which?", "criteria": criteria}
        state = self.shim.render_state({"x": 1}, contract.state_compact)
        text = contract.question_prompt(state, probe)[0]
        base = self.encode(text)
        ids = {}
        for label in contract.labels:
            full = self.encode(text + " " + label)
            assert len(full) == len(base) + 1 and full[: len(base)] == base, label
            ids[label] = full[-1]
        assert len(set(ids.values())) == len(ids)
        return ids

    def truncate(self, text: str) -> str:
        """The shim's `truncate_state`, with vLLM's /tokenize and /detokenize."""
        limit = self.contract.max_state_tokens
        if not limit or len(text.encode("utf-8", "ignore")) <= limit:
            return text
        ids = self.encode(text)
        if len(ids) <= limit:
            return text
        half = limit // 2
        return self.tokenizer.decode(ids[:half]) + "\n...\n" + self.tokenizer.decode(ids[-half:])

    def images(self, images: list) -> tuple:
        batch = self.processor(images=images, return_tensors="pt")
        merge = self.processor.merge_size**2
        counts = [int(g.prod()) // merge for g in batch["image_grid_thw"]]
        return batch["pixel_values"], batch["image_grid_thw"], counts

    def logprobs(self, ids: list[int], labels: list[str], pixels=None, grid=None) -> list[float]:
        torch = self.torch
        kwargs = {}
        if pixels is not None:
            pad = self.model.config.image_token_id
            kinds = torch.tensor([[int(i == pad) for i in ids]])
            kwargs = {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": kinds}
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([ids]), **kwargs).logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)
        return [float(logp[self.label_ids[label]]) for label in labels]

    def request(self, body: dict, images: list) -> dict:
        """Every question's result, or the shim's refusal of the whole request."""
        shim, contract = self.shim, self.contract
        # The shim numbers the images in the state after the top-level ones;
        # stand-ins for those (it takes URLs there unread) keep the count.
        stand_ins = ["https://example.com/image.png"] * len(images)
        state, found = shim.extract_images({"state": body["state"], "images": stand_ins})
        images = images + [decode_data_url(url) for url in found[len(images) :]]
        if len(images) > contract.max_images:
            return {"refused": f"{len(images)} images; at most {contract.max_images}"}
        rendered = shim.render_state(state, contract.state_compact)
        state_text = self.truncate(rendered)
        pixels = grid = counts = None
        if images:
            pixels, grid, counts = self.images(images)
        results = {}
        for qid, question in body["questions"].items():
            results[qid] = self.question(qid, question, state_text, images, pixels, grid, counts)
            if state_text != rendered:
                results[qid]["state_cut"] = True
        return results

    def question(self, qid, question, state_text, images, pixels, grid, counts) -> dict:
        shim, contract = self.shim, self.contract
        kind = question.get("type", "choice")
        try:
            if images:
                urls = ["image"] * len(images)
                prompt = contract.image_prompt(state_text, question, urls, qid)
                _, text, names, descs, labels = prompt
            else:
                text, names, descs, labels = contract.question_prompt(state_text, question, qid)
        except shim.Unprocessable as error:
            return {"refused": str(error)}
        if images:
            # vLLM expands each image's pad into the image's tokens.
            for count in counts:
                text = text.replace("<|image_pad|>", "<|IMG|>" * count, 1)
            text = text.replace("<|IMG|>", "<|image_pad|>")
        ids = self.encode(text)
        result = {"options": len(names), "tokens": len(ids), "ids_sha256": ids_digest(ids)}
        if images:
            result["image_grids"] = grid.tolist()
        if len(ids) > MAX_TOKENS:
            result["refused"] = f"over the {MAX_TOKENS}-token limit"
        elif len(ids) <= TINY_TOKENS:
            logprobs = self.logprobs(ids, labels, pixels, grid)
            t = contract.image_temperature if images else contract.temperature_by_type[kind]
            threshold = shim.noul_threshold(question)
            floor = contract.noul_floor
            answer = shim.format_answer(kind, names, descs, logprobs, t, floor, threshold)
            plain = shim.format_answer(kind, names, descs, logprobs, t, 0, threshold)
            result |= {
                "ids": ids,
                "label_ids": [self.label_ids[label] for label in labels],
                "logprobs": logprobs,
                "temperature": t,
                "answer": answer,
                "answer_no_floor": plain,
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, help="release folder (default: fetch)")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--out", type=Path, default=PARITY / "h2o" / "reference.json")
    args = parser.parse_args()
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request
    from tiny_models import write_h2o

    torch.set_grad_enabled(False)
    release = (args.release or fetch_release()).resolve()
    existing = json.loads(args.out.read_text())["results"] if args.out.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = dict(existing)
    with tempfile.TemporaryDirectory() as folder:
        tiny = write_h2o(Path(folder), seed=SEED, files_from=release)
        reference = Reference(release, tiny)
        for case in cases:
            start = time.perf_counter()
            body = {"state": case["state"], "questions": case["questions"]}
            try:
                parse_request(body)
            except DecisionError as error:
                results[case["id"]] = {"refused": str(error)}
            else:
                images = [open_image(spec) for spec in case.get("images", [])]
                results[case["id"]] = reference.request(body, images)
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
        contract = reference.contract
        meta = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "seed": SEED,
            "label_ids": reference.label_ids,
            "temperature": contract.temperature_by_type,
            "image_temperature": contract.image_temperature,
            "noul_floor": contract.noul_floor,
            "max_state_tokens": contract.max_state_tokens,
            "max_pixels": contract.max_pixels,
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
