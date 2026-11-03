"""Run H2O-Lightning's own serving code on the parity set; store token ids and answers.

usage: python scripts/make_h2o_reference.py [--revision v1.4.2|v1.2.3] [--release PATH]
           [--only ID,ID]
       python scripts/make_h2o_reference.py --model PATH [--revision ...] [--only ID,ID]

The parity set is tests/parity/requests.json (text and image requests),
tests/parity/marker/requests.json and tests/parity/h2o/requests.json
(images inside a state, records over the context).
Needs torch, torchvision, transformers and Pillow (not dependencies of the
package) and the release's small files: without --release they are fetched
into the Hugging Face cache at the pinned revision (everything but the
weights and the adapter, about 30 MB).

The release serves through vLLM behind a standard-library shim
(`h2o_lightning_shim.py`). Its own `Shim` class runs here, imported from the
release folder, so the prompts, the fitting of long records (v1.2.3: a cut at
32,000 tokens; v1.4.2: whole while the prompt fits the context, else BM25
selection per question), the label ids and the answers are the release's.
vLLM's part is a stand-in engine with transformers, as vLLM does it: the
folder's tokenizer as `Qwen2Tokenizer` (the release's training tokenizer:
transformers' `AutoTokenizer` would pick `Qwen3_5Tokenizer`, which splits
Devanagari and vowelled Arabic differently), `Qwen3_5ForConditionalGeneration`
on the CPU in float32 (the release computes the logits in float32), the
log-softmax over the whole vocabulary at the answer slot, the chat template
and the folder's image processor with the shim's `max_pixels` for images.
A two-way read (v1.4.2) runs the full-attention layers without a mask, what
the release's vLLM build does for `attention_mode` "bidirectional"; the
Gated DeltaNet layers stay causal. The model is a tiny random one in the
release's layout (tests/tiny_models.py, `write_h2o`) with the release's
files; the shim's start-up check of the two-way probes is skipped for it
(their answers are the release weights').

Writes tests/parity/h2o/reference.json (v1.4.2) or reference-v1.2.3.json
with, per request and reading mode ("causal"; v1.4.2 also "bidirectional"),
per question: the option count, the token count and a SHA-256 of the token
ids, whether the record was cut or selected, the image grids, and for
prompts of at most 4,096 tokens also the token ids, the label token ids, the
tiny model's log-probabilities of the labels and the shim's answer with and
without the yes/no floor. A request the wire format or the shim refuses is
recorded as refused. Existing entries are kept, so a run can be continued
with --only. For v1.4.2 also, under "limits", the token ids of the text
requests longer than 256, 1,024 and 8,192 tokens as the shim fits them to a vLLM
started with that context (one-way).

With --model (a folder with the release's weights at the pinned revision;
the code and small files still come from the pinned snapshot) the real
model runs instead: bfloat16 on MPS, the logits in float32 (the
release's `"head_dtype": "float32"`: the last hidden state in float32 times
the tied embeddings in float32), every prompt answered, no token ids kept;
the shim's two-way probe check runs (the release's recorded answers, within
0.05). Writes reference-bf16.json (v1.4.2; about 60 min, both modes) or
reference-v1.2.3-bf16.json.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
REPO_ID = "h2oai/h2o-lightning-4b"
REVISIONS = {
    "v1.4.2": "839c5c9523764f0b68c2785944f5541ee66a92b8",
    "v1.2.3": "acaf0d4ea251e54de928c75ef4352670d33192d3",
}
MAX_MODEL_LEN = 40960  # vLLM's --max-model-len in the release's serve.sh
# Lowered contexts (v1.4.2, text, token ids only): how the shim fits prompts
# to a vLLM started with a smaller --max-model-len.
LIMITS = (256, 1024, 8192)
TINY_TOKENS = 4096
SEED = 0
MEMORY_FRACTION = 0.7  # of the GPU's working set, on MPS


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


def data_uri(image) -> str:
    """A parity image as the shim takes it: a PNG data URI (lossless)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def fetch_release(revision: str) -> Path:
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download

    load_dotenv(ROOT / ".env")
    return Path(
        snapshot_download(
            REPO_ID,
            revision=revision,
            ignore_patterns=["*.safetensors", "adapter/*", "assets/*"],
            token=os.environ.get("HF_TOKEN"),
        )
    )


class Engine:
    """vLLM's part for the shim: tokenizer, chat template, images, the model.

    ``real``: the release's own weights in bfloat16 on ``device`` (every
    prompt read); otherwise the tiny model in float32 on the CPU (prompts up
    to TINY_TOKENS read, longer ones only tokenized). ``reads`` collects one
    entry per read, in the order the shim makes them.
    """

    def __init__(self, shim, folder: Path, model_name: str, real=False, device="cpu", files=None):
        import torch
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        self.torch, self.modeling, self.shim = torch, modeling, shim
        self.base = "stand-in engine"
        self.model_name = model_name
        # The weights from ``folder``; the tokenizer, chat template and image
        # processor from ``files`` (the release's small files; default: the same).
        self.real, self.device = real, device
        self.folder = files or folder
        self.tokenizer = Qwen2Tokenizer.from_pretrained(self.folder)
        dtype = torch.bfloat16 if real else torch.float32
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                folder, dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.head = self.model.get_input_embeddings().weight.float()
        self.attention = [
            m for m in self.model.modules() if isinstance(m, modeling.Qwen3_5Attention)
        ]
        self.Processor = Qwen2VLImageProcessor
        self.processors = {}
        self.reads = []
        self.context = MAX_MODEL_LEN

    # What the shim asks of vLLM.
    def call(self, method, path, body=None):
        assert (method, path) == ("GET", "/v1/models"), (method, path)
        return 200, {"data": [{"id": self.model_name, "max_model_len": self.context}]}

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, ids):
        return self.tokenizer.decode(ids)

    def chat_text(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )

    def tokenize_chat(self, messages):
        return self.tokenize(self.chat_text(messages))

    def label_logprobs(self, prompt, ids, attention_mode=None, cache_write=None):
        return self.read(self.tokenize(prompt), ids, attention_mode, prompt)

    def label_logprobs_chat(
        self, messages, ids, max_pixels=0, attention_mode=None, cache_write=None
    ):
        from PIL import Image

        images = []
        for message in messages:
            for part in message["content"] if isinstance(message["content"], list) else []:
                if part.get("type") == "image_url":
                    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
                    images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        if max_pixels not in self.processors:
            self.processors[max_pixels] = self.Processor.from_pretrained(
                self.folder, **({"max_pixels": max_pixels} if max_pixels else {})
            )
        processor = self.processors[max_pixels]
        batch = processor(images=images, return_tensors="pt")
        merge = processor.merge_size**2
        text = self.chat_text(messages)
        # vLLM expands each image's pad into the image's tokens.
        for grid in batch["image_grid_thw"]:
            text = text.replace("<|image_pad|>", "<|IMG|>" * (int(grid.prod()) // merge), 1)
        text = text.replace("<|IMG|>", "<|image_pad|>")
        return self.read(self.tokenize(text), ids, attention_mode, text, batch)

    def read(self, tokens, ids, mode, text, batch=None):
        if len(tokens) > self.context:
            raise self.shim.Unprocessable(f"over the server's context limit: {len(tokens)} tokens")
        entry = {"tokens": tokens, "text": text, "mode": mode or "causal"}
        if batch is not None:
            entry["image_grids"] = batch["image_grid_thw"].tolist()
        if self.real or len(tokens) <= TINY_TOKENS:
            logprobs = self.logprobs(tokens, ids, mode == "bidirectional", batch)
            entry["logprobs"] = [logprobs[i] for i in ids]
        else:
            logprobs = {i: 0.0 for i in ids}  # not read: the answer is not kept
        self.reads.append(entry)
        return logprobs, {"prompt_tokens": len(tokens), "prompt_token_ids": tokens}

    def logprobs(self, tokens, ids, two_way, batch):
        torch, device = self.torch, self.device
        kwargs = {}
        if batch is not None:
            pad = self.model.config.image_token_id
            kwargs = {
                "pixel_values": batch["pixel_values"].to(device, self.model.dtype),
                "image_grid_thw": batch["image_grid_thw"].to(device),
                "mm_token_type_ids": torch.tensor([[int(t == pad) for t in tokens]], device=device),
            }
        create_causal_mask = self.modeling.create_causal_mask
        if two_way:
            # No mask and is_causal False: the full-attention layers read both ways.
            self.modeling.create_causal_mask = lambda **_: None
            for layer in self.attention:
                layer.is_causal = False
        try:
            with torch.no_grad():
                output = self.model.model(input_ids=torch.tensor([tokens], device=device), **kwargs)
                logits = self.head @ output.last_hidden_state[0, -1].float()
        finally:
            self.modeling.create_causal_mask = create_causal_mask
            for layer in self.attention:
                layer.is_causal = True
        logp = torch.log_softmax(logits, -1)
        return {i: float(logp[i]) for i in ids}


class Reference:
    """The release's shim with the stand-in engine, for one release and model."""

    def __init__(self, release: Path, folder: Path, real=False, device="cpu"):
        """``release``: the release's small files (its shim, configs, tokenizer);
        ``folder``: the model whose weights run."""
        # Reads one at a time, in question order, so each read is its question's.
        os.environ["SHIM_QUESTION_THREADS"] = "1"
        os.environ["SHIM_TWO_WAY_FLAG"] = "1"  # one engine, both modes, as serve.sh
        sys.path.insert(0, str(release))
        import h2o_lightning_shim as shim

        self.shim = shim
        config = json.loads((release / "serve_config.json").read_text())
        config["_config_dir"] = str(release)
        self.config = config
        self.contract = shim.Contract(config, env={})
        self.two_way = getattr(self.contract, "two_way", None) is not None
        self.modes = ("causal", "bidirectional") if self.two_way else ("causal",)
        self.engine = Engine(shim, folder, self.contract.model, real, device, files=release)
        self.server = shim.Shim(self.contract, vllm=self.engine)
        self.real = real
        # The shim's start-up checks, before any request: the label ids and the
        # tokenizer; on the real weights also the two-way probes (their reads
        # are not a request's). The probes' answers are the release weights',
        # so the tiny model skips them.
        self.server.check_backend()
        if self.two_way and real:
            self.server.check_two_way_flag()
        elif self.two_way:
            self.server.two_way_ok = True
        self.engine.reads.clear()

    def at_limit(self, body: dict, limit: int) -> dict:
        """The questions' token ids, one-way, from a shim whose vLLM has a
        ``limit``-token context."""
        self.engine.context = limit
        try:
            server = self.shim.Shim(self.contract, vllm=self.engine, min_context=1)
            server.check_backend()
            self.engine.reads.clear()
            try:
                out = server.decide_questions(json.loads(json.dumps(body)), "causal")
            except (self.shim.Unprocessable, self.shim.BadRequest) as error:
                return {"refused": str(error)}
            entries = self.questions(body, out["answers"], "causal")
        finally:
            self.engine.context = MAX_MODEL_LEN
        keep = ("tokens", "ids_sha256", "state_cut")
        return {q: {k: e[k] for k in keep if k in e} for q, e in entries.items()}

    def temperatures(self, mode: str):
        """(by type, image temperature, floor) of a mode."""
        if self.two_way:
            return self.server.temperatures(mode)
        c = self.contract
        return c.temperature_by_type, c.image_temperature, c.noul_floor

    def request(self, body: dict) -> dict:
        results = {}
        for mode in self.modes:
            self.engine.reads.clear()
            try:
                if self.two_way:
                    out = self.server.decide_questions(json.loads(json.dumps(body)), mode)
                else:
                    out = self.server.decide(json.loads(json.dumps(body)))
            except (self.shim.Unprocessable, self.shim.BadRequest) as error:
                results[mode] = {"refused": str(error)}
                continue
            results[mode] = self.questions(body, out["answers"], mode)
        if all("refused" in r for r in results.values()):
            return {"refused": results[self.modes[0]]["refused"]}
        return results

    def questions(self, body: dict, answers: dict, mode: str) -> dict:
        shim = self.shim
        reads = self.engine.reads
        questions = body["questions"]
        assert len(reads) == len(questions), (len(reads), len(questions))
        # The record as the shim renders it whole: images in the state become
        # "[image N]", numbered after the top-level ones. The chat template
        # trims the user turn, so the record is looked for trimmed.
        state = shim.extract_images(json.loads(json.dumps(body)))[0]
        full = shim.render_state(state, self.contract.state_compact).strip()
        by_type, image_t, _floor = self.temperatures(mode)
        out = {}
        for (qid, question), read in zip(questions.items(), reads, strict=True):
            ids = read["tokens"]
            names, descs = shim.options_of(question, len(self.contract.labels))
            entry = {"options": len(names), "tokens": len(ids), "ids_sha256": ids_digest(ids)}
            if full not in read["text"]:
                entry["state_cut"] = True
            if "image_grids" in read:
                entry["image_grids"] = read["image_grids"]
            if "logprobs" in read:
                kind = question.get("type", "choice")
                t = image_t if "image_grids" in read else by_type[kind]
                labels = self.contract.labels[: len(names)]
                entry |= {"ids": ids} if not self.real else {}
                entry |= {
                    "label_ids": [self.contract.label_ids[label] for label in labels],
                    "logprobs": read["logprobs"],
                    "temperature": t,
                    "answer": answers[qid],
                    "answer_no_floor": shim.format_answer(
                        kind, names, descs, read["logprobs"], t, 0, shim.noul_threshold(question)
                    ),
                }
            out[qid] = entry
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", default="v1.4.2", choices=sorted(REVISIONS))
    parser.add_argument("--release", type=Path, help="release folder (default: fetch)")
    parser.add_argument("--model", type=Path, help="run the real model in this release folder")
    parser.add_argument("--device", default="mps", help="device for --model (default: mps)")
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    name = (
        "reference"
        + ("-v1.2.3" if args.revision == "v1.2.3" else "")
        + ("-bf16" if args.model else "")
    )
    args.out = args.out or PARITY / "h2o" / f"{name}.json"
    import torch
    import transformers

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(PARITY))
    from parity_images import open_image

    from mlx_decision.errors import DecisionError
    from mlx_decision.types import parse_request
    from tiny_models import write_h2o

    torch.set_grad_enabled(False)
    if args.model and args.device == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    # The release's code and small files at the pinned revision; with --model
    # only the weights come from that folder (a download may hold no shim).
    release = (args.release or fetch_release(REVISIONS[args.revision])).resolve()
    if (
        args.model
        and (args.model / "serve_config.json").read_bytes()
        != (release / "serve_config.json").read_bytes()
    ):
        raise SystemExit(f"{args.model} is not the release at {args.revision} (serve_config.json)")
    has_two_way = "two_way" in json.loads((release / "serve_config.json").read_text())
    if has_two_way != (args.revision != "v1.2.3"):
        raise SystemExit(
            f"{release} does not look like {args.revision} (two_way block: {has_two_way})"
        )
    existing = json.loads(args.out.read_text())["results"] if args.out.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    results = dict(existing)
    with tempfile.TemporaryDirectory() as folder:
        if args.model:
            reference = Reference(release, args.model.resolve(), real=True, device=args.device)
        else:
            tiny = write_h2o(Path(folder) / "tiny-h2o", seed=SEED, files_from=release)
            reference = Reference(release, tiny)
        for case in cases:
            start = time.perf_counter()
            body = {"state": case["state"], "questions": case["questions"]}
            try:
                parse_request(body)
            except DecisionError as error:
                results[case["id"]] = {"refused": str(error)}
            else:
                if case.get("images"):
                    body["images"] = [data_uri(open_image(spec)) for spec in case["images"]]
                results[case["id"]] = reference.request(body)
            print(f"{case['id']}: {time.perf_counter() - start:.1f} s", flush=True)
        limits = {}
        if reference.two_way and not args.model:
            for limit in LIMITS:
                limits[str(limit)] = {}
                for case in parity_set():
                    result = results.get(case["id"], {})
                    if case.get("images") or "refused" in result:
                        continue
                    longest = max(e["tokens"] for e in result["causal"].values())
                    if longest > limit:
                        body = {"state": case["state"], "questions": case["questions"]}
                        limits[str(limit)][case["id"]] = reference.at_limit(body, limit)
                print(f"limit {limit}: {len(limits[str(limit)])} requests", flush=True)
        contract, server = reference.contract, reference.server
        meta = {
            "repo_id": REPO_ID,
            "tag": args.revision,
            "revision": REVISIONS[args.revision],
            **({"model": "release"} if args.model else {"seed": SEED}),
            "label_ids": contract.label_ids,
            "temperature": contract.temperature_by_type,
            "image_temperature": contract.image_temperature,
            "noul_floor": contract.noul_floor,
            "max_state_tokens": contract.max_state_tokens,
            "max_pixels": contract.max_pixels,
            "max_model_len": MAX_MODEL_LEN,
            **({} if args.model else {"tiny_tokens": TINY_TOKENS}),
            "modes": list(reference.modes),
            "device": reference.engine.device,
            "dtype": str(reference.engine.model.dtype),
            "logits": "float32",
            "tokenizer": "Qwen2Tokenizer",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        if reference.two_way:
            by_type, image_t, floor = server.temperatures("bidirectional")
            meta["two_way"] = {
                "temperature": by_type,
                "image_temperature": image_t,
                "noul_floor": floor,
            }
            if args.model:
                meta["two_way"]["probe_anchor"] = server.two_way_anchor
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "results": results, **({"limits": limits} if limits else {})}
    args.out.write_text(dump(data) + "\n")
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
