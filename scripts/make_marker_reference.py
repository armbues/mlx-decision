"""Run Laya's or Julia's own PyTorch code on the marker parity set and store the results.

usage: python scripts/make_marker_reference.py --model PATH [--only ID,ID]

The parity set is the text requests of tests/parity/requests.json plus
tests/parity/marker/requests.json. Needs torch and transformers (not
dependencies of the package) and the family's code: Laya's from the `laya`
package (pip install laya), Julia's from its release folder, which is put on
the import path. Both run on the CPU in float32, as their code does there:
Laya as `laya.load(path)` serves it, Julia with the settings its README
loads it with (strict encoding, 8,192 tokens, a 512-token question budget).
Julia's README entry point (`load_model`) swaps in a faster rewrite of the
encoder's forward that fails with current transformers; its plain
`TransformerEngine` runs the unchanged model, and `julia.data.sequence`
with `strict=True` decides what is refused.

Writes tests/parity/marker/reference-<model folder>.json with, per request,
either the refusal or per question: the token count and a SHA-256 of the
token ids, the marker positions, whether the state was cut, the raw option
scores and the probabilities. Existing entries are kept, so a run can be
continued with --only.
"""

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"


def parity_set() -> list[dict]:
    main = [c for c in json.loads((PARITY / "requests.json").read_text()) if not c.get("images")]
    extra = json.loads((PARITY / "marker" / "requests.json").read_text())
    return main + extra


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    top = max(values)
    exps = [math.exp((v - top) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LayaReference:
    def __init__(self, path: Path):
        import laya
        import torch

        warnings.filterwarnings("ignore", message="laya: this checkpoint ships")
        self.agent = laya.load(str(path), device="cpu")
        self.meta = {
            "family": "laya",
            "laya": laya.__version__,
            "dtype": str(next(self.agent.model.parameters()).dtype),
            "max_len": self.agent.cfg.get("max_len"),
            "head_max_len": self.agent.cfg.get("head_max_len"),
        }
        self.torch = torch
        self.captured = []
        forward = self.agent.model.forward

        def spy(input_ids, attention_mask, marker_pos, marker_mask, qtype, *args, **kwargs):
            out = forward(
                input_ids, attention_mask, marker_pos, marker_mask, qtype, *args, **kwargs
            )
            logits = out[0] if isinstance(out, tuple) else out
            for row in range(input_ids.shape[0]):
                count = int(marker_mask[row].sum())
                length = int(attention_mask[row].sum())
                self.captured.append(
                    (input_ids[row, :length].tolist(), logits[row, :count].float().tolist())
                )
            return out

        self.agent.model.forward = spy

    def run(self, state, questions) -> dict:
        agent = self.agent
        ids = list(questions)
        try:
            for question_id, question in questions.items():
                agent._check_question(question_id, question)
            internal = {qid: agent._to_internal(q) for qid, q in questions.items()}
            items = agent._encode_state(state, ids, internal)
            self.captured = []
            served = agent.system_one(state, questions)["answers"]
        except ValueError as error:
            return {"refused": str(error)}
        assert len(self.captured) == len(items), "one forward row per question"
        out = []
        for qid, item, (row_ids, scores) in zip(ids, items, self.captured, strict=True):
            assert row_ids == item["ids"], f"{qid}: captured ids differ from the encoding"
            question = internal[qid]
            qtype = {"choice": 0, "score": 1, "noul": 2}[question["t"]]
            from laya.common import temp_bucket

            k = len(item["markers"])
            temperature = agent.temperature_by_options.get(
                temp_bucket(qtype, k), agent.temperature[qtype]
            )
            probabilities = softmax(scores, temperature)
            check_served(served[qid], probabilities)
            out.append(
                question_entry(qid, item["ids"], item["markers"], scores, probabilities)
                | {"truncated": item["state_stats"]["truncated"], "temperature": temperature}
            )
        return {"questions": out}


def check_served(answer: dict, probabilities: list[float]) -> None:
    """The probabilities computed here are the ones the package serves (4 decimals)."""
    served = (
        [1 - answer["noul"], answer["noul"]]
        if answer["type"] == "noul"
        else list(answer["probabilities"].values())
    )
    worst = max(abs(a - b) for a, b in zip(served, probabilities, strict=True))
    assert worst < 1e-4, f"served probabilities differ by {worst}"


def question_entry(qid, ids, markers, scores, probabilities) -> dict:
    return {
        "id": qid,
        "length": len(ids),
        "ids_sha256": ids_digest(ids),
        "markers": markers,
        "scores": scores,
        "probabilities": probabilities,
    }


class JuliaReference:
    def __init__(self, path: Path):
        sys.path.insert(0, str(path))
        from julia.data import sequence
        from julia.inference import TransformerEngine
        from julia.typed import predict_typed

        self.engine = TransformerEngine(
            path, device="cpu", max_length=8192, head_length=512, memory_map=False
        )
        self.sequence = sequence
        self.predict_typed = predict_typed
        self.meta = {
            "family": "julia",
            "strict_encoding": True,
            "max_length": 8192,
            "head_length": 512,
            "dtype": str(next(self.engine.model.parameters()).dtype),
        }

    def run(self, state, questions) -> dict:
        engine, sequence = self.engine, self.sequence
        tokenizer = engine.tokenizer
        seen = {}

        class Recorder:
            def logits(self, rows):
                seen["encoded"] = [sequence(tokenizer, row, 8192, 512, strict=True) for row in rows]
                seen["scores"] = engine.logits(rows)
                return seen["scores"]

        try:
            self.predict_typed(Recorder(), state, questions)
        except ValueError as error:
            return {"refused": str(error)}
        out = []
        for qid, item, scores in zip(questions, seen["encoded"], seen["scores"], strict=True):
            # Strict encoding refuses whatever it would cut, so nothing is truncated.
            out.append(
                question_entry(qid, item["ids"], item["markers"], scores, softmax(scores))
                | {"truncated": item["truncated"], "temperature": 1.0}
            )
        return {"questions": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--only", help="comma-separated request ids")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import torch
    import transformers

    torch.set_grad_enabled(False)
    path = args.model.resolve()
    if (path / "rl_agent_config.json").exists():
        reference = LayaReference(path)
    elif (path / "julia_config.json").exists():
        reference = JuliaReference(path)
    else:
        sys.exit(f"{path}: neither a Laya nor a Julia release")
    out_path = args.out or PARITY / "marker" / f"reference-{args.model.name}.json"
    existing = json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    cases = parity_set()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    results = dict(existing)
    for case in cases:
        start = time.perf_counter()
        results[case["id"]] = reference.run(case["state"], case["questions"])
        status = "refused" if "refused" in results[case["id"]] else "ok"
        print(f"{case['id']}: {status} ({time.perf_counter() - start:.1f} s)", flush=True)
    meta = reference.meta | {
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "weights_sha256": file_digest(path / "model.safetensors"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=1) + "\n")
    print(f"wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
