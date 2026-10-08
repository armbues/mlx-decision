"""The pplx backend: Qwen3.5 backbone with non-causal full attention plus a letter readout.

Each question is its own sequence (see ``prompt``): the last token's hidden
state times the readout gives one logit per answer code; the codes beyond
the question's options are left out, the rest divided by the release's
temperature and normalized. The state attends to the question in the
full-attention layers, so nothing of one question's pass serves another.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ...backbones.qwen3_5.load import load_text_model
from ...backbones.qwen3_5.tokenizer import load_tokenizer
from ...backend import BackendOutput, Capabilities
from ...errors import DecisionError
from ...types import Request
from ..clef.model import precision
from .prompt import options, render

# The release's limit per question's sequence; longer input is refused, as
# its code does, never cut.
MAX_INPUT_TOKENS = 8192
REQUIRED_FILES = ("config.json", "tokenizer.json", "decision_config.json", "readout.safetensors")
ATTENTION_MODES = {"noncausal_full_attention": False, "causal": True}  # mode -> causal


@dataclass
class Encoded:
    question_id: str
    option_ids: list[str]
    input_ids: list[int]


class PplxBackend:
    def __init__(self, name, backbone, readout, tokenizer, codes, temperature, max_input_tokens):
        self.name = name
        self.backbone = backbone
        self.readout = readout
        self.tokenizer = tokenizer
        self.codes = codes
        self.temperature = temperature
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            max_choice_options=len(codes),
            max_score_levels=len(codes),
            truncates_input=False,
        )

    def encode(self, request: Request) -> list[Encoded]:
        """Every question's token ids; refuses the request if one is over the limit."""
        limit = self.capabilities.max_input_tokens
        encoded = []
        for question_id, question in request.questions.items():
            where = f"questions.{question_id}"
            text = render(request.state, question, self.codes, where=where)
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) > limit:
                raise DecisionError(
                    f"the question with its state needs {len(ids)} tokens; "
                    f"the limit is {limit} and input is not cut",
                    param=where,
                )
            encoded.append(Encoded(question_id, options(question)[0], ids))
        return encoded

    def score(self, request: Request) -> BackendOutput:
        encoded = self.encode(request)
        probabilities = {}
        for question in encoded:
            values = self.probabilities(question.input_ids, len(question.option_ids))
            probabilities[question.question_id] = dict(
                zip(question.option_ids, values, strict=True)
            )
        return BackendOutput(
            probabilities=probabilities,
            input_tokens=sum(len(q.input_ids) for q in encoded),
        )

    def probabilities(self, input_ids: list[int], count: int) -> list[float]:
        """One pass: the first ``count`` answer codes' probabilities."""
        hidden = self.backbone(mx.array(input_ids)[None])[0, -1].astype(mx.float32)
        logits = self.readout[:count].astype(mx.float32) @ hidden
        values = mx.softmax(logits / self.temperature)
        mx.eval(values)
        return values.tolist()


def read_decision_config(path: Path) -> tuple[list[str], float, bool]:
    """Answer codes, temperature and whether attention is causal, from ``decision_config.json``."""
    config = json.loads((path / "decision_config.json").read_text())
    if config.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported decision config format")
    mode = config.get("attention_mode", "causal")
    if mode not in ATTENTION_MODES:
        raise ValueError(f"{path}: unknown attention mode {mode!r}")
    if config.get("pooling", "last") != "last":
        raise ValueError(f"{path}: only last-token pooling is supported")
    temperature = config.get("temperature", 1.0)
    number = isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
    if not number or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{path}: the temperature must be a positive number")
    codes = config.get("codes")
    if not isinstance(codes, list) or not codes or not all(isinstance(c, str) for c in codes):
        raise ValueError(f"{path}: decision_config.json has no answer codes")
    return codes, float(temperature), ATTENTION_MODES[mode]


def weight_files(path: Path, options: dict) -> list[tuple[Path, None]]:
    """The weights ``load`` reads (the vision tower shares a shard), kept as stored."""
    files = sorted(path.glob("model*.safetensors")) + [path / "readout.safetensors"]
    return [(file, None) for file in files if file.exists()]


def load(path: Path, max_input_tokens: int | None = None) -> PplxBackend:
    """Load pplx-decider from a release folder or a converted copy.

    ``max_input_tokens`` lowers the limit per question's sequence (default
    and maximum 8,192, the release's); longer input is refused, not cut.
    """
    if max_input_tokens is None:
        max_input_tokens = MAX_INPUT_TOKENS
    if not 1 <= max_input_tokens <= MAX_INPUT_TOKENS:
        raise ValueError(
            f"max_input_tokens must be between 1 and {MAX_INPUT_TOKENS} (the release's limit)"
        )
    # Checked before the backbone, which takes a while to load.
    for name in REQUIRED_FILES:
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    if not any(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"{path}: no model*.safetensors weights")
    codes, temperature, causal = read_decision_config(path)
    readout = mx.load(str(path / "readout.safetensors"))["weight"]
    if readout.shape[0] < len(codes):
        raise ValueError(f"{path}: the readout has fewer rows than answer codes")
    backbone = load_text_model(path, lm_head=False, causal=causal)
    if readout.shape[1] != backbone.language_model.args.hidden_size:
        raise ValueError(f"{path}: the readout does not match the backbone's width")
    mx.eval(readout)
    tokenizer = load_tokenizer(path)
    backend = PplxBackend(
        path.resolve().name, backbone, readout, tokenizer, codes, temperature, max_input_tokens
    )
    backend.precision = precision(path, backbone)
    return backend
