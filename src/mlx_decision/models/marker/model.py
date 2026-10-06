"""The Laya and Julia backends: a ModernBERT encoder plus the marker head.

Both families share the encoder and the head; they differ in how a release is
laid out on disk, how options are written into the input and how scores
become probabilities.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from ...backbones.modernbert import Model as Encoder
from ...backbones.modernbert import ModelArgs as EncoderArgs
from ...backend import BackendOutput, Capabilities
from ...types import Request
from .encode import encode_request
from .head import QUESTION_TYPES, MarkerHead

QUESTION_NAMES = {index: name for name, index in QUESTION_TYPES.items()}
# The questions of a request run in padded batches of at most this many tokens
# (count x longest). Measured: batching short sequences is up to 4x faster
# than one by one; from about 4k tokens per sequence one by one is about 10%
# faster and needs less memory.
BATCH_TOKENS = 4096
# Laya refuses to apply a fitted temperature outside this range: below 0.5 one
# shipped bucket (choice:11+, 0.10) would turn a 0.24 top probability into 0.99.
TEMPERATURE_RANGE = (0.5, 5.0)


@dataclass(frozen=True)
class SpecialTokens:
    cls: int
    sep: int
    mask: int
    pad: int
    mask_text: str  # the marker as text, removed from user text before encoding


@dataclass
class Settings:
    """What a release says about encoding and calibration."""

    family: str  # "laya" or "julia"
    max_length: int  # whole sequence, special tokens included
    head_length: int  # budget for the question and its options
    strict: bool = False  # refuse instead of cutting options, question or state (Julia)
    # Laya's calibration, already clamped: one per question type, and per
    # bucket of type and option count ("choice:3-5"), which wins when present.
    temperatures: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperatures_by_options: dict[str, float] = field(default_factory=dict)

    def temperature(self, question_type: int, options: int) -> float:
        size = (
            "2" if options <= 2 else "3-5" if options <= 5 else "6-10" if options <= 10 else "11+"
        )
        bucket = f"{QUESTION_NAMES[question_type]}:{size}"
        return self.temperatures_by_options.get(bucket, self.temperatures[question_type])


@dataclass
class Sequence:
    """One question's encoded input: token ids, marker positions, question type."""

    input_ids: list[int]
    markers: list[int]
    question_type: int


class MarkerBackend:
    def __init__(
        self, name, encoder, head, tokenizer, special, settings, capabilities, precision=None
    ):
        self.name = name
        self.precision = precision
        self.encoder = encoder
        self.head = head
        self.tokenizer = tokenizer
        self.special = special
        self.settings = settings
        self.capabilities = capabilities

    def score(self, request: Request) -> BackendOutput:
        encoded = encode_request(self.tokenizer, self.special, self.settings, request)
        sequences = [Sequence(q.input_ids, q.markers, q.question_type) for q in encoded]
        scored = [row for batch in batches(sequences) for row in self.logits(batch)]
        probabilities = {}
        for question, scores in zip(encoded, scored, strict=True):
            temperature = self.settings.temperature(question.question_type, len(scores))
            values = mx.softmax(mx.array(scores, dtype=mx.float32) / temperature).tolist()
            probabilities[question.question_id] = dict(
                zip(question.option_ids, values, strict=True)
            )
        return BackendOutput(
            probabilities=probabilities,
            input_tokens=sum(len(q.input_ids) for q in encoded),
            truncated=any(q.truncated for q in encoded),
        )

    def logits(self, sequences: list[Sequence]) -> list[list[float]]:
        """Raw option scores per sequence, all sequences in one padded batch."""
        width = max(len(s.input_ids) for s in sequences)
        count = max(len(s.markers) for s in sequences)
        ids = [s.input_ids + [self.special.pad] * (width - len(s.input_ids)) for s in sequences]
        mask = [[1] * len(s.input_ids) + [0] * (width - len(s.input_ids)) for s in sequences]
        # Rows with fewer options repeat their first marker; those scores are dropped.
        markers = [s.markers + [s.markers[0]] * (count - len(s.markers)) for s in sequences]
        attention = mx.array(mask) if len(sequences) > 1 else None
        hidden = self.encoder(mx.array(ids), attention)
        scores = self.head(
            hidden, attention, mx.array([s.question_type for s in sequences]), mx.array(markers)
        )
        mx.eval(scores)
        return [row[: len(s.markers)] for row, s in zip(scores.tolist(), sequences, strict=True)]


def batches(sequences: list[Sequence], budget: int | None = None):
    """Consecutive sequences grouped while count x longest stays within ``budget``."""
    budget = BATCH_TOKENS if budget is None else budget
    batch, longest = [], 0
    for sequence in sequences:
        length = max(longest, len(sequence.input_ids))
        if batch and length * (len(batch) + 1) > budget:
            yield batch
            batch, length = [], len(sequence.input_ids)
        batch.append(sequence)
        longest = length
    if batch:
        yield batch


def _special_tokens(tokenizer: Tokenizer, folder: Path) -> SpecialTokens:
    config = json.loads((folder / "tokenizer_config.json").read_text())
    ids = {}
    for role in ("cls", "sep", "mask", "pad"):
        token = config.get(f"{role}_token")
        if isinstance(token, dict):  # older tokenizer configs store AddedToken objects
            token = token.get("content")
        token_id = None if token is None else tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"{folder}: the tokenizer has no {role} token")
        ids[role] = token_id
    return SpecialTokens(**ids, mask_text=tokenizer.id_to_token(ids["mask"]))


DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
# Measured against both families' float32 reference on the parity set: float16
# is the closest and the fastest; bfloat16 moves Julia's answers by up to 0.15,
# and float32 matmuls on the GPU (TF32) by up to 0.06.
DEFAULT_DTYPE = "float16"


def _load_parts(
    path: Path,
    weights_file: str,
    encoder_dir: str,
    tokenizer_dir: str,
    layers: int,
    dtype: str | None,
):
    for name in (weights_file, f"{encoder_dir}/config.json", f"{tokenizer_dir}/tokenizer.json"):
        if not (path / name).exists():
            raise FileNotFoundError(f"{path}: {name} is missing")
    config = json.loads((path / encoder_dir / "config.json").read_text())
    args = EncoderArgs.from_config(config)
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}")
    weights = mx.load(str(path / weights_file))
    if dtype is not None:
        weights = {k: v.astype(DTYPES[dtype]) for k, v in weights.items()}
    encoder = Encoder(args)
    encoder.load_weights(list(Encoder.sanitize(weights, "encoder.").items()), strict=True)
    head = MarkerHead(args.hidden_size, layers)
    head.load_weights(list(MarkerHead.sanitize(weights).items()), strict=True)
    mx.eval(encoder.parameters(), head.parameters())
    tokenizer = Tokenizer.from_file(str(path / tokenizer_dir / "tokenizer.json"))
    special = _special_tokens(tokenizer, path / tokenizer_dir)
    precision = str(tree_flatten(encoder.parameters())[0][1].dtype).rsplit(".", 1)[-1]
    return encoder, head, tokenizer, special, args, precision


def load_julia(
    path: Path,
    max_input_tokens: int | None = None,
    dtype: str | None = DEFAULT_DTYPE,
    strict_encoding: bool | None = None,
) -> MarkerBackend:
    """Load Julia 1 from a release folder (``julia_config.json``).

    ``max_input_tokens`` raises or lowers the sequence limit (default: the
    release's ``inference-policy.json``, up to the encoder's 8,192).
    ``dtype`` (``"float32"``, ``"float16"``, ``"bfloat16"``) converts the
    weights (default float16); None keeps them as stored. ``strict_encoding`` (default: the
    policy's, True for Julia 1) refuses requests that would have to be cut
    (state, question or options) instead of cutting them.
    """
    root = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    config = json.loads((path / root.get("julia_config_file", "julia_config.json")).read_text())
    if config.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Julia checkpoint format")
    policy_file = path / "inference-policy.json"
    policy = json.loads(policy_file.read_text()) if policy_file.exists() else {}
    encoder_dir = str(Path(root.get("encoder_config_file", "encoder/config.json")).parent)
    encoder, head, tokenizer, special, args, precision = _load_parts(
        path,
        root.get("weights_file", "model.safetensors"),
        encoder_dir,
        root.get("tokenizer_directory", "tokenizer"),
        config["head_layers"],
        dtype,
    )
    settings = Settings(
        family="julia",
        max_length=_max_length(max_input_tokens, policy.get("max_length"), args),
        head_length=policy.get("head_length", 512),
        strict=policy.get("strict_encoding", True) if strict_encoding is None else strict_encoding,
    )
    capabilities = Capabilities(
        max_input_tokens=settings.max_length,
        max_choice_options=20,
        max_score_levels=20,
        question_tokens=settings.head_length,
        truncates_input=not settings.strict,
    )
    return MarkerBackend(
        path.resolve().name, encoder, head, tokenizer, special, settings, capabilities, precision
    )


def load_laya(
    path: Path, max_input_tokens: int | None = None, dtype: str | None = DEFAULT_DTYPE
) -> MarkerBackend:
    """Load a Laya checkpoint from a release folder (``rl_agent_config.json``).

    ``max_input_tokens`` raises or lowers the sequence limit (default: the
    checkpoint's ``max_len``, up to the encoder's 8,192). ``dtype`` as for
    ``load_julia``.
    """
    config = json.loads((path / "rl_agent_config.json").read_text())
    encoder, head, tokenizer, special, args, precision = _load_parts(
        path, "model.safetensors", "encoder", "tokenizer", config["head_layers"], dtype
    )
    settings = Settings(
        family="laya",
        max_length=_max_length(max_input_tokens, config.get("max_len"), args),
        head_length=config.get("head_max_len", 192),
        temperatures=_laya_temperatures(path, config.get("temperature", [1.0, 1.0, 1.0])),
        temperatures_by_options={
            bucket: _clamp_temperature(value)
            for bucket, value in config.get("temperature_by_options", {}).items()
        },
    )
    capabilities = Capabilities(
        max_input_tokens=settings.max_length, question_tokens=settings.head_length
    )
    return MarkerBackend(
        path.resolve().name, encoder, head, tokenizer, special, settings, capabilities, precision
    )


def _max_length(requested: int | None, default: int | None, args: EncoderArgs) -> int:
    limit = args.max_position_embeddings
    value = requested if requested is not None else (default or limit)
    if not 16 <= value <= limit:
        raise ValueError(f"max_input_tokens must be between 16 and {limit}")
    return value


def _clamp_temperature(value) -> float:
    """A usable temperature, as Laya applies it: clamped, 1.0 when not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 1.0
    low, high = TEMPERATURE_RANGE
    return min(high, max(low, float(value)))


def _laya_temperatures(path: Path, values) -> list[float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{path}: temperature must be a list of 3 numbers")
    return [_clamp_temperature(value) for value in values]
