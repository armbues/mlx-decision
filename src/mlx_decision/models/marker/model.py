"""The Laya and Julia backends: a ModernBERT encoder plus the marker head.

Both families share the encoder and the head; they differ in how a release is
laid out on disk, how options are written into the input and how scores
become probabilities.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.modernbert import Model as Encoder
from ...backbones.modernbert import ModelArgs as EncoderArgs
from ...backend import Capabilities
from .head import MarkerHead


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
    temperatures: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    temperatures_by_options: dict[str, float] = field(default_factory=dict)


@dataclass
class Sequence:
    """One question's encoded input: token ids, marker positions, question type."""

    input_ids: list[int]
    markers: list[int]
    question_type: int


class MarkerBackend:
    def __init__(self, name, encoder, head, tokenizer, special, settings, capabilities):
        self.name = name
        self.encoder = encoder
        self.head = head
        self.tokenizer = tokenizer
        self.special = special
        self.settings = settings
        self.capabilities = capabilities

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
    return encoder, head, tokenizer, special, args


def load_julia(
    path: Path, max_input_tokens: int | None = None, dtype: str | None = None
) -> MarkerBackend:
    """Load Julia 1 from a release folder (``julia_config.json``).

    ``max_input_tokens`` raises or lowers the sequence limit (default: the
    release's ``inference-policy.json``, up to the encoder's 8,192).
    ``dtype`` (``"float32"``, ``"float16"``, ``"bfloat16"``) converts the
    weights; None keeps them as stored.
    """
    root = json.loads((path / "config.json").read_text()) if (path / "config.json").exists() else {}
    config = json.loads((path / root.get("julia_config_file", "julia_config.json")).read_text())
    if config.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Julia checkpoint format")
    policy_file = path / "inference-policy.json"
    policy = json.loads(policy_file.read_text()) if policy_file.exists() else {}
    encoder_dir = str(Path(root.get("encoder_config_file", "encoder/config.json")).parent)
    encoder, head, tokenizer, special, args = _load_parts(
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
        strict=policy.get("strict_encoding", True),
    )
    capabilities = Capabilities(
        max_input_tokens=settings.max_length, max_choice_options=20, max_score_levels=20
    )
    return MarkerBackend(
        path.resolve().name, encoder, head, tokenizer, special, settings, capabilities
    )


def load_laya(
    path: Path, max_input_tokens: int | None = None, dtype: str | None = None
) -> MarkerBackend:
    """Load a Laya checkpoint from a release folder (``rl_agent_config.json``).

    ``max_input_tokens`` raises or lowers the sequence limit (default: the
    checkpoint's ``max_len``, up to the encoder's 8,192). ``dtype`` as for
    ``load_julia``.
    """
    config = json.loads((path / "rl_agent_config.json").read_text())
    encoder, head, tokenizer, special, args = _load_parts(
        path, "model.safetensors", "encoder", "tokenizer", config["head_layers"], dtype
    )
    settings = Settings(
        family="laya",
        max_length=_max_length(max_input_tokens, config.get("max_len"), args),
        head_length=config.get("head_max_len", 192),
        temperatures=list(config.get("temperature", [1.0, 1.0, 1.0])),
        temperatures_by_options=dict(config.get("temperature_by_options", {})),
    )
    capabilities = Capabilities(max_input_tokens=settings.max_length)
    return MarkerBackend(
        path.resolve().name, encoder, head, tokenizer, special, settings, capabilities
    )


def _max_length(requested: int | None, default: int | None, args: EncoderArgs) -> int:
    limit = args.max_position_embeddings
    value = requested if requested is not None else (default or limit)
    if not 16 <= value <= limit:
        raise ValueError(f"max_input_tokens must be between 16 and {limit}")
    return value
