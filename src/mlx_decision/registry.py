"""Model families: how a model folder is recognised and loaded."""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from .backend import Backend, Capabilities

MARKER_FILE = "mlx_decision.json"


@dataclass(frozen=True)
class KnownModel:
    """A model on the Hugging Face Hub that a family loads, offered by ``download``."""

    repo_id: str
    description: str
    size: str  # approximate download size, for the menu


@dataclass(frozen=True)
class Listing:
    """How the ``download`` menu shows a family: its name, maker and a short line."""

    title: str
    maker: str
    summary: str


@dataclass(frozen=True)
class Family:
    name: str
    detect: Callable[[Path], bool]
    loader: str  # "module:function"; imported only when the family is used
    # Same form; writes an MLX copy of a model: (path, output, source, bits=,
    # shards=, revision=, ...), shards being the weight files read in turn.
    converter: str | None = None
    known_models: tuple[KnownModel, ...] = ()
    # Files a download fetches (Hub patterns); None: the whole repository.
    files: tuple[str, ...] | None = None
    # "module:function" (path, load options) -> [(weight file, load dtype or None)],
    # for the memory check; None: every top-level *.safetensors, as stored.
    weight_files: str | None = None
    # (tensor name, shape) -> whether the converter quantizes it, for size
    # estimates before converting; None: every tensor.
    quantizable: Callable[[str, list[int]], bool] | None = None
    # A note on terms of use that ``download`` shows (and asks about in a
    # terminal) and a download by repo id logs; None: nothing to note.
    licence: str | None = None
    # Its line in the ``download`` menu; None: not listed there.
    listing: Listing | None = None
    # "module:function" (path, load options) -> Capabilities: what a model takes,
    # read from its folder's config files without loading weights; the
    # backend's constructor builds its capabilities the same way.
    capabilities: str | None = None


# In registration order, which is the order the docs list the families in.
_FAMILIES: dict[str, Family] = {}


def register_family(
    name: str,
    detect: Callable[[Path], bool],
    loader: str,
    converter: str | None = None,
    known_models: tuple[KnownModel, ...] = (),
    files: tuple[str, ...] | None = None,
    weight_files: str | None = None,
    quantizable: Callable[[str, list[int]], bool] | None = None,
    licence: str | None = None,
    listing: Listing | None = None,
    capabilities: str | None = None,
) -> None:
    _FAMILIES[name] = Family(
        name,
        detect,
        loader,
        converter,
        known_models,
        files,
        weight_files,
        quantizable,
        licence,
        listing,
        capabilities,
    )


def families() -> list[Family]:
    """Every family, in the order messages and menus list them."""
    return list(_FAMILIES.values())


def known_models() -> list[KnownModel]:
    return [model for family in families() for model in family.known_models]


def get_family(name: str) -> Family:
    return _FAMILIES[name]


def resolve(reference: str) -> Callable:
    """Import ``module:function``."""
    module, function = reference.split(":")
    return getattr(import_module(module), function)


# Fitted models (a marker naming a base): loaded through their base's family,
# so not one of the families listed in messages and menus.
FITTED = Family(
    "fitted",
    detect=lambda path: False,
    loader="mlx_decision.fitted:load",
    weight_files="mlx_decision.fitted:weight_files",
    capabilities="mlx_decision.fitted:capabilities",
)


def detect_family(path: Path) -> Family:
    marker = path / MARKER_FILE
    if marker.exists():
        config = json.loads(marker.read_text())
        if isinstance(config, dict) and "base" in config:
            return FITTED
        name = config.get("family")
        if name not in _FAMILIES:
            raise ValueError(f"{path}: unknown model family {name!r}")
        return _FAMILIES[name]
    for family in _FAMILIES.values():
        if family.detect(path):
            return family
    known = ", ".join(_FAMILIES) or "none"
    raise ValueError(f"{path}: not a supported decision model (known families: {known})")


def loader_parameters(family: Family) -> dict[str, inspect.Parameter] | None:
    """The loader's parameters by name; None when it takes any keyword."""
    parameters = inspect.signature(resolve(family.loader)).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return dict(parameters)


def accepted_options(family: Family, options: dict) -> dict:
    """The options the family's loader takes."""
    parameters = loader_parameters(family)
    if parameters is None:
        return dict(options)
    return {name: value for name, value in options.items() if name in parameters}


def check_options(family: Family, options: dict) -> None:
    """Fail unless the family's loader takes every option."""
    unknown = [name for name in options if name not in accepted_options(family, options)]
    if unknown:
        raise ValueError(f"{family.name} models do not take {', '.join(unknown)}")


def load_backend(path: Path, **options) -> Backend:
    from .memory import cap_buffer_cache

    family = detect_family(path)
    check_options(family, options)
    cap_buffer_cache()
    backend = resolve(family.loader)(path, **options)
    if family is not FITTED:  # a fitted model's family is its base's
        backend.family = family.name
    return backend


def static_capabilities(path: Path, **options) -> Capabilities:
    """What the model in ``path`` takes when loaded with ``options``, read from its
    config files: no weights are loaded. Raises as ``load_backend`` would for
    options the family does not take or values its loader refuses."""
    family = detect_family(path)
    check_options(family, options)
    if family.capabilities is None:
        raise ValueError(f"{family.name} models do not report capabilities before loading")
    return resolve(family.capabilities)(path, options)


def _qwen_text_matrix(name: str, shape: list[int]) -> bool:
    """The text backbone's matrices (named as released or as MLX saves them);
    the vision tower and the decision head stay as they are."""
    return len(shape) == 2 and name.startswith(
        ("model.language_model.", "lm_head.", "language_model.")
    )


register_family(
    "clef",
    detect=lambda path: (path / "joint_head_config.json").exists(),
    loader="mlx_decision.models.clef.model:load",
    capabilities="mlx_decision.models.clef.model:capabilities",
    listing=Listing("Clef", "Cloudflare", "joint schema head over Qwen3.5; text and images"),
    converter="mlx_decision.models.clef.convert:convert",
    known_models=(
        KnownModel(
            "Cloudflare/clef-flash",
            "Clef flash: Cloudflare's 9B decision model (Qwen3.5), text and images",
            "19.1 GB",
        ),
        KnownModel(
            "Cloudflare/clef",
            "Clef: Cloudflare's 27B decision model (Qwen3.5), text and images; "
            "below 96 GB of memory it runs quantized (offered after picking it)",
            "55.0 GB",
        ),
    ),
    # Weights, head, tokenizer and configs; not the reference code or chat template.
    files=("*.json", "model*.safetensors", "joint_head.safetensors", "LICENSE*", "README.md"),
    weight_files="mlx_decision.models.clef.model:weight_files",
    quantizable=_qwen_text_matrix,
)


class UnknownFormat(ValueError):
    """A folder of a family's kind that names a prompt or request format no
    family here knows (a newer release): refused, not loaded or fetched."""


# The letter-readout families by the ``prompt`` their ``decision_config.json``
# names (pplx's names none).
LETTER_PROMPTS = {None: "pplx", "d3": "d3"}
# The families by the request format (``template``) a ``joint_config.json``
# names, with the whole protocol it must declare (pplx-decider v1.2's; its
# code refuses any other value too).
JOINT_TEMPLATES: dict[str, str] = {"joint-question-inline-readout-v1": "pplx"}
JOINT_PROTOCOLS: dict[str, dict] = {
    "joint-question-inline-readout-v1": {
        "version": 3,
        "template": "joint-question-inline-readout-v1",
        "marker": "<|fim_suffix|>",
        "readout_offset": 0,
        "max_questions": None,
        "max_length": None,
        "usage": "nonpadding_joint_sequence_including_template_and_image_positions",
    },
}


def _letter_family(path: Path) -> str | None:
    """The letter-readout family of a folder, None if it is not one; refuses a
    prompt or request format no family knows, so no model loads with another
    family's prompt."""
    file = path / "decision_config.json"
    if not file.exists():
        return None
    joint = path / "joint_config.json"
    if joint.exists():
        config = json.loads(joint.read_text())
        template = config.get("template") if isinstance(config, dict) else None
        if template not in JOINT_TEMPLATES:
            raise UnknownFormat(
                f"{path}: joint_config.json names the request format {template!r}, "
                "which no family here knows"
            )
        protocol = JOINT_PROTOCOLS[template]
        different = sorted(
            k for k in protocol.keys() | config.keys() if config.get(k, ...) != protocol.get(k, ...)
        )
        if different:
            raise UnknownFormat(
                f"{path}: joint_config.json declares {template!r} with other settings "
                f"({', '.join(different)}) than the version known here"
            )
        return JOINT_TEMPLATES[template]
    prompt = json.loads(file.read_text()).get("prompt")
    if prompt not in LETTER_PROMPTS:
        raise UnknownFormat(
            f"{path}: decision_config.json names the prompt {prompt!r}, which no family here knows"
        )
    return LETTER_PROMPTS[prompt]


register_family(
    "pplx",
    # The JSON files alone, so that ``download`` recognises the repository.
    detect=lambda path: _letter_family(path) == "pplx",
    loader="mlx_decision.models.pplx.model:load",
    capabilities="mlx_decision.models.pplx.model:capabilities",
    listing=Listing(
        "pplx", "Perplexity", "whole requests in one pass, 4B and 27B; text and images"
    ),
    converter="mlx_decision.models.pplx.convert:convert",
    # v1.1 (``perplexity-ai/pplx-decider-v1.1-27b``) still downloads by repo id.
    known_models=(
        KnownModel(
            "perplexity-ai/pplx-decider-flash-v1.2",
            "pplx decider flash: Perplexity's 4B decision model (Qwen3.5), text and images",
            "9.1 GB",
        ),
        KnownModel(
            "perplexity-ai/pplx-decider-v1.2",
            "pplx decider: Perplexity's 27B decision model (Qwen3.5), text and images; "
            "below 96 GB of memory it runs quantized (offered after picking it)",
            "52.2 GB",
        ),
    ),
    # Not the training and evaluation files or the reference code.
    files=(
        "config.json",
        "decision_config.json",
        "joint_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model*.safetensors",
        "readout.safetensors",
        "LICENSE",
        "NOTICE",
        "README.md",
    ),
    weight_files="mlx_decision.models.letter:weight_files",
    quantizable=_qwen_text_matrix,
)


register_family(
    "d3",
    # The JSON files alone, so that ``download`` recognises the repository.
    detect=lambda path: _letter_family(path) == "d3",
    loader="mlx_decision.models.d3.model:load",
    capabilities="mlx_decision.models.d3.model:capabilities",
    listing=Listing("d3", "vLLM Semantic Router", "pplx's design, 0.6B to 27B; text and images"),
    converter="mlx_decision.models.d3.convert:convert",
    known_models=(
        KnownModel(
            "vllm-sr/d3-flash",
            "d3 flash: vLLM Semantic Router's 9B decision model (Qwen3.5), text and images",
            "16.8 GB",
        ),
        KnownModel(
            "vllm-sr/d3",
            "d3: vLLM Semantic Router's 27B decision model (Qwen3.5), text and images; "
            "below 96 GB of memory it runs quantized (offered after picking it)",
            "52.2 GB",
        ),
        KnownModel(
            "vllm-sr/d3-mini",
            "d3 mini: 4B (Qwen3.5), text and images",
            "9.1 GB",
        ),
        KnownModel(
            "vllm-sr/d3-nano",
            "d3 nano: 2B (Qwen3.5), text and images",
            "4.4 GB",
        ),
        KnownModel(
            "vllm-sr/d3-lite",
            "d3 lite: 0.8B (Qwen3.5), text and images",
            "1.7 GB",
        ),
        KnownModel(
            "vllm-sr/d3-edge",
            "d3 edge: 0.6B (Qwen3.5), text and images",
            "1.2 GB",
        ),
    ),
    # Not the release's own runtime, server, manifest or card images.
    files=(
        "config.json",
        "decision_config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model*.safetensors",
        "readout.safetensors",
        "LICENSE",
        "NOTICE",
        "README.md",
    ),
    weight_files="mlx_decision.models.letter:weight_files",
    quantizable=_qwen_text_matrix,
)


def _h2o_detect(path: Path) -> bool:
    """``serve_config.json`` with the option labels (the JSON alone, for ``download``)."""
    file = path / "serve_config.json"
    if not file.exists():
        return False
    try:
        config = json.loads(file.read_text())
    except ValueError:
        return False
    return isinstance(config, dict) and isinstance(config.get("labels"), list)


register_family(
    "h2o",
    detect=_h2o_detect,
    loader="mlx_decision.models.h2o.model:load",
    capabilities="mlx_decision.models.h2o.model:capabilities",
    listing=Listing("H2O-Lightning", "H2O.ai", "label tokens over Qwen3.5; text and images"),
    converter="mlx_decision.models.h2o.convert:convert",
    known_models=(
        KnownModel(
            "h2oai/h2o-lightning-4b",
            "H2O-Lightning: H2O.ai's 4B decision model (Qwen3.5), text and images",
            "9.3 GB",
        ),
    ),
    # Not the serving shim, its tests, the adapter or the chart images.
    files=(
        "config.json",
        "serve_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "model*.safetensors",
        "LICENSE",
        "README.md",
    ),
    weight_files="mlx_decision.models.h2o.model:weight_files",
    quantizable=_qwen_text_matrix,
)


def _d1_detect(path: Path) -> bool:
    """An LFM2-VL ``config.json`` whose ``auto_map`` names the d1 model (the JSON alone,
    for ``download``)."""
    file = path / "config.json"
    if not file.exists():
        return False
    try:
        config = json.loads(file.read_text())
    except ValueError:
        return False
    if not isinstance(config, dict) or config.get("model_type") != "lfm2_vl":
        return False
    auto_map = config.get("auto_map")
    return isinstance(auto_map, dict) and str(auto_map.get("AutoModel", "")).endswith("D1Model")


register_family(
    "d1",
    detect=_d1_detect,
    loader="mlx_decision.models.d1.model:load",
    capabilities="mlx_decision.models.d1.model:capabilities",
    listing=Listing("d1", "Liquid AI", "LFM2.5 with a decision readout; text"),
    known_models=(
        KnownModel(
            "LiquidAI/d1-3B",
            "d1: LiquidAI's 3B decision model (LFM2.5), text",
            "6.3 GB",
        ),
    ),
    licence=(
        "d1's weights are under the LFM Open License v1.0: commercial use is licensed only "
        "to organisations with less than $10 million in annual revenue (see the LICENSE "
        "file of the model's repository)."
    ),
    # Not the release's code, chat template or image settings (text only).
    files=(
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model*.safetensors",
        "LICENSE",
        "README.md",
    ),
)

# The encoder and tokenizer folders plus the files at the top; the Laya repo
# also holds other checkpoints in sub-folders, Julia's its PyTorch code.
_MARKER_FILES = ("model.safetensors", "encoder/*", "tokenizer/*", "README.md", "LICENSE*")

register_family(
    "laya",
    detect=lambda path: (path / "rl_agent_config.json").exists(),
    loader="mlx_decision.models.marker.model:load_laya",
    capabilities="mlx_decision.models.marker.model:laya_capabilities",
    listing=Listing("Laya", "ConvAI Innovations", "small encoders (ModernBERT, mmBERT); text"),
    known_models=(
        KnownModel(
            "convaiinnovations/laya",
            "Laya: English decision model (ModernBERT-large, 421M), calibrated",
            "843 MB",
        ),
        KnownModel(
            "convaiinnovations/laya-multilingual",
            "Laya multilingual: 100+ languages (mmBERT-base, 322M)",
            "678 MB",
        ),
        KnownModel(
            "convaiinnovations/laya-typed-decisions",
            "Laya fine-tuned for typed-decision workflows (ModernBERT-large)",
            "846 MB",
        ),
    ),
    files=(*_MARKER_FILES, "rl_agent_config.json"),
    weight_files="mlx_decision.models.marker.model:weight_files",
)

register_family(
    "julia",
    detect=lambda path: (path / "julia_config.json").exists(),
    loader="mlx_decision.models.marker.model:load_julia",
    capabilities="mlx_decision.models.marker.model:julia_capabilities",
    listing=Listing("Julia", "Supersonic Labs", "small multilingual encoder (mmBERT); text"),
    known_models=(
        KnownModel(
            "SupersonicLabs/Julia-1",
            "Julia 1: multilingual decision model (mmBERT-small, 144M), 2-20 options",
            "612 MB",
        ),
    ),
    files=(*_MARKER_FILES, "config.json", "julia_config.json", "inference-policy.json"),
    weight_files="mlx_decision.models.marker.model:weight_files",
)
