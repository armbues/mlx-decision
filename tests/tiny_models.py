"""Tiny random models with the real file layout, for tests without weights."""

import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer, models, pre_tokenizers

from mlx_decision.backbones.qwen3_5.load import save_text_model
from mlx_decision.backbones.qwen3_5.qwen3_5 import Model, ModelArgs
from mlx_decision.models.clef.head import JointSchemaHead

QWEN3_5 = {
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 256,
        "tie_word_embeddings": False,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000,
            "rope_type": "default",
        },
    },
}
CLEF_HEAD = {
    "hidden_size": 64,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 4,
    "feedforward": 64,
}
WORDS = ["[UNK]", "yes", "no", "billing", "technical", "sales", "STATE:", "OPTION"]
WORDS += ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
VISION = {
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_heads": 4,
    "in_channels": 3,
    "patch_size": 4,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 64,
    "num_position_embeddings": 36,
    "hidden_act": "gelu_pytorch_tanh",
}
# Patches of 4 x 4 merged 2 x 2: an image costs (H/8) x (W/8) tokens, 1 to 64.
PROCESSOR = {
    "image_processor": {
        "patch_size": 4,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "size": {"shortest_edge": 64, "longest_edge": 4096},
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255,
    }
}


def qwen3_5(seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(QWEN3_5))
    mx.eval(model.parameters())
    return model


def write_clef(folder: Path, seed: int = 0, vision: bool = False) -> Path:
    """A Clef release folder: tiny backbone, head and a word-level tokenizer.

    With ``vision``, the folder also has a tiny vision tower in its own
    shard and a ``processor_config.json``, as the release does.
    """
    config = {**QWEN3_5, "vision_config": VISION} if vision else QWEN3_5
    save_text_model(qwen3_5(seed), folder, config)
    if vision:
        write_vision(folder, seed)
    head = JointSchemaHead(**CLEF_HEAD)
    mx.eval(head.parameters())
    head.save_weights(str(folder / "joint_head.safetensors"))
    (folder / "joint_head_config.json").write_text(json.dumps(CLEF_HEAD))
    tokenizer = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(folder / "tokenizer.json"))
    (folder / "LICENSE").write_text("test licence\n")
    return folder


def write_vision(folder: Path, seed: int) -> None:
    from mlx_decision.backbones.qwen3_5.vision import VisionArgs, VisionModel, save_vision_model

    mx.random.seed(seed + 1)
    model = VisionModel(VisionArgs.from_config({"vision_config": VISION}))
    mx.eval(model.parameters())
    save_vision_model(model, folder)
    (folder / "processor_config.json").write_text(json.dumps(PROCESSOR))


PPLX_TEXT = {
    **QWEN3_5["text_config"],
    "num_hidden_layers": 8,
    "head_dim": 64,
    "vocab_size": 248320,  # the release's, so real token ids can be fed in
    "rope_parameters": {**QWEN3_5["text_config"]["rope_parameters"], "mrope_section": [3, 3, 2]},
}
PPLX_VISION = {
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "out_hidden_size": 64,
    "num_position_embeddings": 16,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
    "model_type": "qwen3_5_vision",
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
}
PPLX_CONFIG = {
    "architectures": ["Qwen3_5Model"],
    "model_type": "qwen3_5",
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "tie_word_embeddings": False,
    "text_config": {
        **PPLX_TEXT,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "mtp_num_hidden_layers": 0,
        "attn_output_gate": True,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
    },
    "vision_config": PPLX_VISION,
}
# The release's small files that describe its prompt and answers.
PPLX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "decision_config.json",
)


PPLX_CODES = [chr(65 + i) for i in range(26)]
PPLX_CODES += [a + b for a in PPLX_CODES for b in PPLX_CODES][: 255 - 26]
PPLX_SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
PPLX_SPECIAL += ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
PPLX_WORDS = ["[UNK]", "system", "user", "assistant", "State", "Question", "Options", ":"]
PPLX_WORDS += ["billing", "technical", "sales", "refund", "please", "crash", "now"]


def write_pplx_tokenizer(folder: Path) -> None:
    """A word-level tokenizer and decision config for a pplx folder without release files."""
    words = PPLX_WORDS + PPLX_SPECIAL + PPLX_CODES
    tokenizer = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(PPLX_SPECIAL)
    tokenizer.save(str(folder / "tokenizer.json"))
    config = {
        "format_version": 1,
        "codes": PPLX_CODES,
        "token_ids": [words.index(code) for code in PPLX_CODES],
        "temperature": 1.0087417621345625,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
    }
    (folder / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _random(name: str, shape: tuple[int, ...], seed: int):
    """Weights drawn with NumPy, seeded by name, so they never depend on the MLX version."""
    import zlib

    import numpy as np

    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    if len(shape) == 1:
        return rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    return rng.normal(0, 0.05, shape).astype(np.float32)


def write_pplx(folder: Path, seed: int = 0, files_from: Path | None = None) -> Path:
    """A pplx release folder: tiny backbone with vision tower and readout, float32.

    Names and layouts are the release's (``language_model.*`` and
    ``visual.*`` without a ``model.`` prefix, PyTorch's convolution kernels,
    no ``lm_head``), the vision tower shares the first shard with the
    embeddings, as in the release. The vocabulary has the release's size;
    ``files_from`` (a release folder) supplies the tokenizer, processor and
    decision config. Without it the folder gets a word-level tokenizer
    (``PPLX_WORDS``, the prompt's special tokens and the answer codes) and a
    decision config with the release's codes and temperature.
    """
    import shutil

    from mlx.utils import tree_flatten

    from mlx_decision.backbones.qwen3_5.vision import VisionArgs, VisionModel

    # Lazily built: only the names and shapes are used, nothing is allocated.
    text = Model(ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": PPLX_TEXT}))
    vision = VisionModel(VisionArgs.from_config(PPLX_CONFIG))
    shapes = {}
    for name, value in tree_flatten(text.parameters()):
        if name.startswith("language_model.lm_head"):
            continue
        shape = tuple(value.shape)
        if "conv1d.weight" in name:
            shape = (shape[0], shape[2], shape[1])
        shapes[name.replace("language_model.model.", "language_model.", 1)] = shape
    for name, value in tree_flatten(vision.parameters()):
        shape = tuple(value.shape)
        if name == "patch_embed.proj.weight":
            v = PPLX_VISION
            patch = (v["in_channels"], v["temporal_patch_size"], v["patch_size"], v["patch_size"])
            shape = (shape[0], *patch)
        shapes[f"visual.{name}"] = shape

    first = {n for n in shapes if n.startswith(("visual.", "language_model.embed_tokens."))}
    shards = {
        "model-00001-of-00002.safetensors": sorted(first),
        "model-00002-of-00002.safetensors": sorted(set(shapes) - first),
    }
    folder.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    total = 0
    for file, names in shards.items():
        tensors = {name: mx.array(_random(name, shapes[name], seed)) for name in names}
        mx.save_safetensors(str(folder / file), tensors, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(names, file))
        total += sum(t.nbytes for t in tensors.values())
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (folder / "config.json").write_text(json.dumps(PPLX_CONFIG, indent=2) + "\n")
    readout = _random("readout", (255, PPLX_TEXT["hidden_size"]), seed) * 3
    mx.save_safetensors(
        str(folder / "readout.safetensors"), {"weight": mx.array(readout)}, {"format": "pt"}
    )
    if files_from:
        for name in PPLX_FILES:
            shutil.copy(Path(files_from) / name, folder / name)
    else:
        write_pplx_tokenizer(folder)
    (folder / "LICENSE").write_text("test licence\n")
    return folder


MODERNBERT = {
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "local_attention": 8,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
}
MARKER_WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "question", ":", "choice"]
MARKER_WORDS += ["score", "noul", "level", "false", "true", "billing", "technical", "other"]
MARKER_WORDS += ["refund", "please", "crash", "now", "0", "1", "2"]
MARKER_TOKENS = {"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]"}
MARKER_TOKENS["pad_token"] = "[PAD]"


def write_marker(folder: Path, family: str, seed: int = 0, max_length: int = 48) -> Path:
    """A Laya or Julia release folder: tiny encoder and head, word-level tokenizer.

    The question and its options get 24 tokens, the whole sequence ``max_length``.
    """
    from mlx.utils import tree_flatten

    from mlx_decision.backbones.modernbert import Model as Encoder
    from mlx_decision.backbones.modernbert import ModelArgs as EncoderArgs
    from mlx_decision.models.marker.head import MarkerHead

    mx.random.seed(seed)
    encoder = Encoder(EncoderArgs.from_config(MODERNBERT))
    head = MarkerHead(MODERNBERT["hidden_size"], 1)
    mx.eval(encoder.parameters(), head.parameters())
    weights = {f"encoder.{k}": v for k, v in tree_flatten(encoder.parameters())}
    for key, value in tree_flatten(head.parameters()):
        # The release names, as PyTorch writes them.
        key = key.replace("self_attn.in_proj.", "self_attn.in_proj_")
        key = key.replace("layers.", "head.layers.", 1) if key.startswith("layers.") else key
        for ours, theirs in (("scorer_norm", "scorer.0"), ("scorer_in", "scorer.1")):
            key = key.replace(ours, theirs)
        weights[key.replace("scorer_out", "scorer.3")] = value
    weights["temperature"] = mx.ones(3)
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "model.safetensors"), weights)
    (folder / "encoder").mkdir(exist_ok=True)
    (folder / "encoder" / "config.json").write_text(json.dumps(MODERNBERT))
    (folder / "tokenizer").mkdir(exist_ok=True)
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(MARKER_WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(folder / "tokenizer" / "tokenizer.json"))
    (folder / "tokenizer" / "tokenizer_config.json").write_text(json.dumps(MARKER_TOKENS))
    if family == "laya":
        config = {
            "head_layers": 1,
            "max_len": max_length,
            "head_max_len": 24,
            "temperature": [2.0, 1.0, 0.5],
            "temperature_by_options": {"choice:3-5": 1.5},
        }
        (folder / "rl_agent_config.json").write_text(json.dumps(config))
    else:
        config = {"format_version": 1, "head_layers": 1, "n_act": 2}
        (folder / "julia_config.json").write_text(json.dumps(config))
        policy = {"max_length": max_length, "head_length": 24, "strict_encoding": True}
        (folder / "inference-policy.json").write_text(json.dumps(policy))
    return folder
