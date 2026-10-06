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
