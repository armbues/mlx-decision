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


def qwen3_5(seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict(QWEN3_5))
    mx.eval(model.parameters())
    return model


def write_clef(folder: Path, seed: int = 0) -> Path:
    """A Clef release folder: tiny backbone, head and a word-level tokenizer."""
    save_text_model(qwen3_5(seed), folder, QWEN3_5)
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
