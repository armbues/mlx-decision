"""The Clef backend: Qwen3.5 text model plus joint schema head."""

import json
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from ...backbones.qwen3_5.load import load_text_model
from ...backend import BackendOutput, Capabilities
from ...types import Request
from .encode import DEFAULT_MAX_LENGTH, encode_request
from .head import JointSchemaHead


class ClefBackend:
    def __init__(self, name, backbone, head, tokenizer, max_input_tokens=DEFAULT_MAX_LENGTH):
        self.name = name
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.capabilities = Capabilities(
            max_input_tokens=max_input_tokens,
            # The release documents no limits; these are the Jev API's.
            max_choice_options=255,
            max_score_levels=10,
        )

    def score(self, request: Request) -> BackendOutput:
        encoded = encode_request(
            self.tokenizer, request, max_length=self.capabilities.max_input_tokens
        )
        input_ids = mx.array(encoded.input_ids)
        hidden = self.backbone(input_ids[None])[0]
        logits = self.head(
            hidden,
            input_ids,
            encoded.questions,
            self.backbone.language_model.output_embedding_rows,
        )
        probabilities = [mx.softmax(values.astype(mx.float32), axis=-1) for values in logits]
        mx.eval(probabilities)
        return BackendOutput(
            probabilities={
                question.question_id: dict(zip(question.option_ids, values.tolist(), strict=True))
                for question, values in zip(encoded.questions, probabilities, strict=True)
            },
            input_tokens=len(encoded.input_ids),
            truncated=encoded.truncated,
        )


def load(path: Path, max_input_tokens: int = DEFAULT_MAX_LENGTH) -> ClefBackend:
    backbone = load_text_model(path)
    head = JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_weights(str(path / "joint_head.safetensors"), strict=True)
    head.eval()
    mx.eval(head.parameters())
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    return ClefBackend(path.resolve().name, backbone, head, tokenizer, max_input_tokens)
