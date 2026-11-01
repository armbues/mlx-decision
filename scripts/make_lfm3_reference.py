"""The LFM3 backbone as transformers runs it: tests/parity/d1lfm3/backbone.json.

    python scripts/make_lfm3_reference.py

Fetches the transformers LFM3 code that d1-2B's release vendors (Apache-2.0,
transformers at 69da1b47; the release's own d1 files are not fetched or run)
at a pinned revision into the Hugging Face cache, imports it as a package,
and runs it on the tiny random LFM3-VL model of
``tests/tiny_models.write_lfm3_vl`` (CPU, float32, eager attention):

- a causal pass over two rows of token ids;
- a pass under a set layout: two right-padded rows, each a state and its
  questions, with the masks mlx-decision builds (on full-attention layers a
  state token sees every real token of its row, a question token the state
  and its own question; sliding layers causal within the window), given to
  transformers as one mask per layer type;
- the vision tower and projector on random patches of two images.

Hidden states of real tokens, tower and projector outputs go into the
fixture with the inputs (a few seconds). Needs torch and transformers.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

REPO = "LiquidAI/d1-2B"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"
# transformers' LFM3 code as the release vendors it (each file says so).
SOURCES = (
    "configuration_lfm3.py",
    "configuration_lfm3_vl.py",
    "integrations_lfm3.py",
    "modeling_lfm3.py",
    "modeling_lfm3_vl.py",
)
OUT = ROOT / "tests" / "parity" / "d1lfm3" / "backbone.json"
SEED = 0
# Two rows of a set layout: the state's tokens, then each question's.
LAYOUT = [(10, [6, 8]), (10, [12])]
CAUSAL_TOKENS = 24
GRIDS = [[1, 4, 6], [1, 2, 2]]


def fetch() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    files = {}
    for name in SOURCES:
        path = Path(hf_hub_download(REPO, name, revision=REVISION))
        if "Vendored from transformers' LFM3 code" not in path.read_text().splitlines()[0]:
            raise SystemExit(f"{name} is not the vendored transformers code")
        files[name] = path
    return files


def import_sources(files: dict[str, Path], into: Path):
    """The vendored files as a package (their relative imports need one)."""
    import importlib

    package = into / "lfm3_reference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    for name, path in files.items():
        shutil.copy(path, package / name)
    sys.path.insert(0, str(into))
    configuration = importlib.import_module("lfm3_reference.configuration_lfm3")
    configuration_vl = importlib.import_module("lfm3_reference.configuration_lfm3_vl")
    modeling = importlib.import_module("lfm3_reference.modeling_lfm3")
    modeling_vl = importlib.import_module("lfm3_reference.modeling_lfm3_vl")
    from transformers import AutoConfig, AutoModel

    # What the release's own module does for its model types.
    for config, model in (
        (configuration.Lfm3Config, modeling.Lfm3Model),
        (configuration_vl.Lfm3VlVisionConfig, modeling_vl.Lfm3VlVisionModel),
    ):
        AutoConfig.register(config.model_type, config, exist_ok=True)
        AutoModel.register(config, model, exist_ok=True)
    return configuration_vl, modeling_vl


def build(folder: Path, configuration_vl, modeling_vl):
    import torch
    from safetensors.torch import load_file

    config = configuration_vl.Lfm3VlConfig(**json.loads((folder / "config.json").read_text()))
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = modeling_vl.Lfm3VlForConditionalGeneration(config).to(torch.float32).eval()
    missing, unexpected = model.load_state_dict(
        load_file(str(folder / "model.safetensors")), strict=False
    )
    if unexpected or set(missing) - {"lm_head.weight"}:
        raise SystemExit(f"weights do not fit: missing {missing}, unexpected {unexpected}")
    return model


def set_masks(layout, window: int):
    """Masks per layer type for right-padded rows of packed sets (additive, float)."""
    import torch

    length = max(state + sum(questions) for state, questions in layout)
    rows = len(layout)
    question = torch.full((rows, length), -1)
    real = torch.zeros((rows, length), dtype=torch.bool)
    states = torch.tensor([state for state, _ in layout])
    for row, (state, questions) in enumerate(layout):
        start = state
        for index, size in enumerate(questions):
            question[row, start : start + size] = index
            start += size
        real[row, :start] = True
    q = torch.arange(length)[None, :, None]
    k = torch.arange(length)[None, None, :]
    s = states[:, None, None]
    own = question[:, :, None]
    same = (own == question[:, None, :]) & (own >= 0)
    full = ((q < s) | (k < s) | same) & real[:, None, :]
    sliding = ((k <= q) & (k > q - window)).expand(rows, -1, -1)

    def additive(mask):
        return torch.where(mask, 0.0, torch.finfo(torch.float32).min)[:, None]

    masks = {
        "full_attention": additive(full),
        "sliding_attention": additive(sliding),
        "conv": real.long(),
    }
    return masks, real, question, length


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    import numpy as np
    import torch

    from tiny_models import write_lfm3_vl

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    files = fetch()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        configuration_vl, modeling_vl = import_sources(files, tmp / "src")
        folder = write_lfm3_vl(tmp / "tiny-lfm3", seed=SEED)
        model = build(folder, configuration_vl, modeling_vl)
        text = json.loads((folder / "config.json").read_text())["text_config"]
        language = model.model.language_model
        vocab = 300  # below the image and video tokens

        with torch.no_grad():
            causal_ids = rng.integers(1, vocab, (2, CAUSAL_TOKENS)).tolist()
            causal = language(input_ids=torch.tensor(causal_ids), use_cache=False)
            causal_hidden = causal.last_hidden_state

            masks, real, question, length = set_masks(LAYOUT, text["sliding_window"])
            set_ids = rng.integers(1, vocab, (len(LAYOUT), length)).tolist()
            out = language(input_ids=torch.tensor(set_ids), attention_mask=masks, use_cache=False)
            set_hidden = out.last_hidden_state

            patches = sum(t * h * w for t, h, w in GRIDS)
            vision = json.loads((folder / "config.json").read_text())["vision_config"]
            width = (
                vision["in_channels"] * vision["temporal_patch_size"] * vision["patch_size"] ** 2
            )
            pixels = rng.normal(0, 1, (patches, width)).astype(np.float32)
            grid = torch.tensor(GRIDS)
            tower = model.model.vision_tower(torch.from_numpy(pixels), grid_thw=grid)
            features = model.model.multi_modal_projector(tower.last_hidden_state)

    lengths = real.sum(1).tolist()
    fixture = {
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "files": {
                name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()
            },
            "torch": torch.__version__,
        },
        "seed": SEED,
        "causal": {"ids": causal_ids, "hidden": causal_hidden.tolist()},
        "set": {
            "ids": set_ids,
            "lengths": lengths,
            "state": [state for state, _ in LAYOUT],
            "question": question.tolist(),
            # Real tokens only: padding positions are not defined to match.
            "hidden": [set_hidden[row, :n].tolist() for row, n in enumerate(lengths)],
        },
        "vision": {
            "grids": GRIDS,
            "pixels": pixels.tolist(),
            "tower": tower.last_hidden_state.tolist(),
            "features": features.tolist(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
