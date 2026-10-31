"""How a Base Labs folder says it was trained, read from its JSON files alone.

Detection uses it too, so it imports nothing heavy.
"""

import json
from pathlib import Path

OVERRIDES = "modeling_overrides"
# The mask per layer type the models were trained with.
ATTENTION = {"full_attention": "block_bidirectional", "linear_attention": "causal"}
READOUT = "slot_lm"
# Files of OneJev's other readouts and calibration, which no Base Labs model uses.
OTHER_READOUTS = {
    "readout_head.pt": "a trained pointer-head readout",
    "calibration.json": "a calibration file",
}


def overrides(path: Path) -> dict | None:
    """``config.json``'s ``modeling_overrides`` of a Qwen3.5 folder, if it names a readout."""
    file = path / "config.json"
    if not file.exists():
        return None
    try:
        config = json.loads(file.read_text())
    except ValueError:
        return None
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        return None
    found = config.get(OVERRIDES)
    return found if isinstance(found, dict) and "readout" in found else None


def check_config(path: Path) -> None:
    """Refuse a folder trained or served another way than Base Labs' models, naming
    the setting: its attention per layer type, its readout, and the files of
    OneJev's other readout and of a calibration."""
    found = overrides(path)
    if found is None:
        raise ValueError(f"{path}: config.json has no {OVERRIDES} with a readout")
    attention = found.get("attention")
    if not isinstance(attention, dict):
        raise ValueError(f"{path}: config.json's {OVERRIDES}.attention must be an object")
    for layer_type in sorted(attention.keys() | ATTENTION.keys()):
        value = attention.get(layer_type)
        if value != ATTENTION.get(layer_type):
            expected = ATTENTION.get(layer_type)
            known = f"only {expected!r} is supported" if expected else "no such layer type here"
            raise ValueError(
                f"{path}: config.json's {OVERRIDES}.attention.{layer_type} is {value!r}; {known}"
            )
    if found["readout"] != READOUT:
        raise ValueError(
            f"{path}: config.json's {OVERRIDES}.readout is {found['readout']!r}; "
            f"only {READOUT!r} is supported"
        )
    if not (path / "readout.json").exists():
        raise FileNotFoundError(f"{path}: readout.json is missing")
    readout = json.loads((path / "readout.json").read_text())
    if not isinstance(readout, dict):
        raise ValueError(f"{path}: readout.json must be an object")
    if readout.get("readout", READOUT) != READOUT:
        raise ValueError(
            f"{path}: readout.json's readout is {readout['readout']!r}; "
            f"only {READOUT!r} is supported"
        )
    if readout.get("attention", "noncausal") != "noncausal":
        raise ValueError(
            f"{path}: readout.json's attention is {readout['attention']!r}, but config.json's "
            f"{OVERRIDES} says block-bidirectional"
        )
    for name, what in OTHER_READOUTS.items():
        if (path / name).exists():
            raise ValueError(f"{path}: {name} is {what}, which this family does not read")
