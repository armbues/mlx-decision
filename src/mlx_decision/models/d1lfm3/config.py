"""How a d1 LFM3 folder (d1-2B) is set up, read from its JSON files alone.

Detection and static capabilities use it too, so it imports nothing heavy.
The ``d1_*`` fields of ``config.json`` say how the release serves the model;
only the release's settings are taken, anything else is refused naming the
field. ``calibration.json`` holds the temperature the answers are read with.
"""

import json
from dataclasses import dataclass
from pathlib import Path

ATTENTION = "set"
CODES = 2
# The release's cap on pictures, when the config gives none.
VISION_MAX_PIXELS = 1638400
# Answer types this package asks; a per-type temperature for one of them is
# not something the release ships.
TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class Settings:
    set_size: int
    max_pixels: int
    temperature: float


def read_json(file: Path) -> dict | None:
    if not file.exists():
        return None
    try:
        data = json.loads(file.read_text())
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def is_d1lfm3(path: Path) -> bool:
    """An LFM3-VL ``config.json`` whose ``auto_map`` names the d1 model."""
    config = read_json(path / "config.json")
    if config is None or config.get("model_type") != "lfm3_vl":
        return False
    auto_map = config.get("auto_map")
    return isinstance(auto_map, dict) and str(auto_map.get("AutoModel", "")).endswith("D1Model")


def settings(path: Path) -> Settings:
    """The folder's serving settings, refusing any the release does not use."""
    config = read_json(path / "config.json")
    if config is None:
        raise ValueError(f"{path}: config.json is missing or not a JSON object")
    attention = config.get("d1_attention")
    if attention != ATTENTION:
        raise ValueError(
            f"{path}: d1_attention {attention!r} is not supported (only {ATTENTION!r})"
        )
    codes = config.get("d1_codes")
    if codes != CODES:
        raise ValueError(f"{path}: d1_codes {codes!r} is not supported (only {CODES})")
    set_size = config.get("d1_set_size")
    if isinstance(set_size, bool) or not isinstance(set_size, int) or set_size < 1:
        raise ValueError(f"{path}: d1_set_size {set_size!r} is not a whole number from 1")
    max_pixels = config.get("d1_vision_max_pixels", VISION_MAX_PIXELS)
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < 1:
        raise ValueError(f"{path}: d1_vision_max_pixels {max_pixels!r} is not a pixel count")
    calibration = read_json(path / "calibration.json")
    if calibration is None:
        raise ValueError(f"{path}: calibration.json is missing or not a JSON object")
    temperature = calibration.get("default")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or temperature <= 0
    ):
        raise ValueError(f"{path}: calibration.json has no positive default temperature")
    own = calibration.get("temperatures") or {}
    if not isinstance(own, dict):
        raise ValueError(f"{path}: calibration.json's temperatures are not an object")
    if typed := sorted(set(own) & set(TYPES)):
        raise ValueError(
            f"{path}: calibration.json has its own temperature for {', '.join(typed)} "
            "(not supported: the release reads these with the default)"
        )
    return Settings(set_size, max_pixels, float(temperature))
