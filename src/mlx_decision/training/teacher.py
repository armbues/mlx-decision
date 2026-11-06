"""Teacher labels: a model's answers as another model's training labels.

A *soft* label is the answer's whole distribution (the training format's
soft labels: an object from option to probability, a noul's probability of
true), a *hard* one its top answer. Training on soft labels passes the
teacher's uncertainty on, and with it its calibration: a student fitted on a
teacher whose probabilities are off is off the same way. ``describe_teacher``
reads what a model folder says about that (a fitted model's closing
calibration and the ECE its report measured) and ``teacher_note`` says when
to fit the teacher first.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..fitted import is_fitted, read_marker

# A teacher whose answers are off by more than this ECE gets a note: its
# students inherit it (a teacher with a closing calibration is at about 2%).
ECE_NOTE = 0.05
LABELS = ("soft", "hard")


def soft_label(answer: Mapping[str, Any]) -> Any:
    """An answer (the wire format) as a soft label: a noul's probability of
    true, a choice's or score's probability per option or level."""
    if answer["type"] == "noul":
        return float(answer["noul"])
    return {key: float(value) for key, value in answer["probabilities"].items()}


def target(answer: Mapping[str, Any]) -> dict[str, float]:
    """An answer's distribution as ``Example.targets`` holds a soft label:
    option id (a level as its key, "true" / "false") to probability."""
    if answer["type"] == "noul":
        return {"true": float(answer["noul"]), "false": 1 - float(answer["noul"])}
    return {key: float(value) for key, value in answer["probabilities"].items()}


def hard_label(answer: Mapping[str, Any]) -> Any:
    """An answer as a hard label: true or false, the option, the level."""
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    probabilities = answer["probabilities"]
    top = (
        answer["choice"]
        if answer["type"] == "choice"
        else max(probabilities, key=probabilities.get)
    )
    return int(top) if answer["type"] == "score" else top


def confidence(answer: Mapping[str, Any]) -> float:
    """The answer's confidence, as the Studio's batches filter by it: a
    choice's or score's own (above an even split), a noul's top probability."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return float(answer["confidence"])


def label(answer: Mapping[str, Any], kind: str) -> Any:
    if kind not in LABELS:
        raise ValueError(f"labels are soft or hard, not {kind!r}")
    return soft_label(answer) if kind == "soft" else hard_label(answer)


@dataclass(frozen=True)
class Teacher:
    """What a teacher's folder says: its ``name``; for a fitted model its
    ``base``, ``kinds``, the closing calibration's ``calibration`` choice
    (None: none, or a release's own), the questions it calibrated and per
    question the ECE its report measured after the fit."""

    name: str
    base: str | None = None
    kinds: tuple[str, ...] = ()
    calibration: str | None = None
    calibrated: frozenset[str] = frozenset()
    ece: Mapping[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {"name": self.name}
        if self.base is not None:
            body.update(
                base=self.base,
                kinds=list(self.kinds),
                calibration=self.calibration,
                calibrated=sorted(self.calibrated),
            )
        if self.ece:
            body["ece"] = dict(self.ece)
        return body


def describe_teacher(path: str | Path, name: str | None = None) -> Teacher:
    """The teacher at ``path`` (a model folder), named ``name`` or by its folder."""
    path = Path(path)
    name = name or path.name
    if not is_fitted(path):
        return Teacher(name)
    try:
        marker = read_marker(path)
    except (OSError, ValueError):
        return Teacher(name)
    block = marker.get("fit") or {}
    base = marker.get("base") or {}
    base_name = (base.get("repo_id") or base.get("path") or "").rstrip("/").rsplit("/", 1)[-1]
    kinds = tuple(block.get("kinds") or ())
    closing = block.get("calibration") or {}
    choice = closing.get("choice", "auto" if "calibrate" in kinds else None)
    # The questions the closing calibration covers; a marker without that list
    # (written before it was kept) calibrated every question of its fit.
    listed = closing.get("questions")
    if listed is None:
        listed = [q.get("id") for q in block.get("questions") or [] if isinstance(q, dict)]
    calibrated = frozenset(listed) if choice not in (None, "none") else frozenset()
    ece = {
        question: float(entry["after"]["ece"])
        for question, entry in (block.get("metrics") or {}).items()
        if (entry.get("after") or {}).get("ece") is not None
    }
    return Teacher(
        name,
        base_name or None,
        kinds,
        None if choice == "none" else choice,
        calibrated,
        ece,
    )


def _percents(values: Mapping[str, float]) -> str:
    return ", ".join(f"{question} {100 * value:.1f}%" for question, value in values.items())


def teacher_note(
    teacher: Teacher, questions: Iterable[str], measured: Mapping[str, float] | None = None
) -> str | None:
    """Why a student of ``teacher`` on ``questions`` would inherit poor or
    unknown calibration, or None: the ECE ``measured`` on labelled rows above
    ``ECE_NOTE``; else questions its fit gave no closing calibration (every
    one, for a release); else its fit's report measuring an ECE above it."""
    questions = list(dict.fromkeys(questions))
    advice = (
        f"a student trained on its probabilities inherits that; fit {teacher.name} on "
        "labelled rows first (a calibration alone takes seconds) and label again"
    )
    if measured:
        off = {q: e for q, e in measured.items() if q in questions and e > ECE_NOTE}
        if off:
            where = f"on the labelled rows (ECE {_percents(off)})"
            return f"{teacher.name}'s probabilities are off {where}: {advice}"
        return None
    unknown = [q for q in questions if q not in teacher.calibrated]
    if unknown:
        what = (
            "has no closing calibration"
            if not teacher.calibrated
            else f"has no closing calibration for {', '.join(unknown)}"
        )
        trust = "so how far its probabilities can be trusted is unknown"
        return f"{teacher.name} {what}, {trust}: {advice}"
    off = {q: teacher.ece[q] for q in questions if teacher.ece.get(q, 0) > ECE_NOTE}
    if off:
        return f"{teacher.name}'s fit left an ECE of {_percents(off)} on its report: {advice}"
    return None
