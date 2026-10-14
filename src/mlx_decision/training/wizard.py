"""The ``training fit`` and ``training eval`` menus: what the flags leave open, asked.

``training fit``: the base and the data if missing, then the kind (a menu of
the kinds the base's family takes, each with its estimated peak memory for
this data, ``auto``'s pick first), the holdout share or a
validation file, the epochs (for the kinds that train in epochs) and the
output folder; then a summary with the command line that does the same
without menus, and a confirmation. ``training eval`` asks only for a missing
model or data file. Everything lands in the same ``FitPlan`` the flags fill.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import Validator

from ..flags import command_line
from ..interactive import ask, confirm, pick, say
from . import plan as kinds_plan
from .console import plan_lines

EPOCH_KINDS = ("head", "finetune")
DEFAULT_HOLDOUT = 0.2
DEFAULT_EPOCHS = 3


class Cancelled(Exception):
    """The menus were left (Ctrl-C, Ctrl-D, or no at the end)."""


@dataclass
class FitPlan:
    """What a ``training fit`` run does, as its flags say it (None: left open)."""

    model: str | None = None
    data: str | None = None
    output: str | None = None
    method: str | None = None
    validation: str | None = None
    holdout: float | None = None
    seed: int = 0
    epochs: int | None = None
    head_only: bool = False
    early_stop: bool = True
    load_options: dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False

    def is_open(self) -> bool:
        """Whether something has no value yet (the menus ask for it)."""
        return (
            self.model is None
            or self.data is None
            or self.method is None
            or (self.validation is None and self.holdout is None)
            or self.output is None
        )

    def with_defaults(self, default_output) -> "FitPlan":
        """Open values filled with the defaults (``auto``, a 0.2 holdout, 3 epochs,
        ``default_output(model, data)``); model and data stay as they are."""
        self.method = self.method or "auto"
        if self.validation is None and self.holdout is None:
            self.holdout = DEFAULT_HOLDOUT
        self.epochs = self.epochs or DEFAULT_EPOCHS
        if self.output is None and self.model is not None and self.data is not None:
            self.output = str(default_output(self.model, Path(self.data)))
        return self

    def arguments(self) -> list[str]:
        words = ["-m", self.model, "--data", self.data, "-o", self.output]
        if self.method not in (None, "auto"):
            words += ["--method", self.method]
        if self.validation is not None:
            words += ["--validation", self.validation]
        elif self.holdout not in (None, DEFAULT_HOLDOUT):
            words += ["--holdout", f"{self.holdout:g}"]
        if self.seed:
            words += ["--seed", str(self.seed)]
        if self.epochs not in (None, DEFAULT_EPOCHS):
            words += ["--epochs", str(self.epochs)]
        if self.head_only:
            words.append("--head-only")
        if not self.early_stop:
            words.append("--no-early-stop")
        if "max_input_tokens" in self.load_options:
            words += ["--max-input-tokens", str(self.load_options["max_input_tokens"])]
        if self.overwrite:
            words.append("--overwrite")
        return [str(word) for word in words]

    def command(self) -> str:
        return command_line(["mlx-decision", "training", "fit", *self.arguments()])


def _ask(message: str, **options) -> str:
    try:
        return ask(message, **options)
    except (KeyboardInterrupt, EOFError):
        raise Cancelled() from None


def _pick(message: str, options: list, default=None):
    try:
        return pick(message, options, default)
    except (KeyboardInterrupt, EOFError):
        raise Cancelled() from None


def _existing(what: str) -> Validator:
    return Validator.from_callable(
        lambda text: bool(text.strip()) and Path(text.strip()).expanduser().exists(),
        error_message=f"no such {what}",
    )


def ask_model(message: str = "Model (folder or Hugging Face repo id): ") -> str:
    return _ask(
        message,
        completer=PathCompleter(expanduser=True, only_directories=True),
        validator=Validator.from_callable(lambda text: bool(text.strip()), error_message="a model"),
    )


def ask_data_file(message: str = "Labelled data (JSON lines): ") -> str:
    return _ask(message, completer=PathCompleter(expanduser=True), validator=_existing("file"))


def _choices(prepared) -> list[tuple[tuple[str, bool], str]]:
    """The kinds the family takes, ``auto``'s pick first, each with its estimate."""
    from ..memory import working_set

    family = prepared.family
    main = kinds_plan.MAIN_KIND.get(family)
    entries = []
    if main is not None:
        entries.append((main, False))
        if main == "finetune":
            entries.append((main, True))
    entries.append(("calibrate", False))
    limit = working_set()
    options = []
    for method, head_only in entries:
        kinds = kinds_plan.resolve(family, method)
        estimate = prepared.estimate(kinds, head_only)
        name = ", then ".join(kinds) + (" (head only)" if head_only else "")
        if method == main:
            name += "  [auto]" if not head_only else ""
        peak = estimate.peak_bytes / 1e9
        text = f"{name:<36} peak {peak:.1f} GB"
        if estimate.disk_bytes:
            text += f", {estimate.disk_bytes / 1e9:.1f} GB on disk"
        if limit and estimate.peak_bytes > limit:
            text += "  (more than this Mac's working set)"
        options.append(((method, head_only), text))
    return options


def ask_fit(plan: FitPlan, default_output) -> FitPlan:
    """``plan`` with what its flags left open asked, then confirmed. Raises
    ``Cancelled`` when the menus are left or the summary is declined."""
    from .fitting import prepare

    if plan.model is None:
        plan.model = ask_model("Base model (folder, Hugging Face repo id, or a fitted model): ")
    if plan.data is None:
        plan.data = ask_data_file()
    holdout = DEFAULT_HOLDOUT if plan.holdout is None else plan.holdout
    prepared = prepare(
        plan.model,
        plan.data,
        validation=plan.validation,
        holdout=holdout,
        seed=plan.seed,
        load_options=plan.load_options,
    )
    if plan.method is None:
        say(f"{prepared.origin.name} ({prepared.family}); memory estimated for this data:")
        plan.method, plan.head_only = _pick("Method:", _choices(prepared))
    kinds = kinds_plan.resolve(prepared.family, plan.method)
    if plan.validation is None and plan.holdout is None:
        how = _pick(
            "Report on:",
            [
                ("holdout", f"a holdout of {DEFAULT_HOLDOUT:.0%} of the data (stratified)"),
                ("share", "a holdout of another share..."),
                ("file", "a validation file..."),
            ],
        )
        if how == "file":
            plan.validation = ask_data_file("Validation data (JSON lines): ")
        elif how == "share":
            plan.holdout = float(_ask("Share (0.05 to 0.9): ", default="0.2", validator=_share()))
        else:
            plan.holdout = DEFAULT_HOLDOUT
    if plan.epochs is None and kinds[0] in EPOCH_KINDS:
        answer = _ask(
            "Epochs (the best is kept): ", default=str(DEFAULT_EPOCHS), validator=_whole()
        )
        plan.epochs = int(answer)
    if plan.output is None:
        default = str(default_output(plan.model, Path(plan.data)))
        plan.output = _ask(
            "Write the fitted model to: ",
            default=default,
            completer=PathCompleter(expanduser=True, only_directories=True),
        )
        if Path(plan.output).exists():
            if not confirm(f"{plan.output} exists. Replace it?"):
                raise Cancelled()
            plan.overwrite = True
    plan.with_defaults(default_output)
    prepared = prepare(
        plan.model,
        plan.data,
        validation=plan.validation,
        holdout=plan.holdout or DEFAULT_HOLDOUT,
        seed=plan.seed,
        load_options=plan.load_options,
    )
    _summary(plan, prepared, kinds)
    if not confirm("Start?"):
        raise Cancelled()
    return plan


def _summary(plan: FitPlan, prepared, kinds: list[str]) -> None:
    from .fitting import label_counts

    splits = prepared.splits
    estimate = prepared.estimate(kinds, plan.head_only)
    metrics = {
        "examples": len(prepared.source.examples),
        "holdout": len(splits.holdout),
        "fit": len(splits.fit),
        "dev": len(splits.dev),
        "labels": label_counts(splits.fit + splits.dev),
        "kinds": kinds,
        "estimate": estimate.as_json(),
    }
    say("")
    for line in plan_lines(metrics):
        say(line)
    for warning in prepared.warnings:
        say(f"warning: {warning}")
    say(f"base: {prepared.origin.name}; output: {plan.output}")
    say("The same without menus:")
    say("  " + plan.command())


def _share() -> Validator:
    def valid(text: str) -> bool:
        try:
            return 0.05 <= float(text) <= 0.9
        except ValueError:
            return False

    return Validator.from_callable(valid, error_message="a number from 0.05 to 0.9")


def _whole() -> Validator:
    return Validator.from_callable(
        lambda text: text.strip().isdigit() and int(text) >= 1, error_message="1 or more"
    )


def ask_eval(model: str | None, data: str | None) -> tuple[str, str]:
    """A missing model or data file asked; raises ``Cancelled``."""
    return model or ask_model(), data or ask_data_file()
