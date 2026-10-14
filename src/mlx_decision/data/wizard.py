"""The ``data`` menus: what the flags leave open, asked step by step.

Source (a preset, a Hugging Face dataset or a file), config and split
(menus from the Hub), shape and columns (the guess preselected, each
column shown with its kind and a sample), the question (``chat``'s
builder, prefilled from the guess: one option per class, in the classes'
order), and the output file; then a preview, a confirmation, and the
command line that does the same without menus. Everything the menus pick
lands in the same ``DataPlan`` the flags fill.
"""

from pathlib import Path
from typing import Any

from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.validation import Validator

from ..flags import command_line
from ..interactive import ask, build_question, confirm, pick, say
from . import presets as preset_data
from .make import DataPlan, Made
from .shapes import (
    KINDS,
    OPTIONAL,
    ROLES,
    SHAPES,
    candidates,
    class_names,
    default_id,
    guess,
    rating_values,
)
from .sources import (
    HUB_PREFIX,
    SourceError,
    Table,
    hub_configs,
    hub_splits,
    open_source,
    saved_splits,
)

SHAPE_TEXT = {
    "text": "a text and its class",
    "pair": "two texts (premise and hypothesis) and how they relate",
    "choice": "a question and its choices (multiple choice)",
    "yesno": "a passage and a yes/no question about it",
    "rating": "a text and a number on a scale",
}
PRESET_TEXT = {
    "trec": "TREC: the kind of answer a question asks for (6 classes)",
    "offensive": "TweetEval Offensive: is a tweet offensive (yes/no)",
    "sst5": "SST-5: sentiment of movie reviews (5 levels)",
    "openbookqa": "OpenBookQA: science questions (4 choices)",
    "anli-r3": "ANLI round 3: premise and hypothesis (3 classes)",
}
HUB, FILE = "hub", "file"
NONE = "(none)"


class Cancelled(Exception):
    """The menus were left (Ctrl-C, Ctrl-D, or no at the end)."""


def _pick(message: str, options: list[tuple[Any, str]], default=None):
    try:
        return pick(message, options, default)
    except (KeyboardInterrupt, EOFError):
        raise Cancelled() from None


def _ask(message: str, **options) -> str:
    try:
        return ask(message, **options)
    except (KeyboardInterrupt, EOFError):
        raise Cancelled() from None


def _not_empty(what: str) -> Validator:
    return Validator.from_callable(lambda text: bool(text.strip()), error_message=f"{what}, please")


def ask_source(plan: DataPlan) -> None:
    options = [(name, PRESET_TEXT.get(name, name)) for name in preset_data.PRESETS]
    options += [(HUB, "A Hugging Face dataset..."), (FILE, "A file or a saved dataset folder...")]
    picked = _pick("Data:", options)
    if picked == HUB:
        repo = _ask("Hugging Face dataset (ORG/NAME): ", validator=_not_empty("a dataset id"))
        plan.source = repo if repo.startswith(HUB_PREFIX) else HUB_PREFIX + repo
    elif picked == FILE:
        plan.source = _ask(
            "File (.csv, .jsonl, .parquet) or folder: ",
            completer=PathCompleter(expanduser=True),
            validator=Validator.from_callable(
                lambda text: Path(text.strip()).expanduser().exists(),
                error_message="no such file or folder",
            ),
        )
    else:
        plan.preset = picked


def ask_config_and_split(plan: DataPlan) -> None:
    if plan.preset is not None:
        pinned = preset_data.PRESETS[plan.preset].source
        repo, config, revision = pinned.repo, pinned.config, pinned.revision
    elif plan.source and plan.source.startswith(HUB_PREFIX):
        repo, config, revision = plan.source[len(HUB_PREFIX) :], plan.config, plan.revision
        if config is None:
            configs = hub_configs(repo, revision)
            if len(configs) > 1:
                plan.config = config = _pick("Config:", [(c, c) for c in configs])
            elif configs:
                config = configs[0]
    else:
        if plan.split is None and Path(plan.source).expanduser().is_dir():
            splits = saved_splits(Path(plan.source).expanduser())
            if len(splits) > 1:
                plan.split = _pick("Split:", [(s, s) for s in splits], _training_split(splits))
        return
    if plan.split is None:
        splits = hub_splits(repo, config, revision)
        if len(splits) > 1:
            plan.split = _pick("Split:", [(s, s) for s in splits], _training_split(splits))


def _training_split(splits: list[str]) -> str | None:
    return next((s for s in splits if s.startswith("train")), None)


def ask_shape_and_columns(plan: DataPlan, table: Table) -> None:
    given = {role: name for role, name in plan.columns.items() if name}
    try:
        guessed = guess(table, plan.shape, given)
    except SourceError:
        guessed = None
    if plan.shape is None:
        default = guessed.shape if guessed else None
        plan.shape = _pick(
            "Shape:", [(shape, f"{shape:<7} {SHAPE_TEXT[shape]}") for shape in SHAPES], default
        )
        if guessed is not None and guessed.shape != plan.shape:
            guessed = None
    used = set(given.values())
    for role in ROLES[plan.shape]:
        if role in given:
            continue
        fitting = [c for c in table.columns.values() if c.kind in KINDS[role]]
        columns = fitting or list(table.columns.values())
        options = [(c.name, c.describe()) for c in columns if c.name not in used]
        if role in OPTIONAL:
            options.append((NONE, NONE))
        default = guessed.columns.get(role) if guessed else None
        if default is None:
            found = candidates(table, role, used)
            default = found[0] if len(found) == 1 else (NONE if role in OPTIONAL else None)
        picked = _pick(f"{role.capitalize()} column:", options, default)
        plan.columns[role] = None if picked == NONE else picked
        if picked != NONE:
            used.add(picked)


def ask_question(plan: DataPlan, table: Table) -> None:
    """The question, through chat's builder, turned back into flags."""
    plan.question_id = plan.question_id or default_id(table)
    if plan.shape == "yesno":
        return  # each row asks its own question
    if plan.shape == "choice":
        built = _build(plan, {"type": "choice"}, ("choice",), criteria=False)
        plan.question_id, plan.instructions = built[0], built[1].get("instructions")
        return
    label = table.columns[plan.columns["label"]]
    if plan.shape == "rating":
        classes = rating_values(table, label)
        levels = list(plan.levels or classes)
        say(f"One level per value, lowest first ({', '.join(classes)}).")
        current = {"type": "score", "criteria": levels}
        plan.question_id, question = _build(plan, current, ("score",), count=len(classes))
        plan.instructions = question.get("instructions")
        plan.levels = None if question["criteria"] == classes else list(question["criteria"])
        return
    classes = class_names(label, table, None)
    types = ("choice", "noul") if len(classes) == 2 else ("choice",)
    current = {"type": "choice", "criteria": dict(zip(classes, classes, strict=True))}
    say(f"One option per class, in this order: {', '.join(classes)}.")
    plan.question_id, question = _build(plan, current, types, count=len(classes))
    plan.instructions = question.get("instructions")
    if question["type"] == "noul":
        plan.noul = True
        return
    ids = list(question["criteria"])
    descriptions = {
        key: key if value is None else value for key, value in question["criteria"].items()
    }
    plan.label_map = None if ids == classes else dict(zip(classes, ids, strict=True))
    plan.options = None if all(descriptions[key] == key for key in ids) else descriptions


def _build(
    plan: DataPlan,
    current: dict,
    types: tuple[str, ...],
    count: int | None = None,
    criteria: bool = True,
):
    current = {**current, "instructions": plan.instructions} if plan.instructions else current

    def check(question_id: str, question: dict) -> None:
        found = question.get("criteria")
        if count is not None and question["type"] != "noul" and len(found or []) != count:
            raise ValueError(f"give one entry per class or value: {count}, not {len(found or [])}")

    built = build_question(
        set(),
        plan.question_id,
        current,
        check=check,
        types=types,
        criteria=criteria,
        instructions_required=True,
    )
    if built is None:
        raise Cancelled()
    return built


def ask_output(plan: DataPlan, table: Table | None) -> None:
    name = table.name if table is not None else (plan.preset or "data")
    split = f"-{plan.split}" if plan.split else ""
    default = f"{default_id(table) if table is not None else name}{split}.jsonl"
    plan.output = _ask("Write to: ", default=default, completer=PathCompleter(expanduser=True))


def ask_data(plan: DataPlan, preview: int = 3) -> tuple[DataPlan, Made]:
    """``plan`` with what its flags left open asked; the examples made. Raises
    ``Cancelled`` when the menus are left."""
    from ..cli import show_made

    if plan.source is None and plan.preset is None:
        ask_source(plan)
    ask_config_and_split(plan)
    table = None
    if plan.preset is None:
        table = open_source(plan.source, plan.config, plan.split, plan.revision)
        if plan.source.startswith(HUB_PREFIX):
            plan.split = table.split  # the only one, when none was picked: said in the command
        ask_shape_and_columns(plan, table)
        if not plan.has_wording():
            ask_question(plan, table)
    if plan.output is None and plan.show is None:
        ask_output(plan, table)
    made = plan.make()
    say("")
    show_made(made, preview if plan.show is None else plan.show)
    say("")
    if plan.show is None and not confirm(
        f"Write {len(made.examples):,} examples to {plan.output}?"
    ):
        raise Cancelled()
    return plan, made


def same_without_menus(plan: DataPlan) -> str:
    """The command that repeats ``plan`` without menus, to print at the end."""
    words = ["mlx-decision", "data", *plan.arguments()]
    return "The same without menus:\n  " + command_line(words)
