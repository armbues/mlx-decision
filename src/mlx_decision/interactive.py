"""The ``chat`` session: states typed one after another, questions built in between.

A prompt_toolkit session reads each state with line editing and history; a
blank line submits it. A single line starting with ``/`` is a command that
lists, adds, edits, removes or saves the questions, or attaches images that
go with every following state.
"""

import json
import sys
from pathlib import Path
from typing import Any

import typer
from prompt_toolkit import PromptSession, shortcuts
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.validation import ValidationError, Validator

from .cli import (
    CHAT_DESCRIBE_HOW,
    QUESTION_ID,
    describe_options_hint,
    format_result,
    parse_state,
    undescribed_choices,
)
from .errors import DecisionError
from .fitted import unfitted_warnings

TYPES = {
    "choice": "pick one of several options",
    "score": "rate on ordered levels",
    "noul": "a yes/no question",
}
COMMANDS = {
    "list": "show the questions",
    "add": "build a new question",
    "edit": "edit a question: /edit ID",
    "remove": "remove a question: /remove ID",
    "save": "write the questions to a file that -q reads: /save FILE",
    "image": "attach an image to the following states: /image FILE; /image clear",
    "help": "show this help",
    "quit": "end the session (or Ctrl-D)",
}
HINT = "Type a state and finish it with a blank line. /help lists the commands, Ctrl-D quits."
NEXT_STEPS = [("start", "Start answering states"), ("add", "Add another question")]


def say(message: str = "") -> None:
    typer.echo(message, err=True)


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def is_command(text: str) -> bool:
    return text.startswith("/") and not text.startswith("//") and "\n" not in text


def block_bindings(commands: bool) -> KeyBindings:
    """Enter adds a line; Enter on a blank last line submits the block.

    With ``commands``, a single line starting with ``/`` submits at once.
    Without, an empty block can be submitted (to end a list).
    """
    bindings = KeyBindings()

    @bindings.add("enter")
    @bindings.add("c-j")
    def _(event):
        buffer = event.current_buffer
        text = buffer.text
        if commands and is_command(text):
            buffer.validate_and_handle()
        elif buffer.cursor_position == len(text) and not buffer.document.current_line.strip():
            if text.strip() or not commands:
                buffer.text = text.rstrip()
                buffer.validate_and_handle()
        else:
            buffer.insert_text("\n")

    return bindings


def cancel_menu(event) -> None:
    event.app.exit(exception=EOFError())


# Ctrl-D leaves a menu like any other prompt (Ctrl-C is built in).
MENU_BINDINGS = KeyBindings()
MENU_BINDINGS.add("c-d")(cancel_menu)


def continuation(width: int, line_number: int, wrap_count: int) -> str:
    return " " * width


class CommandCompleter(Completer):
    """Completes command names after ``/`` and question ids after ``/edit`` and ``/remove``."""

    def __init__(self, session: "Session"):
        self.session = session
        self.paths = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        name, space, arg = text[1:].partition(" ")
        if not space:
            for command in COMMANDS:
                if command.startswith(name):
                    yield Completion(f"/{command}", start_position=-len(text))
        elif name in ("edit", "remove") and " " not in arg:
            for question_id in self.session.questions:
                if question_id.startswith(arg):
                    yield Completion(question_id, start_position=-len(arg))
        elif name in ("image", "save"):
            yield from self.paths.get_completions(Document(arg), complete_event)


class Session:
    """Answers states with ``model`` and keeps the questions of ``body`` editable."""

    def __init__(self, model, body: dict[str, Any], state_json: bool, as_json: bool):
        self.model = model
        self.body = body
        self.body.setdefault("questions", {})
        self.state_json = state_json
        self.as_json = as_json
        self.states = PromptSession(
            history=InMemoryHistory(),
            multiline=True,
            key_bindings=block_bindings(commands=True),
            prompt_continuation=continuation,
            completer=CommandCompleter(self),
            bottom_toolbar=self.toolbar,
        )

    @property
    def questions(self) -> dict[str, Any]:
        return self.body["questions"]

    @property
    def images(self) -> list[Any]:
        if self.body.get("images") is None:  # left out, or null in a -q file
            self.body["images"] = []
        return self.body["images"]

    def toolbar(self) -> str:
        count = len(self.questions)
        text = f" {count} question{'' if count == 1 else 's'}"
        if self.images:
            text += f", {len(self.images)} image{'' if len(self.images) == 1 else 's'}"
        return text + " · blank line: answer · /help: commands · Ctrl-D: quit"

    def run(self) -> None:
        if not self.questions:
            say("No questions yet: build the first one (Ctrl-C cancels).")
            self.build()
            while self.questions and self.next_step() == "add":
                self.build()
            if not self.questions:
                return
        say(HINT)
        while True:
            try:
                text = self.states.prompt("state> ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                return
            if is_command(text):
                if not self.command(text):
                    return
            else:
                self.answer(text.removeprefix("/") if text.startswith("//") else text)

    def answer(self, text: str) -> None:
        if not self.questions:
            say("error: no questions; /add one")
            return
        for message in unfitted_warnings(self.model, self.body):
            say(f"warning: {message}")
        busy = sys.stderr.isatty()
        if busy:
            typer.echo("answering...", err=True, nl=False)
        try:
            state = parse_state(text, self.state_json)
            body = {**self.body, "state": state}
            if not body.get("images"):
                body.pop("images", None)
            result = self.model.decide_request(body)
        except DecisionError as error:
            result, failure = None, error
        except KeyboardInterrupt:
            # Stops this answer only; the session and its questions stay.
            result, failure = None, None
        if busy:
            typer.echo("\r\033[K", err=True, nl=False)
        if result is None:
            say("cancelled" if failure is None else f"error: {failure}")
            return
        if self.as_json:
            typer.echo(json.dumps(result.to_wire(), ensure_ascii=False))
            if result.truncated:  # the body has no field for it
                say("note: the state was truncated to fit the model's input limit")
        else:
            typer.echo(format_result(result))
        say()

    # Commands

    def command(self, text: str) -> bool:
        """Run one command; False ends the session."""
        name, _, arg = text[1:].partition(" ")
        arg = arg.strip()
        if name == "quit":
            return False
        if name == "help":
            width = max(len(command) for command in COMMANDS)
            for command, description in COMMANDS.items():
                say(f"  /{command:<{width}}  {description}")
            say("  A state that starts with / is typed as //.")
        elif name == "list":
            say(describe(self.questions) if self.questions else "no questions; /add one")
            if self.images:
                say("images: " + ", ".join(_short(image) for image in self.images))
        elif name == "image":
            self.attach(arg)
        elif name == "add":
            self.build()
        elif name in ("edit", "remove"):
            if not self.questions:
                say("no questions; /add one")
                return True
            if not arg:
                arg = self.pick_question(f"Question to {name}:")
                if arg is None:
                    return True
            if arg not in self.questions:
                say(f"error: no question {arg!r} ({', '.join(self.questions)})")
            elif name == "edit":
                self.build(arg)
            else:
                del self.questions[arg]
                say(f"removed {arg}")
        elif name == "save":
            self.save(arg)
        else:
            say(f"error: unknown command /{name}; /help lists the commands")
        return True

    def attach(self, arg: str) -> None:
        """``/image FILE`` adds an image for the following states; ``/image clear`` drops them."""
        if arg == "clear":
            self.images.clear()
            say("images cleared")
            return
        if not arg:
            say("error: /image needs a file name (or clear)")
            return
        if not self.model.backend.capabilities.supports_images:
            say(f"error: {self.model.name} does not support images")
            return
        from .images import load_image

        path = Path(arg).expanduser()
        try:
            load_image(path, len(self.images))  # check it now, not with the next state
        except DecisionError as error:
            say(f"error: {error.message}")
            return
        self.images.append(str(path))
        count = len(self.images)
        say(f"attached {path} ({count} image{'' if count == 1 else 's'})")

    def save(self, arg: str) -> None:
        if not arg:
            say("error: /save needs a file name")
            return
        path = Path(arg).expanduser()
        if path.exists() and not self.confirm(f"{path} exists. Overwrite?"):
            return
        try:
            path.write_text(json.dumps(self.questions, indent=2, ensure_ascii=False) + "\n")
        except OSError as error:
            say(f"error: cannot write {path}: {error}")
            return
        say(f"saved {len(self.questions)} questions to {path}")

    # The question builder

    def build(self, question_id: str | None = None) -> None:
        """Add a question, or edit ``question_id``; repeats until the model accepts it."""

        def check(draft_id: str, draft: dict[str, Any]) -> None:
            self.model.check({"state": "", "questions": {draft_id: draft}})

        built = build_question(
            set(self.questions) - {question_id},
            question_id,
            dict(self.questions[question_id]) if question_id else {},
            check=check,
            suggest=self.free_id,
        )
        if built is None:
            return
        draft_id, draft = built
        if question_id is None:
            self.questions[draft_id] = draft
        else:
            # Keep the question's place when its id changes.
            self.body["questions"] = {
                (draft_id if key == question_id else key): (draft if key == question_id else value)
                for key, value in self.questions.items()
            }
        say(f"{'updated' if question_id else 'added'}:")
        say(describe({draft_id: draft}, indent="  "))
        if bare := undescribed_choices(self.model, {draft_id: draft}):
            say(describe_options_hint(bare, CHAT_DESCRIBE_HOW))

    def next_step(self) -> str:
        try:
            return pick("Next:", NEXT_STEPS)
        except (KeyboardInterrupt, EOFError):
            return "start"

    def pick_question(self, message: str) -> str | None:
        options = [
            (question_id, describe({question_id: question}).splitlines()[0])
            for question_id, question in self.questions.items()
        ]
        try:
            return pick(message, options)
        except (KeyboardInterrupt, EOFError):
            return None

    def confirm(self, message: str) -> bool:
        return confirm(message)

    def free_id(self, kind: str) -> str:
        n = 1
        while f"{kind}_{n}" in self.questions:
            n += 1
        return f"{kind}_{n}"


# Prompts and menus, shared by chat and the data and training wizards.


def ask(message: str, **options) -> str:
    """One line, stripped."""
    return PromptSession().prompt(message, **options).strip()


def pick(message: str, options: list[tuple[Any, str]], default: Any = None) -> Any:
    """A menu: arrow keys or the option's number, then Enter."""
    if default not in dict(options):
        default = None
    return shortcuts.choice(message, options=options, default=default, key_bindings=MENU_BINDINGS)


def confirm(message: str) -> bool:
    """Yes or no; cancelling says no."""
    try:
        return shortcuts.confirm(message)
    except (KeyboardInterrupt, EOFError):
        return False


def ask_block(default: list[str], validator: Validator | None) -> str:
    """Several lines, prefilled with ``default``; a blank line ends them."""
    return PromptSession().prompt(
        "  ",
        default="\n".join(default) + "\n" if default else "",
        multiline=True,
        key_bindings=block_bindings(commands=False),
        prompt_continuation=continuation,
        validator=validator,
        validate_while_typing=False,
    )


def build_question(
    taken: set[str],
    question_id: str | None = None,
    current: dict[str, Any] | None = None,
    *,
    check=None,
    suggest=lambda kind: f"{kind}_1",
    types: tuple[str, ...] = tuple(TYPES),
    criteria: bool = True,
    instructions_required: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """Ask for a question, or edit ``current`` (as ``question_id``), until
    ``check(id, question)`` accepts it (it raises ``DecisionError`` or
    ``ValueError`` with what is wrong); None when cancelled.

    ``taken`` are ids already in use; ``suggest(kind)`` the id offered when
    none is typed. ``types`` limits the question types (one: no menu);
    without ``criteria`` the options and levels are not asked (they come
    from elsewhere); ``instructions_required`` asks for instructions on
    every type (a yes/no question always needs them).
    """
    draft_id, draft = question_id, dict(current or {})
    while True:
        try:
            draft_id, draft = ask_question(
                taken, draft_id, draft, suggest, types, criteria, instructions_required
            )
        except (KeyboardInterrupt, EOFError):
            say("cancelled")
            return None
        try:
            if check is not None:
                check(draft_id, draft)
        except (DecisionError, ValueError) as error:
            say(f"error: {error}")
            continue
        return draft_id, draft


def ask_question(
    taken: set[str],
    draft_id: str | None,
    current: dict[str, Any],
    suggest,
    types: tuple[str, ...],
    criteria: bool,
    instructions_required: bool,
) -> tuple[str, dict[str, Any]]:
    if len(types) == 1:
        kind = types[0]
    else:
        kind = pick(
            "Type:",
            [(name, f"{name:<7} {TYPES[name]}") for name in types],
            default=current.get("type"),
        )
    suggestion = suggest(kind)
    new_id = (
        ask(
            "id: ",
            default=draft_id or "",
            placeholder=None if draft_id else [("fg:ansibrightblack", suggestion)],
            validator=IdValidator(taken, empty_ok=not draft_id),
        )
        or suggestion
    )
    old_instructions = current.get("instructions")
    default = as_text(old_instructions) if old_instructions is not None else ""
    required = kind == "noul" or instructions_required
    text = ask(
        "instructions: " if required else "instructions (optional): ",
        default=default,
        validator=Validator.from_callable(
            lambda text: bool(text.strip()) or not required,
            error_message=(
                "a yes/no question needs its text"
                if kind == "noul"
                else "the question needs its text"
            ),
        ),
    )
    question: dict[str, Any] = {"type": kind}
    if text.strip():
        question["instructions"] = old_instructions if text == default else text.strip()
    same_type = current.get("type") == kind
    if not criteria:
        return new_id, question
    if kind == "choice":
        found = current.get("criteria") if same_type else None
        question["criteria"] = ask_options(found if isinstance(found, dict) else {})
    elif kind == "score":
        found = current.get("criteria") if same_type else None
        question["criteria"] = ask_levels(found if isinstance(found, list) else [])
    elif same_type and "criteria" in current:
        question["criteria"] = current["criteria"]
    return new_id, question


def ask_options(current: dict[str, Any]) -> dict[str, Any]:
    shown = [key if value is None else f"{key}: {as_text(value)}" for key, value in current.items()]
    if not one_per_line(shown, option_key):
        say(KEPT.format(what="options"))
        return current
    say("options, one per line as 'option' or 'option: description' (blank line ends):")
    lines = dict(zip(shown, current.items(), strict=True))
    text = ask_block(shown, OptionsValidator())
    options = {}
    for line in filter(str.strip, text.splitlines()):
        if line in lines:
            key, value = lines[line]
        else:
            key, _, description = line.partition(":")
            key, value = key.strip(), description.strip() or None
        options[key] = value
    return options


def ask_levels(current: list[Any]) -> list[Any]:
    shown = [as_text(value) for value in current]
    if not one_per_line(shown):
        say(KEPT.format(what="levels"))
        return current
    say("levels, lowest first, one per line (blank line ends):")
    lines = dict(zip(shown, current, strict=True))
    text = ask_block(shown, None)
    return [lines.get(line, line.strip()) for line in filter(str.strip, text.splitlines())]


# Criteria the line editor would change without being asked to.
KEPT = (
    "{what} kept as they are: some look alike on a line, span lines or are "
    "blank, so editing them here would lose them; edit them in a -q file"
)


def option_key(line: str) -> str:
    """The option a line names, as ``OptionsValidator`` reads it."""
    return line.partition(":")[0].strip()


def one_per_line(shown: list[str], key=lambda line: line) -> bool:
    """Whether entries shown as ``shown`` come back from the line editor as they
    are: each one non-blank line, no two with the same ``key``."""
    if any(len(line.splitlines()) != 1 or not line.strip() for line in shown):
        return False
    keys = [key(line) for line in shown]
    return len(set(keys)) == len(keys)


class IdValidator(Validator):
    def __init__(self, taken: set[str], empty_ok: bool):
        self.taken = taken
        self.empty_ok = empty_ok

    def validate(self, document) -> None:
        text = document.text.strip()
        if not text and self.empty_ok:
            return
        if not QUESTION_ID.fullmatch(text):
            raise ValidationError(message="letters, digits, '_', '.' and '-' only")
        if text in self.taken:
            raise ValidationError(message=f"{text} is taken")


class OptionsValidator(Validator):
    def validate(self, document) -> None:
        seen = set()
        for line in filter(str.strip, document.text.splitlines()):
            key = option_key(line)
            if key in seen:
                raise ValidationError(message=f"option {key!r} given twice")
            seen.add(key)


def describe(questions: dict[str, Any], indent: str = "") -> str:
    """The questions as the builder shows them."""
    lines = []
    for question_id, question in questions.items():
        kind = question.get("type", "?")
        head = f"{question_id} ({kind})"
        if question.get("instructions") is not None:
            head += f": {as_text(question['instructions'])}"
        lines.append(head)
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            for key, value in criteria.items():
                lines.append(f"  {key}" if value is None else f"  {key}: {as_text(value)}")
        elif isinstance(criteria, list):
            for level, value in enumerate(criteria):
                lines.append(f"  {level}  {as_text(value)}")
    return "\n".join(indent + line for line in lines)


def _short(value: Any) -> str:
    """A path or data URL cut to 60 characters; a path keeps its end (the file name)."""
    text = str(value)
    if len(text) <= 60:
        return text
    return text[:57] + "..." if text.startswith("data:") else "..." + text[-57:]
