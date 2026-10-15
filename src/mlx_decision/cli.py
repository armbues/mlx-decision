"""The ``mlx-decision`` command."""

import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import typer

from .errors import DecisionError, PlatformError

app = typer.Typer(
    help="Run and fine-tune decision models on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)

QUESTION_ID = re.compile(r"[\w.\-]+")
BAR_WIDTH = 24
LEGEND_WIDTH = 48


def show_version(value: bool) -> None:
    if value:
        from . import __version__

        typer.echo(f"mlx-decision {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    pass


def fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def split_id(value: str, flag: str, required: bool) -> tuple[str | None, str]:
    """Split ``id=rest``; the id is optional for a noul."""
    head, sep, rest = value.partition("=")
    if sep and QUESTION_ID.fullmatch(head.strip()):
        return head.strip(), rest.strip()
    if required:
        raise typer.BadParameter(
            f"expected ID=OPTION,OPTION,... but got {value!r}", param_hint=flag
        )
    return None, value.strip()


def shorthand_questions(nouls: list[str], choices: list[str], scores: list[str]) -> dict:
    questions: dict[str, dict] = {}

    def add(question_id: str, question: dict, flag: str) -> None:
        if question_id in questions:
            raise typer.BadParameter(f"question id {question_id!r} given twice", param_hint=flag)
        questions[question_id] = question

    for index, value in enumerate(nouls, 1):
        question_id, text = split_id(value, "--noul", required=False)
        add(question_id or f"noul_{index}", {"type": "noul", "instructions": text}, "--noul")
    for flag, kind, values in (("--choice", "choice", choices), ("--score", "score", scores)):
        for value in values:
            question_id, rest = split_id(value, flag, required=True)
            options = [option.strip() for option in rest.split(",") if option.strip()]
            if kind == "choice":
                if repeated := sorted({o for o in options if options.count(o) > 1}):
                    raise typer.BadParameter(
                        f"option {repeated[0]!r} given twice in {question_id!r}", param_hint=flag
                    )
                add(question_id, {"type": kind, "criteria": dict.fromkeys(options)}, flag)
            else:
                add(question_id, {"type": kind, "criteria": options}, flag)
    return questions


def build_questions(questions_file: Path | None, shorthand: dict) -> dict[str, Any]:
    """The request body from a questions file and shorthand flags, without a state."""
    from .types import decode_json

    body: dict[str, Any] = {}
    if questions_file is not None:
        try:
            data = decode_json(questions_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as error:
            fail(f"cannot read questions from {questions_file}: {error}")
        except DecisionError as error:
            fail(f"cannot read questions from {questions_file}: {error.message}")
        body = dict(data) if isinstance(data, dict) and "questions" in data else {"questions": data}
    if shorthand:
        questions = body.setdefault("questions", {})
        if not isinstance(questions, dict):
            fail("the questions file must hold a JSON object")
        if clash := sorted(set(questions) & set(shorthand)):
            fail(f"question id given twice: {', '.join(clash)}")
        questions.update(shorthand)
    if "questions" not in body:
        fail("no questions: give --questions FILE or --noul/--choice/--score")
    return body


def undescribed_choices(model, questions: Any) -> list[str]:
    """Choice questions with an option left undescribed, if the model is Julia.

    Julia reads an option only through its description and answers much
    worse with bare ids or one-word labels (README, Julia).
    """
    if model.info()["family"] != "julia" or not isinstance(questions, dict):
        return []
    return [
        question_id
        for question_id, question in questions.items()
        if isinstance(question, dict)
        and question.get("type") == "choice"
        and isinstance(question.get("criteria"), dict)
        and any(value is None or value == "" for value in question["criteria"].values())
    ]


def describe_options_hint(question_ids: list[str], how: str) -> str:
    return (
        f"hint: {', '.join(question_ids)}: Julia answers much worse without option "
        f"descriptions; describe each option in a short phrase of what it covers ({how})"
    )


RUN_DESCRIBE_HOW = 'in a -q file: "billing": "Billing and payment disputes"'
CHAT_DESCRIBE_HOW = "/edit ID, then 'billing: Billing and payment disputes'"


def parse_state(state: str, state_json: bool) -> Any:
    if not state_json:
        return state
    from .types import decode_json

    return decode_json(state, "the state", param="state")


def build_request(
    questions_file: Path | None,
    shorthand: dict,
    state: str | None,
    state_file: Path | None,
    state_json: bool,
) -> dict[str, Any]:
    body = build_questions(questions_file, shorthand)
    if state is not None and state_file is not None:
        fail("give either --state or --state-file, not both")
    if state_file is not None:
        try:
            state = state_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            fail(f"cannot read the state: {error}")
    if state is None and "state" not in body:
        if sys.stdin.isatty():
            fail("no state: give --state, --state-file, or pipe it to stdin")
        state = sys.stdin.read()
    if state is not None:
        try:
            body["state"] = parse_state(state, state_json)
        except DecisionError as error:
            fail(f"--state-json: {error.message}")
    return body


def answer_states(model, defaults: dict[str, Any], lines) -> None:
    """One response body per request line, in order; a bad line gives an error body.

    Each line is a request body (text, or bytes decoded as UTF-8); fields it
    leaves out (usually the questions) come from ``defaults``.
    """
    from .types import decode_json

    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            if isinstance(line, bytes):
                try:
                    line = line.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise DecisionError(f"not valid UTF-8: {error}") from None
            request = decode_json(line)
            if not isinstance(request, dict):
                raise DecisionError('a line must be a request body: {"state": ...}')
            result = model.decide_request({**defaults, **request})
        except DecisionError as error:
            # The API's error body, plus the input line it belongs to.
            detail = {"message": error.message, "type": "invalid_request_error"}
            detail["param"] = error.param
            if error.code:
                detail["code"] = error.code
            typer.echo(json.dumps({"error": detail, "line": number}, ensure_ascii=False))
            continue
        if result.truncated:
            typer.echo(f"note: line {number}: the state was truncated", err=True)
        typer.echo(json.dumps(result.to_wire(), ensure_ascii=False))


def bar(probability: float) -> str:
    return "█" * round(probability * BAR_WIDTH)


def short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= LEGEND_WIDTH else text[: LEGEND_WIDTH - 1] + "…"


def format_result(result) -> str:
    """A readable summary: one block per question, then model and usage."""
    lines = []
    for question_id, answer in result.answers.items():
        if answer.type == "noul":
            verdict = "yes" if answer.noul >= 0.5 else "no"
            lines.append(f"{question_id} (noul): {verdict}, p(yes) {answer.noul:.3f}")
            lines.append(f"  {bar(answer.noul)}")
        elif answer.type == "choice":
            lines.append(
                f"{question_id} (choice): {answer.choice}, confidence {answer.confidence:.2f}"
            )
            width = max(len(option) for option in answer.probabilities)
            for option, p in sorted(answer.probabilities.items(), key=lambda item: -item[1]):
                lines.append(f"  {option:<{width}}  {p:.3f}  {bar(p)}")
        else:
            top = len(answer.probabilities) - 1
            lines.append(
                f"{question_id} (score): {answer.score:.2f} on 0-{top}, "
                f"confidence {answer.confidence:.2f}"
            )
            for level, p in answer.probabilities.items():
                legend = short(answer.legend[level])
                lines.append(f"  {level:>2}  {p:.3f}  {bar(p):<{BAR_WIDTH}}  {legend}")
        lines.append("")
    footer = f"{result.model} · {result.usage.input_tokens} input tokens"
    if result.truncated:
        footer += " · state truncated to fit the input limit"
    lines.append(footer)
    return "\n".join(lines)


# Options shared by run and chat.
ModelOption = Annotated[
    str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
]
QuestionsOption = Annotated[
    Path | None,
    typer.Option("--questions", "-q", help="JSON file: a questions map, or a whole request body."),
]
NoulOption = Annotated[
    list[str] | None, typer.Option("--noul", help="A yes/no question: '[ID=]text'. Repeatable.")
]
ChoiceOption = Annotated[
    list[str] | None,
    typer.Option("--choice", help="A choice: 'ID=option,option,...'. Repeatable."),
]
ScoreOption = Annotated[
    list[str] | None,
    typer.Option("--score", help="A score: 'ID=lowest,...,highest'. Repeatable."),
]
StateJsonOption = Annotated[
    bool, typer.Option("--state-json", help="Parse the state as JSON (object, array, ...).")
]
JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Print the response body as JSON (always on with run --states)."),
]
ImageOption = Annotated[
    list[Path] | None,
    typer.Option(
        "--image",
        help="An image file to ask about, with every state. Repeatable. "
        "Needs the images extra: pip install 'mlx-decision\\[images]'.",
    ),
]

MaxImageOption = Annotated[
    float | None,
    typer.Option(
        "--max-image-mp",
        min=0,
        help="Shrink larger images to this many megapixels (1 MP = 2^20 pixels = 1,024 "
        "tokens; Clef and d3). Default: Clef 2, d3 its release's 1.56. 0 = no cap beyond the "
        "model's own, as in Cloudflare's reference.",
    ),
]


MaxInputOption = Annotated[
    int | None,
    typer.Option(
        "--max-input-tokens",
        min=16,
        help="Input limit in tokens; longer states are cut (or refused by models that do not "
        "cut). Default: the model's own (Clef 16,384; pplx 8,192; d3 262,144; "
        "H2O-Lightning 40,960; d1 32,768; Laya 512 or 1,024; Julia 8,192). Laya and Julia "
        "take up to 8,192; pplx, d3, H2O-Lightning and d1 only a lower limit.",
    ),
]


PrefixCacheOption = Annotated[
    float | None,
    typer.Option(
        "--prefix-cache",
        min=0,
        metavar="GB",
        help="Memory for kept states (Clef, H2O-Lightning, d1): a request whose state came "
        "recently, with other questions, computes only the questions. Default 2; 0 turns "
        "it off.",
    ),
]


NoulFloorOption = Annotated[
    float | None,
    typer.Option(
        "--noul-floor",
        min=0,
        max=0.999,
        help="Yes/no answers less sure than this are reported at it, on their side "
        "(H2O-Lightning; default: the release's 0.801, as evaluated). 0 turns it off, "
        "for routing uncertain answers elsewhere.",
    ),
]


NoMemoryCheckOption = Annotated[
    bool,
    typer.Option(
        "--no-memory-check",
        help="Load the model even if its weights plus 10% exceed the GPU's recommended "
        "working set (refused by default, as it would swap).",
    ),
]


def load_options(
    max_image_mp: float | None,
    max_input_tokens: int | None,
    prefix_cache: float | None = None,
    noul_floor: float | None = None,
) -> dict[str, Any]:
    """Load options for the flags that were given."""
    options = image_options(max_image_mp)
    if max_input_tokens is not None:
        options["max_input_tokens"] = max_input_tokens
    if prefix_cache is not None:
        options["prefix_cache_gb"] = prefix_cache
    if noul_floor is not None:
        options["noul_floor"] = noul_floor
    return options


def image_options(max_image_mp: float | None) -> dict[str, Any]:
    """Load options for ``--max-image-mp``; none when the flag is not given."""
    if max_image_mp is None:
        return {}
    return {"max_image_pixels": round(max_image_mp * 2**20) if max_image_mp else None}


def add_images(body: dict[str, Any], images: list[Path] | None) -> dict[str, Any]:
    """Append ``--image`` files to the body's images, checking that they exist."""
    if images:
        from .images import require_pillow

        try:
            require_pillow()  # before the model loads, not with the first state
        except DecisionError as error:
            fail(f"--image: {error.message}")
        for path in images:
            if not path.is_file():
                fail(f"--image: no such file: {path}")
        body["images"] = list(body.get("images") or []) + [str(path) for path in images]
    return body


@app.command()
def run(
    model: ModelOption,
    questions: QuestionsOption = None,
    noul: NoulOption = None,
    choice: ChoiceOption = None,
    score: ScoreOption = None,
    state: Annotated[
        str | None,
        typer.Option("--state", "-s", help="The state as text. Default: read stdin."),
    ] = None,
    state_file: Annotated[
        Path | None, typer.Option("--state-file", help="Read the state from a file.")
    ] = None,
    state_json: StateJsonOption = False,
    image: ImageOption = None,
    max_image_mp: MaxImageOption = None,
    max_input_tokens: MaxInputOption = None,
    prefix_cache: PrefixCacheOption = None,
    noul_floor: NoulFloorOption = None,
    no_memory_check: NoMemoryCheckOption = False,
    states: Annotated[
        str | None,
        typer.Option(
            "--states",
            help="Many requests: a JSON lines file ('-' for stdin), one request body per "
            'line, e.g. {"state": "..."}; questions from -q or flags apply where a '
            "line has none. Prints one response body per line.",
        ),
    ] = None,
    as_json: JsonOption = False,
) -> None:
    """Answer questions about a state."""
    from .model import load
    from .types import parse_request

    shorthand = shorthand_questions(noul or [], choice or [], score or [])
    if states is not None:
        if state is not None or state_file is not None or state_json:
            fail(
                "--states reads whole requests from one file; "
                "drop --state/--state-file/--state-json"
            )
        # Questions and images from flags are defaults; lines may bring their own.
        body = build_questions(questions, shorthand) if questions or shorthand else {}
        body.pop("state", None)
        add_images(body, image)
        try:
            # Bytes, so that a line that is not UTF-8 is reported, not repaired.
            source = (
                nullcontext(getattr(sys.stdin, "buffer", sys.stdin))
                if states == "-"
                else open(states, "rb")  # noqa: SIM115
            )
        except OSError as error:
            fail(f"cannot read the states: {error}")
        with source as lines:
            try:
                if "questions" in body:
                    parse_request({**body, "state": ""})
                loaded = load(
                    model,
                    check_memory=not no_memory_check,
                    **load_options(max_image_mp, max_input_tokens, prefix_cache, noul_floor),
                )
            except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
                fail(str(error))
            if bare := undescribed_choices(loaded, body.get("questions")):
                typer.echo(describe_options_hint(bare, RUN_DESCRIBE_HOW), err=True)
            warn_unfitted(loaded, body)
            answer_states(loaded, body, lines)
        return
    body = add_images(build_request(questions, shorthand, state, state_file, state_json), image)
    try:
        parse_request(body)  # catch request errors before loading
        loaded = load(
            model,
            check_memory=not no_memory_check,
            **load_options(max_image_mp, max_input_tokens, prefix_cache, noul_floor),
        )
        if bare := undescribed_choices(loaded, body.get("questions")):
            typer.echo(describe_options_hint(bare, RUN_DESCRIBE_HOW), err=True)
        warn_unfitted(loaded, body)
        result = loaded.decide_request(body)
    except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
        fail(str(error))
    if as_json:
        if result.truncated:
            typer.echo("note: the state was truncated to fit the model's input limit", err=True)
        typer.echo(json.dumps(result.to_wire(), indent=2, ensure_ascii=False))
    else:
        typer.echo(format_result(result))


@app.command()
def chat(
    model: ModelOption,
    questions: QuestionsOption = None,
    noul: NoulOption = None,
    choice: ChoiceOption = None,
    score: ScoreOption = None,
    state_json: StateJsonOption = False,
    as_json: JsonOption = False,
    image: ImageOption = None,
    max_image_mp: MaxImageOption = None,
    max_input_tokens: MaxInputOption = None,
    prefix_cache: PrefixCacheOption = None,
    noul_floor: NoulFloorOption = None,
    no_memory_check: NoMemoryCheckOption = False,
) -> None:
    """Load a model once, then answer states typed one after another.

    Without questions, a builder asks for them first. Between states, /help
    lists the commands that show, add, edit, remove and save questions.
    """
    from .model import load
    from .types import parse_request

    if not sys.stdin.isatty():
        fail("chat needs a terminal; for many states in a script use: run --states FILE")
    shorthand = shorthand_questions(noul or [], choice or [], score or [])
    body: dict[str, Any] = {}
    if questions is not None or shorthand:
        body = build_questions(questions, shorthand)
        body.pop("state", None)
    add_images(body, image)
    try:
        if "questions" in body:
            parse_request({**body, "state": ""})  # catch question errors before loading
        loaded = load(
            model,
            check_memory=not no_memory_check,
            **load_options(max_image_mp, max_input_tokens, prefix_cache, noul_floor),
        )
        if body.get("images"):
            loaded.check({"state": "", "questions": {"q": {"type": "noul"}}, **body})
    except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
        fail(str(error))
    count = len(body.get("questions", {}))
    typer.echo(f"{loaded.name} loaded, {count} question{'s' * (count != 1)}.", err=True)
    if bare := undescribed_choices(loaded, body.get("questions")):
        typer.echo(describe_options_hint(bare, CHAT_DESCRIBE_HOW), err=True)
    start_chat(loaded, body, state_json, as_json)


def warn_unfitted(model, body: dict[str, Any]) -> None:
    """Warn about questions a fitted model was not fitted on (answered without the fit's
    calibration and readouts)."""
    from .fitted import unfitted_warnings

    for message in unfitted_warnings(model, body):
        typer.echo(f"warning: {message}", err=True)


def start_chat(model, body: dict[str, Any], state_json: bool, as_json: bool) -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.output import create_output

    from .interactive import Session

    # Prompts go to the terminal even when stdout is redirected.
    with create_app_session(output=create_output(always_prefer_tty=True)):
        Session(model, body, state_json, as_json).run()


def pick_repo() -> str | None:
    """A menu of the known models plus "Other...", where an id is typed."""
    from prompt_toolkit import prompt
    from prompt_toolkit.shortcuts import choice

    from .interactive import MENU_BINDINGS
    from .registry import known_models

    models = known_models()
    width = max(len(model.repo_id) for model in models)
    options = [
        (model.repo_id, f"{model.repo_id:<{width}}  {model.size:>6}  {model.description}")
        for model in models
    ]
    try:
        picked = choice(
            "Model to download:", options=[*options, ("", "Other...")], key_bindings=MENU_BINDINGS
        )
        return picked or prompt("Hugging Face repo id (org/name): ").strip() or None
    except (KeyboardInterrupt, EOFError):
        return None


def ask_models_dir() -> Path | None:
    """The folder that gets the model's own folder, or None for the Hub cache.

    Raises ``typer.Exit`` when the prompt is cancelled.
    """
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter

    try:
        parent = prompt(
            "Download into this folder instead of the Hugging Face cache (empty: the cache): ",
            completer=PathCompleter(only_directories=True, expanduser=True),
        ).strip()
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(1) from None
    return Path(parent).expanduser() if parent else None


def pick_bits(check) -> int:
    """A menu of full size and the quantized sizes; 0 is full size.

    The default is the largest size that fits (the smallest when none
    does). Raises ``typer.Exit`` when the menu is cancelled.
    """
    from prompt_toolkit.shortcuts import choice

    from .download import QUANTIZED_BITS, format_size
    from .interactive import MENU_BINDINGS

    def label(text: str, bits: int) -> str:
        return text if check.fits(bits) else f"{text}  (does not fit in this Mac's GPU working set)"

    options = [(0, label(f"full size  {format_size(check.size_bytes)}", 0))] + [
        (bits, label(f"{bits}-bit      about {format_size(check.quantized_bytes(bits))}", bits))
        for bits in QUANTIZED_BITS
    ]
    best = check.fitting_bits()
    try:
        return choice(
            "Store the model:",
            options=options,
            default=QUANTIZED_BITS[-1] if best is None else best,
            key_bindings=MENU_BINDINGS,
        )
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(1) from None


def stdin_is_terminal() -> bool:
    return sys.stdin.isatty()


@app.command()
def download(
    repo_id: Annotated[
        str | None,
        typer.Argument(help="Hugging Face repo id, e.g. Cloudflare/clef-flash. Default: a menu."),
    ] = None,
    local_dir: Annotated[
        Path | None,
        typer.Option(
            help="Download into this folder instead of the Hugging Face cache. "
            "Default in a terminal: asks for a folder, and the model goes into a "
            "folder named after the repo inside it."
        ),
    ] = None,
    bits: Annotated[
        int | None,
        typer.Option(
            help="Store the model quantized to 8 or 4 bits instead of at full size; "
            "the full-size weights are fetched one file at a time and deleted after "
            "quantizing. Default in a terminal: asks (8-bit when the model does not fit "
            "in memory at full size).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Download even if no model family can load it, and without asking about "
            "the terms of use some models note.",
        ),
    ] = False,
) -> None:
    """Download a model from the Hugging Face Hub (HF_TOKEN for private or gated ones)."""
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    from .download import QUANTIZED_BITS, check_repo, download_quantized, format_size
    from .download import download as fetch
    from .hub import http_errors
    from .registry import get_family

    if bits is not None and bits not in QUANTIZED_BITS:
        raise typer.BadParameter("must be 8 or 4", param_hint="--bits")

    terminal = stdin_is_terminal()
    parent = ask_models_dir() if local_dir is None and terminal else None
    if repo_id is None:
        if not terminal:
            fail("give a repo id, e.g. mlx-decision download Cloudflare/clef-flash")
        repo_id = pick_repo()
        if repo_id is None:
            raise typer.Exit(1)
    try:
        check = check_repo(repo_id)
        if check.family is None:
            typer.echo(f"warning: {repo_id}: {check.reason}", err=True)
            if not yes:
                from prompt_toolkit.shortcuts import confirm

                if not terminal or not confirm("Download it anyway?"):
                    fail("not downloaded (use --yes to download anyway)")
        if check.family is not None and (licence := get_family(check.family).licence):
            typer.echo(f"note: {licence}", err=True)
            if terminal and not yes:
                from prompt_toolkit.shortcuts import confirm

                if not confirm("Download it under these terms?"):
                    fail("not downloaded")
        if bits is not None and not check.convertible:
            fail(f"{repo_id}: {check.family or 'these'} models cannot be stored quantized")
        if bits is None and check.convertible:
            if terminal:
                bits = pick_bits(check) or None
            elif not check.fits():
                best = check.fitting_bits()
                hint = (
                    f"--bits {best} stores it quantized"
                    if best
                    else f"not even at {QUANTIZED_BITS[-1]} bits"
                )
                typer.echo(
                    f"warning: {repo_id} does not fit in this Mac's GPU working set at full "
                    f"size; {hint}",
                    err=True,
                )
        if parent is not None:
            local_dir = parent / repo_id.rsplit("/", 1)[-1]
        target = f" to {local_dir}" if local_dir else ""
        if bits is None:
            size = format_size(check.size_bytes)
            typer.echo(f"downloading {repo_id} ({size}){target} ...", err=True)
            path = fetch(repo_id, local_dir, check.files, check.revision)
        else:
            size = format_size(check.quantized_bytes(bits))
            typer.echo(
                f"downloading {repo_id} as {bits}-bit (about {size}){target}; "
                "fetching one weight file at a time ...",
                err=True,
            )

            def on_file(number: int, total: int, name: str) -> None:
                typer.echo(f"{number}/{total} {name}", err=True)

            path = download_quantized(check, bits, local_dir, on_file)
    except RepositoryNotFoundError:
        fail(f"{repo_id}: not found on the Hub (or private: set HF_TOKEN)")
    except (HfHubHTTPError, *http_errors(), OSError, ValueError) as error:
        fail(f"{repo_id}: {str(error).splitlines()[0]}")
    typer.echo(f"downloaded {repo_id} to {path}", err=True)
    if check.family is not None:  # none could load it, as the warning said
        model = str(local_dir) if local_dir else repo_id
        typer.echo(f"try it: mlx-decision chat -m {model}", err=True)


@app.command()
def server(
    model: Annotated[
        list[str],
        typer.Option(
            "--model",
            "-m",
            help="Model folder, Hugging Face repo id, or a folder of models (each subfolder "
            "a model named after it). Repeat for several.",
        ),
    ],
    default_model: Annotated[
        str | None,
        typer.Option(
            help="Model for requests that name none or an unknown one (such as the Jev SDK's "
            "jev-latest). Default: the only model; with several and no default, such "
            "requests get a 422.",
        ),
    ] = None,
    memory_budget: Annotated[
        float | None,
        typer.Option(
            min=0,
            metavar="GB",
            help="Memory for loaded models; the least recently used are unloaded to stay "
            "within it. Default: the GPU's recommended working set.",
        ),
    ] = None,
    host: Annotated[str, typer.Option(help="Address to bind to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    max_image_mp: MaxImageOption = None,
    max_input_tokens: MaxInputOption = None,
    prefix_cache: PrefixCacheOption = None,
    noul_floor: NoulFloorOption = None,
    no_memory_check: NoMemoryCheckOption = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            envvar="MLX_DECISION_API_KEY",
            help="Require 'Authorization: Bearer KEY' on /v1/* (API clients send "
            "TYPESAFE_API_KEY this way).",
            show_envvar=True,
        ),
    ] = None,
) -> None:
    """Serve models over HTTP (Jev-compatible API: POST /v1/systemone).

    The request's "model" field picks the model. Models load on their first
    request (the default model at startup) and are unloaded, least recently
    used first, when the next one would exceed the memory budget.
    """
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        fail("the server needs extra packages: pip install 'mlx-decision[server]'")
    import logging

    from .pool import ModelPool, discover

    # Loads and unloads; uvicorn configures only its own loggers.
    package_logger = logging.getLogger("mlx_decision")
    if not package_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    options = load_options(max_image_mp, max_input_tokens, prefix_cache, noul_floor)
    budget = round(memory_budget * 1e9) if memory_budget is not None else None
    try:
        # Fail here, not inside the server: folders, options, the default model's size.
        specs = discover(
            model, options, on_skip=lambda path, reason: typer.echo(f"skipped {reason}", err=True)
        )
        pool = ModelPool(
            specs, default=default_model, budget=budget, check_memory=not no_memory_check
        )
        if pool.default is not None:
            pool.size(pool.default)
    except (FileNotFoundError, ValueError) as error:
        fail(str(error))
    names = ", ".join(pool.names)
    if pool.default is None:
        typer.echo(
            f"models: {names}; no default model: requests must name one in 'model' (or "
            f"start with --default-model NAME)",
            err=True,
        )
    elif len(pool.names) > 1:
        typer.echo(f"models: {names}; default {pool.default}", err=True)
    if api_key:
        typer.echo("API key required on /v1/*", err=True)
    app = create_app(pool, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command()
def mcp(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    http: Annotated[
        bool,
        typer.Option(
            "--http",
            help="Serve over streamable HTTP at http://HOST:PORT/mcp instead of stdio, so "
            "several clients share one loaded model.",
        ),
    ] = False,
    host: Annotated[str, typer.Option(help="Address to bind to (with --http).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on (with --http).")] = 8000,
    max_image_mp: MaxImageOption = None,
    max_input_tokens: MaxInputOption = None,
    prefix_cache: PrefixCacheOption = None,
    noul_floor: NoulFloorOption = None,
    no_memory_check: NoMemoryCheckOption = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            envvar="MLX_DECISION_API_KEY",
            help="With --http: require 'Authorization: Bearer KEY' (ignored on stdio, "
            "which only its own client can reach).",
            show_envvar=True,
        ),
    ] = None,
) -> None:
    """Serve a model to agents over MCP (tools: decide, model_info).

    On stdio by default, as MCP clients start local servers; --http for one
    shared process.
    """
    try:
        import anyio

        from .mcp_server import serve_http, serve_stdio
    except ImportError:
        fail("the MCP server needs extra packages: pip install 'mlx-decision[mcp]'")
    import logging

    from .fitted import model_family
    from .hub import resolve_model_path
    from .memory import check_fits
    from .model import load
    from .registry import check_options

    # stdout carries the protocol: everything else goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    options = load_options(max_image_mp, max_input_tokens, prefix_cache, noul_floor)
    try:
        # Fail here, before a client is waiting for the handshake.
        path = resolve_model_path(model)
        check_options(model_family(path), options)
        if not no_memory_check:
            check_fits(path, options, name=model)
    except (FileNotFoundError, ValueError) as error:
        fail(str(error))
    typer.echo(f"loading {model} ...", err=True)
    try:
        if http:
            if api_key:
                typer.echo("API key required", err=True)
            serve_http(lambda: load(model, check_memory=False, **options), host, port, api_key)
        else:
            anyio.run(serve_stdio, lambda: load(model, check_memory=False, **options))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        # The transport's task group wraps a failed load in exception groups.
        while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        if isinstance(error, DecisionError | FileNotFoundError | ValueError | PlatformError):
            fail(str(error))
        raise


@app.command()
def convert(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Folder to write. Default: <model>-q<bits>, -mq<target> or -mlx.",
        ),
    ] = None,
    quantize: Annotated[
        bool, typer.Option("--quantize", "-q", help="Quantize the backbone.")
    ] = False,
    bits: Annotated[
        int | None,
        typer.Option(help="Bits per weight (implies --quantize). Default with -q: 8."),
    ] = None,
    group_size: Annotated[
        int, typer.Option(help="Weights per quantization group: 32, 64 or 128.")
    ] = 64,
    keep_output_embeddings: Annotated[
        bool,
        typer.Option(
            "--keep-output-embeddings",
            help="Leave the output embedding matrix (lm_head; for H2O-Lightning the tied "
            "embeddings) unquantized (Clef, H2O-Lightning).",
        ),
    ] = False,
    target_bits: Annotated[
        float | None,
        typer.Option(
            min=2,
            max=8,
            help="Mixed precision: average bits per weight, allocated by measured "
            "sensitivity (Clef, H2O-Lightning; implies --quantize; takes a few minutes).",
        ),
    ] = None,
) -> None:
    """Write an MLX copy of a model, optionally quantized."""
    from .convert import convert as convert_model
    from .convert import default_output

    if bits is not None and target_bits is not None:
        raise typer.BadParameter("give either --bits or --target-bits", param_hint="--bits")
    if bits is not None and bits not in (2, 3, 4, 5, 6, 8):
        raise typer.BadParameter("must be 2, 3, 4, 5, 6 or 8", param_hint="--bits")
    if group_size not in (32, 64, 128):
        raise typer.BadParameter("must be 32, 64 or 128", param_hint="--group-size")
    quantize = quantize or bits is not None or target_bits is not None
    if keep_output_embeddings and not quantize:
        raise typer.BadParameter(
            "only applies when quantizing (-q, --bits or --target-bits)",
            param_hint="--keep-output-embeddings",
        )
    if quantize and bits is None:
        bits = 8
    options = {}
    if target_bits is not None:

        def progress(done, total, unit, value):
            typer.echo(f"  sensitivity {done}/{total} {unit.name}: {value:.2e}", err=True)

        options = {"target_bits": target_bits, "progress": progress}
    if output is None:
        output = default_output(model, bits if quantize else None, target_bits)
    typer.echo(f"converting {model} to {output} ...", err=True)
    try:
        convert_model(
            model,
            output,
            bits=bits if quantize else None,
            group_size=group_size,
            quantize_output_embeddings=not keep_output_embeddings,
            **options,
        )
    except (FileExistsError, FileNotFoundError, ValueError, PlatformError) as error:
        fail(str(error))
    typer.echo(f"wrote {output}")


def int_list(value: str, flag: str) -> tuple[int, ...]:
    try:
        numbers = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise typer.BadParameter(
            f"expected numbers like 250,1000 but got {value!r}", param_hint=flag
        ) from None
    if not numbers or min(numbers) < 0:
        raise typer.BadParameter("numbers must not be negative", param_hint=flag)
    return numbers


def is_folder_of_models(model: str) -> bool:
    """Whether ``model`` is a local folder that no family loads itself.

    Its subfolders may be models; ``discover`` says whether any is.
    """
    from .hub import is_repo_id
    from .registry import detect_family

    path = Path(model).expanduser()
    if is_repo_id(model) or not path.is_dir():
        return False
    try:
        detect_family(path)
    except ValueError:
        return True
    return False


def benchmark_models(
    folder: str, arguments: list[str], check_memory: bool, as_json: bool, out: Path | None
) -> None:
    """``benchmark`` for a folder of models: one process per model, then a summary."""
    from .benchmark import format_report, format_summary, from_dict, run_models
    from .memory import ModelTooLargeError, check_fits
    from .pool import discover

    if out is not None and out.exists() and not out.is_dir():
        raise typer.BadParameter("with a folder of models, --out is a folder", param_hint="--out")
    skipped: list[str] = []
    try:
        specs = discover(
            [folder],
            on_skip=lambda path, reason: skipped.append(reason.replace(str(path), path.name, 1)),
        )
    except (FileNotFoundError, ValueError) as error:
        fail(str(error))
    if check_memory:
        fitting = []
        for spec in specs:
            try:
                check_fits(spec.path, spec.options, name=spec.name, source=str(spec.path))
                fitting.append(spec)
            except ModelTooLargeError as error:
                skipped.append(str(error))
        specs = fitting
    if as_json:  # otherwise listed in the summary
        for reason in skipped:
            typer.echo(f"skipped {reason}", err=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    def on_start(number: int, total: int, name: str) -> None:
        typer.echo(f"benchmarking {name} ({number}/{total}) ...", err=True)

    def on_done(run) -> None:
        if run.data is None:
            typer.echo(f"error: {run.name}: {run.error}", err=True)
            return
        if out is not None:
            (out / f"{run.name}.json").write_text(json.dumps(run.data, indent=2) + "\n")
        if not as_json:
            typer.echo(format_report(from_dict(run.data)) + "\n")

    runs = run_models(specs, arguments, on_start, on_done)
    if as_json:
        typer.echo(json.dumps([run.data for run in runs if run.data is not None], indent=2))
    else:
        typer.echo(format_summary(runs, skipped))
    if out is not None:
        typer.echo(f"results written to {out}", err=True)
    if any(run.data is None for run in runs):
        raise typer.Exit(1)


@app.command()
def benchmark(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model folder, Hugging Face repo id, or a folder of models (each "
            "subfolder benchmarked in a process of its own, in alphabetical order).",
        ),
    ],
    lengths: Annotated[
        str | None,
        typer.Option(
            help="State lengths in tokens, comma-separated. Default: 250,1000,4000,15000, "
            "those below the model's input limit or state cut (plus one near it when "
            "they stay below half of it)."
        ),
    ] = None,
    questions: Annotated[
        str, typer.Option(help="Numbers of questions, comma-separated.")
    ] = "1,5,20",
    repeats: Annotated[int, typer.Option(min=1, help="Timed runs per row.")] = 5,
    no_memory_check: NoMemoryCheckOption = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the results as JSON (with machine and versions)."),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Also write the results as JSON to this file, with the machine's "
            "configuration and the versions, to compare runs across Macs. With a "
            "folder of models: a folder, one <model>.json per model.",
        ),
    ] = None,
) -> None:
    """Measure load time and latency per request across state length and question count.

    For Clef, H2O-Lightning and d1, each row is also timed with its state's computed prefix kept
    (the warm columns), as a later request on the same state would run.
    """
    from .benchmark import benchmark as run_benchmark
    from .benchmark import format_report, to_dict

    grid = {
        "lengths": int_list(lengths, "--lengths") if lengths else None,
        "question_counts": int_list(questions, "--questions"),
    }
    if min(grid["question_counts"]) < 1:
        raise typer.BadParameter("at least one question per request", param_hint="--questions")
    if is_folder_of_models(model):
        arguments = ["--questions", questions, "--repeats", str(repeats)]
        arguments += ["--lengths", lengths] if lengths else []
        arguments += ["--no-memory-check"] if no_memory_check else []
        benchmark_models(model, arguments, not no_memory_check, as_json, out)
        return
    try:
        report = run_benchmark(model, check_memory=not no_memory_check, repeats=repeats, **grid)
    except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
        fail(str(error))
    if out is not None:
        out.write_text(json.dumps(to_dict(report), indent=2) + "\n")
    if as_json:
        typer.echo(json.dumps(to_dict(report), indent=2))
    else:
        typer.echo(format_report(report))
    if out is not None:
        typer.echo(f"results written to {out}", err=True)


training_app = typer.Typer(
    help="Fit a model to your labelled data, and score models on it.",
    no_args_is_help=True,
)
app.add_typer(training_app, name="training")

DataOption = Annotated[
    Path | None,
    typer.Option(
        "--data",
        help='Labelled examples as JSON lines: a request body plus "labels" per line, '
        'e.g. {"state": "...", "questions": {...}, "labels": {"team": "billing"}}.',
    ),
]


def echo_event(event) -> None:
    """An event as one JSON line on stdout, at once."""
    typer.echo(json.dumps(event.as_json(), ensure_ascii=False))
    sys.stdout.flush()


def default_output(model: str, data: Path) -> Path:
    """``<base>-<data name>`` in the current folder."""
    base = model.rstrip("/").rsplit("/", 1)[-1] if "/" in model else Path(model).name
    return Path(f"{base or 'model'}-{data.stem}")


YesOption = Annotated[
    bool,
    typer.Option(
        "--yes", "-y", help="Take the defaults for what the flags leave open; ask nothing."
    ),
]


@training_app.command("fit")
def training_fit(
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="The base: a model folder or Hugging Face repo id, or a fitted model (its "
            "fit is replaced, its base kept).",
        ),
    ] = None,
    data: DataOption = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Folder for the fitted model. Default: <base>-<data name> in this folder.",
        ),
    ] = None,
    method: Annotated[
        str | None,
        typer.Option(
            "--method",
            help="auto (the family's own kind, then calibrate; the default), calibrate (every "
            "family), head (Clef), readout (pplx, d3, H2O-Lightning, d1) or finetune (Laya, "
            "Julia).",
        ),
    ] = None,
    validation: Annotated[
        Path | None,
        typer.Option(
            "--validation",
            help="Labelled examples that score the report, instead of a holdout of --data.",
        ),
    ] = None,
    holdout: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=0.9,
            help="Share of --data held out (by label) to score the report. Default: 0.2.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed of the splits (and the training order).")] = 0,
    epochs: Annotated[
        int | None,
        typer.Option(
            min=1, help="Epochs of the head and finetune kinds; the best is kept. Default: 3."
        ),
    ] = None,
    head_only: Annotated[
        bool, typer.Option("--head-only", help="finetune: train the marker head alone.")
    ] = False,
    no_early_stop: Annotated[
        bool,
        typer.Option(
            "--no-early-stop",
            help="Train every epoch, even after one is worse on the dev part than the best "
            "before it (by default that ends the training). The best epoch is kept either way.",
        ),
    ] = False,
    max_input_tokens: MaxInputOption = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace the output folder if it exists.")
    ] = False,
    events: Annotated[
        bool,
        typer.Option(
            "--events",
            help="Print the progress as JSON lines on stdout instead of a progress bar and "
            'the report; a failed run ends with {"stage": "error"}, a cancelled one with '
            '{"stage": "cancelled"}. Asks nothing.',
        ),
    ] = False,
    yes: YesOption = False,
) -> None:
    """Fit a model to labelled examples and write the fitted model.

    The holdout (or --validation) only scores the report; the rest is the
    training part. In a terminal, what the flags leave open is asked in
    menus (the kind with its peak memory, the holdout, epochs, the output). Ctrl-C
    stops at the next step and writes nothing.
    """
    from .training import Cancelled, fit
    from .training.console import Progress, format_fit, interrupt_cancels
    from .training.wizard import Cancelled as MenusLeft
    from .training.wizard import FitPlan, ask_fit

    plan = FitPlan(
        model=model,
        data=str(data) if data is not None else None,
        output=str(output) if output is not None else None,
        method=method,
        validation=str(validation) if validation is not None else None,
        holdout=holdout,
        seed=seed,
        epochs=epochs,
        head_only=head_only,
        early_stop=not no_early_stop,
        load_options=load_options(None, max_input_tokens),
        overwrite=overwrite,
    )
    asked = plan.is_open() and not (yes or events) and terminal()
    if asked:
        try:
            plan = ask_fit(plan, default_output)
        except MenusLeft:
            fail("cancelled")
        except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
            fail(str(error))
    else:
        if plan.model is None:
            fail("pass -m MODEL (the base)")
        if plan.data is None:
            fail("pass --data FILE.jsonl (labelled examples)")
        plan.with_defaults(default_output)
    progress = Progress(sys.stderr, sys.stderr.isatty(), plan=not asked)
    on_event = echo_event if events else progress
    with interrupt_cancels(sys.stderr) as stop:
        try:
            result = fit(
                plan.model,
                plan.data,
                output=plan.output,
                method=plan.method,
                validation=plan.validation,
                holdout=plan.holdout if plan.holdout is not None else 0.2,
                seed=plan.seed,
                epochs=plan.epochs,
                head_only=plan.head_only,
                early_stop=plan.early_stop,
                on_event=on_event,
                cancel=stop,
                overwrite=plan.overwrite,
                **plan.load_options,
            )
        except Cancelled:
            progress.close()
            if events:
                echo_event(stage_event("cancelled", "nothing was written"))
            typer.echo("cancelled; nothing was written", err=True)
            raise typer.Exit(130) from None
        except (
            DecisionError,
            FileExistsError,
            FileNotFoundError,
            ValueError,
            PlatformError,
        ) as error:
            progress.close()
            message = str(error)
            if isinstance(error, FileExistsError):
                message = f"{plan.output} exists; pass --overwrite to replace it"
            if events:
                echo_event(stage_event("error", message))
            fail(message)
    progress.close()
    if not events:
        typer.echo(format_fit(result.report, str(result.path)))


def stage_event(stage: str, message: str):
    """How a run ended other than done: ``error`` or ``cancelled``."""
    from .training import Event

    return Event(stage, message=message)


@training_app.command("eval")
def training_eval(
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Model folder or Hugging Face repo id (or a fitted model)."
        ),
    ] = None,
    data: DataOption = None,
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help="A second model scored after the first (typically the base of a fitted "
            "model), shown beside it. One model is loaded at a time.",
        ),
    ] = None,
    max_input_tokens: MaxInputOption = None,
    no_memory_check: NoMemoryCheckOption = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the results as JSON.")] = False,
) -> None:
    """Score models on labelled examples: accuracy, macro-F1, ECE (and MAE for
    scores), refusals and a confusion matrix per question."""
    from .training import Cancelled, evaluate
    from .training.console import Progress, format_eval, interrupt_cancels

    if (model is None or data is None) and terminal():
        from .training.wizard import Cancelled as MenusLeft
        from .training.wizard import ask_eval

        try:
            model, data = ask_eval(model, data)
        except MenusLeft:
            fail("cancelled")
    if model is None:
        fail("pass -m MODEL")
    if data is None:
        fail("pass --data FILE.jsonl (labelled examples)")
    progress = Progress(sys.stderr, sys.stderr.isatty())
    with interrupt_cancels(sys.stderr) as stop:
        try:
            result = evaluate(
                model,
                data,
                compare=compare,
                on_event=progress,
                cancel=stop,
                check_memory=not no_memory_check,
                **load_options(None, max_input_tokens),
            )
        except Cancelled:
            progress.close()
            typer.echo("cancelled", err=True)
            raise typer.Exit(130) from None
        except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
            progress.close()
            fail(str(error))
    progress.close()
    if as_json:
        typer.echo(json.dumps(result.as_json(), indent=2, ensure_ascii=False))
    else:
        typer.echo(format_eval(result))


def pairs(value: str | None, flag: str) -> dict[str, str] | None:
    """``a=x,b=y`` as a dict, in order (``\\,`` for a comma in a value)."""
    from .flags import split_pairs

    if value is None:
        return None
    try:
        return split_pairs(value)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint=flag) from None


def terminal() -> bool:
    """Whether menus can be shown: input and output are a terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def question_from_file(path: Path) -> tuple[str, dict]:
    """The one question of a ``-q`` file (a questions map or a whole request body)."""
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise typer.BadParameter(f"cannot read {path}: {error}", param_hint="-q") from None
    questions = loaded.get("questions", loaded) if isinstance(loaded, dict) else None
    if not isinstance(questions, dict) or len(questions) != 1:
        raise typer.BadParameter(
            "the file must hold exactly one question (a questions map or a request body)",
            param_hint="-q",
        )
    [(question_id, question)] = questions.items()
    return question_id, question


def show_made(made, count: int) -> None:
    """The first examples as JSON lines, then the label counts."""
    for example in made.examples[:count]:
        typer.echo(json.dumps(example.to_json(), ensure_ascii=False))
    typer.echo("")
    typer.echo(summary_of_made(made))


def summary_of_made(made) -> str:
    lines = [f"{len(made.examples):,} examples"]
    if made.mapping is not None:
        used = ", ".join(f"{role}={name}" for role, name in made.mapping.columns.items() if name)
        lines[0] += f" ({made.mapping.shape} shape: {used})"
    for question_id, counts in made.label_counts().items():
        listed = ", ".join(f"{label} {count:,}" for label, count in counts.items())
        lines.append(f"labels: {question_id}: {listed}")
    for reason, count in made.skipped.items():
        lines.append(f"skipped {count:,} rows: {reason}")
    return "\n".join(lines)


ColumnFlag = Annotated[str | None, typer.Option(help="Column name (overrides the guess).")]


@app.command("data")
def data_command(
    source: Annotated[
        str | None,
        typer.Argument(
            help="hf:ORG/NAME (a Hugging Face dataset), a .csv, .jsonl or .parquet file, or "
            "a folder saved by the datasets library.",
            show_default=False,
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="The JSON lines file to write.")
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            help="A coverage-suite dataset with its wording, pinned to its commit: trec, "
            "offensive, sst5, openbookqa or anli-r3 (with --split)."
        ),
    ] = None,
    config: Annotated[str | None, typer.Option(help="The dataset's config.")] = None,
    split: Annotated[str | None, typer.Option(help="The dataset's split.")] = None,
    revision: Annotated[str | None, typer.Option(help="A commit, branch or tag.")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Only the first N rows.")] = None,
    sample: Annotated[
        int | None, typer.Option(min=1, help="N rows drawn at random (with --seed).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Seed of --sample.")] = 0,
    show: Annotated[
        int | None,
        typer.Option(min=0, help="Print the first N lines and the label counts; write nothing."),
    ] = None,
    shape: Annotated[
        str | None,
        typer.Option(
            help="text, pair, choice, yesno or rating. Default: guessed from the columns."
        ),
    ] = None,
    text: ColumnFlag = None,
    label: ColumnFlag = None,
    premise: ColumnFlag = None,
    hypothesis: ColumnFlag = None,
    question_column: ColumnFlag = None,
    choices: ColumnFlag = None,
    answer: ColumnFlag = None,
    context: ColumnFlag = None,
    question: Annotated[
        str | None,
        typer.Option(help="The question's instructions (yesno: each row's question is used)."),
    ] = None,
    question_id: Annotated[
        str | None, typer.Option("--id", help="The question's id. Default: the dataset's name.")
    ] = None,
    options: Annotated[
        str | None,
        typer.Option(
            help="Option ids and descriptions: 'id=description,...' (\\, for a comma). "
            "Default: the label names."
        ),
    ] = None,
    questions_file: Annotated[
        Path | None,
        typer.Option("--questions", "-q", help="JSON file with the whole question (as run's -q)."),
    ] = None,
    label_map: Annotated[
        str | None,
        typer.Option(help="Raw labels to option ids: '0=negative,1=positive'."),
    ] = None,
    noul: Annotated[
        bool,
        typer.Option("--noul", help="Two classes as a yes/no question (the second class is yes)."),
    ] = False,
    levels: Annotated[
        str | None,
        typer.Option(
            help="rating: level descriptions, lowest first: 'bad,ok,good' (\\, for a comma)."
        ),
    ] = None,
    no_input: Annotated[
        bool,
        typer.Option(
            "--no-input",
            help="Never ask (also without a terminal); anything left open is an error.",
        ),
    ] = False,
) -> None:
    """Turn a dataset into labelled examples for training fit (JSON lines).

    In a terminal, what the flags leave open is asked in menus, which end by
    printing the command that does the same without them.
    """
    from .data.format import write_examples
    from .data.make import DataPlan
    from .flags import split_list

    plan = DataPlan(
        source=source,
        output=str(output) if output is not None else None,
        preset=preset,
        config=config,
        split=split,
        revision=revision,
        limit=limit,
        sample=sample,
        seed=seed,
        show=show,
        shape=shape,
        columns={
            "text": text,
            "label": label,
            "premise": premise,
            "hypothesis": hypothesis,
            "question": question_column,
            "choices": choices,
            "answer": answer,
            "context": context,
        },
        question_id=question_id,
        instructions=question,
        options=pairs(options, "--options"),
        label_map=pairs(label_map, "--label-map"),
        noul=noul,
        levels=split_list(levels) if levels else None,
    )
    if questions_file is not None:
        if question is not None or options is not None:
            fail("-q holds the whole question; drop --question and --options")
        file_id, plan.question = question_from_file(questions_file)
        plan.questions_file = str(questions_file)
        plan.question_id = question_id or file_id
    open_steps = (
        (plan.source is None and plan.preset is None)
        or (plan.output is None and plan.show is None)
        or not plan.has_wording()
    )
    try:
        if open_steps and not no_input and terminal():
            from .data.wizard import Cancelled, ask_data, same_without_menus

            try:
                plan, made = ask_data(plan)
            except Cancelled:
                fail("cancelled")
            if plan.show is not None:
                typer.echo(same_without_menus(plan), err=True)
                return
            write_examples(plan.output, made.examples)  # the summary was in the preview
            typer.echo(f"wrote {plan.output}", err=True)
            typer.echo(same_without_menus(plan), err=True)
            return
        else:
            if plan.output is None and plan.show is None:
                fail("pass -o FILE.jsonl (or --show N to preview)")
            made = plan.make()
            if plan.show is not None:
                show_made(made, plan.show)
                return
    except (DecisionError, FileNotFoundError, ValueError) as error:
        fail(str(error))
    write_examples(plan.output, made.examples)
    typer.echo(summary_of_made(made), err=True)
    typer.echo(f"wrote {plan.output}", err=True)
