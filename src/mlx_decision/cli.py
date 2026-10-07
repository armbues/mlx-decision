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
    help="Run decision models on Apple Silicon.",
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
    body: dict[str, Any] = {}
    if questions_file is not None:
        try:
            data = json.loads(questions_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:  # ValueError: not UTF-8 or not JSON
            fail(f"cannot read questions from {questions_file}: {error}")
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
    try:
        return json.loads(state)
    except json.JSONDecodeError as error:
        raise DecisionError(f"the state is not valid JSON: {error}", param="state") from None


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

    Each line is a request body; fields it leaves out (usually the questions)
    come from ``defaults``.
    """
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                raise DecisionError(f"not valid JSON: {error}") from None
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
        "tokens). Default: 2. 0 = no cap beyond the model's own, as in Cloudflare's reference.",
    ),
]


MaxInputOption = Annotated[
    int | None,
    typer.Option(
        "--max-input-tokens",
        min=16,
        help="Input limit in tokens; longer states are cut (or refused by models that do not "
        "cut). Default: the model's own (Clef 16,384; Laya 512 or 1,024; Julia 8,192). Laya "
        "and Julia take up to 8,192.",
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


def load_options(max_image_mp: float | None, max_input_tokens: int | None) -> dict[str, Any]:
    """Load options for the flags that were given."""
    options = image_options(max_image_mp)
    if max_input_tokens is not None:
        options["max_input_tokens"] = max_input_tokens
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
            source = (
                nullcontext(sys.stdin)
                if states == "-"
                else open(states, encoding="utf-8", errors="replace")  # noqa: SIM115
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
                    **load_options(max_image_mp, max_input_tokens),
                )
            except (DecisionError, FileNotFoundError, ValueError, PlatformError) as error:
                fail(str(error))
            if bare := undescribed_choices(loaded, body.get("questions")):
                typer.echo(describe_options_hint(bare, RUN_DESCRIBE_HOW), err=True)
            answer_states(loaded, body, lines)
        return
    body = add_images(build_request(questions, shorthand, state, state_file, state_json), image)
    try:
        parse_request(body)  # catch request errors before loading
        loaded = load(
            model, check_memory=not no_memory_check, **load_options(max_image_mp, max_input_tokens)
        )
        if bare := undescribed_choices(loaded, body.get("questions")):
            typer.echo(describe_options_hint(bare, RUN_DESCRIBE_HOW), err=True)
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
            model, check_memory=not no_memory_check, **load_options(max_image_mp, max_input_tokens)
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
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Download even if no model family can load it.")
    ] = False,
) -> None:
    """Download a model from the Hugging Face Hub (HF_TOKEN for private or gated ones)."""
    import httpx
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    from .download import check_repo, format_size
    from .download import download as fetch

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
        if parent is not None:
            local_dir = parent / repo_id.rsplit("/", 1)[-1]
        target = f" to {local_dir}" if local_dir else ""
        typer.echo(f"downloading {repo_id} ({format_size(check.size_bytes)}){target} ...", err=True)
        path = fetch(repo_id, local_dir, check.files)
    except RepositoryNotFoundError:
        fail(f"{repo_id}: not found on the Hub (or private: set HF_TOKEN)")
    except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError) as error:
        fail(f"{repo_id}: {str(error).splitlines()[0]}")
    typer.echo(f"downloaded {repo_id} to {path}", err=True)
    model = str(local_dir) if local_dir else repo_id
    typer.echo(f"try it: mlx-decision chat -m {model}", err=True)


@app.command()
def server(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    host: Annotated[str, typer.Option(help="Address to bind to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    max_image_mp: MaxImageOption = None,
    max_input_tokens: MaxInputOption = None,
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
    """Serve a model over HTTP (Jev-compatible API: POST /v1/systemone)."""
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        fail("the server needs extra packages: pip install 'mlx-decision[server]'")
    from .hub import resolve_model_path
    from .memory import check_fits
    from .model import load
    from .registry import check_options, detect_family

    typer.echo(f"loading {model} ...", err=True)
    options = load_options(max_image_mp, max_input_tokens)
    try:
        # Fail here, not inside the server.
        path = resolve_model_path(model)
        check_options(detect_family(path), options)
        if not no_memory_check:
            check_fits(path, options, name=model)
    except (FileNotFoundError, ValueError) as error:
        fail(str(error))
    if api_key:
        typer.echo("API key required on /v1/*", err=True)
    app = create_app(lambda: load(model, check_memory=False, **options), api_key=api_key)
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

    from .hub import resolve_model_path
    from .memory import check_fits
    from .model import load
    from .registry import check_options, detect_family

    # stdout carries the protocol: everything else goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    options = load_options(max_image_mp, max_input_tokens)
    try:
        # Fail here, before a client is waiting for the handshake.
        path = resolve_model_path(model)
        check_options(detect_family(path), options)
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
            help="Leave the output embedding matrix (lm_head) unquantized.",
        ),
    ] = False,
    target_bits: Annotated[
        float | None,
        typer.Option(
            min=2,
            max=8,
            help="Mixed precision: average bits per weight, allocated by measured "
            "sensitivity (implies --quantize; takes a few minutes).",
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


@app.command()
def benchmark(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    lengths: Annotated[
        str | None,
        typer.Option(
            help="State lengths in tokens, comma-separated. Default: 250,1000,4000,15000, "
            "those below the model's input limit (plus one near it for small limits)."
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
            "configuration and the versions, to compare runs across Macs.",
        ),
    ] = None,
) -> None:
    """Measure load time and latency per request across state length and question count."""
    from .benchmark import benchmark as run_benchmark
    from .benchmark import format_report, to_dict

    grid = {
        "lengths": int_list(lengths, "--lengths") if lengths else None,
        "question_counts": int_list(questions, "--questions"),
    }
    if min(grid["question_counts"]) < 1:
        raise typer.BadParameter("at least one question per request", param_hint="--questions")
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
