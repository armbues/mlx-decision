"""The ``mlx-decision`` command."""

import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from .errors import DecisionError

app = typer.Typer(
    help="Run decision models on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)

QUESTION_ID = re.compile(r"[\w.\-]+")
BAR_WIDTH = 24
LEGEND_WIDTH = 48


@app.callback()
def main() -> None:
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
    for index, value in enumerate(nouls, 1):
        question_id, text = split_id(value, "--noul", required=False)
        questions[question_id or f"noul_{index}"] = {"type": "noul", "instructions": text}
    for flag, kind, values in (("--choice", "choice", choices), ("--score", "score", scores)):
        for value in values:
            question_id, rest = split_id(value, flag, required=True)
            options = [option.strip() for option in rest.split(",") if option.strip()]
            criteria = {option: None for option in options} if kind == "choice" else options
            if question_id in questions:
                raise typer.BadParameter(
                    f"question id {question_id!r} given twice", param_hint=flag
                )
            questions[question_id] = {"type": kind, "criteria": criteria}
    return questions


def build_request(
    questions_file: Path | None,
    shorthand: dict,
    state: str | None,
    state_file: Path | None,
    state_json: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if questions_file is not None:
        try:
            data = json.loads(questions_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
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

    if state is not None and state_file is not None:
        fail("give either --state or --state-file, not both")
    if state_file is not None:
        try:
            state = state_file.read_text()
        except OSError as error:
            fail(f"cannot read the state: {error}")
    if state is None and "state" not in body:
        if sys.stdin.isatty():
            fail("no state: give --state, --state-file, or pipe it to stdin")
        state = sys.stdin.read()
    if state is not None:
        if state_json:
            try:
                state = json.loads(state)
            except json.JSONDecodeError as error:
                fail(f"--state-json: the state is not valid JSON: {error}")
        body["state"] = state
    return body


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


@app.command()
def run(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    questions: Annotated[
        Path | None,
        typer.Option(
            "--questions", "-q", help="JSON file: a questions map, or a whole request body."
        ),
    ] = None,
    noul: Annotated[
        list[str] | None,
        typer.Option("--noul", help="A yes/no question: '[ID=]text'. Repeatable."),
    ] = None,
    choice: Annotated[
        list[str] | None,
        typer.Option("--choice", help="A choice: 'ID=option,option,...'. Repeatable."),
    ] = None,
    score: Annotated[
        list[str] | None,
        typer.Option("--score", help="A score: 'ID=lowest,...,highest'. Repeatable."),
    ] = None,
    state: Annotated[
        str | None,
        typer.Option("--state", "-s", help="The state as text. Default: read stdin."),
    ] = None,
    state_file: Annotated[
        Path | None, typer.Option("--state-file", help="Read the state from a file.")
    ] = None,
    state_json: Annotated[
        bool, typer.Option("--state-json", help="Parse the state as JSON (object, array, ...).")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the response body as JSON.")
    ] = False,
) -> None:
    """Answer questions about a state."""
    from .model import load

    shorthand = shorthand_questions(noul or [], choice or [], score or [])
    body = build_request(questions, shorthand, state, state_file, state_json)
    try:
        result = load(model).decide_request(body)
    except (DecisionError, FileNotFoundError, ValueError) as error:
        fail(str(error))
    if as_json:
        if result.truncated:
            typer.echo("note: the state was truncated to fit the model's input limit", err=True)
        typer.echo(json.dumps(result.to_wire(), indent=2, ensure_ascii=False))
    else:
        typer.echo(format_result(result))


@app.command()
def server(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    host: Annotated[str, typer.Option(help="Address to bind to.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Serve a model with the Jev API (POST /v1/systemone)."""
    try:
        import uvicorn

        from .server import create_app
    except ImportError:
        fail("the server needs extra packages: pip install 'mlx-decision[server]'")
    from .model import load

    typer.echo(f"loading {model} ...", err=True)
    uvicorn.run(create_app(lambda: load(model)), host=host, port=port, log_level="info")


@app.command()
def convert(
    model: Annotated[
        str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Folder to write.")],
    quantize: Annotated[
        bool, typer.Option("--quantize", "-q", help="Quantize the backbone.")
    ] = False,
    bits: Annotated[int, typer.Option(help="Bits per weight when quantizing.")] = 8,
    group_size: Annotated[int, typer.Option(help="Weights per quantization group.")] = 64,
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

    if quantize and bits not in (2, 3, 4, 5, 6, 8):
        raise typer.BadParameter("must be 2, 3, 4, 5, 6 or 8", param_hint="--bits")
    options = {}
    if target_bits is not None:

        def progress(done, total, unit, value):
            typer.echo(f"  sensitivity {done}/{total} {unit.name}: {value:.2e}", err=True)

        options = {"target_bits": target_bits, "progress": progress}
    typer.echo(f"converting {model} ...", err=True)
    try:
        convert_model(
            model,
            output,
            bits=bits if quantize else None,
            group_size=group_size,
            quantize_output_embeddings=not keep_output_embeddings,
            **options,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
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
        str, typer.Option(help="State lengths in tokens, comma-separated.")
    ] = "250,1000,4000,15000",
    questions: Annotated[
        str, typer.Option(help="Numbers of questions, comma-separated.")
    ] = "1,5,20",
    repeats: Annotated[int, typer.Option(min=1, help="Timed runs per row.")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Print the numbers as JSON.")] = False,
) -> None:
    """Measure load time and latency per request across state length and question count."""
    from .benchmark import benchmark as run_benchmark
    from .benchmark import format_report, to_dict

    grid = {
        "lengths": int_list(lengths, "--lengths"),
        "question_counts": int_list(questions, "--questions"),
    }
    if min(grid["question_counts"]) < 1:
        raise typer.BadParameter("at least one question per request", param_hint="--questions")
    try:
        report = run_benchmark(model, repeats=repeats, **grid)
    except (DecisionError, FileNotFoundError, ValueError) as error:
        fail(str(error))
    if as_json:
        typer.echo(json.dumps(to_dict(report), indent=2))
    else:
        typer.echo(format_report(report))
