"""The ``mlx-decision`` command."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .errors import DecisionError

app = typer.Typer(
    help="Run decision models on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    pass


@app.command()
def run(
    model: Annotated[str, typer.Option("--model", "-m", help="Model folder or Hugging Face repo id.")],
    questions: Annotated[
        Path,
        typer.Option(
            "--questions", "-q", help="JSON file: a questions map, or a whole request body."
        ),
    ],
    state: Annotated[
        str | None, typer.Option("--state", "-s", help="The state as text. Default: read stdin.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the response body as JSON.")] = True,
) -> None:
    """Answer questions about a state."""
    from .model import load

    body = json.loads(questions.read_text())
    if not (isinstance(body, dict) and "questions" in body):
        body = {"questions": body}
    if state is not None:
        body["state"] = state
    elif "state" not in body:
        body["state"] = sys.stdin.read()

    try:
        result = load(model).decide_request(body)
    except (DecisionError, FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from None
    if result.truncated:
        typer.echo("warning: the state was truncated to fit the model's input limit", err=True)
    typer.echo(json.dumps(result.to_wire(), indent=2, ensure_ascii=False))
