"""An MCP server with two tools: ``decide`` and ``model_info``.

Needs the ``mcp`` extra. Built on the SDK's low-level server so the tools
have schemas and descriptions written for agents, and arguments are checked
by the same validation as the Python API, ``run`` and ``server``. All model
work runs on the one worker thread the model was loaded on.
"""

import asyncio
import hmac
import json
import logging
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from . import __version__
from .errors import DecisionError
from .images import check_data_urls
from .model import DecisionModel

logger = logging.getLogger(__name__)

# Room for several large images as data URLs, as on the HTTP server.
MAX_BODY_BYTES = 64 * 2**20

READ_ONLY = types.ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)

DECIDE = """\
Answer typed questions about a text (the state) with {name}, a local \
decision model. It generates no text: for each question it returns a \
probability for every allowed answer, in one pass. Use it to classify, \
route, triage, rate or check text: which team should handle a ticket, how \
angry a message is, whether a document mentions a person.

Question types, keyed by an id of your choice in `questions`:
- noul: a yes/no question. Answer: `noul`, the probability of yes.
- choice: pick one option of `criteria` (option id -> short description). \
Answer: `choice` (the most likely id), `confidence`, `probabilities` per option.
- score: rate on ordered levels; `criteria` lists their descriptions from \
lowest to highest. Answer: `score`, the expected level as a 0-based index \
into `criteria` (fractional: 1.7 lies between levels 1 and 2), `confidence`, \
`legend` (index -> level description) and `probabilities` per level. To name \
one level, take the most probable one and report its `legend` text; do not \
cut `score` down to an integer.

Example arguments:
{{"state": "I was charged twice for my subscription this month.", \
"questions": {{"team": {{"type": "choice", "instructions": "Which team should \
handle this request?", "criteria": {{"billing": "Billing and payment disputes", \
"technical": "Technical problems and errors", "sales": "Sales and new purchases"}}}}, \
"urgent": {{"type": "noul", "instructions": "Does this need a reply today?"}}}}}}

The state is one item (a message, a review, a document). Ask all \
questions about it in one call; for several items, make one call per item \
(calls may run in parallel), since every question reads the whole state and \
cannot address a part of it. `confidence` runs from 0 \
to 1; a low value (below about 0.5) means the model hesitates between \
answers: report such an answer as uncertain, with the runner-up. Report \
the model's answers as they are; if you disagree with one, say so and why \
rather than replacing it."""

MODEL_INFO = """\
Describe the loaded decision model: name, family, precision, input limit \
in tokens, question types, option and level limits, and whether it reads \
images. Call it when a request might hit a limit."""

INSTRUCTIONS = """\
Local decision model ({name}): `decide` answers yes/no, choice and score \
questions about a text with probabilities, fast and without generating \
text. `model_info` lists its limits."""


def option_limit(info: dict[str, Any]) -> int | None:
    """Options per question that ``decide`` takes, or None for no limit.

    The family's own limit, and for a model that cuts options to fit, the
    count that keeps about 12 tokens each: beyond it the descriptions
    shrink to a word or two and the answers turn confidently wrong.
    """
    limits = [info["max_choice_options"]]
    if info["question_tokens"] and info["truncates_input"]:
        limits.append(info["question_tokens"] // 12)
    return min((limit for limit in limits if limit), default=None)


def check_option_counts(body: Any, limit: int) -> None:
    questions = body.get("questions") if isinstance(body, dict) else None
    if not isinstance(questions, dict):
        return
    for question_id, question in questions.items():
        criteria = question.get("criteria") if isinstance(question, dict) else None
        if isinstance(criteria, (dict, list)) and len(criteria) > limit:
            raise DecisionError(
                f"{len(criteria)} options; this model takes at most {limit}. Choose "
                "among groups of options first, then ask again with the options of the "
                "chosen group",
                param=f"questions.{question_id}.criteria",
            )


def family_hints(info: dict[str, Any], allow_image_paths: bool) -> list[str]:
    """What an agent should know about this model beyond the general description."""
    hints = []
    if info["max_input_tokens"] and info["truncates_input"]:
        words = info["max_input_tokens"] * 3 // 4
        hints.append(
            f"It reads at most {info['max_input_tokens']:,} tokens (about {words:,} words, "
            "questions included); a longer state is cut, the model sees only part of it "
            "and the result has `truncated: true`. Split longer texts into parts (by "
            "section or paragraph), ask about each part in its own call and combine "
            "the answers."
        )
    elif info["max_input_tokens"]:
        hints.append(
            f"Requests longer than {info['max_input_tokens']:,} tokens are refused; "
            "split long texts."
        )
    if limit := option_limit(info):
        why = ""
        if info["truncates_input"] and limit != info["max_choice_options"]:
            why = f" (a question and its options share {info['question_tokens']} tokens)"
        hints.append(
            f"A choice or score takes at most {limit} options or levels{why}. For more, "
            "first choose among groups of options, then ask again with the options of "
            "the chosen group."
        )
    if info["family"] == "julia":
        hints.append(
            "Describe every choice option in `criteria` with a short phrase of what it "
            'covers ("Billing and payment disputes"): with bare ids or one-word labels '
            "this model answers much worse."
        )
    if info["supports_images"]:
        where = "data URLs (data:image/png;base64,...)"
        if allow_image_paths:
            where += " or absolute file paths"
        hints.append(f"`images` attaches images the state refers to, as {where}.")
    return hints


def decide_schema(info: dict[str, Any], allow_image_paths: bool) -> dict[str, Any]:
    text = {"type": "string", "description": "The question or option in plain words."}
    question = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"const": "noul"},
                    "instructions": {**text, "description": "The yes/no question."},
                    "criteria": {
                        "type": "object",
                        "description": "Optional: what yes and no mean.",
                        "properties": {"true": text, "false": text},
                        "additionalProperties": False,
                    },
                },
                "required": ["type", "instructions"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "choice"},
                    "instructions": {**text, "description": "What to choose."},
                    "criteria": {
                        "type": "object",
                        "description": "Option id -> short description of the option.",
                        "additionalProperties": text,
                        "minProperties": 1,
                    },
                },
                "required": ["type", "instructions", "criteria"],
            },
            {
                "type": "object",
                "properties": {
                    "type": {"const": "score"},
                    "instructions": {**text, "description": "What to rate."},
                    "criteria": {
                        "type": "array",
                        "description": "Level descriptions, lowest first.",
                        "items": text,
                        "minItems": 2,
                    },
                },
                "required": ["type", "instructions", "criteria"],
            },
        ]
    }
    properties: dict[str, Any] = {
        "state": {
            "description": "The text to decide about (or any JSON value).",
            "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}],
        },
        "questions": {
            "type": "object",
            "description": "Questions by an id of your choice.",
            "additionalProperties": question,
            "minProperties": 1,
        },
    }
    if info["supports_images"]:
        kind = "data URL or absolute file path" if allow_image_paths else "data URL"
        properties["images"] = {
            "type": "array",
            "description": f"Optional images the state refers to, each a {kind}.",
            "items": {"type": "string"},
        }
    return {"type": "object", "properties": properties, "required": ["state", "questions"]}


def tool_list(model: DecisionModel, allow_image_paths: bool) -> list[types.Tool]:
    info = model.info()
    hints = family_hints(info, allow_image_paths)
    description = DECIDE.format(name=model.name)
    if hints:
        description += "\n\nThis model: " + " ".join(hints)
    return [
        types.Tool(
            name="decide",
            title="Decide",
            description=description,
            input_schema=decide_schema(info, allow_image_paths),
            annotations=READ_ONLY,
        ),
        types.Tool(
            name="model_info",
            title="Decision model info",
            description=MODEL_INFO,
            input_schema={"type": "object", "properties": {}},
            annotations=READ_ONLY,
        ),
    ]


def json_result(body: dict[str, Any]) -> types.CallToolResult:
    """``body`` as structured content and, for clients that read only text, as JSON text."""
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return types.CallToolResult(content=[types.TextContent(text=text)], structured_content=body)


def error_result(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def create_server(
    model: DecisionModel, worker: Executor, allow_image_paths: bool = False
) -> Server:
    """The MCP server for ``model``; tool calls run one at a time on ``worker``.

    ``allow_image_paths`` lets ``decide`` read image files named in a call:
    only for stdio, where the client already has the server's file access.
    """
    tools = tool_list(model, allow_image_paths)
    limit = option_limit(model.info())

    async def on_worker(function: Callable, *args):
        return await asyncio.get_running_loop().run_in_executor(worker, function, *args)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        if params.name == "model_info":
            return json_result(model.info())
        if params.name != "decide":
            return error_result(f"unknown tool {params.name!r}")
        body = dict(params.arguments or {})
        body.pop("model", None)
        try:
            if not allow_image_paths:
                check_data_urls(body)
            if limit:
                check_option_counts(body, limit)
            result = await on_worker(model.decide_request, body)
        except DecisionError as error:
            return error_result(str(error))
        except Exception:
            logger.exception("tool call failed")
            return error_result("the model failed to answer the request")
        return json_result({**result.to_wire(), "truncated": result.truncated})

    return Server(
        "mlx-decision",
        version=__version__,
        instructions=INSTRUCTIONS.format(name=model.name),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve_stdio(load_model: Callable[[], DecisionModel]) -> None:
    """Serve over stdin/stdout until the client closes stdin.

    The transport is opened first: it moves anything else written to stdout
    over to stderr, so output during the load cannot corrupt the protocol.
    The client's handshake waits until the model is loaded.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision") as worker:
        async with stdio_server() as (read_stream, write_stream):
            model = await asyncio.get_running_loop().run_in_executor(worker, load_model)
            logger.info("model %s loaded", model.name)
            server = create_server(model, worker, allow_image_paths=True)
            await server.run(read_stream, write_stream, server.create_initialization_options())


class RequireApiKey:
    """ASGI middleware: HTTP requests need ``Authorization: Bearer <key>``, else 401."""

    def __init__(self, app, api_key: str):
        self.app = app
        self.expected = f"Bearer {api_key}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                body = json.dumps({"error": "missing or invalid API key"}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def http_app(server: Server, host: str, api_key: str | None = None):
    """The streamable HTTP app at ``/mcp``, behind the API key if there is one.

    The SDK turns on DNS rebinding protection when ``host`` is a loopback address.
    """
    app = server.streamable_http_app(host=host, max_request_body_size=MAX_BODY_BYTES)
    return RequireApiKey(app, api_key) if api_key else app


def serve_http(
    load_model: Callable[[], DecisionModel], host: str, port: int, api_key: str | None = None
) -> None:
    """Load the model, then serve over streamable HTTP until interrupted.

    Image paths are refused: a client must not make the server read its files.
    """
    import uvicorn

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-decision") as worker:
        model = worker.submit(load_model).result()
        logger.info("model %s loaded", model.name)
        server = create_server(model, worker, allow_image_paths=False)
        uvicorn.run(http_app(server, host, api_key), host=host, port=port, log_level="info")
