# Server and MCP

Serving a model to programs: over HTTP with an API compatible with
Jev's (`server`), and to AI agents over the Model Context Protocol
(`mcp`).

- [Server](#server), [several models](#several-models)
- [Agents: MCP](#agents-mcp): [Claude Code](#claude-code),
  [Claude Desktop](#claude-desktop), [over HTTP](#over-http)

## Server

```bash
mlx-decision server -m Cloudflare/clef-flash --port 8000
```

Serves `POST /v1/systemone`, compatible with the Jev API's request and
response bodies, plus `GET /v1/models` and `GET /health`. Clients written
for that API work by changing the base URL, for example with its Python
SDK:

```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8000 TYPESAFE_API_KEY=unused python my_script.py
```

The server binds to localhost by default (`--host` to change), answers
one request at a time (others queue),
returns `422` with the API's error body for invalid requests, and sets
`X-MLX-Decision-Truncated: true` when the state was truncated. Request
bodies larger than 64 MB get `413`. Bodies may be zstd-compressed
(`Content-Encoding: zstd`), as the SDK sends them on Python 3.14 or with
its `zstd` extra; the limit applies to the decoded body.

Images go in the request's `images` list as base64 data URLs
(`data:image/png;base64,...`; PNG, JPEG or WebP). The server reads no files
and fetches no URLs, so anything else is refused with `422` naming the
image (`images.0`). With the SDK, pass them as
`extra_body={"images": [...]}`, or put them in the state as data URLs
([images inside the state](PYTHON.md#images)). The Jev API's own image format
(`assets`, the SDK's image uploads) and its `box` questions are not
served yet: such requests are refused (`422` or `415`), not answered
without them.

To require a key, start it with `--api-key KEY` (or set
`MLX_DECISION_API_KEY`). Requests to `/v1/*` then need
`Authorization: Bearer KEY`, which is what the SDK sends from
`TYPESAFE_API_KEY`; others get `401`. `/health` stays open. Use a long
random key (`python -c "import secrets; print(secrets.token_urlsafe(24))"`):
after 10 wrong keys in a row a client waits, from one second doubling up
to a minute (`429` with `Retry-After`), and each wrong key is logged
with the client's address.

A web page in your browser cannot use the server: requests must name the
server's address in `Host` (so DNS rebinding does not work), and those
sent by another site's pages (with a foreign `Origin`) get `403`.
Clients that send no `Origin`, such as the SDK and curl, are unaffected.

### Several models

One server can serve several models: repeat `-m`, or pass a folder of
models (each subfolder a model, named after it; folders no model family
loads are skipped). The request's `model` field picks one, which is what
the SDK sends with `model=`:

```bash
mlx-decision server -m ~/Models --default-model clef-flash --memory-budget 30
```

Models load on their first request and stay loaded while they fit in
`--memory-budget` (in GB; default: the GPU's recommended working set);
when the next one would not fit, the least recently used are unloaded
first. Sizes are estimated from the weights, plus the memory Clef,
H2O-Lightning and d1 keep for recent states. A request without a `model`, or with one the server
does not have (such as the SDK's default `jev-latest`), goes to
`--default-model`, which also loads at startup. With a single model that
model is the default; with several and no `--default-model`, such
requests get `422` listing the models. `GET /v1/models` lists them
(`release_date` is empty: a local model has none), and
`/health` names the default and the loaded ones. A model too large for
the budget gets `422` with the memory check's message.

## Agents: MCP

`mlx-decision mcp` serves a model to AI agents over the
[Model Context Protocol](https://modelcontextprotocol.io), so an agent in
Claude Code, Claude Desktop, Cursor or any other MCP client can call a
decision as a tool, answered on your Mac. It needs the `mcp` extra.

The server has two tools:

- `decide` takes the fields of a request body (`state`, `questions`, and
  `images` for Clef, pplx, d3, Base Labs, H2O-Lightning and d1 LFM3) and
  returns the response body as structured content, plus `truncated`. An invalid request comes back as a tool
  error naming the field, which the agent can read and correct.
- `model_info` describes the loaded model: family, precision,
  H2O-Lightning's reading, input limit, option limits and whether it
  reads images.

The tool descriptions explain decision models to an agent that has never
heard of them, with an example, and add what matters for the loaded
model: its input limit and how to split longer texts, its option limit,
and for Julia that options need short descriptions. `decide` refuses
more options than the model tells apart (16 to 21 for Laya, 20 for
Julia) and tells the agent to choose among groups first; `run`, `server`
and the Python API keep Laya's own behaviour of shortening the list.

### Claude Code

```bash
claude mcp add decision -- mlx-decision mcp -m convaiinnovations/laya
```

`--scope user` makes it available in all your projects. Then ask
something like "Use the decision tool to sort these five tickets by
team", and the agent calls `decide` once per ticket.

### Claude Desktop

Add the server to `claude_desktop_config.json` (Settings, Developer,
Edit Config) and restart the app. Claude Desktop does not search your
shell's `PATH`, so give the full path that `which mlx-decision` prints:

```json
{
  "mcpServers": {
    "decision": {
      "command": "/Users/you/.venv/bin/mlx-decision",
      "args": ["mcp", "-m", "convaiinnovations/laya"]
    }
  }
}
```

Other clients take the same command and arguments. On stdio, each client
starts its own process with its own copy of the model, loaded before the
client's handshake is answered (about a second, longer the first time
the weights are read from disk). A repo id that is not downloaded yet is
downloaded first, which can take longer than a client waits, so fetch
the model beforehand with `mlx-decision download` or pass a local
folder. Logs go to stderr, which clients show in their MCP log.

`images` takes data URLs and, on stdio only, absolute file paths, which
the server reads.

### Over HTTP

To share one loaded model between several agents (or keep Clef loaded
once), start one process with `--http`:

```bash
mlx-decision mcp --http -m Cloudflare/clef-flash --port 8001 --api-key KEY
claude mcp add --transport http decision http://127.0.0.1:8001/mcp \
  --header "Authorization: Bearer KEY"
```

It serves streamable HTTP at `http://HOST:PORT/mcp`, binds to localhost
by default (`--host` to change; port 8000 by default, as `server`),
answers one call at a time (others queue), and takes images as data URLs
only: it reads no files on behalf of a client. The API key works as for
[`server`](#server) (`--api-key` or `MLX_DECISION_API_KEY`); without one,
requests are not checked.
