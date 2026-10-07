# mlx-decision

Run decision models locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

A decision model reads a *state* (a message, a document, any JSON) and a set
of typed *questions*, and returns a probability for every allowed answer in
a single forward pass, without generating text. mlx-decision is to decision
models what `mlx-lm` is to LLMs: load a model, ask it questions from Python
or the command line, serve it over HTTP with an API compatible with
[Jev's](https://typesafe.ai), or hand it to agents as an
[MCP](#agents-mcp) tool.

| Model | Size | Input | Limit | Memory | Short request | Speedup | Good at |
|---|---|---|---|---|---|---|---|
| Cloudflare's [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 9B | text, images | 16,384 tokens | 19 GB (8-bit: 11 GB) | 0.25 s | 3.5x | knowledge, entailment, many options, images |
| Cloudflare's [clef](https://huggingface.co/Cloudflare/clef) | 27B | text, images | 16,384 tokens | 54 GB (8-bit: 32 GB, 4-bit: 19 GB) | 1.1 s (8-bit) | 3.7x (bf16, M2 Ultra) | as clef-flash, on Macs with more memory |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | English text | 512 tokens per question | 1.8 GB | 16 ms | 3.0x | topic classification of English text |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | English text | 1,024 tokens per question | 1.7 GB | 16 ms | 2.9x | as laya, tuned for typed decision workflows |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | text, 100+ languages | 1,024 tokens per question | 1.6 GB | 8 ms | 2.5x | topic classification in many languages |
| Supersonic Labs' [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144M | text, many languages | 8,192 tokens per question | 1.4 GB | 6 ms | 2.0x | emotion and routing with 2 to 20 options, long states |

"Short request": one question about a state of about 250 tokens, on an
Apple M5 Pro. "Speedup": the same request against the model's own PyTorch
code on the same Mac (MPS), with the same answers; across all measured
requests it ranges from 1.4 to 6 times ([Speed](#speed)). How the models differ is in
[Models](#models).

This is alpha software: the interface may still change. Questions and bug
reports go to the [issue tracker](https://github.com/armbues/mlx-decision/issues).

## Install

Requirements: a Mac with Apple Silicon (M1 or later) and macOS 14 or newer,
Python 3.12 or newer, and the memory the model needs (see the table; for
clef-flash a Mac with 32 GB or more, or a smaller copy made with
[convert](#quantizing-clef-flash-convert)).

```bash
pip install mlx-decision                    # library and the mlx-decision command
pip install "mlx-decision[images]"          # plus image input (Pillow, NumPy)
pip install "mlx-decision[server]"          # plus the HTTP server (FastAPI, uvicorn)
pip install "mlx-decision[mcp]"             # plus the MCP server for agents
pip install "mlx-decision[server,images]"   # several extras at once
```

From source: clone the [repository](https://github.com/armbues/mlx-decision)
and run `pip install -e ".[server,images,mcp]"` in it. MLX also has builds for
Linux and Windows, so pip may install mlx-decision there, but loading a
model stops with an error: the models run on Apple Silicon only.

### Getting a model

Models load from a local folder or a Hugging Face repo id; a repo id is
downloaded into the Hugging Face cache on first use. This is enough to
start:

```bash
mlx-decision run -m SupersonicLabs/Julia-1 -s "My card was charged twice." --noul "Is it a billing problem?"
```

`mlx-decision download` fetches a model ahead of time, from a menu of the
supported models or by repo id. In a terminal it asks for a folder to
download into (the model goes into a folder named after the repo inside
it, which you then pass to `-m`; an empty answer keeps the Hugging Face
cache); `--local-dir` names the folder directly. It warns before
downloading a repository no supported model family can load. Private or
gated repos need `HF_TOKEN`.

## Python

```python
import mlx_decision
from mlx_decision import Choice, Noul, Score

model = mlx_decision.load("Cloudflare/clef-flash")

ticket = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP."
)
result = model.decide(
    ticket,
    {
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
    },
)

print(result.answers["department"].choice)      # technical
print(result.answers["department"].confidence)  # 0.90
print(result.answers["frustration"].score)      # 1.29 (expected level, 0-based)
print(result.answers["is_urgent"].noul)         # 0.94 (probability of yes)
print(result.to_wire())                         # the response body, as JSON data
```

`confidence` follows the definitions of the Jev API (choice: how far the top
probability sits above an even split; score: how concentrated the
probability is around the most likely level), whatever the model's own
definition.

Questions can also be plain dicts in the API's wire format
(`{"type": "choice", "instructions": ..., "criteria": {...}}`), and
`model.decide_request(body)` takes a whole request body. Invalid requests
raise `mlx_decision.DecisionError`, whose `param` names the offending field
(building an invalid `Choice`, `Score` or `Noul` object directly raises
pydantic's `ValidationError` instead). `result.truncated` tells you whether
the state was cut to fit the model's input limit. The answer types
(`ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`) and `Usage` can be imported
from `mlx_decision`.

`load` takes options per model: `max_input_tokens` changes the input limit
(Laya and Julia up to 8,192), `strict_encoding=False` lets Julia cut what it
would otherwise refuse, `dtype` sets Laya's and Julia's precision (default
float16), and `max_image_pixels` caps image size for clef-flash.
`model.info()` describes the loaded model as JSON data: family,
precision, input limit, option limits and whether it takes images.

With the `images` extra, clef-flash takes images as file paths, bytes, PIL
images or data URLs:

```python
result = model.decide(
    "Customer says checkout fails; screenshot attached.",
    {"shows_error": Noul(instructions="The screenshot shows an error message")},
    images=["screenshot.png"],
)
```

## Command line

Ask questions about a state:

```bash
mlx-decision run -m Cloudflare/clef-flash \
  -s "My card was charged twice this month." \
  --noul "refund=The customer wants money back" \
  --choice "team=billing,technical,sales" \
  --score "anger=calm,annoyed,angry"
```

```
refund (noul): no, p(yes) 0.110
  ███

team (choice): billing, confidence 0.98
  billing    0.989  ████████████████████████
  technical  0.007
  sales      0.004

anger (score): 1.05 on 0-2, confidence 0.54
   0  0.128  ███                       calm
   1  0.695  █████████████████         annoyed
   2  0.177  ████                      angry

clef-flash · 291 input tokens
```

A score's answer is the expected level, weighted by the probabilities:
0 × 0.128 + 1 × 0.695 + 2 × 0.177 = 1.05 on a scale of 0 to 2.

The state can also come from `--state-file` or stdin (`--state-json` parses
it as JSON), and questions from a JSON file with `-q questions.json`
(a questions map or a whole request body). `--json` prints the exact
response body. `--image FILE` (repeatable) sends images along with the
state (clef-flash). `--max-input-tokens N` (also on `chat` and `server`)
changes the model's input limit, e.g. to give Laya up to 8,192 tokens.

`mlx-decision --version` prints the installed version.

### Many states

`--states FILE` answers a JSON lines file with one request body per line
and prints one response body per line, in the same order. `-` reads
stdin. A line is the same body the server takes, so it can bring its own
questions; questions from `-q` or the shorthand flags apply to lines
without them:

```bash
mlx-decision run -m Cloudflare/clef-flash -q questions.json --states tickets.jsonl > answers.jsonl
```

```
{"state": "My card was charged twice this month."}
{"state": {"ticket": 4711, "text": "The app crashes on login."}}
{"state": "Is this spam?", "questions": {"spam": {"type": "noul", "instructions": "Is it spam?"}}}
```

Lines can carry images as file paths or data URLs
(`{"state": "...", "images": ["shots/4711.png"]}`); `--image` flags apply
to lines without their own. A line that cannot be answered yields an
error body with its line number, and the run goes on:

```
{"error": {"message": "state is required", "type": "invalid_request_error", "param": "state"}, "line": 3}
```

## Interactive: `chat`

`mlx-decision chat` loads the model once and answers one state after
another: type or paste a state, finish it with a blank line, and quit with
Ctrl-D. Lines can be edited, Up recalls earlier states, and a toolbar at
the bottom shows the keys. Questions come from the same flags and files as
in `run`:

```bash
mlx-decision chat -m Cloudflare/clef-flash --choice "team=billing,technical,sales"
```

Without any questions, `chat` first asks for them: the type from a menu
(arrow keys or the number, then Enter), an id (Enter takes the one
suggested), the instructions, and then the options of a choice (`option`
or `option: description`) or the levels of a score (lowest first), one per
line, ending with a blank line. Each question is checked against the model
as soon as it is complete, and shown as understood.

```
$ mlx-decision chat -m Cloudflare/clef-flash
clef-flash loaded, 0 questions.
No questions yet: build the first one (Ctrl-C cancels).
 Type:
   >  1. choice  pick one of several options
      2. score   rate on ordered levels
      3. noul    a yes/no question
id: team
instructions (optional): Which team should handle this?
options, one per line as 'option' or 'option: description' (blank line ends):
  billing: payments, invoices, refunds
  technical
  sales

added:
  team (choice): Which team should handle this?
    billing: payments, invoices, refunds
    technical
    sales
 Next:
      1. Start answering states
   >  2. Add another question
 Type:
      1. choice  pick one of several options
      2. score   rate on ordered levels
   >  3. noul    a yes/no question
id: refund
instructions: The customer wants money back
added:
  refund (noul): The customer wants money back
 Next:
   >  1. Start answering states
      2. Add another question
Type a state and finish it with a blank line. /help lists the commands, Ctrl-D quits.
state> My card was charged twice this month.

team (choice): billing, confidence 0.93
  billing    0.951  ███████████████████████
  technical  0.028  █
  sales      0.021  █

refund (noul): no, p(yes) 0.030
  █

clef-flash · 225 input tokens

state> /save questions.json
saved 2 questions to questions.json
```

Between states, commands change the questions, whether they came from the
builder, flags or a file. `/edit` and `/remove` without an id offer a menu
of the questions.

| Command | |
|---|---|
| `/list` | show the questions |
| `/add` | build another question |
| `/edit [ID]` | change a question; the current values are filled in |
| `/remove [ID]` | remove a question |
| `/save FILE` | write the questions as a file that `-q` reads |
| `/image FILE`, `/image clear` | attach an image to the following states, or drop them (`--image` attaches at start) |
| `/help`, `/quit` | |

A state that itself starts with `/` is typed as `//`. `chat` needs a
terminal; scripts use `run --states`. With `--json`, the response bodies
go to stdout and can be redirected while the prompts stay on screen.

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

The server binds to localhost by default (`--host` to change), ignores the
request's `model` field, answers one request at a time (others queue),
returns `422` with the API's error body for invalid requests, and sets
`X-MLX-Decision-Truncated: true` when the state was truncated. Request
bodies larger than 64 MB get `413`.

Images go in the request's `images` list as base64 data URLs
(`data:image/png;base64,...`; PNG, JPEG or WebP). The server reads no files
and fetches no URLs, so anything else is refused with `422` naming the
image (`images.0`). With the SDK, pass them as
`extra_body={"images": [...]}`.

To require a key, start it with `--api-key KEY` (or set
`MLX_DECISION_API_KEY`). Requests to `/v1/*` then need
`Authorization: Bearer KEY`, which is what the SDK sends from
`TYPESAFE_API_KEY`; others get `401`. `/health` stays open.

## Agents: MCP

`mlx-decision mcp` serves a model to AI agents over the
[Model Context Protocol](https://modelcontextprotocol.io), so an agent in
Claude Code, Claude Desktop, Cursor or any other MCP client can call a
decision as a tool, answered on your Mac. It needs the `mcp` extra.

The server has two tools:

- `decide` takes the fields of a request body (`state`, `questions`, and
  `images` for clef-flash) and returns the response body as structured
  content, plus `truncated`. An invalid request comes back as a tool
  error naming the field, which the agent can read and correct.
- `model_info` describes the loaded model: family, precision, input
  limit, option limits and whether it reads images.

The tool descriptions explain decision models to an agent that has never
heard of them, with an example, and add what matters for the loaded
model: its input limit and how to split longer texts, its option limit,
and for Julia that options need short descriptions. Laya cuts long
option lists to fit its question budget, and with more than about 16
options the answers turn confidently wrong, so `decide` refuses more
options than the model tells apart (16 for laya, 21 for
laya-typed-decisions and laya-multilingual; Julia's own limit is 20) and
tells the agent to choose among groups of options first. `run`,
`server` and the Python API keep Laya's own behaviour.

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

To share one loaded model between several agents (or keep clef-flash's
19 GB loaded once), start one process with `--http`:

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

## Models

The models differ in size and in how they read a request. clef-flash reads
the state once and answers all questions together; Laya and Julia read each
question on its own, with the whole state, so their input limit applies per
question and their time grows with the number of questions.

Accuracy on five public benchmarks, 500 test examples each, with the same
questions and option descriptions for every model (preliminary; Jev is
TypeSafe AI's hosted model, for comparison):

| Model | AG News (topic, 4 options) | DAIR Emotion (6) | ANLI (entailment, 3) | BANKING77 (intent, 77) | MMLU (knowledge, 4) |
|---|---|---|---|---|---|
| clef-flash | 91.4 | 60.0 | 58.2 | **96.0** | **93.0** |
| laya | **94.6** | 59.8 | 48.6 | 36.0 | 35.2 |
| laya-typed-decisions | **94.6** | 61.2 | 47.4 | 36.2 | 37.6 |
| laya-multilingual | 93.8 | 49.0 | 39.2 | 35.0 | 30.2 |
| Julia-1 | 83.0 | **73.8** | 33.0 | n/a | 32.2 |
| Jev | 86.4 | 62.2 | **71.6** | 80.2 | 91.6 |

Accuracy in percent; macro-F1 and calibration error are in
[docs/BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md). The small models match or beat
clef-flash on topic and emotion and are near chance on knowledge and
entailment, as their model cards say. Each model matches its own reference
code token for token on a fixed test set, with probabilities within 0.035
(clef-flash), 0.004 (Laya) and 0.018 (Julia).

### clef-flash

Takes text and images (videos are rejected) up to 16,384 tokens; a longer
state is cut to fit. Runs in bf16 as released, or as a smaller copy made
with [convert](#quantizing-clef-flash-convert).

**Images** need the `images` extra. The Python API takes file paths,
bytes, PIL images or data URLs; `run` and `chat` take files (`--image`,
`/image`); the server takes data URLs only. They must be PNG, JPEG or WebP,
at most about 89 million pixels. Each image is resized the way Cloudflare's
processor does it and costs one token per 32x32 pixels. Images larger than
2 megapixels (about 2,048 tokens) are shrunk to that size, which changed no
answer in our tests, even on screenshots with small text; `--max-image-mp`
(Python: `max_image_pixels`) sets another cap, and `0` lifts it to the
processor's own maximum (16.7 MP), as Cloudflare's reference does. Time
with a short state and three questions:

| Image | Tokens | bf16 |
|---|---|---|
| 640x480 | 300 | 0.47 s |
| 1024x768 | 768 | 0.84 s |
| 1920x1080, or larger (shrunk to 2 MP) | about 2,048 | 2.3 s |

Images count against the input limit together with the questions: the
state is cut first, and a request whose images and questions alone do not
fit fails with an error on `images`. The vision tower (0.85 GB) loads with
the first image request.

### Laya

Encoder models from Convai Innovations: laya and laya-typed-decisions
(ModernBERT-large, English; non-Latin scripts fail, per Laya's README) and
laya-multilingual (mmBERT-base). A state that does not fit is cut, keeping
the end of a list such as a conversation, and long option lists are
shortened to fit the model's question budget (which is why BANKING77's 77
options cost so much). `--max-input-tokens` raises the limit up to 8,192.
Probabilities use Laya's calibrated temperatures, as its own package
applies them.

### Julia

Supersonic Labs' encoder model (mmBERT-small). It answers 2 to 20 options
per question and, as its release recommends, refuses what it would have to
cut: a state that does not fit, an option over 48 tokens, its marker token
`<mask>` in the text. In Python, `strict_encoding=False` cuts instead.

Laya and Julia need instructions for every question, and Julia a
description for every option. Where a question leaves them out (as the
short forms `--choice` and `--noul` do), the question id stands in for the
instructions and the option id for the description. Both run in float16.

Julia reads an option only through its description, and how that is
worded matters a lot. Describe each option in a short phrase of what it
covers: on 11 support tickets routed to billing, technical or sales,
Julia got 8 to 11 right with phrases such as "Billing and payment
disputes", but 4 to 6 with the bare ids and 1 to 3 with one-word labels
such as "Billing" (Laya: 9 to 10 with any of them). Check the wording on
a few of your own cases before relying on it. `run` and `chat` print a
hint when Julia gets choice options without descriptions.

## Quantizing clef-flash: `convert`

Laya and Julia load as they are (0.3 to 0.85 GB). clef-flash can be
written as a smaller copy:

```bash
# 8-bit: half the memory, answers within the parity tolerance
mlx-decision convert -m Cloudflare/clef-flash -q                 # -> clef-flash-q8

# mixed precision: bits go where the answers are most sensitive
mlx-decision convert -m Cloudflare/clef-flash --target-bits 5    # -> clef-flash-mq5
```

Without `-o` the folder is named after the model plus `-q<bits>`,
`-mq<target>` or `-mlx` (no quantization). A converted folder loads like
the original (`mlx_decision.load("clef-flash-q8")`). Only the text backbone
is quantized; the decision head and the vision tower stay as released.
With `--target-bits` the converter first measures, on a built-in
calibration set, how far the answers move when each block of the model is
quantized alone (about 6 minutes), then gives the sensitive blocks more
bits within the average you asked for.

| Copy | Disk | Peak memory | Changed answers (of 139) |
|---|---|---|---|
| bf16 (as released) | 17.8 GB | 19.0 GB | - |
| 8-bit (`-q`) | 9.1 GB | 11.2 GB | 0 |
| mixed 5 (`--target-bits 5`) | 6.0 GB | 8.1 GB | 0 |
| mixed 4 (`--target-bits 4`) | 4.9 GB | 7.1 GB | 1 |
| uniform 4-bit (`-q --bits 4`) | 4.9 GB | 7.1 GB | 9 |

Use the released model if 19 GB fit; otherwise 8-bit; on smaller Macs
`--target-bits 5`, then `--target-bits 4`. Quantization saves memory, not
time: quantized copies are 10-20% slower.

## Speed

Median time per request, end to end, against each model's own PyTorch code
on the same Mac (MPS; Apple M5 Pro with 64 GB):

| Model | Request | PyTorch | mlx-decision | Speedup |
|---|---|---|---|---|
| clef-flash (bf16) | 394 tokens, 1 question | 0.88 s | 0.25 s | 3.5x |
| clef-flash (bf16) | 6,099 tokens, 20 questions | 12.96 s | 3.80 s | 3.4x |
| clef-flash (bf16) | 16,384 tokens, 20 questions | 36.12 s | 11.39 s | 3.2x |
| clef-flash (bf16) | 1024x768 image, 3 questions | 2.73 s | 1.01 s | 2.7x |
| laya | 250-token state, 1 question | 46 ms | 15 ms | 3.0x |
| laya | 450-token state, 20 questions | 543 ms | 386 ms | 1.4x |
| laya-multilingual | 1,000-token state, 20 questions | 558 ms | 337 ms | 1.7x |
| Julia-1 | 1,000-token state, 20 questions | 557 ms | 182 ms | 3.1x |
| Julia-1 | 7,350-token state, 20 questions | 12.4 s | 2.0 s | 6.1x |

clef-flash peaks at 19.0 GB against the reference's 23.9 GB. Full tables,
the 8-bit copy and how to reproduce them: [docs/BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md).

To measure a model on your machine:

```bash
mlx-decision benchmark -m Cloudflare/clef-flash
```

It reports load time, peak memory and median / p95 time per request
across state lengths (those that fit the model's limit) and numbers of
questions. As a rule of thumb, clef-flash reads about 1,700 tokens per
second; laya about 25,000, laya-multilingual about 60,000 and Julia-1
about 100,000, counting the state once per question. `--out FILE` also
writes the results as JSON, with the Mac's chip, CPU and GPU cores,
memory, macOS and the versions used, so runs on different Macs can be
compared.

## Development

```bash
make setup   # editable install with dev tools
make test    # unit tests; parity tests run when the weights are found
make lint
```

Tests that need weights look for them in `$MLX_DECISION_MODELS/<name>`
(`clef-flash`, `laya`, `laya-typed-decisions`, `laya-multilingual`,
`Julia-1`), then in the Hugging Face cache, and are skipped otherwise. The
parity fixtures are rebuilt with `scripts/make_parity_set.py` and
`scripts/make_parity_reference.py` (runs Cloudflare's reference with
PyTorch), and for Laya and Julia with `scripts/make_marker_parity_set.py`
and `scripts/make_marker_reference.py` (runs the `laya` package or Julia's
code from its release folder).

## Affiliation

mlx-decision is an independent open-source project, not affiliated with or
endorsed by Cloudflare, TypeSafe AI, Convai Innovations or Supersonic Labs.
Its server speaks a request and response format compatible with TypeSafe
AI's public Jev API. clef-flash (Cloudflare), Laya (Convai Innovations) and
Julia 1 (Supersonic Labs) are used under their Apache-2.0 licences; this
package downloads them from Hugging Face and does not ship their weights.

## Licence

MIT, see `LICENSE`. The Qwen3.5 model code is adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); the Clef
input encoding and head are ported from Cloudflare's release; the image
preprocessing, vision tower, image positions and the ModernBERT encoder
are ported from [transformers](https://github.com/huggingface/transformers);
the Laya and Julia head and input encoding are ported from the
[laya](https://pypi.org/project/laya/) package and Julia 1's release (all
Apache-2.0). Their licence texts are in `LICENSES/`. The test photos are
public domain or CC0 (`tests/parity/images/SOURCES.md`). Model weights keep
their own licence (clef-flash, Laya, Julia 1: Apache-2.0).
