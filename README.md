# mlx-decision

Run and fine-tune decision models locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

A decision model reads a *state* (a message, a document, any JSON) and a set
of typed *questions*, and returns a probability for every allowed answer in
a single forward pass, without generating text. mlx-decision is to decision
models what `mlx-lm` is to LLMs: load a model, ask it questions from Python
or the command line, serve it over HTTP with an API compatible with
[Jev's](https://typesafe.ai), hand it to agents as an
[MCP](#agents-mcp) tool, or [fit it to your own data](#fine-tune-on-your-data).

| Model | Size | Input | Limit | Memory | Short request | Speedup | Good at |
|---|---|---|---|---|---|---|---|
| Cloudflare's [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 9B | text, images | 16,384 tokens | 19 GB (8-bit: 11 GB) | 0.25 s | 3.5x | knowledge, classification, many options, images |
| Cloudflare's [clef](https://huggingface.co/Cloudflare/clef) | 27B | text, images | 16,384 tokens | 54 GB (8-bit: 32 GB, 4-bit: 19 GB) | 1.1 s (8-bit) | 3.7x (bf16) | as clef-flash, a little better on entailment (8-bit) |
| Perplexity's [pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | 27B | text, images | 8,192 tokens per question | 52 GB (8-bit: 29 GB, 4-bit: 17 GB) | 0.83 s (8-bit) | - | knowledge, entailment, images; the most accurate here (8-bit) |
| H2O.ai's [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) | 4B | text, images | 40,960 tokens per question | 13 GB (8-bit: 9 GB, 4-bit: 7 GB) | 0.18 s | - | fast with long states and many questions, images, calibrated scores |
| LiquidAI's [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 3B | text, many languages | 32,768 tokens per question | 12 GB | 0.06 s | 1.1x | small and fast, many questions on one state; commercial use limited by its licence |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | English text | 512 tokens per question | 1.8 GB | 16 ms | 3.0x | topic classification of English text |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | English text | 1,024 tokens per question | 1.7 GB | 16 ms | 2.9x | as laya, tuned for typed decision workflows |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | text, 100+ languages | 1,024 tokens per question | 1.6 GB | 8 ms | 2.5x | topic classification in many languages |
| Supersonic Labs' [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144M | text, many languages | 8,192 tokens per question | 1.4 GB | 6 ms | 2.0x | emotion and routing with 2 to 20 options, long states |

"Short request": one question about a state of about 250 tokens, on an
Apple M5 Pro. "Speedup": the same request against the model's own PyTorch
code on the same Mac (MPS), with the same answers; across all measured
requests it ranges from 1.4 to 6 times ([Speed](#speed)), except for
d1-3B, whose own code is about as fast (0.9 to 1.2 times). Clef 27B's
reference does not fit that Mac in bf16, so its speedup was measured on
an M2 Ultra; pplx's was not measured, nor H2O-Lightning's, whose release
runs on vLLM with CUDA. How the models differ is in [Models](#models).

This is alpha software: the interface may still change. Questions and bug
reports go to the [issue tracker](https://github.com/armbues/mlx-decision/issues).

## Install

Requirements: a Mac with Apple Silicon (M1 or later) and macOS 14 or newer,
Python 3.12 or newer, and the memory the model needs (see the table; which
copy fits which Mac is in [Clef](#which-copy-fits), [pplx](#pplx) and
[H2O-Lightning](#h2o-lightning)).

```bash
pip install mlx-decision                    # library and the mlx-decision command
pip install "mlx-decision[images]"          # plus image input (Pillow, NumPy)
pip install "mlx-decision[server]"          # plus the HTTP server (FastAPI, uvicorn)
pip install "mlx-decision[mcp]"             # plus the MCP server for agents
pip install "mlx-decision[training]"        # plus Hugging Face datasets for `data`
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
downloading a repository no supported model family can load, and shows a
model's own terms of use where it has them (d1-3B: commercial use only
below $10 million annual revenue), asking before it downloads; `--yes`
skips both questions. Private or gated repos need `HF_TOKEN`.

Clef, pplx and H2O-Lightning models can also be stored quantized. After the model, the menu asks
for full size, 8-bit or 4-bit, with the size of each and which of them fit
in the Mac's memory (the largest that fits is the default); `--bits 8` or
`--bits 4` chooses without asking:

```bash
mlx-decision download Cloudflare/clef --bits 8
mlx-decision run -m Cloudflare/clef -s "My card was charged twice." --noul "Is it a billing problem?"
```

The full-size weight files are then fetched one at a time, quantized and
deleted, so the full-size model never has to fit in memory or on disk:
Clef 27B in 8-bit takes about 30 GB on disk, plus one 5 GB file while
the download runs. The result is the same as running
[convert](#quantizing-convert) on a full download. In the Hugging
Face cache it takes the place of the repo's files, so `-m
Cloudflare/clef` loads it without fetching anything. `hf cache ls` and
`hf cache rm` treat it like any download; `hf cache verify` reports it
as changed, since it no longer holds the repo's files.

Before reading the weights, loading checks that the model fits: its
weights plus 10% must be within the GPU's recommended working set
(macOS swaps beyond it). A model that does not fit is refused with both
sizes and, for Clef, pplx and H2O-Lightning, the `convert` command for a
smaller copy that fits.
`--no-memory-check` loads it anyway.

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
pydantic's `ValidationError` instead). The state, instructions and criteria
take any JSON data nested up to 100 levels deep, with finite numbers
(no `NaN` or infinity). `result.truncated` tells you whether
the state was cut to fit the model's input limit. The answer types
(`ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`) and `Usage` can be imported
from `mlx_decision`.

`load` takes options per model: `max_input_tokens` changes the input limit
(Laya and Julia up to 8,192; pplx, H2O-Lightning and d1 only lower),
`strict_encoding=False` lets Julia cut what it would otherwise refuse,
`dtype` sets Laya's and Julia's precision (default float16),
`max_image_pixels` caps image size for Clef, `noul_floor` sets
H2O-Lightning's [yes/no floor](#h2o-lightning), and `prefix_cache_gb` sets
how much memory Clef, H2O-Lightning and d1 keep for
[recent states](#reusing-a-state) (default 2, 0 turns it off).
`check_memory=False` skips the memory check; a model that does not fit
raises `mlx_decision.ModelTooLargeError` otherwise.
`model.info()` describes the loaded model as JSON data: family,
precision, input limit, option limits and whether it takes images.

With the `images` extra, Clef, pplx and H2O-Lightning take images as file
paths, bytes, PIL images or data URLs:

```python
result = model.decide(
    "Customer says checkout fails; screenshot attached.",
    {"shows_error": Noul(instructions="The screenshot shows an error message")},
    images=["screenshot.png"],
)
```

Images can also come inside the state, as H2O-Lightning's release takes
them: a string value that is a data URL (`data:image/png;base64,...`) or
bare base64 of an image, or a chat-style image part such as
`{"type": "image_url", "image_url": {"url": "data:..."}}`, anywhere in a
string, object or list state. Each is taken out of the state, replaced
by `[image N]` (numbered after the images in `images`) and read as if it
had been given in `images`; errors name its place in the state
(`state.photos.1`). Image URLs are refused, not fetched. This works the
same in `run`, `server` and `mcp`; models without images read such a
state as text.

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
state (Clef, pplx, H2O-Lightning). `--max-input-tokens N` (also on `chat`,
`server` and `mcp`) changes the model's input limit, e.g. to give Laya up
to 8,192 tokens; `--noul-floor` changes H2O-Lightning's
[yes/no floor](#h2o-lightning).

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
as soon as it is complete, and shown as understood. `/save FILE` keeps the
questions for the next time:

```
$ mlx-decision chat -m Cloudflare/clef-flash -q questions.json
clef-flash loaded, 2 questions.
Type a state and finish it with a blank line. /help lists the commands, Ctrl-D quits.
state> My card was charged twice this month.

team (choice): billing, confidence 0.93
  billing    0.951  ███████████████████████
  technical  0.028  █
  sales      0.021  █

refund (noul): no, p(yes) 0.030
  █

clef-flash · 225 input tokens
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

The server binds to localhost by default (`--host` to change), answers
one request at a time (others queue),
returns `422` with the API's error body for invalid requests, and sets
`X-MLX-Decision-Truncated: true` when the state was truncated. Request
bodies larger than 64 MB get `413`.

Images go in the request's `images` list as base64 data URLs
(`data:image/png;base64,...`; PNG, JPEG or WebP). The server reads no files
and fetches no URLs, so anything else is refused with `422` naming the
image (`images.0`). With the SDK, pass them as
`extra_body={"images": [...]}`, or put them in the state as data URLs
([images inside the state](#python)).

To require a key, start it with `--api-key KEY` (or set
`MLX_DECISION_API_KEY`). Requests to `/v1/*` then need
`Authorization: Bearer KEY`, which is what the SDK sends from
`TYPESAFE_API_KEY`; others get `401`. `/health` stays open.

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
first. Sizes are estimated from the weights, plus the memory Clef keeps
for recent states. A request without a `model`, or with one the server
does not have (such as the SDK's default `jev-latest`), goes to
`--default-model`, which also loads at startup. With a single model that
model is the default; with several and no `--default-model`, such
requests get `422` listing the models. `GET /v1/models` lists them, and
`/health` names the default and the loaded ones. A model too large for
the budget gets `422` with the memory check's message.

## Agents: MCP

`mlx-decision mcp` serves a model to AI agents over the
[Model Context Protocol](https://modelcontextprotocol.io), so an agent in
Claude Code, Claude Desktop, Cursor or any other MCP client can call a
decision as a tool, answered on your Mac. It needs the `mcp` extra.

The server has two tools:

- `decide` takes the fields of a request body (`state`, `questions`, and
  `images` for Clef, pplx and H2O-Lightning) and returns the response body as structured
  content, plus `truncated`. An invalid request comes back as a tool
  error naming the field, which the agent can read and correct.
- `model_info` describes the loaded model: family, precision, input
  limit, option limits and whether it reads images.

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

## Fine-tune on your data

Every model here can be fitted to labelled examples of your own task,
on the Mac it runs on, in minutes. The result is a small folder that
points at its base and loads like any model (`-m`, `load`, the
server's `model=`). `training fit` picks the kind of fit the family
takes:

| Model | Kind | Data (train split) | Accuracy | ECE | Time | Peak |
|---|---|---|---|---|---|---|
| clef-flash (8-bit) | head | ANLI-R3, 2,000 rows | 50.3 → 52.7% | 20.9 → 21.4 | 10 min | 15.1 GB |
| clef-flash (8-bit) | calibrate | 500 rows of each of five datasets | ANLI-R3: 50.3 → 52.2% | 20.9 → 2.8 | 11 min | 11.9 GB |
| H2O-Lightning-4B | readout | TREC, 5,452 rows | 85.2 → 97.4% | 4.7 → 1.4 | 13 min | 10.6 GB |
| d1-3B | readout | TREC, 5,452 rows | 86.6 → 96.4% | 5.5 → 1.4 | 4 min | 6.1 GB |
| laya | finetune | TweetEval Offensive, 11,916 rows | 77.8 → 84.7% | 7.7 → 5.0 | 17 min | 8.9 GB |
| Julia-1 | finetune | TREC, 5,452 rows | 16.0 → 97.4% | 63.1 → 1.9 | 2 min | 4.8 GB |

Scored on each dataset's test split, on an Apple M5 Pro. ECE (expected
calibration error, lower is better) says how far the probabilities are
from how often the model is right. The calibration row is one fit for
five questions (ANLI-R3's rows from its dev split); it lowered ECE on
the other four datasets too. pplx takes a readout fit like H2O-Lightning and d1,
but needs a Mac with 32 GB or more for it and has not been measured.

The training data is JSON lines: a request body plus the correct
answers.

```json
{"state": "My card was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team handles this ticket?", "criteria": {"billing": "Payments, refunds, invoices", "tech": "Crashes, bugs, errors"}}}, "labels": {"team": "billing"}}
```

`mlx-decision data` writes such a file from a Hugging Face dataset (the
`training` extra), a CSV, JSONL or Parquet file. It guesses which
columns hold the text and the label, and takes the question's wording
from you. Five benchmark datasets come as presets:

```bash
pip install "mlx-decision[training]"
mlx-decision data hf:dair-ai/emotion --config split --split train -o emotion.jsonl \
  --question "Which emotion does the text express most?"
mlx-decision data --preset trec --split train -o trec-train.jsonl
mlx-decision data --preset trec --split test -o trec-test.jsonl
mlx-decision data --preset anli-r3 --split train_r3 --sample 2000 -o anli-train.jsonl
```

One run per kind:

```bash
# Laya, Julia: fine-tune the encoder and head (--head-only: the head alone)
mlx-decision training fit -m SupersonicLabs/Julia-1 --data trec-train.jsonl -o julia-trec

# pplx, H2O-Lightning, d1: retrain the option readout on the frozen network
mlx-decision training fit -m LiquidAI/d1-3B --data trec-train.jsonl -o d1-trec

# Clef: train the decision head on the frozen backbone (8-bit copy on a 24 GB Mac)
mlx-decision training fit -m clef-flash-q8 --data anli-train.jsonl -o clef-anli

# every family: calibrate the probabilities only (seconds after one scoring pass)
mlx-decision training fit -m convaiinnovations/laya --data emotion.jsonl --method calibrate
```

Each run holds back a fifth of the data, reports accuracy, macro-F1 and
ECE on it before and after, and ends with a calibration fitted on a
slice of the training part. That calibration can only be as good as
the data is like what the model will be asked: fitted on data the model
finds much easier than its real work, it makes the model overconfident.
Then compare on data the fit has not seen, and use the result:

```bash
mlx-decision training eval -m julia-trec --data trec-test.jsonl --compare SupersonicLabs/Julia-1
mlx-decision chat -m julia-trec
```

In a terminal without flags, `data` and `training fit` ask for what they
need in menus (the kinds with their peak memory, for example) and end by
printing the equivalent command. Before loading, `training fit` checks
its estimated peak memory against the Mac's working set, and refuses a
run that does not fit, naming smaller ones that would. The kinds, their
defaults, memory and time per family, the dataset shapes and the Python
API (`mlx_decision.training.fit`) are in
[docs/TRAINING.md](https://github.com/armbues/mlx-decision/blob/main/docs/TRAINING.md).

## Models

The models differ in size and in how they read a request. Clef reads the
state once and answers all questions together; pplx, Laya and Julia read
each question on its own, with the whole state, so their input limit
applies per question and their time grows with the number of questions.
H2O-Lightning and d1 also read each question on its own, but after the
state, which they compute once per request, so more questions cost
little.

Accuracy on five public benchmarks, the complete test split of each
(5,270 examples), covering the three question types, with the same
questions and option descriptions for every model (Jev is TypeSafe AI's
hosted model, for comparison):

| Model | TREC (answer type, `choice`, 6 options) | TweetEval Offensive (`noul`) | ANLI round 3 (entailment, `choice`, 3) | OpenBookQA (science, `choice`, 4) | SST-5 (sentiment, `score`, 5 levels) |
|---|---|---|---|---|---|
| clef-flash | **96.8** | 78.1 | 50.8 | 95.4 | 57.6 / 0.51 |
| clef (8-bit) | 96.4 | **81.0** | 55.2 | 95.4 | 57.6 / 0.51 |
| pplx (8-bit) | 94.0 | 79.6 | 66.8 | **96.8** | **58.0** / **0.47** |
| H2O-Lightning-4B | 85.2 | 76.3 | 59.9 | 84.6 | 57.3 / 0.52 |
| d1-3B | 86.6 | 78.7 | 55.7 | 86.2 | 56.8 / 0.54 |
| laya | 79.0 | 61.9 | 39.5 | 39.2 | 35.1 / 0.99 |
| laya-typed-decisions | 78.8 | 68.4 | 38.5 | 41.4 | 44.3 / 0.68 |
| laya-multilingual | 89.8 | 71.9 | 35.5 | 29.2 | 27.7 / 1.40 |
| Julia-1 | 16.0 | 37.9 | 33.2 | 33.2 | 36.7 / 0.82 |
| Jev | 92.8 | 77.7 | **69.2** | 96.0 | 57.7 / 0.49 |

Accuracy in percent for `choice`, macro-F1 in percent for `noul` (yes at
a probability of 0.5 or more), and for `score` the accuracy of the most
likely level and the mean absolute error of the expected level (in
levels, lower is better). Macro-F1 and calibration error for every cell
are in
[docs/BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md).
The three large models are level with Jev or ahead of it on four of the
five benchmarks; on entailment pplx comes close to Jev, Clef does not.
H2O-Lightning, at 4B about a seventh of the size of the 27B models, trails them on
answer type and science questions, is ahead of clef-flash on entailment
and about level on sentiment, with the best calibrated scores of all
(calibration error 1.6%). d1-3B, smaller still, is level with
H2O-Lightning: ahead on answer type, offensive language and science
questions, behind on entailment.
The small models are near chance on entailment and science questions,
as their model cards lead one to expect; Julia, made for emotion and
routing, answers TREC almost only with its first two options and calls
most tweets offensive. Each model matches its own
reference code token for token on a fixed test set, with probabilities
within 0.035 (clef-flash), 0.031 (clef), 0.020 (H2O-Lightning), 0.030
(d1-3B), 0.004 (Laya) and 0.018 (Julia); pplx's probabilities were compared on a small
random model in its layout (within 0.00001), not on the 27B weights.

### Clef

Cloudflare's clef-flash (9B) and clef (27B) take text and images (videos
are rejected) up to 16,384 tokens; a longer state is cut to fit.

#### Which copy fits

Each Clef model runs in bf16 as released or as a quantized copy. The fit
check asks for the weights plus 10% within the GPU's working set, which
is about two thirds of a Mac's memory on a 16 GB Mac and four fifths on a
64 GB Mac (`mlx-decision benchmark` prints it):

| Copy | clef-flash needs | Clef 27B needs |
|---|---|---|
| bf16 (as released) | 21.0 GB: 32 GB Macs and up | 60.5 GB: 96 GB Macs and up |
| 8-bit | 11.7 GB: 24 GB Macs and up | 32.7 GB: 48 GB Macs and up |
| 4-bit | 6.8 GB: 16 GB Macs and up | 17.9 GB: 32 GB Macs and up |

Peak memory at the full 16,384 tokens: clef-flash 19.0 GB in bf16 and
11.2 GB in 8-bit; Clef 27B 32 GB in 8-bit and 19.5 GB in 4-bit.
[`download --bits`](#getting-a-model) stores a quantized copy directly;
[convert](#quantizing-convert) makes one from a full download and
offers mixed precision, which changes fewer answers than uniform 4-bit.

Clef 27B is three to four times slower than clef-flash and only a little
more accurate on our benchmarks, mainly on entailment, with better
calibrated probabilities (8-bit, see the table above). In bf16 it
matches Cloudflare's reference within 0.031 per probability (measured
on an M2 Ultra, where it is 3.7 times faster than the reference); the
8-bit copy changed one of 177 answers, the 4-bit copy three.

#### Images

Images need the `images` extra. The Python API takes file paths,
bytes, PIL images or data URLs; `run` and `chat` take files (`--image`,
`/image`); the server takes data URLs only. They must be PNG, JPEG or WebP,
at most about 89 million pixels. Each image is resized the way Cloudflare's
processor does it and costs one token per 32x32 pixels. Images larger than
2 megapixels (about 2,048 tokens) are shrunk to that size, which changed no
answer in our tests, even on screenshots with small text; `--max-image-mp`
(Python: `max_image_pixels`) sets another cap, and `0` lifts it to the
processor's own maximum (16.7 MP), as Cloudflare's reference does. A
1024x768 image (768 tokens) with a short state and three questions takes
0.84 s on clef-flash ([more sizes](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md#images)).

Images count against the input limit together with the questions: the
state is cut first, and a request whose images and questions alone do not
fit fails with an error on `images`. The vision tower (0.85 GB) loads with
the first image request.

#### Reusing a state

Clef keeps what it computed for recent states (up
to 2 GB per model), so a request with the same state and images as a
recent one, and other questions, computes only the new questions: on
clef-flash, a 16,384-token state takes about 10 s the first time and
0.2 s for one new question after that (1.5 s for twenty). The answers
are the same as without reuse, within the parity tolerance.
`--prefix-cache GB` on `run`, `chat`, `server` and `mcp` (Python:
`prefix_cache_gb`) changes the amount; 0 turns it off. H2O-Lightning and
d1 keep recent states the same way (d1 only states long enough to be
worth it). pplx, Laya and Julia read each question together with the
state, so they have nothing to reuse.

### pplx

Perplexity's pplx-decider-v1.1-27b, fine-tuned from Qwen3.8-27B (the
same architecture as Clef 27B) with a decision head of up to 255 options.
It reads every question as its own sequence with the whole state and its
images, so its input limit, 8,192 tokens, applies per question. Longer
input is refused, never cut (`truncated` is never set), as the release
does; `--max-input-tokens` can lower the limit, not raise it.

| Copy | Disk | The fit check needs | Peak memory (7,350 tokens) |
|---|---|---|---|
| bf16 (as released) | 52.2 GB | 57.4 GB: 96 GB Macs and up | - |
| 8-bit | 28.2 GB | 31.0 GB: 48 GB Macs and up | 29.1 GB |
| 4-bit | 15.3 GB | 16.9 GB: 32 GB Macs and up | 17.1 GB |

`download --bits 8` or `--bits 4` stores a copy without keeping the
full-size files, and [convert](#quantizing-convert) writes
one from a full download (uniform 8-bit or 4-bit; the decision head and
the vision tower stay as released).

One question about a 250-token state takes 0.83 s (8-bit), a little less
than Clef 27B; five questions take five times as long, where Clef 27B
reads the state once (4,000-token state: 9.5 s for one question, 49 s for
five; Clef 27B 10.6 s and 11.9 s). The 4-bit copy is about as fast. There
is nothing to reuse between requests, since the state is read together
with each question.

Images, as for Clef, need the `images` extra; each image is scaled to
between 64 and 256 tokens (the release's fixed bounds, so
`--max-image-mp` does not apply), and the vision tower loads with the
first image request.

On the benchmarks in [Models](#models) the 8-bit copy is the most
accurate local model overall: as good as Clef 27B (8-bit) on answer type,
offensive language and sentiment, a little better on science questions,
and well ahead on entailment (ANLI round 3: 66.8% against 55.2%; Jev
69.2%). Perplexity reports a Decision Index of 61.6
for the release (Jev: 57.9). Its prompt and token ids match the
release's code exactly; its probabilities were compared with that code on
a small random model in the same layout, since the release's code needs
about 104 GB to run the 27B on a Mac.

### H2O-Lightning

H2O.ai's H2O-Lightning-4B, built on Qwen3.5-4B, takes text and images and
answers 1 to 255 options. Each question is its own sequence, but the
system prompt and the state come first: they are computed once per
request and shared by its questions, and kept for later requests on the
same state as for Clef ([reusing a state](#reusing-a-state)). The input
limit, 40,960 tokens, applies per question. As in the release, a state
over 32,000 tokens keeps its first and last 16,000 around a `...` line
(`truncated` is set), and a question still over the limit is refused.
`--max-input-tokens` lowers the limit, not raises it; the state is then
cut to leave room for the request's longest question.

As released, a yes/no answer less sure than 0.801 is reported at 0.801 on
its side: the answer stays, its probability moves. `--noul-floor 0`
(Python: `noul_floor=0`, also for `server` and `mcp`) reports the model's
own probability, for routing uncertain answers elsewhere; a value
between 0.5 and 1 sets another floor. Other families refuse the option.

| Copy | Disk | The fit check needs | Peak memory (longest states) |
|---|---|---|---|
| bf16 (as released) | 9.1 GB | 10.0 GB: 16 GB Macs and up | 12.6 GB |
| 8-bit | 5.1 GB | 5.7 GB: 16 GB Macs and up | 8.9 GB |
| mixed 4.5 | 3.3 GB | 3.6 GB: 16 GB Macs and up | 7.2 GB |
| 4-bit | 3.0 GB | 3.3 GB: 16 GB Macs and up | 7.0 GB |

On a 16 GB Mac the full-size model fits, but long states take it past
the GPU's working set; use the 8-bit copy there. `download --bits` and
[convert](#quantizing-convert) write the copies; the vision tower stays as
released.

One question about a 250-token state takes 0.18 s, five take 0.41 s;
a 28,800-token state takes 12.2 s with one question and 12.7 s with five,
and 0.69 s for five new questions once the state is kept. As for Clef,
the quantized copies save memory, not time: from 1,000 tokens on they
are 5-17% slower.

Images, as for Clef, need the `images` extra: up to 4 per request, each
scaled to at most 1,048,576 pixels as in the release (`--max-image-mp`
does not apply), placed ahead of the text. The vision tower loads with
the first image request.

On the benchmarks in [Models](#models) it is behind the 27B models, ahead
of clef-flash on entailment (ANLI round 3: 59.9% against 50.8%) and has the
best calibrated sentiment scores. H2O.ai reports 205 of the 231 public
JevBench items for the release. Its prompt and token ids match the
release's code (its shim, with transformers in place of vLLM) for every
question of the test set; in bf16 its probabilities are within 0.020 of
that code run in bf16 on the same Mac (mean 0.0014), and the 3 answers
that changed of 271 were near ties.

### d1

LiquidAI's d1-3B, built on LFM2.5-VL-3B (LFM2: mostly short convolutions
with a few attention layers), reads text in many languages; the release
also reads images, which mlx-decision does not support for it yet
(`images` are refused). It answers up to 255 options and 2 to 10 score
levels (the model card's range). Each question is its own prompt with the
state first: a request's questions run together in one pass, and a long
state is computed once and kept for later requests, as for Clef
([reusing a state](#reusing-a-state)). The input limit, 32,768 tokens
(the model's context), applies per question; a longer state is cut,
keeping its start and end around a `...` line (`truncated` is set), and
only a question too long on its own is refused. `--max-input-tokens`
lowers the limit. A question without instructions uses its id, as for
Laya and Julia.

Its weights are not under Apache-2.0 but the LFM Open License v1.0,
which licenses commercial use only to organisations with less than $10
million in annual revenue; read it before using the model. `download`
shows this and asks before downloading (`--yes` skips the question), and
loading by repo id notes it on stderr when it downloads.

It runs in bf16 as released (6.2 GB): the fit check needs 6.9 GB, so 16
GB Macs and up, though its peak of 11.6 GB (29,450 tokens, twenty
questions) is past a 16 GB Mac's working set. There are no quantized copies: they would save memory but no
time. One question about a 250-token state takes 62 ms, five 125 ms,
twenty 245 ms; a 4,000-token state takes 0.77 s with one question and
1.0 s with twenty, and twenty new questions on a kept state take 0.2 to
1.0 s. Its release's own PyTorch code is about as fast on the same Mac
(mlx-decision 0.9 to 1.2 times as fast): it reads the state once for all
questions too.

On the benchmarks in [Models](#models) it is level with H2O-Lightning-4B
in a third of the time, ahead on answer type, offensive language and
science questions and behind on entailment (ANLI round 3: 55.7% against
59.9%). LiquidAI reports it as the best decision model under 10B on the
Decision Index 0.2.1 (scored by themselves). Its prompt and token ids
match the release's code for every question of the test set (22
languages included); in bf16 its probabilities are within 0.030 of that
code run in bf16 on the same Mac (mean 0.0054), with no clear answer
changed. The prompt and answer reading were written from the release's
documented and observed behaviour; the release's own code (under the LFM
licence) is not part of mlx-decision.

### Laya

Encoder models from Convai Innovations: laya and laya-typed-decisions
(ModernBERT-large, English; non-Latin scripts fail, per Laya's README) and
laya-multilingual (mmBERT-base). A state that does not fit is cut, keeping
the end of a list such as a conversation, and long option lists are
shortened to fit the model's question budget (which costs accuracy when
there are many options). `--max-input-tokens` raises the limit up to 8,192.
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

## Quantizing: `convert`

Laya, Julia and d1 load as released. Clef, pplx and H2O-Lightning can be
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

clef-flash's copies, against the released model:

| Copy | Disk | Peak memory | Changed answers (of 139) |
|---|---|---|---|
| bf16 (as released) | 17.8 GB | 19.0 GB | - |
| 8-bit (`-q`) | 9.1 GB | 11.2 GB | 0 |
| mixed 5 (`--target-bits 5`) | 6.0 GB | 8.1 GB | 0 |
| mixed 4 (`--target-bits 4`) | 4.9 GB | 7.1 GB | 1 |
| uniform 4-bit (`-q --bits 4`) | 4.9 GB | 7.1 GB | 9 |

Use the released model if it fits ([which copy fits](#which-copy-fits)),
otherwise 8-bit, then a 4-bit copy. Quantization saves memory, not time:
quantized copies are 10-20% slower.

pplx takes `-q` (8-bit) and `-q --bits 4`; mixed precision is not
offered for it, since measuring it needs the full-size model loaded.

H2O-Lightning takes all of these, mixed precision included (about 4
minutes to measure). Against the release's code on its test set (271
questions), the 8-bit copy changed 6 answers, all near ties (largest
difference 0.035); mixed 4.5 changed 11, 4 of them clear ones (0.16);
uniform 4-bit 19, 12 of them clear (0.33). For this model 4-bit costs
more than for clef-flash: prefer 8-bit, or mixed precision on a small Mac.

`convert` reads one weight file at a time, so a model larger than the
Mac's memory can be converted: Clef 27B to 8-bit takes about 10 s and
9 GB of memory. Mixed precision runs the model to measure it, so making
such a copy needs the full-size model to fit (for Clef 27B, a 96 GB
Mac); the copy then runs on smaller Macs.
[`download --bits`](#getting-a-model) stores a uniform copy without the
full-size download.

## Speed

Median time per request, end to end, against each model's own PyTorch code
on the same Mac (MPS; Apple M5 Pro with 64 GB); short requests are in the
table at the top:

| Model | Request | PyTorch | mlx-decision | Speedup |
|---|---|---|---|---|
| clef-flash (bf16) | 6,099 tokens, 20 questions | 12.96 s | 3.80 s | 3.4x |
| clef-flash (bf16) | 16,384 tokens, 20 questions | 36.12 s | 11.39 s | 3.2x |
| clef-flash (bf16) | 1024x768 image, 3 questions | 2.73 s | 1.01 s | 2.7x |
| laya | 450-token state, 20 questions | 543 ms | 386 ms | 1.4x |
| laya-multilingual | 1,000-token state, 20 questions | 558 ms | 337 ms | 1.7x |
| d1-3B (bf16) | 3,400-token state, 1 question | 809 ms | 676 ms | 1.2x |
| d1-3B (bf16) | 15,000-token state, 20 questions | 4.74 s | 4.01 s | 1.2x |
| Julia-1 | 7,350-token state, 20 questions | 12.4 s | 2.0 s | 6.1x |

clef-flash peaks at 19.0 GB against the reference's 23.9 GB.
H2O-Lightning is not in the table: its release runs on vLLM, which needs
CUDA. Full tables, the quantized copies and how to reproduce them: [docs/BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md).

To measure a model on your machine:

```bash
mlx-decision benchmark -m Cloudflare/clef-flash
```

It reports load time, peak memory and median / p95 time per request
across state lengths (those that fit the model's limit) and numbers of
questions; for Clef, H2O-Lightning and d1 also each request again on a
state it has kept. As a rule of thumb, clef-flash reads about 1,700 tokens per
second and Clef 27B (8-bit) about 380; pplx (8-bit) about 420;
H2O-Lightning about 2,400 to 3,100 for one question, and more questions
add little since they share the state; d1-3B about 4,400 to 5,200 for
one question, likewise; laya about 25,000,
laya-multilingual about 60,000 and Julia-1 about 100,000, counting the
state once per question. `--out FILE` also
writes the results as JSON, with the Mac's chip, CPU and GPU cores,
memory, macOS and the versions used, so runs on different Macs can be
compared.

Given a folder of models, `benchmark` measures each in turn, each in a
process of its own, and ends with a summary table; folders no family
loads and models that do not fit are listed as skipped. `--out DIR`
then writes one file per model:

```bash
mlx-decision benchmark -m ~/Models --out results/
```

## Development

```bash
make setup   # editable install with dev tools
make test       # unit tests; parity tests run when the weights are found
make test-slow  # Clef 27B 8-bit, H2O-Lightning and d1-3B against their references (about 12 min)
make lint
```

Tests that need weights look for them in `$MLX_DECISION_MODELS/<name>`
(`clef-flash`, `clef-8bit`, `laya`, `laya-typed-decisions`,
`laya-multilingual`, `Julia-1`), then in the Hugging Face cache, and are
skipped otherwise. The
parity fixtures are rebuilt with `scripts/make_parity_set.py` and
`scripts/make_parity_reference.py` (runs Cloudflare's reference with
PyTorch), and for Laya and Julia with `scripts/make_marker_parity_set.py`
and `scripts/make_marker_reference.py` (runs the `laya` package or Julia's
code from its release folder). pplx's fixture comes from
`scripts/make_pplx_reference.py`, which runs the release's code on a
small random model with the release's tokenizer; its tests need only the
release's small files in the Hugging Face cache. H2O-Lightning's comes
from `scripts/make_h2o_reference.py` in the same way (the release's shim,
with transformers in place of vLLM); with `--model` it runs the real
weights in bf16 for the slow test, which takes the release from
`$MLX_DECISION_MODELS/h2o-lightning-4b`. d1-3B's come from
`scripts/make_d1_reference.py` (its release's code, which runs only
there; `--model` for the slow test, from `$MLX_DECISION_MODELS/d1-3B`).

## Affiliation

mlx-decision is an independent open-source project, not affiliated with or
endorsed by Cloudflare, Perplexity, H2O.ai, LiquidAI, TypeSafe AI, Convai
Innovations or Supersonic Labs. Its server speaks a request and response format
compatible with TypeSafe AI's public Jev API. Clef and clef-flash
(Cloudflare), pplx-decider (Perplexity), H2O-Lightning (H2O.ai), Laya
(Convai Innovations) and Julia 1 (Supersonic Labs) are used under their
Apache-2.0 licences, d1-3B (LiquidAI) under the LFM Open License v1.0;
this package downloads them from Hugging Face and does not ship their
weights.

## Licence

MIT, see `LICENSE`. The Qwen3.5 and LFM2 model code and the gradient
checkpointing used for fine-tuning are adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); the Clef
input encoding and head are ported from Cloudflare's release; the image
preprocessing, vision tower, image positions and the ModernBERT encoder
are ported from [transformers](https://github.com/huggingface/transformers);
the pplx prompt is ported from pplx-decider's release; the H2O-Lightning
prompt and the lifting of images out of the state from H2O-Lightning's
release; the d1 prompt and answer reading were written from d1-3B's
documented and observed behaviour, without its release's code; the Laya and Julia
head and input encoding are ported from the
[laya](https://pypi.org/project/laya/) package and Julia 1's release (all
Apache-2.0). Their licence texts are in `LICENSES/`. The test photos are
public domain or CC0 (`tests/parity/images/SOURCES.md`). Model weights keep
their own licence (Clef, clef-flash, pplx-decider, H2O-Lightning, Laya,
Julia 1: Apache-2.0; d1-3B: LFM Open License v1.0).
