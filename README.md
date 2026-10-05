# mlx-decision

Run decision models locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

A decision model reads a *state* (a message, a document, any JSON) and a set
of typed *questions*, and returns a probability for every allowed answer in
a single forward pass, without generating text. mlx-decision is to decision
models what `mlx-lm` is to LLMs: load a model, ask it questions from Python
or the command line, serve it behind the same API as the hosted
[Jev](https://typesafe.ai) service, and convert it to smaller quantized
copies.

The first supported model is Cloudflare's
[clef-flash](https://huggingface.co/Cloudflare/clef-flash) (Qwen3.5-9B
backbone plus a joint schema head), text input only.

## Install

Requires an Apple Silicon Mac and Python 3.12 or newer.

From a clone of this repository:

```bash
pip install .            # library and the mlx-decision command
pip install ".[server]"  # plus the HTTP server (FastAPI, uvicorn)
```

Models load from a local folder or a Hugging Face repo id; repo ids are
downloaded through the normal Hugging Face cache (clef-flash: 18 GB).

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
print(result.to_wire())                         # the Jev response body
```

Questions can also be plain dicts in the Jev wire format
(`{"type": "choice", "instructions": ..., "criteria": {...}}`), and
`model.decide_request(body)` takes a whole request body. Invalid requests
raise `mlx_decision.DecisionError`, whose `param` names the offending field.
`result.truncated` tells you whether the state was cut to fit the model's
input limit (16,384 tokens for clef-flash).

`confidence` follows Jev's documented formulas (choice: how far the top
probability sits above an even split; score: how concentrated the
probability is around the most likely level), not Clef's own top
probability.

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

The state can also come from `--state-file` or stdin (`--state-json` parses
it as JSON), and questions from a JSON file with `-q questions.json`
(a questions map or a whole request body). `--json` prints the exact
response body.

`mlx-decision --version` prints the installed version.

### Interactive

With `--interactive` (`-i`) the model loads once and answers one state
after another: type or paste a state, finish it with a blank line, and
quit with Ctrl-D. Lines can be edited, and Up recalls earlier states.

```bash
mlx-decision run -m Cloudflare/clef-flash -i --choice "team=billing,technical,sales"
```

Without any questions, `-i` first asks for them: a type (choice, score or
noul), an id (Enter takes the one suggested), the instructions, and then
the options of a choice (`option` or `option: description`) or the levels
of a score (lowest first), one per line, ending with a blank line. Each
question is checked against the model as soon as it is complete.

```
$ mlx-decision run -m Cloudflare/clef-flash -i
clef-flash loaded, 0 questions.
No questions yet: build the first one (Ctrl-C cancels).
type (choice, score, noul): choice
id: team
instructions (optional): Which team should handle this?
options, one per line as 'option' or 'option: description' (blank line ends):
  billing: payments, invoices, refunds
  technical
  sales

added team
Add another question? [y/N] y
type (choice, score, noul): noul
id: refund
instructions: The customer wants money back
added refund
Add another question? [y/N]
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
builder, flags or a file:

| Command | |
|---|---|
| `/list` | show the questions |
| `/add` | build another question |
| `/edit ID` | change a question; the current values are filled in |
| `/remove ID` | remove a question |
| `/save FILE` | write the questions as a file that `-q` reads |
| `/help`, `/quit` | |

A state that itself starts with `/` is typed as `//`. Interactive mode
needs a terminal; with piped input it reads blank-line-separated states
without editing or commands.

### Many states

`--states FILE` answers a JSON lines file, one state per line (a JSON
string for text, or any other JSON value), and prints one response body per
line in the same order. `-` reads stdin. A line that cannot be answered
yields an error body with its line number, and the run goes on:

```bash
mlx-decision run -m Cloudflare/clef-flash -q questions.json --states tickets.jsonl > answers.jsonl
```

```
{"error": {"message": "not valid JSON: ...", "type": "invalid_request_error", "param": "state"}, "line": 3}
```

## Server

```bash
mlx-decision server -m Cloudflare/clef-flash --port 8000
```

Serves `POST /v1/systemone` with Jev's request and response bodies, plus
`GET /v1/models` and `GET /health`. Existing Jev clients work by changing
the base URL, for example the official SDK:

```bash
TYPESAFE_BASE_URL=http://127.0.0.1:8000 TYPESAFE_API_KEY=unused python my_jev_script.py
```

The server binds to localhost by default (`--host` to change), ignores the
request's `model` field, answers one request at a time (others queue),
returns `422` with Jev's error body for invalid requests, and sets
`X-MLX-Decision-Truncated: true` when the state was truncated.

To require a key, start it with `--api-key KEY` (or set
`MLX_DECISION_API_KEY`). Requests to `/v1/*` then need
`Authorization: Bearer KEY`, which is what the Jev SDK sends from
`TYPESAFE_API_KEY`; others get `401`. `/health` stays open.

## Smaller models: `convert`

```bash
# 8-bit: half the memory, answers within the parity tolerance
mlx-decision convert -m Cloudflare/clef-flash -q                 # -> clef-flash-q8

# mixed precision: bits go where the answers are most sensitive
mlx-decision convert -m Cloudflare/clef-flash --target-bits 5    # -> clef-flash-mq5
```

Without `-o` the folder is named after the model plus `-q<bits>`,
`-mq<target>` or `-mlx` (no quantization). A converted folder loads like
the original (`mlx_decision.load("clef-flash-q8")`). Only the backbone is quantized;
the decision head stays as released. With `--target-bits` the converter
first measures, on a built-in calibration set, how far the answers move
when each block of the model is quantized alone (about 6 minutes for
clef-flash), then gives the sensitive blocks more bits within the average
you asked for.

Measured on a 50-request test set against the unquantized model (139
questions):

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| bf16 (as released) | 17.8 GB | 19.0 GB | - | - |
| 8-bit (`-q`) | 9.1 GB | 11.2 GB | 0.001 / 0.014 | 0 |
| mixed 5 (`--target-bits 5`) | 6.0 GB | 8.1 GB | 0.003 / 0.070 | 0 |
| mixed 4 (`--target-bits 4`) | 4.9 GB | 7.1 GB | 0.006 / 0.084 | 1 |
| uniform 4-bit (`-q --bits 4`) | 4.9 GB | 7.1 GB | 0.014 / 0.212 | 9 |

Recommendation: use the released bf16 model if 19 GB fit; otherwise 8-bit;
on smaller Macs `--target-bits 5`, then `--target-bits 4`. Uniform 4-bit
loses noticeably more than mixed precision of the same size. Quantization
saves memory, not time: a decision is one compute-bound pass, and quantized
models are 10-20% slower.

## Speed: `benchmark`

```bash
mlx-decision benchmark -m Cloudflare/clef-flash
```

Reports load time, peak memory, and median / p95 latency per request across
state lengths and numbers of questions. Median latency per request for
clef-flash on an Apple M5 Pro (20-core GPU) with 64 GB:

| Input tokens | Questions | bf16 | 8-bit | 4-bit |
|---|---|---|---|---|
| 394 | 1 | 0.25 s | 0.27 s | 0.25 s |
| 1,552 | 5 | 0.84 s | 1.07 s | 1.03 s |
| 4,149 | 1 | 2.41 s | 2.89 s | 2.76 s |
| 6,103 | 20 | 3.63 s | 4.33 s | 4.13 s |
| 16,384 | 20 | 10.7 s | 12.1 s | 11.4 s |
| Peak memory | | 19.0 GB | 11.2 GB | 7.1 GB |

Latency grows linearly with input length, at about 1,550-1,850 tokens per
second. All numbers in this README were measured on that machine.

## Supported models

| Family | Models | Input | Notes |
|---|---|---|---|
| Clef | `Cloudflare/clef-flash` | text | images and videos are rejected for now |

On a 50-request test set, clef-flash on MLX matches Cloudflare's PyTorch
reference token for token and within 0.035 per probability (mean 0.001),
with no changed answers.

### Accuracy (preliminary)

Preliminary results: clef-flash (bf16, this package) and the hosted Jev
service on five public benchmarks, 500 test examples each (fixed sample;
larger runs and quantized models will follow), with the same questions
and option descriptions for both. Accuracy / macro-F1 in percent; ECE is
the expected calibration error of the top probability (lower is better).
In brackets: macro-F1 published by Cloudflare (clef-flash / Jev), from
their own, unpublished prompts.

| Benchmark | Options | clef-flash acc / F1 / ECE | Jev acc / F1 / ECE | Published F1 or acc |
|---|---|---|---|---|
| AG News (topic) | 4 | **91.4** / **91.5** / 2.5 | 86.4 / 86.4 / 9.5 | - |
| DAIR Emotion | 6 | 60.0 / 54.6 / 19.4 | **62.2** / **56.2** / 26.0 | - |
| ANLI r1-r3 (entailment) | 3 | 58.2 / 57.9 / 14.1 | **71.6** / **71.9** / 11.5 | 59.1 / 74.8 |
| BANKING77 (intent) | 77 | **96.0** / **95.8** / 3.5 | 80.2 / 79.4 / 8.2 | 90.9 / 79.7 |
| MMLU (accuracy) | 4 | **93.0** / 93.0 / 6.0 | 91.6 / 91.6 / 3.7 | 91.8 / 91.7 |

clef-flash is stronger at classification with many or well-described
labels (BANKING77, AG News) and Jev at adversarial entailment (ANLI). Both
are poorly calibrated on Emotion, whose labels overlap. Reproduce with
`scripts/accuracy_benchmark.py` (datasets are downloaded separately).

## Development

```bash
make setup   # editable install with dev tools
make test    # unit tests; parity tests run when the weights are found
make lint
```

Tests that need clef-flash look for it in `$MLX_DECISION_MODELS/clef-flash`,
then in the Hugging Face cache, and are skipped otherwise. The parity
fixtures are rebuilt with `scripts/make_parity_set.py` and
`scripts/make_parity_reference.py` (the latter runs Cloudflare's reference
with PyTorch).

## Licence

MIT, see `LICENSE`. The Qwen3.5 model code is adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); the Clef
input encoding and head are ported from Cloudflare's release (Apache-2.0).
Their licence texts are in `LICENSES/`. Model weights keep their own
licence (clef-flash: Apache-2.0).
