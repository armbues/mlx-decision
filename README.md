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
`Authorization` header and the request's `model` field, answers one request
at a time (others queue), returns `422` with Jev's error body for invalid
requests, and sets `X-MLX-Decision-Truncated: true` when the state was
truncated.

## Smaller models: `convert`

```bash
# 8-bit: half the memory, answers within the parity tolerance
mlx-decision convert -m Cloudflare/clef-flash -o clef-flash-8bit -q

# mixed precision: bits go where the answers are most sensitive
mlx-decision convert -m Cloudflare/clef-flash -o clef-flash-mixed-5 --target-bits 5
```

A converted folder loads like the original
(`mlx_decision.load("clef-flash-8bit")`). Only the backbone is quantized;
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
state lengths and numbers of questions. clef-flash in bf16 on an Apple
Silicon Mac with 64 GB:

| Input tokens | Questions | Median latency |
|---|---|---|
| 394 | 1 | 0.25 s |
| 1,552 | 5 | 0.84 s |
| 4,149 | 1 | 2.41 s |
| 6,103 | 20 | 3.63 s |
| 16,384 | 20 | 10.7 s |

Latency grows linearly with input length, at about 1,550-1,850 tokens per
second.

## Supported models

| Family | Models | Input | Notes |
|---|---|---|---|
| Clef | `Cloudflare/clef-flash` | text | images and videos are rejected for now |

On a 50-request test set, clef-flash on MLX matches Cloudflare's PyTorch
reference token for token and within 0.035 per probability (mean 0.001),
with no changed answers. Compared with the hosted Jev service it gives the
same choice answer 83% of the time and the same side of 0.5 for yes/no
questions 91% of the time; they are different models.

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
