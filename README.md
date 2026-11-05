# mlx-decision

Run and fine-tune decision models locally on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

A decision model reads a *state* (a message, a document, a screenshot,
any JSON) and a set of typed *questions*: pick one of several options,
rate on a scale, or answer yes or no. It returns a probability for every
allowed answer in one pass over the input, without generating text, so an
answer is always one you allowed and comes in a fraction of a second.

mlx-decision is for anyone who classifies, routes, moderates or checks
text and images and wants to do it on their own Mac: no data leaves it,
and there is no per-call cost. The **Studio**, a web app on your Mac,
lets you write questions and try them, compare models, answer a whole
dataset and fit a model to your own examples, without writing code;
developers use it from Python, the command line, an HTTP server and
MCP for AI agents.

![The Studio: a support ticket answered by a model, compared across three models, and a model trained on a dataset](https://raw.githubusercontent.com/armbues/mlx-decision/main/docs/images/studio.webp)

*The Studio on a MacBook Air: a support ticket answered by
pplx-decider-flash-v1.2, the same request compared with d1-3B and
d3-nano, and d3-edge trained on TREC.*

## Quick start

```bash
pip install mlx-decision
mlx-decision web
```

New to Python? It needs Python 3.12 or newer, which macOS does not
have. In Terminal, install [uv](https://docs.astral.sh/uv/), which
brings its own Python, open a new Terminal window, and install
mlx-decision with it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --python 3.13 mlx-decision
mlx-decision web
```

`web` starts the Studio and opens it in your browser (it is tested with
Chrome); keep the Terminal window open while you use it. With no models
yet, it asks where models should go and offers three that fit your
Mac's memory; download one, start a question set from a template such
as Support triage, and press Run All.
[Your first decision](https://github.com/armbues/mlx-decision/blob/main/docs/STUDIO.md#your-first-decision)
walks through it step by step.

The same from the command line, with a yes/no question (`--noul`), a
choice and a score, on a model from Hugging Face (downloaded on first
use, 9 GB):

```bash
mlx-decision run -m perplexity-ai/pplx-decider-flash-v1.2 \
  -s "My card was charged twice this month." \
  --noul "refund=The customer wants money back" \
  --choice "team=billing,technical,sales" \
  --score "anger=calm,annoyed,angry"
```

```
refund (noul): yes, p(yes) 0.872
  █████████████████████

team (choice): billing, confidence 0.90
  billing    0.934  ██████████████████████
  technical  0.059  █
  sales      0.008

anger (score): 2.32 on 1-3, confidence 0.24
   1  0.094  ██                        calm
   2  0.492  ████████████              annoyed
   3  0.414  ██████████                angry

pplx-decider-flash-v1.2 · 147 input tokens
```

And from Python:

```python
import mlx_decision
from mlx_decision import Choice, Noul

model = mlx_decision.load("perplexity-ai/pplx-decider-flash-v1.2")
result = model.decide("My card was charged twice this month.", {
    "team": Choice(criteria={
        "billing": "Payments and refunds",
        "technical": "Bugs and errors",
        "sales": "Plans and pricing",
    }),
    "refund": Noul(instructions="The customer wants money back"),
})
print(result.answers["team"].choice, round(result.answers["refund"].noul, 2))  # billing 0.89
```

## Interfaces

**Studio.** `mlx-decision web`. The Playground holds
question sets with several examples, runs them on one model or compares
up to three, and keeps every run in a history. Batches answer a
dataset's rows and show accuracy, a confusion matrix and how well the
probabilities are calibrated; Training fits a model and reports what it
gained. Get Code turns any request into curl, Python or a command line.
Everything is a plain file in a workspace the command line reads.
[docs/STUDIO.md](https://github.com/armbues/mlx-decision/blob/main/docs/STUDIO.md)

**Command line.** `run` answers one state or a JSON lines file of them,
`chat` one state after another, `download` fetches models (quantized
on the way if you like), `convert` makes 8-bit, 4-bit or mixed-precision
copies, and `benchmark` measures speed on your Mac. Defaults live in
`~/.mlx-decision/config.toml`.
[docs/CLI.md](https://github.com/armbues/mlx-decision/blob/main/docs/CLI.md)

**Python.** `mlx_decision.load` and `model.decide` as above, with
questions as objects or as the API's JSON, whole request bodies, images
as paths or bytes, and the load options of each family.
[docs/PYTHON.md](https://github.com/armbues/mlx-decision/blob/main/docs/PYTHON.md)

**Server.** `mlx-decision server -m MODEL` serves an API compatible with
TypeSafe AI's [Jev](https://typesafe.ai), so clients written for it,
such as its Python SDK, work by changing the base URL. One server can
hold several models within a memory budget and pick one per request.
[docs/SERVER.md](https://github.com/armbues/mlx-decision/blob/main/docs/SERVER.md)

**MCP.** `mlx-decision mcp -m MODEL` gives AI agents a `decide` tool over
the [Model Context Protocol](https://modelcontextprotocol.io), on stdio
or HTTP:
`claude mcp add decision -- mlx-decision mcp -m convaiinnovations/laya`.
[docs/SERVER.md](https://github.com/armbues/mlx-decision/blob/main/docs/SERVER.md#agents-mcp)

**Fine-tuning.** `mlx-decision data` turns a Hugging Face dataset, CSV,
JSONL or Parquet file into labelled examples; `training fit` fits a model
to them in minutes, on the Mac it runs on, and `training eval` compares
it with its base. The result is a small folder that loads like any
model. On TREC, a fit takes d1-3B from 86.6% to 96.4% in 4 minutes on
an M5 Pro; a LoRA fit takes d3-edge from 30% to 90% on photos of 37 pet
breeds in 17 minutes. In the Studio, the Datasets and Training pages do the same
without the command line.
[docs/TRAINING.md](https://github.com/armbues/mlx-decision/blob/main/docs/TRAINING.md)

## Models

Not sure which to pick?
[Which model to start with](https://github.com/armbues/mlx-decision/blob/main/docs/MODELS.md#which-model-to-start-with)
suggests one by your Mac's memory.

| Model | Size | Input | Memory | Good at |
|---|---|---|---|---|
| Cloudflare's [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 9B | text, images | 19 GB (8-bit: 11 GB) | knowledge, classification, many options, images |
| Cloudflare's [clef](https://huggingface.co/Cloudflare/clef) | 27B | text, images | 54 GB (8-bit: 32 GB, 4-bit: 19 GB) | as clef-flash, a little better on entailment |
| Perplexity's [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | 4B | text, images | 9.1 GB (8-bit: 5.1 GB, 4-bit: 3.0 GB) | all questions of a request in one pass; distilled from the 27B |
| Perplexity's [pplx-decider-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) | 27B | text, images | 52 GB (8-bit: 29 GB, 4-bit: 17 GB) | as flash, and more accurate |
| vLLM Semantic Router's [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | 9B | text, images | 19 GB (8-bit: 12 GB) | entailment, knowledge, calibrated scores; five more sizes from 0.6B to 27B |
| Base Labs' [elephant-1](https://huggingface.co/baselabs/elephant-1) | 9B | text, images | 18 GB (8-bit: 11 GB) | entailment and knowledge, calibrated; a request's questions share one pass over the state |
| Base Labs' [mammoth-1](https://huggingface.co/baselabs/mammoth-1) | 27B | text, images | 8-bit: 33 GB | as elephant-1; not measured here beyond a check of its 8-bit copy |
| Base Labs' [joey-1](https://huggingface.co/baselabs/joey-1) | 2B | text, images | 5.0 GB (8-bit: 3.4 GB) | small and fast, as elephant-1 in its reading; offensive tweets, weaker on answer type and sentiment |
| H2O.ai's [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) | 4B | text, images | 11 GB (8-bit: 9 GB) | sentiment, long states, images; each question read whole by default, or one pass over the state for many questions |
| LiquidAI's [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | 1.9B | text, images, many languages | 5.3 GB | small and fast, with pictures; answers depend on which questions are asked together; commercial use limited by its licence |
| LiquidAI's [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 3B | text, many languages | 12 GB | small and fast, many questions on one state; commercial use limited by its licence |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | English text | 1.8 GB | topic classification of English text |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | English text | 1.7 GB | as laya, tuned for typed decision workflows |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | text, 100+ languages | 1.6 GB | topic classification in many languages |
| Supersonic Labs' [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144M | text, many languages | 1.4 GB | emotion and routing with 2 to 20 options, long states |

Memory is the peak on the longest inputs measured (about 30,000 tokens,
or each model's limit); loading and short states need less. A model
loads if its weights plus 10% fit the GPU's working set, about
three-quarters of the Mac's memory, which is what the Studio checks
before a download; the quantized copies are made by `download` or
`convert`. Input limits, speed,
the speedup over each model's own PyTorch code, which copy fits which
Mac and each family's details are in
[docs/MODELS.md](https://github.com/armbues/mlx-decision/blob/main/docs/MODELS.md).

## Accuracy

Five public benchmarks, the complete test split of each (5,270
examples), with the same questions for every model. Jev is TypeSafe
AI's hosted model, for comparison.

| Model | TREC | TweetEval Offensive | ANLI round 3 | OpenBookQA | SST-5 |
|---|---|---|---|---|---|
| clef-flash | **96.8** | 78.1 | 50.8 | 95.4 | 57.6 |
| clef (8-bit) | 96.4 | **81.0** | 55.2 | 95.4 | 57.6 |
| pplx v1.1 27B (8-bit) | 94.0 | 79.6 | 66.8 | **96.8** | 58.0 |
| pplx-decider-flash-v1.2 | 93.6 | 77.9 | 61.8 | 93.8 | 60.6 |
| d3-flash | 85.8 | 77.7 | 64.3 | 95.2 | 57.2 |
| elephant-1 | 89.6 | 75.9 | **74.5** | **96.8** | 53.9 |
| joey-1 | 80.0 | 80.5 | 56.5 | 83.0 | 43.7 |
| H2O-Lightning-4B | 89.0 | 80.0 | 60.4 | 93.0 | **61.4** |
| d1-2B | 88.2 | 80.7 | 52.2 | 86.2 | 56.3 |
| d1-3B | 86.6 | 78.7 | 55.7 | 86.2 | 56.8 |
| laya | 79.0 | 61.9 | 39.5 | 39.2 | 35.1 |
| Julia-1 | 16.0 | 37.9 | 33.2 | 33.2 | 36.7 |
| Jev | 92.8 | 77.3 | 70.2 | 96.0 | 57.2 |

TREC (the kind of answer a question asks for, 6 options), ANLI
(entailment, 3) and OpenBookQA (science, 4) are choices, scored by
accuracy in percent; TweetEval Offensive is a yes/no question, scored
by macro-F1; SST-5 rates sentiment on 5 levels, scored by the accuracy
of the most likely level. Clef and pplx v1.1 are level with Jev or
ahead of it on four of the five; on entailment pplx comes close to Jev,
Clef does not. elephant-1, a 9B model, is the best on entailment, ahead
of Jev, and level with the 27B pplx on science questions (its model card
lists public decision data among its training data without naming the
datasets). joey-1, its 2B sibling, is close to the best on offensive
tweets and weaker on answer type and sentiment. d1-2B, at 1.9B, is level
with d1-3B. The
small models are near chance on entailment and science questions, as
their model cards lead one to expect. pplx-decider-flash-v1.2, at 4B,
is close to the 27B on answer type; H2O-Lightning-4B, also 4B, is the
best of all on sentiment (SST-5), read as its release reads by default
(each question's prompt whole). pplx v1.1 is the earlier release of the 27B; v1.2 27B, the
release in the model table, is not measured yet. Every cell's macro-F1 and calibration error, the other
Laya models and how each model matches its own reference code are in
[docs/BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md#accuracy).

## Install

Requirements: a Mac with Apple Silicon (M1 or later) and macOS 14 or
newer, Python 3.12 or 3.13 (later versions may work), and the memory the
model needs. Without a Python of your own, [uv](https://docs.astral.sh/uv/)
installs mlx-decision with its own Python, as in the
[quick start](#quick-start): `uv tool install --python 3.13 mlx-decision`
(with extras: `"mlx-decision[training]"`), later
`uv tool upgrade mlx-decision`.

```bash
pip install mlx-decision                    # library, command line, Studio, server, images
pip install "mlx-decision[training]"        # plus datasets from the Hugging Face Hub
pip install "mlx-decision[mcp]"             # plus the MCP server for agents
pip install "mlx-decision[all]"             # both
```

Training needs no extra for CSV and JSONL files; the `training` extra
adds Hub datasets, Parquet files and saved `datasets` folders, in the
Studio and for `data`. The `web`, `server` and `images` extras of
earlier versions are part of the base install now; install lines that
name them still work.

MLX also has builds for Linux and Windows, so pip may install
mlx-decision there, but loading a model stops with an error: the models
run on Apple Silicon only. Installing from source and running the tests
is in
[docs/DEVELOPMENT.md](https://github.com/armbues/mlx-decision/blob/main/docs/DEVELOPMENT.md).

## Documentation

- [STUDIO.md](https://github.com/armbues/mlx-decision/blob/main/docs/STUDIO.md):
  the Studio, page by page, and its workspace's files
- [CLI.md](https://github.com/armbues/mlx-decision/blob/main/docs/CLI.md):
  `run`, `chat`, `download`, `convert`, `benchmark` and the config file
- [PYTHON.md](https://github.com/armbues/mlx-decision/blob/main/docs/PYTHON.md):
  the Python API
- [SERVER.md](https://github.com/armbues/mlx-decision/blob/main/docs/SERVER.md):
  the Jev-compatible server and MCP for agents
- [TRAINING.md](https://github.com/armbues/mlx-decision/blob/main/docs/TRAINING.md):
  training data, the kinds of fit, fitted models and evaluation
- [MODELS.md](https://github.com/armbues/mlx-decision/blob/main/docs/MODELS.md):
  every model family, its limits and which copy fits your Mac
- [BENCHMARKS.md](https://github.com/armbues/mlx-decision/blob/main/docs/BENCHMARKS.md):
  speed, quantization, agreement with the reference code and accuracy
- [STABILITY.md](https://github.com/armbues/mlx-decision/blob/main/docs/STABILITY.md):
  what stays stable until 2.0
- [DEVELOPMENT.md](https://github.com/armbues/mlx-decision/blob/main/docs/DEVELOPMENT.md):
  working on mlx-decision itself
- [CHANGELOG.md](https://github.com/armbues/mlx-decision/blob/main/CHANGELOG.md):
  what changed in each release

## Stability

From 1.0.0, mlx-decision follows [Semantic Versioning](https://semver.org/).
The Python API for answering, the request and response format, MCP's
`decide` tool, the commands and their flags, fitted models, training
files, the config file and the Studio's workspace stay stable until
2.0; the flags of `data` and `training` and the training Python API are
provisional and may change in a minor release. The details are in
[docs/STABILITY.md](https://github.com/armbues/mlx-decision/blob/main/docs/STABILITY.md).
Questions and bug reports go to the
[issue tracker](https://github.com/armbues/mlx-decision/issues).

## Affiliation

mlx-decision is an independent open-source project, not affiliated with or
endorsed by Cloudflare, Perplexity, the vLLM Semantic Router project,
Baseten (Base Labs), the OneJev project, H2O.ai, LiquidAI, TypeSafe AI, Convai Innovations or Supersonic Labs. Its server speaks a request and response format
compatible with TypeSafe AI's public Jev API. Clef and clef-flash
(Cloudflare), pplx-decider (Perplexity), the d3 models (vLLM Semantic
Router), H2O-Lightning (H2O.ai), Laya
(Convai Innovations) and Julia 1 (Supersonic Labs) are used under their
Apache-2.0 licences, elephant-1, mammoth-1 and joey-1 (Base Labs) under the MIT
licence, d1-2B and d1-3B (LiquidAI) under the LFM Open License v1.0;
this package downloads them from Hugging Face and does not ship their
weights.

## Licence

MIT, see [LICENSE](https://github.com/armbues/mlx-decision/blob/main/LICENSE). The Qwen3.5 and LFM2 model code and the gradient
checkpointing used for fine-tuning are adapted from
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.); the LFM3
model code, vision tower and projector from
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (Apache-2.0, ported there from
transformers); the Clef
input encoding and head are ported from Cloudflare's release; the image
preprocessing, vision tower, image positions and the ModernBERT encoder
are ported from [transformers](https://github.com/huggingface/transformers);
the pplx prompts are ported from pplx-decider's releases (v1.1, and v1.2's
joint request format), the d3 prompt from
d3-flash's, the Base Labs prompt from [OneJev](https://github.com/OmniJev/OneJev),
which serves Base Labs' models; the H2O-Lightning
prompt, its selection of a long state's parts and the lifting of images
out of the state from H2O-Lightning's release; the d1 LFM3 and d1 prompts and answer reading were written from
d1-2B's and d1-3B's documented and observed behaviour, without their
releases' code; the Laya and Julia
head and input encoding are ported from the
[laya](https://pypi.org/project/laya/) package and Julia 1's release (all
Apache-2.0). The Studio's page ships
[Preact](https://preactjs.com) and
[Preact Signals](https://github.com/preactjs/signals) (MIT) and
[htm](https://github.com/developit/htm) (Apache-2.0). Their licence texts are in `LICENSES/`. The test photos are
public domain or CC0 (`tests/parity/images/SOURCES.md`). Model weights keep
their own licence (Clef, clef-flash, pplx-decider, d3, H2O-Lightning,
Laya, Julia 1: Apache-2.0; elephant-1, mammoth-1, joey-1: MIT; d1-2B, d1-3B: LFM
Open License v1.0).
