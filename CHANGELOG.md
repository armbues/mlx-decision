# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may
change the interface).

## [Unreleased]

### Added

- `benchmark --out FILE` writes the results as JSON together with the
  machine they were measured on (Mac model, chip, CPU cores per kind,
  GPU cores, memory, GPU working set, macOS), the versions of
  mlx-decision, MLX and Python, the model's details and the options.
  `--json` prints the same object, and the table now starts with a line
  naming the machine.

## [0.5.0] - 2026-10-07

### Added

- `mlx-decision mcp` serves a model to AI agents over the Model Context
  Protocol (`mcp` extra), with the tools `decide` (a request body in, the
  response body as structured content out, invalid requests as tool
  errors naming the field) and `model_info` (family, precision, input
  and option limits, image support). Tool descriptions are written for
  agents and include hints for the loaded model. On stdio by default,
  where `images` also takes local file paths; `--http` serves streamable
  HTTP at `/mcp` so several agents share one loaded model, with the same
  optional API key as `server`. The README shows the setup for Claude
  Code, Claude Desktop and HTTP clients.
- Over MCP, `decide` refuses a choice or score with more options than the
  model tells apart and asks the agent to choose among groups first: for
  Laya, which shortens long option lists until the descriptions are a word
  or two, 16 options (laya) or 21 (laya-typed-decisions,
  laya-multilingual); for Julia its limit of 20. `run`, `server` and the
  Python API keep each model's own behaviour.
- `DecisionModel.info()` returns the model's name, family, precision or
  quantization, input limit and capabilities.

### Changed

- Julia: `run` and `chat` print a hint when a choice question's options
  have no descriptions, and the README explains how to word them. Julia
  reads an option only through its description and answers routing
  questions much worse with bare ids or one-word labels than with a short
  phrase of what each option covers.

## [0.4.0] - 2026-10-06

### Models

- Laya ([laya](https://huggingface.co/convaiinnovations/laya),
  [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual),
  [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions))
  and Supersonic Labs' [Julia 1](https://huggingface.co/SupersonicLabs/Julia-1):
  ModernBERT and mmBERT encoders with a decision head, matching each
  family's own PyTorch code token for token and within 0.004 (Laya) and
  0.018 (Julia) per probability on an 84-request test set, 1.4 to 6 times
  faster than that code on MPS. They run in float16 by default (`dtype=`
  changes it).
- Laya cuts long states and option lists as its package does and applies
  its calibrated temperatures (clamped to 0.5-5); Julia refuses what it
  would have to cut (`strict_encoding=False` cuts instead). Questions
  without instructions, and for Julia options without a description, get
  their ids instead, since both models need them.

### Added

- `--max-input-tokens` on `run`, `chat` and `server` (`max_input_tokens=`
  in Python) changes a model's input limit, e.g. Laya's 512 tokens up to
  8,192.
- `download` lists the Laya and Julia models and fetches only the files a
  model needs; the size shown counts only those. Loading by repo id does
  the same.
- `benchmark` picks state lengths that fit the model's input limit by
  default.
- `scripts/marker_reference_speed.py` compares the speed with Laya's and
  Julia's own code; `scripts/accuracy_benchmark.py` reports benchmarks a
  model cannot answer (too many options) as n/a and counts refused
  examples as wrong.
- `docs/BENCHMARKS.md` with the full measurements (speed against the
  reference code, latency, images, quantization, agreement, accuracy); the
  README is reorganised around the supported models and keeps compact
  tables.
- `scripts/reference_speed.py` compares the speed of mlx-decision with
  Cloudflare's PyTorch reference on the same requests; the README shows
  the result (about 3.5x faster in bf16 on an M5 Pro).

### Changed

- A load option a model family does not take (e.g. `max_image_pixels` for
  Laya) is an error naming the option, also before `server` starts.

## [0.3.0] - 2026-10-05

First public release. Earlier versions were not published.

### Models

- Cloudflare's [clef-flash](https://huggingface.co/Cloudflare/clef-flash)
  (Qwen3.5-9B backbone plus joint schema head) on MLX, matching
  Cloudflare's PyTorch reference token for token and within 0.035 per
  probability on a 63-request test set.
- Image input for Clef: PNG, JPEG and WebP, resized as Cloudflare's
  processor does, shrunk to 2 megapixels by default (`--max-image-mp`).
  The vision tower loads with the first image request.

### Python API

- `mlx_decision.load()` takes a local folder or a Hugging Face repo id;
  `decide()` and `decide_request()` answer noul, choice and score questions
  with the answer and confidence definitions of the Jev API.
- Invalid requests raise `DecisionError` naming the offending field.

### Command line

- `run`: answer questions given as flags or a JSON file, about a state
  from a flag, a file or stdin; `--json` for the response body; `--states`
  answers a JSON lines file of request bodies with one model load.
- `chat`: interactive session with a question builder, menus and commands
  to list, add, edit, remove and save questions.
- `server`: HTTP server compatible with the Jev API (`POST /v1/systemone`),
  optional API key, one request at a time.
- `convert`: quantized MLX copies (8 bits by default, also 2 to 6), or
  mixed precision allocated by measured sensitivity (`--target-bits`).
- `benchmark`: load time, memory and latency across state length and
  number of questions.
- `download`: fetch a model from the Hugging Face Hub, from a menu of
  supported models or by repo id.
