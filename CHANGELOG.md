# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may
change the interface).

## [Unreleased]

### Added

- `scripts/reference_speed.py` compares the speed of mlx-decision with
  Cloudflare's PyTorch reference on the same requests; the README shows
  the result (about 3.5x faster in bf16 on an M5 Pro).

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
