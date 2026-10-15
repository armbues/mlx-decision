# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (before 1.0, minor versions may
change the interface).

## [Unreleased]

## [0.9.1] - 2026-10-15

### Added

- vLLM Semantic Router's d3 models (Decision 3.0): `vllm-sr/d3-flash`
  (9B), `vllm-sr/d3` (27B), `d3-mini` (4B), `d3-nano` (2B), `d3-lite`
  (0.8B) and `d3-edge` (0.6B, pruned to 12 layers), with text and
  images, through the same API, `run`, `chat`, `server`, `mcp`,
  `benchmark` and `training`, and in the `download` menu after pplx.
  They share pplx's design (each question its own sequence, answers
  read out under letter codes) with their own prompt: up to 255
  options, 2 to 10 score levels, images of up to 1.6 MP each
  (`--max-image-mp` lowers that). Input is never cut: 32,768 tokens per
  question by default, and since the release sets no limit,
  `--max-input-tokens` (Python: `max_input_tokens`) raises it up to
  262,144 (d3-flash: about 136 KB of memory a token; 128k tokens take
  17 GB more and 5 minutes a question). `convert` writes 8-bit, 4-bit
  and mixed-precision copies, `download --bits` stores 8-bit or 4-bit
  ones; the readout and the vision tower stay as released. The prompt
  matches the release's runtime token for token on the whole test set;
  d3-flash in bf16 is within 0.038 of that runtime in bf16, no clear
  answer changed, and the other sizes agree on a sample of requests. On
  the five benchmarks d3-flash comes close to pplx's 27B on entailment
  (ANLI round 3: 64.3%) at a third of its size.

- `mlx-decision download` without a repo id now lists the families
  first (maker, a short line, how many models and their sizes), then
  the picked family's models with a "Back" entry. Escape goes back from
  the models and cancels in the families; a family with one model is
  picked directly. `download REPO_ID` is unchanged.

### Fixed

- A long request to a model that reads its input without a causal mask
  (pplx, d3) no longer holds all attention scores at once: for these
  models' head size MLX has no tiled attention kernel and fell back to
  the whole score matrix, which took about 8 GB at 16k tokens and
  failed at 40k (51 GB). The attention now runs in blocks of queries
  with the same results; d3-flash at 16k tokens peaks at 17 GB instead
  of 25 GB and is faster.

- A model folder whose `decision_config.json` names a prompt that no
  family here knows is refused with an error naming the prompt. Before,
  any such folder loaded as pplx with pplx's prompt, which is how a d3
  model loaded in 0.9.0.

- Qwen3.5 models whose `config.json` lists their layer types
  (`layer_types`) now load with those types, not the usual pattern of a
  full-attention layer every fourth layer; d3-edge, pruned to 12
  layers, depends on it.

## [0.9.0] - 2026-10-14

### Added

- Training: every model can be fitted to labelled examples of your own
  task on the Mac it runs on, and the result is served like any other
  model. The guide is `docs/TRAINING.md`; the README has a summary
  ("Fine-tune on your data"). Measured on test splits, for example:
  Julia-1 on TREC 16.0 → 97.4% in 2 minutes, H2O-Lightning-4B 85.2 →
  97.4% and d1-3B 86.6 → 96.4% on TREC, laya on TweetEval Offensive
  77.8 → 84.7%, clef-flash's ECE on ANLI-R3 20.9 → 2.8.

- `mlx-decision training fit -m BASE --data FILE.jsonl` writes a fitted
  model. Four kinds (`--method`; `auto` picks the family's own and
  follows it with a calibration): `calibrate` for every family (a
  temperature per question, plus a bias per option for a question asked
  the same way throughout), `readout` for pplx, H2O-Lightning and d1
  (each option's readout row and a bias, on the frozen network), `head`
  for Clef (the decision head, on backbone states stored on disk for
  the run), `finetune` for Laya and Julia (the whole encoder and head,
  or the head alone with `--head-only`). A fifth of the data is held back
  for the before / after report (`--holdout`, or `--validation FILE`); a
  dev slice of the rest picks the readout's settings and the best epoch,
  and stops a run early when an epoch gets worse (`--no-early-stop`).
  The peak memory is estimated and checked against the Mac's working
  set before the model loads. `--events` prints the progress as JSON
  lines, and Ctrl-C stops the run without writing anything.

- Fitted models are folders holding only the trained tensors, a report
  and a reference to their base (a Hub repo id at its commit, or a
  path), checked by a fingerprint when loaded, so a missing or changed
  base is refused. They load wherever a model does: `load`, `-m` on
  `run`, `chat`, `server` and `mcp`, a folder of models, the Jev SDK's
  `model=`. Fits that leave the weights alone (calibrate, readout) share
  their loaded base, so several of them cost the memory of one model.
  Questions a fit was not made for are answered without it, with a
  warning in `run` and `chat`.

- `mlx-decision training eval -m MODEL --data FILE.jsonl` scores models on
  labelled data: accuracy, macro-F1, ECE (and MAE for scores), refusals
  and a confusion matrix per question; `--compare OTHER` adds a second
  model, loaded after the first is released; `--json`.

- `mlx-decision data SOURCE -o FILE.jsonl` turns a dataset into training
  data: Hugging Face datasets (`hf:ORG/NAME`, needs the new `training`
  extra), CSV, JSONL and Parquet files and saved `datasets` folders, in
  five shapes (text, pair, multiple choice, yes/no, rating) guessed from
  the columns. `--show N` previews. Presets give the coverage suite's
  five datasets with their wording, at pinned commits.

- In a terminal, `data` and `training fit` ask in menus for what their
  flags leave open (the source, columns and question; the kind with its
  peak memory, the holdout, the output) and print the equivalent command
  line. `--no-input` and `--yes` turn the menus off.

- Python: `mlx_decision.training.fit(...)` and `evaluate(...)`, with
  progress events and a cancel flag.

### Fixed

- The error for a folder that is no supported decision model (also shown
  by `download`) listed the families alphabetically, and the error for
  `-m` options no model takes in the order of the model folders; both now
  list them in the order the documentation does (Clef, pplx, H2O, d1,
  Laya, Julia).
- `download --yes` of a repository no model family can load ended with
  "try it: mlx-decision chat -m ...", which could not load it either. The
  hint is now left out for such a download.

## [0.8.1] - 2026-10-12

### Changed

- `huggingface_hub` from 1.1 up to 2.x (was: 1.0 and later). 1.0 cannot
  hand a single file's progress to the caller, which `download` now
  needs; 2.x, which a new install gets, is now tested and supported.

### Fixed

- `download` drew the Hub's progress bars while it checked the model, and
  some of them again after the licence notice, over it and next to the
  question (seen with `LiquidAI/d1-3B`). The check is now quiet, and a
  download shows one progress bar, closed before anything else is
  written, that follows the bytes as they arrive (for a large file the
  Hub's own bar moved only when a piece was written, once in 15 s); a
  download that finds its files already there shows none.
  The Hub's setting to turn bars off is respected, and none are drawn
  outside a terminal.
- With `huggingface_hub` 2.x, a failed connection in `download` or in
  `-m org/model` ended in a traceback instead of a one-line error.
- d1-3B kept far more memory for a computed state than it counted: a
  kept 8,000-token state held 852 MB, of which the prefix cache's limit
  and the server's memory budget saw 131 MB, so `--prefix-cache` and
  shared states could use several times the memory they were allowed.
  It now holds what it counts.
- `server` with several models kept the model it unloaded alive while
  it loaded the next one, so a switch between two models that fit the
  memory budget only one at a time briefly needed memory for both. The
  unloaded model is now freed before the next one loads.
- The server's API key was not checked when the app ran under a URL
  prefix: mounted in another FastAPI app, or behind a proxy with a root
  path (`/decision/v1/systemone` answered without a key). The
  `mlx-decision server` command itself, which serves at `/`, was not
  affected. The key is now checked on the API's routes, wherever they
  are mounted.
- `server` with several models loaded the model an invalid request named
  (unloading another to make room) before it found the request invalid,
  so a malformed request could cost two model loads. A request is now
  checked before its model is loaded.
- When the default model failed to load at startup, the server's worker
  thread was left running (only noticeable when the app is embedded and
  started again); it is now stopped however the server ends.
- A request with a number JSON cannot write back (`1e400` or `NaN`) or a
  question id that is not valid text (a lone surrogate such as
  `"\ud800"`) reached the model, and `server` then failed with a 500
  while writing the answer (`mcp` gave the score's legend as `Infinity`
  in its text and `null` in its structured result). Both are now
  refused up front with an error naming the field.
- JSON that Python's decoder refuses in other ways than a syntax error
  (nested about 10,000 levels deep, or an integer over 4,300 digits)
  ended `run --states` at that line without an answer for the following
  ones, and `server` answered a too-long integer with a plain-text 500.
  Each is now an input error: an error line in `--states`, a `400` from
  `server`, a message for `--state-json` and question files.
- A state, instructions or criteria nested more than about 1,000 levels
  deep made H2O-Lightning and d1-3B fail while writing their prompt
  (`server` answered 500). Requests may now nest up to 100 levels; deeper
  ones are refused with an error naming the field.
- `Request` objects passed to `decide_request` or `check` skipped the
  checks a JSON body gets (a state that is not JSON data, an empty
  question id, a question changed after it was built): they now get the
  same checks and errors.
- Clef and H2O-Lightning could reuse a kept state computed for another
  image: two images whose resized pixels are the same but laid out
  differently (a plain image and the same image turned on its side) got
  the same key, so the second answer depended on the first request.
  The key now includes each image's layout.
- H2O-Lightning under a lowered `--max-input-tokens` that left room for
  one token of state kept the whole state instead of cutting it, and
  then refused the request as too long. It now cuts the state.
- `convert --target-bits` for H2O-Lightning measured the quantized
  embeddings with the model's output rows still unquantized, so their
  sensitivity in `quantization_sensitivity.json` was too low (0.00034
  instead of 0.00154 at 4 bits). For `--target-bits 4.5` the chosen bits
  and the weights written are unchanged; other targets could choose
  differently.
- `run --states` replaced bytes that are not UTF-8 with `�` and
  answered the changed text; such a line now gets an error line
  ("not valid UTF-8"), from a file and from stdin.
- `chat --json` did not say when the state was truncated; it now notes
  it on stderr, as `run --json` does.
- `chat` with `"images": null` in its `-q` file crashed on `/image`; null
  now means no images.
- `chat`'s `/edit` lost score levels or choice options that look alike
  on a line (`[1, "1"]`, a repeated level, option `a: b` next to `a`
  described as `b`) or that span lines, even when nothing was changed.
  Such criteria are now kept as they are, with a note to edit them in
  the `-q` file; the rest of the question stays editable.
- `mcp`'s `decide` tool advertised `state` as a string, object or array,
  so clients that check arguments against the schema could not send a
  number, `true`/`false` or `null`, which the tool takes. The schema now
  allows any JSON value, as its description says.
- `mcp --http` on port 80 refused requests to `http://localhost/mcp` or
  `http://127.0.0.1/mcp` (`421 Invalid Host header`), since clients
  leave out the port there. Loopback hosts and origins are now accepted
  with or without a port; other hosts are still refused.

## [0.8.0] - 2026-10-11

### Added

- H2O.ai's H2O-Lightning-4B (`h2oai/h2o-lightning-4b`, release `v1.2.3`),
  a 4B model with text and images, through the same API, `run`, `chat`,
  `server`, `mcp` and `benchmark`, and in the `download` menu after pplx.
  It reads up to 40,960 tokens per question and answers up to 255
  options; a state over 32,000 tokens is cut as the release cuts it
  (`truncated` is set). The questions of a request share the computed
  state, and recent states are kept for later requests as for Clef
  (`--prefix-cache`), so five questions on a 28,800-token state take
  12.7 s, and 0.7 s once the state is kept. Up to 4 images per request.
  It runs in bf16 as released (9.1 GB, 12.6 GB peak on long states);
  `convert` writes 8-bit, 4-bit and mixed-precision copies with the
  vision tower kept, and `download --bits` stores them without the
  full-size files. Its prompt matches the release's code token for
  token, and in bf16 its probabilities are within 0.020 of that code
  run on the same Mac.

- LiquidAI's d1-3B (`LiquidAI/d1-3B`), a 3B model, for text, through the
  same API, `run`, `chat`, `server`, `mcp` and `benchmark`, and in the
  `download` menu after H2O-Lightning. Its weights are under the LFM Open
  License v1.0, which licenses commercial use only to organisations with
  less than $10 million in annual revenue: `download` shows this and asks
  before downloading (`--yes` skips the question), and a download by repo
  id notes it on stderr. It reads up to 32,768 tokens per question (a
  longer state is cut, keeping its start and end; `truncated` is set),
  answers up to 255 options and 2 to 10 score levels, runs a request's
  questions in one pass and keeps long states for later requests
  (`--prefix-cache`). One question about a 250-token state takes 62 ms;
  it is about as fast as its release's own code (0.9 to 1.2 times) and
  about as accurate as H2O-Lightning-4B. Its prompt and token ids match
  the release's, and in bf16 its probabilities are within 0.030 of the
  release's code. The prompt and answer reading were written from the
  release's documented and observed behaviour; none of the release's
  code is part of mlx-decision.

- `--noul-floor` on `run`, `chat`, `server` and `mcp` (Python:
  `noul_floor`) sets H2O-Lightning's yes/no floor. As released, a yes/no
  answer less sure than 0.801 is reported at 0.801 on its side; 0 turns
  that off, to route uncertain answers elsewhere. Other models refuse
  the option.

- Images can come inside the state, for every model that reads images
  (Clef, pplx, H2O-Lightning): a string value that is a data URL or
  bare base64 of an image, or a chat-style image part (`image_url`,
  `image`, `input_image`, `base64`), anywhere in the state. Each is replaced by
  `[image N]`, numbered after the images in `images`, and read as if it
  had been given there; errors name its place (`state.photos.1`). Image
  URLs are refused, not fetched.

### Changed

- The `download` menu lists ten models now; "Other..." is chosen with the
  arrow keys (number keys reach the first nine entries).

## [0.7.0] - 2026-10-09

### Added

- Perplexity's pplx decider (`perplexity-ai/pplx-decider-v1.1-27b`), a
  27B model with text and images, through the same API, `run`, `chat`,
  `server`, `mcp` and `benchmark`, and in the `download` menu after the
  Clef models. It reads each question as its own sequence (up to 8,192
  tokens; longer input is refused, not cut) and answers up to 255
  options. `convert -q` and `--bits 4`, or `download --bits 8` / `--bits
  4`, store an 8-bit copy (29 GB peak, Macs with 48 GB or more) or a
  4-bit copy (17 GB peak, 32 GB or more); the decision head and the
  vision tower stay as released. Its prompt matches the release's code
  token for token; probabilities were compared with that code on a small
  random model in the release's layout.

### Changed

- The accuracy results in the README and `docs/BENCHMARKS.md` now come
  from five complete public test splits that cover all three question
  types: TREC (`choice`), TweetEval Offensive (`noul`), ANLI round 3,
  OpenBookQA (`choice`) and SST-5 (`score`), 5,270 examples per model,
  for every supported model including pplx, and Jev for comparison.
  They replace the earlier table of 500 examples from five `choice`
  benchmarks. `scripts/accuracy_benchmark.py` runs this suite by default
  (`--suite classic` for the earlier one), checks each dataset against
  the Hub commit it was saved from, reports macro-F1 for `noul` and the
  mean absolute error for `score`, and resumes an interrupted run.

### Fixed

- Clef split text with combining marks (Hindi, Bengali, Thai, Arabic
  with vowel signs, and other scripts that use them) into other tokens
  than Cloudflare's reference: the release's `tokenizer.json` carries an
  older pre-tokenizer pattern, which transformers replaces when it loads
  the tokenizer. Clef now loads it the same way, so such requests get
  the reference's tokens and answers (Thai: 329 tokens instead of 343 on
  the test request). Converted copies are fixed too; nothing needs to be
  converted again.

- When a model in a `server` or `benchmark` folder does not fit in
  memory, the hint now names the model's folder in its `convert`
  command (`convert -m ~/Models/clef -q`), not the model's name, which
  `convert` could not find.

## [0.6.0] - 2026-10-08

### Added

- Cloudflare's Clef 27B (`Cloudflare/clef`), with text and images, in
  the `download` menu after clef-flash. In bf16 it needs a Mac with
  96 GB or more and matches Cloudflare's reference within 0.031 per
  probability; as an 8-bit copy it runs from 48 GB (32 GB peak), as a
  4-bit copy from 32 GB (19.5 GB peak).

- `download` can store a Clef model quantized: `--bits 8` or `--bits 4`,
  or a choice of full size, 8-bit or 4-bit in the menu, which marks the
  sizes that do not fit in memory and defaults to the largest that does. The full-size
  weight files are fetched one at a time, quantized and deleted, so
  Clef 27B in 8-bit needs about 30 GB of disk, not 55. Stored in the
  Hugging Face cache, the copy loads by repo id without fetching the
  full-size files.

- `server` serves several models: repeat `-m`, or pass a folder of
  models. The request's `model` field picks one; `--default-model`
  answers requests that name none or an unknown one (such as the SDK's
  `jev-latest`). Models load on first use and the least recently used
  are unloaded to stay within `--memory-budget` (default: the GPU's
  recommended working set). `/v1/models` lists the models, `/health`
  the default and the loaded ones.

- `benchmark -m FOLDER` measures every model in a folder, each in a
  process of its own, and ends with a summary table; folders no family
  loads and models that do not fit are listed as skipped. `--out DIR`
  writes one file per model, `--json` prints a list.

- Loading checks first that the model fits in memory: if its weights
  plus 10% exceed the GPU's recommended working set, it is refused
  before anything is read, with both sizes and, for a model that can be
  quantized, the `convert` command for the largest copy that fits, or
  that none does (instead of the Mac swapping). `--no-memory-check` (or `load(..., check_memory=False)`)
  loads it anyway. `server` and `mcp` check before they start.

- Clef reuses what it computed for a state: a request with the same
  state (and images) as a recent one and other questions reads only the
  new questions. On by default for the Python API, `run --states`,
  `chat`, `server` and `mcp`, with up to 2 GB of kept states per model;
  `--prefix-cache GB` (or `load(..., prefix_cache_gb=...)`) changes the
  amount, 0 turns it off. Answers stay within the parity tolerance of
  those without reuse (at most 0.004 apart on clef-flash) and are the
  same for a first and a repeated request.

- `python -m mlx_decision` runs the command line, as `mlx-decision`.

- `scripts/clef_parity_bundle.py` writes a standalone folder that runs
  Cloudflare's reference and mlx-decision side by side on the parity
  set on another Mac, for models too large for this one (how Clef 27B
  in bf16 was checked). `scripts/quantization_report.py` takes such a
  results file with `--reference` and keeps each model's results with
  `--cache DIR`, so adding a model runs only that model.

### Changed

- `server` no longer ignores the request's `model` field. With one
  model, every request still goes to it; with several, a request without
  a known `model` gets `422` unless `--default-model` is set.

- `convert` reads, quantizes and writes one weight shard at a time, so
  it no longer needs memory for the whole model: an 8-bit copy of
  clef-flash peaks at about 10 GB instead of 19 GB, and a model larger
  than the Mac's memory can be quantized. The output has the same
  weights as before; how they are split into files may differ. Mixed
  precision (`--target-bits`) still loads the whole model.

- `benchmark --out FILE` writes the results as JSON together with the
  machine they were measured on (Mac model, chip, CPU cores per kind,
  GPU cores, memory, GPU working set, macOS), the versions of
  mlx-decision, MLX and Python, the model's details and the options.
  `--json` prints the same object, and the table now starts with a line
  naming the machine.

- `benchmark` also times each row for Clef with its state already
  computed (the "warm" columns, `warm_median_s` and `warm_p95_s` in the
  JSON): on clef-flash, a 16,384-token request takes about 10 s the
  first time and 0.2 s with one new question, 1.5 s with twenty.

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
