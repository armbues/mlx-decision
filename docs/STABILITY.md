# Stability

From 1.0.0, mlx-decision follows [Semantic Versioning](https://semver.org/).
This page says what that covers. There are two tiers: **stable** parts do
not break before 2.0, and **provisional** parts may still change in a
minor release, which the [CHANGELOG](../CHANGELOG.md) then says. Anything
not listed here is internal.

## Stable until 2.0

What works with 1.0 keeps working with every 1.x release.

**Python.** The names `mlx_decision` exports:

- `load(model, *, check_memory=True, **options)` and the options listed
  in [PYTHON.md](PYTHON.md#loading-options).
- `DecisionModel`: `name`, `info()`, `decide(state, questions, *,
  images=None)`, `decide_request(request)` and `check(request)`. The
  `info()` method is stable; the contents of its dict are provisional.
  `check` raises what answering would (without loading images or
  counting tokens) and returns the request as checked; that object's
  contents beyond its own fields are not promised.
  `DecisionModel.backend` is not part of the promise.
- The question types `Choice`, `Score` and `Noul`; `Request`; the
  answers `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`; `Usage`,
  `Response` and `Result` (with `truncated` and `to_wire()`).
  `Request.videos` is reserved: an empty list or null is fine, and any
  video is refused (`code: "unsupported_media"`) until a family reads
  video.
- The errors: `DecisionError` (`message`, `param`, `code`; a
  `ValueError`), `ModelTooLargeError` (`required` and `available`, in
  bytes; a `DecisionError`), `DataError` (`line`, `param`, `file`; a
  `DecisionError`) and `PlatformError` (a `RuntimeError`).

**The wire format.** `POST /v1/systemone` takes and answers the Jev API's
request and response bodies (see [The Jev API](#the-jev-api)), with the
error body and statuses in [Errors](#errors). Also stable are the
documented extensions:
`GET /v1/models`, `GET /health`, the request's `model` field choosing
among several models, images as data URLs in `images` or in the state
(mlx-decision's own way; see [The Jev API](#the-jev-api)), the
`X-MLX-Decision-Truncated` header, zstd-compressed bodies, and the
API key as `Authorization: Bearer KEY`.

**MCP.** The `decide` tool and its arguments (a request body) and the
shape of its result.

**Commands.** `run`, `chat`, `server`, `mcp`, `download`, `convert`,
`benchmark` and `web`, with every flag their `--help` shows: its name,
short form and what it takes; and `mlx-decision --version`. What `run
--json` and `run --states` print is stable: response bodies in the wire
format, and for a line `--states` cannot answer `{"error": ..., "line":
N}`. Other output (tables, progress, messages) is for people and may
change.

**Events.** Each line `--events` prints is a JSON object with `stage`,
and when they apply `step`, `total`, `eta_s`, `message` and `metrics`.
A run's last line has the stage `done`, `error` or `cancelled`, also
when it fails before its first step or is stopped. A run its arguments
refuse (exit 2) prints no lines. The other stage names and everything
inside `metrics` are provisional.

**Files.**

- Fitted models: every 1.x release loads a fitted model that 1.0 wrote
  (or 0.9), as long as its base model is unchanged.
- Training data as JSON lines: a request body plus `labels` and an
  optional `id` per line, as [TRAINING.md](TRAINING.md) describes.
- The config file, `config.toml`: its keys `models`, `default_model`,
  `memory_budget_gb`, `host`, `port`, `workspace` and `models_dir`.
- The environment variables: `MLX_DECISION_HOME` (the folder of the
  config file and the default workspace), `MLX_DECISION_API_KEY` (the
  API key of `server`, `mcp` and `web`, as `--api-key`) and
  `MLX_DECISION_WATCH_STDIN` (set to `1`, a run with `--events` stops
  when a pipe on its stdin closes).
- The Studio's workspace: a later 1.x Studio opens a workspace that 1.0
  made. The question sets in it stay request bodies, its models stay
  model folders `-m` loads, and its datasets stay readable by that
  release's `training` commands (a single JSON lines file stays training
  data as above; a dataset folder follows its provisional format).

## Provisional

These may change in a minor release. The CHANGELOG says so, and where it
is cheap the old form keeps working for a while. Most become stable once
training on several datasets at once is done.

- The training Python API, `mlx_decision.training`: `fit`, `evaluate`,
  `label`, `FitResult`, `EvalResult`, `LabelResult`, `Event`,
  `Cancelled` and the report dictionaries.
- The flags of `data`, `training fit`, `training eval` and `training
  label`.
- `dataset.json` in a dataset folder (its `format` number guards it: a
  dataset from a later release is refused).
- What `metrics` holds in `--events` lines, and the stages before the
  last.
- The contents of `info()` and of MCP's `model_info`.

## Internal

Anything not listed above, among them the Studio's own API under
`/studio/api`, the workspace's other files (history, batches, jobs,
speed records), log and message texts, and the modules not exported from
`mlx_decision`.

## What is not a breaking change

- Additions: new optional parameters, flags, fields, routes, commands,
  stages and model families.
- Answers whose numbers move: an MLX upgrade, a fix that brings a model
  closer to its release, another quantization.
- The texts of messages, warnings and errors (their `param` and `code`
  stay).
- Speed and memory use.

## Renamed names

A stable name that is renamed in 1.x keeps working under its old
spelling until 2.0: the old flag or argument is hidden from `--help` and
prints a warning on stderr when used (never as an event line). In 1.0,
`benchmark --out` is now `-o` / `--output`, `benchmark --questions` is
now `--question-counts` and `data --config` is now `--subset`.

## Files from a later version

A 1.x release reads what earlier releases wrote. It does not promise to
read what a later release writes, but it never misreads it: a fitted
model with a kind, tensors or a fingerprint it does not know, a
`dataset.json` or a workspace with a higher `format` is refused with a
message that names it and asks for a newer mlx-decision.

## The Jev API

`/v1` follows version 1 of the Jev API as served on 2026-10-20
(`typesafe/jev-1.14.0`), except for three of its parts, which are not
served yet:

- `assets`, its images by placeholder: a request with them gets `422`
  (`param: "assets"`, `code: "unsupported_media"`);
- `multipart/form-data` bodies, its image uploads: `415`;
- `box` questions: `422` naming the question.

They are refused, never answered without them. Images go in
mlx-decision's own way instead: data URLs in `images` or inside the
state. Serving any of the three later is an addition. Further additions
Jev makes to version 1 may follow in minor releases. Should Jev change
it in a way that breaks clients, the new version is served beside the
old one, which stays.

## Errors

An error answer from `/v1` has the body

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "questions.q.type"}}
```

`param` names the field at fault, or is `null`; `code` is there only when
set. The message's text may change, its `type`, `param` and `code` do not.

| Status | When | `type` | `code` |
|---|---|---|---|
| `400` | the body is not JSON, or not valid zstd data (at most 16 frames) | `invalid_request_error` | |
| `401` | the API key is missing or wrong | `authentication_error` | |
| `403` | a foreign `Host`, or another site's page (`Origin`) | `permission_error` | |
| `413` | the body is larger than 64 MB (decoded) | `invalid_request_error` | |
| `415` | another `Content-Encoding`, or a `multipart` body | `invalid_request_error` | `unsupported_media` for multipart |
| `422` | the request is not valid, or asks what the model cannot do | `invalid_request_error` | `unsupported_media` for images, `videos` or `assets` it does not read |
| `429` | too many wrong API keys from the address (`Retry-After`) | `rate_limit_error` | |
| `500` | the model failed to answer | `api_error` | |
| `503` | under `web`: a job or batch holds the GPU (`Retry-After`) | `api_error` | `busy` |

A model too large for the memory budget is a `422` with `param:
"model"`. Jev answers a busy service with `529`; mlx-decision keeps
`503`, which its SDK retries.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done; also `server`, `mcp` and `web` stopped with Ctrl-C |
| 1 | the command failed: `error: ...` on stderr |
| 2 | refused by the argument parser: an unknown flag, a missing or bad value |
| 130 | a run stopped (Ctrl-C, or the program watching it ended) |

Which check refuses a value decides between 1 and 2: the argument parser
answers 2, a check after it (a file that cannot be read, a model that
does not load) answers 1.

## Platform

Python 3.12 and 3.13, the versions each release is tested on, on macOS
with Apple Silicon. Later Python versions may work; the package does not
refuse them. Without a Metal GPU, loading a model raises
`PlatformError`.

A test (`tests/unit/test_stable_surface.py`) compares what this page
promises (the names, signatures, flags, event fields, config keys,
environment variables, file names and format numbers, the routes,
headers and error body, MCP's tools) with a snapshot, so a change to
them does not go unnoticed.
