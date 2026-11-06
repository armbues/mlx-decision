# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/): from 1.0, what
[docs/STABILITY.md](docs/STABILITY.md) calls stable does not break before
2.0 (before 1.0, minor versions could change the interface).

## [Unreleased]

### Added

- Images in training data: a line's `images` (file paths relative to
  the file or dataset folder, or data URLs) and images inside the state,
  for every kind of fit and `training eval` on the families that read
  images (all but d1-3B, Laya and Julia, which refuse such data before
  loading). Every image is opened in the check before a model loads,
  and a fit's data hash covers the image files. A Clef head fit counts
  each image's tokens in its memory and disk estimate. d3-edge's readout
  fit on Oxford-IIIT Pet (37 breeds, 3,680 photos, 3.3 minutes) went
  from 31.6% to 86.2% on 1,850 test photos.
- The `data` command's `image` shape: a Hugging Face `Image` column, or
  a column of image paths in a CSV or JSON-lines file, and its class
  label (`--image`, `--label`, optionally `--text` as the state),
  written into a dataset folder's `images/`. `--show` lists each line's
  image.
- `training fit --calibration auto|full|temperature|none` (Python:
  `calibration=`; the Studio's New Fit: Options) picks the closing
  calibration. `none` leaves the fitted kind's answers as they are, for
  a program whose thresholds were tuned on the base's probabilities.
  The report and `training.json` say which calibration each question
  got.
- A `lora` kind of fit for d3 and d1 LFM3 models (`training fit
  --method lora`, Python `method="lora"`): low-rank adapters on the MLP
  projections of every layer, trained with each question's readout rows
  through the model's own prompt and readout, on text and images.
  `--rank` and `--learning-rate` change its defaults (rank 8, 1e-4),
  `--epochs` defaults to 2. The fitted folder holds the adapters, fused
  into the base's weights when it loads, so it answers at the base's
  speed. `auto` still picks the readout fit, which takes seconds where
  LoRA takes minutes; the summary before a fit names both. The memory
  estimate counts the adapters, one recomputed layer and the vision
  tower's rows it keeps for every training image. In the Studio's New
  Fit, a panel below the method cards offers it with its own options
  and Start LoRA Fit.
- Image datasets in the Studio: Download a Dataset previews image
  columns as thumbnails and downloads them in the image shape; the
  Datasets page shows each row's thumbnail and the image in the
  inspector; the Playground on a dataset shows the rows' images and
  sends them with Run; a batch links the images into its own folder and
  keeps standing alone, and Save Answers as a Dataset keeps them; Fit
  and Evaluate take image datasets. Adding a JSON-lines file whose lines
  name image files beside it explains how to make it a dataset folder.
- Mixes: a dataset folder whose `dataset.json` lists other datasets
  (`"mix": {"sources": [{"path": "../trec"}, ...], "cap": 2000}`) is
  read by `training fit` and `training eval` as one dataset. Its train,
  report and evaluate splits are the sources' together, train rows
  capped per source and sampled with the mix's seed; a `held_out`
  source is scored in the report but not trained on. Example ids start
  with their source's name, and the data hash covers every source. A
  mix is written as `dataset.json` format 2; plain dataset folders stay
  at format 1. A fit on a mix reports each source dataset and question
  before and after (held-out ones marked) in its output, `training.json`
  and `FitResult.sources`; `training eval` on a mix adds a table per
  source (`--json` and Python: each model's `sources`). The Studio's
  Datasets page shows a mix with its sources (rows per split, caps, the
  held-out ones, links to them), and batches, fits and evaluations
  read it as one dataset. New Fit's Mix Datasets… makes one: several
  datasets, a cap per dataset and for the others, held-out marks, a
  seed and a name. A fit's and an evaluation's reports on a mix have a
  Per dataset table.
- A `finetune`, `head` or `lora` fit warns that it changes the answers
  to questions outside its data too, without a new calibration, and
  that a mix keeps a question. [Training](docs/TRAINING.md#mixes-of-datasets)
  explains mixes, with what one keeps and what a fit on one dataset
  does to the others' questions, measured on d3-edge and Laya.

### Changed

- The closing calibration after a `readout`, `head` or `finetune` fit
  is a temperature per question only: the per-option biases, fitted on
  the dev part (a tenth of the training part), cost accuracy on
  questions with many options (on Oxford-IIIT Pet 2.9 points) and
  changed little elsewhere. `--calibration full` gives the biases as
  before. A calibration alone (`--method calibrate`) keeps them when
  the training part holds 10 or more examples of every option.
- The Studio's batch command (From a Terminal, Copy Command) runs in the
  batch's folder: `cd` into `batches/NAME`, then `mlx-decision run -m
  MODEL -q questions.json --states rows.jsonl --json`.

### Fixed

- A readout fit of a pplx or d3 model answered its yes/no questions
  with the release's rows once loaded: the fitted folder names the
  options true, false, these families read them false, true, and the
  served model did not match the two. Its choice and score questions
  were not affected. Fitted folders written before are now served
  with their fitted rows, without fitting again.

## [1.2.0] - 2026-11-04

Base Labs' joey-1 and images for the Base Labs family, H2O-Lightning-4B
at its current release v1.4.2 read two-way, soft labels for training,
and the Studio deleting downloads. Nothing stable changes.

### Added

- Base Labs' `baselabs/joey-1` (2B, MIT), the family's third model,
  through the same API, `run`, `chat`, `server`, `mcp`, `benchmark`,
  `training` and the Studio, and in the `download` menu after
  mammoth-1. A release without a stored LM head reads its answer rows
  from the tied embeddings, as the release does. In bf16 it is within
  0.010 of the release's runtime on the whole test set and peaks at
  5.0 GB (8-bit: 3.4 GB), about four times as fast as elephant-1; its
  8-bit copy changed no answer, its 4-bit copy 5 of them. On the five
  benchmarks it has the best Offensive accuracy of all local models.
- Images for the Base Labs models (elephant-1, mammoth-1 and joey-1), as
  for Clef: any number per request, each scaled to between 65,536 and
  16,777,216 pixels, `--max-image-mp` lowering the maximum. The
  releases' serving runtime reads images with another attention than
  the models were trained with, so mlx-decision reads them as their
  training code does: on the image test requests elephant-1 is within
  0.017 of that code and joey-1 within 0.021, both in bf16.
- H2O-Lightning-4B read two-way, the default of its release v1.4.2: the
  full-attention layers look both ways over the whole prompt, images
  included. `--reading one-way` (on `run`, `chat`, `server`, `web`,
  `mcp` and `benchmark`; Python: `reading=`; `info()` names it) keeps
  the reading of earlier releases, which shares the state between a
  request's questions and keeps it for later requests: for many
  questions about one long state. A fitted model answers in the reading
  it was fitted in. In bf16 v1.4.2 is within 0.031 of the release's own
  code in both readings, 4 near ties changed of 276 answers, and the
  release's three probe answers within its tolerance; on the five
  benchmarks it is the best local model on sentiment (SST-5: 61.4%).
- Soft labels in training data: a `choice` or `score` label may be a
  distribution over the options and a `noul` label the probability of
  true. Every kind of fit and the calibration train on them; a one-hot
  distribution trains exactly as the hard label. Fit and evaluation
  reports gain the log loss. The `data` command's preset
  `typed-decisions` brings LocalLLaMA's typed-decisions dataset with its
  gold distributions as soft labels.
- `scripts/accuracy_benchmark.py --suite typed-decisions` scores a model
  on the typed-decisions test split with the leaderboard's published
  scorer; [docs/BENCHMARKS.md](docs/BENCHMARKS.md) has the results.
- The Studio deletes downloaded models: a Hugging Face cache repo with
  every revision of it (files another repo uses stay), and a folder or
  a link in the models folder (for a link only the link goes). Models
  named in the config file and the default model are not deleted, and
  the confirmation shows what deleting frees.

### Changed

- H2O-Lightning-4B: `download` and loading by repo id take release
  v1.4.2, the commit mlx-decision was checked against, so a later
  release reaches you only with an mlx-decision that has checked it;
  1.1 fetched the Hub's latest. v1.4.2 has new weights and settings: no
  floor for yes/no answers, the image temperature and pixel bound of
  its `serve_config.json`, and its own temperatures for two-way. Long
  input is read whole up to the 40,960-token context; over it each
  question keeps the parts of the state most relevant to it (ranked by
  BM25, ported from the release), so no request is refused for its
  length. A v1.2.3 folder still loads with that release's settings and
  32,000-token cut, read one-way only.
- `perplexity-ai/pplx-decider-v1.1-27b`: `download` and loading by repo
  id take the commit mlx-decision was checked against; a later update to that repo
  reads larger images than mlx-decision does.
- In the Studio, an 8-bit or 4-bit download holds the GPU while it
  runs, as a conversion does, since it converts each file as it
  arrives: it does not start beside another GPU job or a batch, and the
  models answer "busy" until it is done. Full-size downloads still run
  beside everything.
- In the Studio's "Another dataset…", each split shows its rows and
  size as soon as the splits are listed, and a split larger than 50 MB
  is previewed by its first 5,000 rows, streamed, instead of being
  downloaded whole for the preview. The download still reads every
  row.

### Fixed

- H2O-Lightning text is tokenized as H2O.ai trained the model, for
  v1.2.3 too: Devanagari and vowelled Arabic were split differently.
- A `training fit --overwrite` (or `fitted.write(overwrite=True)`)
  stopped while it replaced a folder no longer loses both fits: the
  next write or load of the folder finishes the replacement.
- The Studio's Playground keeps a dataset's "Ready for a fit and a
  batch" banner after saving its questions: the page took its own save
  for a change made elsewhere, read the dataset again and dropped the
  banner a moment after showing it.

## [1.1.0] - 2026-11-01

Two new model families: Base Labs' elephant-1 and mammoth-1, and
LiquidAI's d1-2B with pictures, each checked against its release's own
code. Nothing stable changes.

### Added

- Base Labs' decision models (Baseten): `baselabs/elephant-1` (9B) and
  `baselabs/mammoth-1` (27B), text only, through the same API, `run`,
  `chat`, `server`, `mcp`, `benchmark`, `training` and the Studio, and
  in the `download` menu after d3. A request's state is read once, its
  attention looking both ways over it as the models were trained; the
  questions then run together off it, each seeing the state and itself
  but not the other questions, and the answer is read out under letter
  codes. Recent states are kept, as for Clef, so new questions on a
  recent state cost a fraction of a second. Up to 255 options and 2 to
  10 score levels; the input limit is per question, 16,384 tokens by
  default (the length they were trained on), raised up to 262,144 with
  `--max-input-tokens`; longer input is refused, never cut. Images are
  refused. `convert` writes 8-bit, 4-bit and mixed-precision copies and
  `download --bits` stores 8-bit or 4-bit ones; only the LM head's rows
  for the answer codes are kept, in float32. The prompt matches the
  release's runtime token for token on the whole test set; elephant-1
  in bf16 is within 0.039 of that runtime in bf16, no answer changed, and
  3.6 times as fast. On the five benchmarks elephant-1 is the best local
  model on entailment, ahead of Jev (ANLI round 3: 74.5%), and level with
  pplx's 27B on science questions (OpenBookQA: 96.8%). mammoth-1 was only
  checked as an 8-bit copy.
- LiquidAI's d1-2B (`LiquidAI/d1-2B`, 1.9B, on LFM3-VL-2B) as the
  family "d1 LFM3": noul, choice and score questions about text and
  pictures, through the same API, `run`, `chat`, `server`, `mcp`,
  `benchmark`, `training` and the Studio, and in the `download` menu
  before d1. A request's questions are read in sets of 16, as the model
  was trained: each set is one pass with the pictures and the state, the
  state seeing the set's questions and each question the state and
  itself, so answers depend on which questions share a set and their
  order, and nothing is kept between requests (`--prefix-cache` is
  refused). The input limit is per set, 32,768 tokens by default and up
  to 128,000 with `--max-input-tokens`; a longer state is cut, keeping
  its start and end. Pictures over 1.6 MP are shrunk, as the release
  does. bf16 only (3.8 GB); no `convert` or quantized download. The
  prompt and answer reading were written in a clean room from the
  release's behaviour; prompts, packed sets and scored tokens match the
  release's code for every request of the test set, and in bf16 the
  answers are within 0.035 of it, no answer changed, at 1.4 to 1.9 times
  its speed. On the five benchmarks it is level with d1-3B. Its weights
  are under the LFM Open License v1.0, as d1-3B's: `download` shows it
  and asks first.
- The readout fit (`training fit --method readout`, `auto`'s choice)
  trains Base Labs' answer-code rows and d1-2B's code rows, as for pplx,
  d3, H2O-Lightning and d1.

### Changed

- Requires MLX 0.32.4 or newer, the version this release was tested
  with.

### Fixed

- `convert` and `download --bits` hold one tensor of a weight file at a
  time instead of the whole file, so a release stored as one large file
  converts on a Mac with less memory than the file: elephant-1's 8-bit
  copy (one 18.8 GB file) now peaks at 10 GB instead of 23 GB.

## [1.0.0] - 2026-10-24

The Studio: mlx-decision in the browser, for trying questions, comparing
models, answering datasets and fitting models without the command line.
From this release, what [docs/STABILITY.md](docs/STABILITY.md) calls
stable does not break before 2.0; training stays provisional. Code
written for 0.9 may need the changes listed first under Changed.

### Added

- **The Studio**, `mlx-decision web`: one process
  on your Mac that serves the Studio's page and, at the same address,
  the Jev-compatible API of `server`, with the same models and memory
  budget. It lists the models `-m`, the config file, the workspace and
  the Hugging Face cache hold, and loads nothing until one is asked
  (or `--default-model`). The guide is `docs/STUDIO.md`.
  - **Playground**: question sets with several examples, eight
    templates (support triage, review rating, content moderation, fact
    check, intent routing, many languages, a conversation, a screenshot
    check), a live check of the questions against the model's limits,
    images by drop or paste, Run and Run All, Compare on two or three
    models side by side, a History of every run (what changed in the
    questions, how each answer moved, restore), and Get Code: the
    request as curl, Python with the Jev SDK, a `run` line or an MCP
    client's configuration.
  - **Models**: each model's memory, what it reads and its measured
    speed on this Mac, Load and Unload within the memory budget (with a
    preview of what loading would unload), Download a Model (families,
    sizes, whether each fits, 8-bit or 4-bit as it downloads, the
    model's terms of use), Convert to 8-bit or 4-bit, and Delete for the
    models the Studio made. With no models yet, the first start asks
    where models go and offers three that fit the Mac.
  - **Datasets**: Download a Dataset (the five benchmark datasets or any
    on the Hugging Face Hub, a role per split, samples and stratified
    parts), Add a JSON-lines or CSV file, and each dataset's labels, rows
    and files. A dataset's questions are written and tried in the
    Playground on a sample of its rows and saved to its `dataset.json`,
    each label value mapped to an answer.
  - **Batches**: a dataset's rows answered by a model, with accuracy,
    macro-F1, calibration error, a confusion matrix and a reliability
    diagram per question; rows filtered by an answer, wrong, unsure or
    refused; Run the Rest, Run Again, Export as CSV or JSON Lines, and
    Save Answers as a Dataset, so one model's answers can train another.
  - **Training**: New Fit, with the kinds of fit the base model takes,
    each with its estimated peak memory, and a report on a share kept
    aside, a split or another dataset; its stages, time left, loss and
    log while it runs; a before-and-after report and the command that
    does the same. New Evaluation scores a model alone or against
    another, such as a fitted model against its base.
  - **Settings**: the config file's keys, where each value comes from,
    and a change written to the file at once; the workspace's folders
    with their sizes and Clear.
  - **Jobs**: downloads, conversions, fits, evaluations and dataset
    downloads run as the `mlx-decision` commands with `--events`, with
    progress and Stop. Conversions, fits and evaluations hold the GPU
    alone, one at a time: every page says so, and the API answers `503`
    with `Retry-After` until the job ends.
  - **The workspace**: everything the Studio makes is a plain file the
    command line reads as it is (question sets are request bodies for
    `run -q`, datasets are training data, models load with `-m`), in
    `~/.mlx-decision` or the folder that `--workspace` or the config
    file's new `workspace` key names. `workspace.json` holds its format,
    and a Studio older than a workspace refuses to open it; one Studio
    uses a workspace at a time (`studio.lock`); a folder that is not
    empty and not a workspace is refused.
  - **Security**: it answers only on this Mac by default. The printed
    address carries a session token that sets a cookie, and requests
    with another `Host` or a foreign `Origin` are refused, so other web
    pages cannot use it. `--host` serves it on the network and then
    needs an API key (`--api-key` or `MLX_DECISION_API_KEY`), with which
    a browser on another machine logs in. Show in Finder, Clear and
    changes to the config file are offered only on this Mac.
  - Flags: `-m`, `--default-model`, `--memory-budget`, `--workspace`,
    `--host`, `--port`, `--api-key`, `--no-browser` and the load
    options of `server`. Chrome on macOS is the supported browser.
  - The page needs no build step and loads nothing from other hosts:
    Preact 10.29.8 and Preact Signals 2.11.2 (MIT) and htm 3.1.1
    (Apache-2.0) ship in the package, unchanged, with their licences in
    `LICENSES/` and the wheel's licence files.

- `docs/STUDIO.md`, the Studio page by page and its workspace's files
  from a terminal; `docs/MODELS.md`, `docs/CLI.md`, `docs/PYTHON.md`,
  `docs/SERVER.md` and `docs/DEVELOPMENT.md` with the reference that was
  in the README.
- `docs/STABILITY.md`: what stays stable until 2.0 (the Python inference
  API, the wire format, MCP `decide`, the inference and serving commands,
  the `--events` envelope, fitted models, training files, the config
  file, the Studio's workspace), what is still provisional (training),
  how renamed names keep working, and the exit codes.

- Perplexity's pplx-decider-v1.2 (27B) and pplx-decider-flash-v1.2
  (4B), in their joint request format: the state and all questions of a
  request in one sequence, each answer read at its question's marker,
  so more questions cost little and the questions see each other. The
  input limit applies per request: 32,768 tokens by default, up to
  262,144 with `--max-input-tokens`; a longer request is answered in
  parts of its questions, each with the whole state and images, as the
  release does, and a question that does not fit alone is refused.
  Images up to 1.6 MP (`--max-image-mp` lowers it). `convert` (also
  mixed precision), quantized `download` and the readout fit of
  `training fit` work as for the other families. Token for token and on
  small random models in both layouts, the answers match the release's
  own code, and flash's also on its weights (bf16: within 0.06 per
  probability). pplx-decider-v1.1-27b still loads with its own prompt.
- A config file, `~/.mlx-decision/config.toml` (`MLX_DECISION_HOME`
  moves the folder), with defaults for existing flags: `models` (`-m` of
  `server`, `web` and `benchmark`), `default_model` (`run`, `chat`,
  `mcp` and `training` without `-m`; `--default-model` of `server` and
  `web`), `memory_budget_gb`, `host`, `port` and `workspace`. Flags win
  over the file; `download`, the Studio's first start and its Settings
  page write it. An unknown key is a warning, a value of the wrong type
  an error naming the file and the key.
- `models_dir` in the config file: the folder where models live (or
  `"hf-cache"`). `-m NAME` finds a model there by name in every command
  that takes `-m`; `server`, `web` and `benchmark` without `-m` serve its
  models; `download` stores into it. When it is unset, the first
  `download` in a terminal asks for it and adds it to the file.
- A model menu for `run` and `chat` without `-m` or a default model: the
  models in `models_dir`, in `models` and in the Hugging Face cache (a
  stopped download is left out), with family and size on disk.
- `training eval --events`: the progress as JSON lines, as `training fit`
  has it, ending with the results.
- `download --events` and `convert --events`: the progress as JSON
  lines, as `training fit` prints it (`download`: bytes as steps;
  `convert`: tensors as steps), ending with `done` and the folder.
  `download --events` asks nothing, as with `--yes`.
- `data --events`: the progress as JSON lines (`check`, `read` with the
  examples and label counts, `write`, then `done` with the file); asks
  nothing, writes the file whole, and a stop leaves nothing.
- Dataset folders: `data -o FOLDER` (a path without `.jsonl`) writes
  `dataset.json` and one JSON-lines file per split. The lines keep the
  raw values of the label column; the questions and a map from each
  value to an answer are in `dataset.json` once, so rewording, merging
  classes or a yes/no question about some of them changes only that
  file. Without `--question` the folder has no questions yet, to be
  added later. `data` takes `--split` more than once into a folder
  (`--sample` once for every split or once per `--split`; written
  whole, a stop leaves nothing), and `--roles` says what each split is
  for (`train`, `report`, `evaluate`; by default from the split names)
  or cuts one into seeded, stratified parts
  (`--roles "train:0.8+report:0.1+evaluate:0.1"`). `--value-names
  "0=Analyst Update,..."` names the label's values in `dataset.json`.
- `data --shape object --columns A,B --label L`: several columns
  together are the state, an object keyed by their names (a list of
  choices beside a question, as in MMLU, or any other columns). Naming
  `--columns` picks the shape; like `text` it asks a `choice` from the
  label's classes, or nothing into a dataset folder.
- `training fit --data FOLDER` trains on a dataset folder's train split
  and reports on its report split (unless `--validation` or `--holdout`
  is given), and so does Python's `fit` (its `holdout` now defaults to
  None: the folder's report split, else a fifth as before);
  `training eval --data FOLDER` scores its evaluate split. A split's
  own file works too, and plain JSON-lines files as before.
  `training fit` says which split it trains on and reports on. Roles
  come from split names such as `train_r3`, `dev_r3` and `test_r3`, also
  for a folder whose `dataset.json` lists no splits. Into a folder,
  `--label-map`, `--options`, `--id`, `--noul` and `--levels` need
  `--question` (or `-q`). `run -q` reads the questions of a dataset
  folder's `dataset.json`.
- `download --local-dir` notes the repo in `.download.json` beside the
  model, so another repo of the same name is not taken for it.
- With `--events`, `MLX_DECISION_WATCH_STDIN=1` and a pipe on stdin,
  `training fit`, `training eval`, `download` and `convert` stop when
  the pipe closes (the program that started them is gone): at the next
  step, or after 4 seconds at once, leaving no half-written folder.
  Without the variable, or with a terminal, `/dev/null` or a file on
  stdin, nothing changes.

### Changed

Changes that code or scripts written for 0.9 may need to follow, with
what to do:

- `load` takes `check_memory` and `decide` takes `images` by keyword
  only: a 0.9 call passing them by position (`load(path, False)`,
  `decide(state, questions, images)`) now raises `TypeError`. Pass them
  by name: `load(path, check_memory=False)`, `decide(state, questions,
  images=images)`.
- Score levels must be text, an object or a list, as the Jev API takes
  them: a number, a boolean or null is refused (`422`, naming
  `questions.<id>.criteria.<n>`), since their answers' `legend` broke
  the Jev SDK's parser. Write numeric levels as text (`"1"`, not `1`).
- Requests using parts of the Jev API that are not served yet are refused
  instead of answered without them: `assets` (images by placeholder;
  `422`, `code: "unsupported_media"`), `multipart/form-data` bodies
  (`415`) and `box` questions (`422`). Send images in `images` or inside
  the state, as before.
- `decide` with questions that are not a mapping raises `DecisionError`
  (`param: "questions"`), not `TypeError`; catch `DecisionError` (a
  `ValueError`) as for every other invalid request.
- `benchmark --questions` is now `--question-counts` (it takes numbers of
  questions; elsewhere `--questions` is a questions file), `benchmark
  --out` is now `-o` / `--output`, and `data --config` is now `--subset`
  (the Studio says Subset too). The old names still work, print a
  warning and are no longer in the help; use the new ones.
- A fitted model from a newer release (a kind of fit, tensors or a base
  fingerprint this release does not know) is refused with a message
  naming it, instead of loading without the parts it does not know. Fits
  now record how their base's fingerprint was made; fits from 0.9 load
  as before.
- `dataset.json` has a `format` (1; a file without it reads as 1). A
  dataset from a newer release is refused with a message naming its
  format, and keys this release does not know are kept when it writes
  the file again.
- Requires typer 0.27 or later (it exits 130 on Ctrl-C).

Other changes:

- `pip install mlx-decision` is the whole install for the Studio, the
  server and images: FastAPI, uvicorn, Pillow and NumPy are now
  dependencies. The `server` and `images` extras (and `web`) are empty
  and still accepted, so existing install lines work. `mcp` and
  `training` stay extras; the new `all` extra installs both.
- The README is shorter and starts with the Studio and a quick start;
  the reference it held moved, unchanged, into the topic files in
  `docs/`.
- The `download` menu lists pplx-decider-flash-v1.2 and
  pplx-decider-v1.2 instead of pplx-decider-v1.1-27b, which still
  downloads by repo id.
- `run` and `chat` print a score's levels counted from 1 (`2.05 on 1-3`),
  as people rate; `--json`, the server, MCP and the API keep the
  Jev-compatible value, which counts from 0.
- `download` in a terminal no longer asks for a folder on every run: it
  follows `models_dir`.
- `download` into a new folder writes `.NAME.partial` beside it and
  renames it when complete; Ctrl-C (or a failed download) removes it, and
  removes a model's new folder in the Hugging Face cache too. `convert`
  stops at the next tensor on Ctrl-C. Neither leaves a half model behind.
- `ModelTooLargeError` is a `DecisionError` (still a `ValueError`), and
  `DataError` and `PlatformError` can be imported from `mlx_decision`.
- MCP `decide`'s schema takes what the tool answers: any JSON value as
  instructions and option descriptions, options without a description,
  levels as objects or lists, and questions without instructions.

### Fixed

- `benchmark` (and the Studio's speed) find the Mac's chip and cores
  also when started with a `PATH` without `/usr/sbin`, as from launchd
  or some editors; before, the chip was left out of `-o` files.
- A fitted model with an unknown tensor in a known group (a later
  version's `calibration.0.scale`) is refused, naming it, instead of
  read as a bias.
- Every `--events` run ends with a `done`, `error` or `cancelled` line,
  also when it fails before its first step (a bad config file, a missing
  `-m`), meets an unexpected error, is stopped with Ctrl-C before its
  steps begin, or is ended by its watching program after the grace time.
- An unexpected failure prints `error: ...` and exits 1 instead of a
  traceback (`MLX_DECISION_DEBUG=1` shows it). `--port` outside 0-65535
  is refused as wrong usage (2). `server` whose default model does not
  load exits 1 with the error (was 3, after a traceback). Escape or
  Ctrl-C in `download`'s menus exits 130.
- `data` on a local CSV or JSON lines file: a row with fewer cells than
  the header reads its missing cells as empty, and a cell over 128 KB is
  read; an unclosed quote, a line nested too deeply or holding a number
  too long, and a label column of lists or objects end with an error
  naming the file and line instead of a traceback.
- `server` now refuses requests from other sites' pages and requests that
  name another address in `Host` (`403`), as the Studio does: a web page
  could make it run a model, and with DNS rebinding read the answers.
- A compressed body made of many tiny zstd frames no longer freezes the
  server: at most 16 frames, decoded off the event loop.
- After 10 wrong API keys in a row a client waits before its next try
  (`429`, from one second doubling up to a minute), on `server`'s `/v1`
  and the Studio's login alike; wrong keys are logged with the client's address.
- Answering many requests in one process (`run --states`, `server`,
  `mcp`, the Studio's batches) no longer runs out of memory: MLX kept
  the GPU buffers of finished requests for reuse, up to more than a
  16 GB Mac has, and requests of varying length rarely reuse each
  other's, so `run --states` with Laya was killed by macOS after 251 of
  400 requests. Loading a model now caps that cache at a tenth of the
  GPU's working set (at least 1 GB; a lower limit set before is kept).
  Speed is unchanged.
- A model whose `joint_config.json` names a request format no family
  knows, or a known one with other settings, is refused with a message
  naming it: `load` by repo id fetches no weights, `convert` writes no
  copy, and `download` names the format and fetches the weights only
  after a confirmation or with `--yes`. 0.9.1 took pplx-decider v1.2
  for pplx v1.1: `download` fetched it, `convert` wrote a copy, and
  loading failed only then, with "unsupported decision config format".
- `server` accepts request bodies compressed with zstd
  (`Content-Encoding: zstd`), which Jev's Python SDK sends for bodies of
  1 KB or more on Python 3.14 or with its `zstd` extra; such requests
  used to fail. The 64 MB limit applies to the decoded body; other
  encodings get `415`. Before Python 3.14 the package installs
  `backports.zstd` for it.

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
