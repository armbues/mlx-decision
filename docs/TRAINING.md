# Training

mlx-decision can fit a decision model to your own labelled examples, on
the Mac it runs on, and serve the result like any other model. Four kinds
of fit cover the six model families, from a calibration in seconds to
fine-tuning a whole encoder. None of them needs more than MLX.

The workflow has four steps:

1. **Data**: turn a dataset into labelled examples, one JSON request per
   line plus its labels (`mlx-decision data`, or your own script).
2. **Fit**: `mlx-decision training fit -m BASE --data FILE` trains and
   writes a *fitted model*. This is a small folder that points at its
   base and holds only what was trained.
3. **Compare**: `mlx-decision training eval -m FITTED --data TEST --compare
   BASE` scores both on data neither has seen.
4. **Use**: pass the folder to `-m` in `run`, `chat`, `server` or `mcp`,
   or to `mlx_decision.load`.

```bash
pip install "mlx-decision[training]"     # `data` reading from the Hugging Face Hub

mlx-decision data --preset trec --split train -o trec-train.jsonl
mlx-decision data --preset trec --split test -o trec-test.jsonl
mlx-decision training fit -m SupersonicLabs/Julia-1 --data trec-train.jsonl -o julia-trec
mlx-decision training eval -m julia-trec --data trec-test.jsonl --compare SupersonicLabs/Julia-1
mlx-decision run -m julia-trec -s "Who wrote Hamlet?" -q questions.json
```

(`questions.json` holds the question as the training file asks it, the
`questions` object of any of its lines.)

Run in a terminal without flags, `data` and `training fit` ask for what
they need in menus. When they finish, they print the command line that
does the same thing without menus.

## What a fit gains

Each model was fitted on its dataset's train split with the default
settings (`--method auto`) and scored on the test split. Before → after:
accuracy / macro-F1 / ECE in percent (ECE: expected calibration error,
lower is better). Measured on an Apple M5 Pro (20-core GPU, 64 GB):

| Model | Kind | Training data | Base | Fitted | Time | Peak memory |
|---|---|---|---|---|---|---|
| clef-flash (8-bit) | head | ANLI-R3, 2,000 rows | 50.3 / 49.9 / 20.9 | 52.7 / 52.5 / 21.4 | 10.4 min | 15.1 GB |
| clef-flash (8-bit) | calibrate | 2,500 rows, five datasets | ANLI-R3: 50.3 / 49.9 / 20.9 | 52.2 / 52.3 / 2.8 | 10.6 min | 11.9 GB |
| H2O-Lightning-4B | readout | TREC, 5,452 rows | 85.2 / 86.8 / 4.7 | 97.4 / 97.0 / 1.4 | 12.8 min | 10.6 GB |
| d1-3B | readout | TREC, 5,452 rows | 86.6 / 86.8 / 5.5 | 96.4 / 94.4 / 1.4 | 4.0 min | 6.1 GB |
| laya | finetune | TweetEval Offensive, 11,916 rows | 77.8 / 61.9 / 7.7 | 84.7 / 78.4 / 5.0 | 17.2 min | 8.9 GB |
| Julia-1 | finetune | TREC, 5,452 rows | 16.0 / 14.0 / 63.1 | 97.4 / 96.0 / 1.9 | 2.3 min | 4.8 GB |

Every kind also fits a calibration at the end (see below). A fit holds
back a fifth of the training data to report on, so these runs trained on
four fifths of each train split. The clef-flash calibration was a single
fit over 500 rows from each of the five datasets, one question each. On
the other four datasets it lowered ECE as well (TREC 4.4 → 2.1, Offensive
9.2 → 3.5, OpenBookQA 4.5 → 1.9, SST-5 10.8 → 6.5) and left accuracy
within half a point.

The laya run trained three epochs and kept the first. With early
stopping, the default since then, the same run stops after the second
epoch and takes about 11.5 minutes.

What to expect in general:

- **Julia** gains the most: on TREC its release guesses (16%), and after
  2 minutes of fine-tuning it answers 97% right. Fine-tuning helps it on
  any task with fixed options.
- **H2O-Lightning and d1** gain 5 to 12 points on a task with fixed
  options and enough examples. A readout fit leaves the network alone and
  retrains only the answer step, so a fitted model shares its base in
  memory.
- **laya** gains on tasks its pre-training covered less well, such as
  offensive language.
- **Clef** gains little from its head alone (2 to 3 points on entailment).
  Its calibration fit is the useful one: it makes the probabilities
  trustworthy when Clef is overconfident, as on ANLI-R3 (ECE 20.9 → 2.8).

## Training data

### Format

A training file has JSON lines. Each line is a request body as the
server takes it (a `state` and its `questions`) plus `labels`, the
correct answer for each question, by question id:

```json
{"id": "t1", "state": "My card was charged twice.", "questions": {"team": {"type": "choice", "instructions": "Which team handles this ticket?", "criteria": {"billing": "Payments, refunds, invoices", "tech": "Crashes, bugs, errors", "sales": "Plans, pricing, upgrades"}}, "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?"}}, "labels": {"team": "billing", "urgent": true}}
```

- `choice`: the label is an option id (a key of `criteria`).
- `noul`: `true` or `false` (also as strings).
- `score`: the level, as one of the `criteria` keys (`2` or `"2"`).
- `id` is optional. It must be unique in the file and defaults to the
  line number.
- A question without a label is asked but not trained on.
- Images are not accepted in training data yet.

The whole file is checked before any model loads: every request must be
valid (the same checks as `run` and the server), and every label must
name an option of its question. An error names its line.

Two rules decide what a fit can learn:

- **Ask a question the same way on every line.** A fit applies to a
  question by its id and its options. Calibration biases and readout rows
  are fitted per option, so they need the same option ids in the same
  order everywhere. Option biases also need the same instructions and
  criteria. A question asked in several wordings is still fitted (a
  temperature, readout rows) but gets no option biases.
- **The data must look like what the model will be asked.** See
  [Calibration](#calibration-needs-data-like-the-target).

### `data`: datasets to training files

```bash
mlx-decision data SOURCE -o FILE.jsonl [options]
mlx-decision data SOURCE --show 3        # preview: three lines and the label counts
```

SOURCE is `hf:ORG/NAME` (a Hugging Face dataset, with `--config`,
`--split` and `--revision`), a `.csv`, `.jsonl` or `.parquet` file, or a
folder written by the `datasets` library's `save_to_disk`. Everything
except CSV and JSONL files needs the `training` extra. A dataset with
several configs or splits must be told which one to use; the error lists
them. `--limit N` takes the first rows, `--sample N --seed S` draws rows
at random (in the dataset's order). The same flags always write the same
file.

Each dataset row becomes one example with one question. The *shape*
says how columns map onto it. It is guessed from the column names and
types (a class-label column gives the label names); `--shape` and the
column flags override the guess. When a guess is ambiguous, the command
stops and lists the columns.

| Shape | Columns | The question | Example |
|---|---|---|---|
| `text` | `--text` → the state, `--label` → the answer | a `choice` with one option per class, or a `noul` with `--noul` (two classes, the second is yes) | emotion, TREC |
| `pair` | `--premise` and `--hypothesis` → the state as `{"premise": ..., "hypothesis": ...}`, `--label` | a `choice` (e.g. entailment / neutral / contradiction) | SNLI, ANLI |
| `choice` | `--choices` → the options, `--answer`, `--context` (if any) → the state | a `choice` per row with the row's options | ARC, OpenBookQA |
| `yesno` | the passage → the state, `--question-column` → the question, a bool label | each row's own `noul` | BoolQ |
| `rating` | `--text` → the state, a numeric `--label` | a `score` with one level per value (`--levels "bad,ok,good"` describes them) | yelp_review_full |

The question's wording has no default. Without `--question`, `-q` or a
preset, the command stops and names these flags. In a terminal, it asks
instead.

- `--question TEXT`: the instructions.
- `--id ID`: the question id (default: the dataset's name).
- `--options "id=description,..."`: the option ids and their
  descriptions (default: the label names). Use `\,` for a comma inside a
  description.
- `--label-map "0=negative,1=positive"`: raw labels to option ids.
- `-q FILE`: the whole question as JSON, as in `run`.

Rows without a usable label are left out and counted in the summary.

```bash
mlx-decision data hf:dair-ai/emotion --config split --split train -o emotion.jsonl \
  --question "Which emotion does the text express most?"

mlx-decision data hf:Yelp/yelp_review_full --split train --sample 5000 \
  --shape rating --question "How positive is this review?" \
  --levels "very negative,negative,neutral,positive,very positive" -o yelp.jsonl
```

**Presets.** `--preset trec|offensive|sst5|openbookqa|anli-r3` (with
`--split`) gives the five datasets of the accuracy benchmark's coverage
suite, with its wording and label maps, pinned to the commits it was
measured on. These are the datasets of the table above. Their splits:
the dataset's own (`train`, `test`; ANLI's `train_r3`, `dev_r3` and
`test_r3`).

**Menus.** Without a source in a terminal, `data` asks step by step,
with its guesses preselected: the source (presets, a Hub dataset, a
file), config and split, shape, columns (with each column's type and a
sample value), label names, the question (the same builder as `chat`)
and the output file. It then shows the preview, asks for confirmation
and finally prints the equivalent command. `--no-input` turns the menus
off.

## Fitting

```bash
mlx-decision training fit -m BASE --data FILE.jsonl [-o FOLDER] [--method KIND]
```

BASE is any model `-m` takes: a folder or a repo id, in any precision. It
can also be a fitted model; its fit is then replaced and its base kept.
The output folder defaults to `<base>-<data name>` in the current folder.
Progress goes to stderr and the report to stdout. The report shows, per
question, accuracy, macro-F1 and ECE before and after, the kept epoch and
the time taken. It is also saved in the folder as `training.json`.

### Splits

A fifth of the data (`--holdout 0.2`, stratified by label, seeded by
`--seed`) is held out. It only scores the report: nothing is tuned on it.
`--validation FILE` replaces the holdout with a file of its own. The rest
is the training part. A tenth of it (at least 50 examples) is the *dev*
part, which picks the readout fit's settings and the best epoch and fits
the closing calibration. The rest is what the weights are trained on.

Fewer than 50 training examples, or fewer than 10 for an option, is a
warning. The fit still runs, but its numbers are noisy.

### Kinds

`--method auto`, the default, picks the family's own kind and follows it
with a calibration:

| Family | `auto` | What is trained | Written to the fitted folder | Also available |
|---|---|---|---|---|
| Clef | `head` | the decision head on top of the frozen backbone | the head (clef-flash: 244 MB) | `calibrate` |
| pplx, H2O-Lightning, d1 | `readout` | each option's readout row and a bias, on the frozen network | rows and biases (kilobytes) | `calibrate` |
| Laya, Julia | `finetune` | the whole encoder and the marker head (`--head-only`: the head alone) | the encoder and head (Julia 0.3 GB, laya 0.8 GB) | `calibrate` |

- **`calibrate`** (every family): one temperature per question, and one
  bias per option when the question is asked the same way on every line.
  It is fitted to the model's probabilities by minimum log loss, in under
  a second after one scoring pass. Accuracy changes only through the
  biases. The probabilities then match how often the model is right.
- **`readout`** (pplx, H2O-Lightning, d1): these models read an answer
  from the last hidden state through one row per option of the language
  model's output layer. The fit trains those rows plus a bias per
  option, starting from the release's rows. The hidden states are
  computed once per example and kept. The learning rate (1e-5, 1e-4,
  1e-3) and the pull towards the release's rows (L2 0 or 0.01) are picked
  on the dev part, over up to 400 full-batch steps. A question whose
  release rows are best on dev is only calibrated.
- **`head`** (Clef): the backbone runs once per example and its hidden
  states are stored on disk, beside the output folder. Their size (2.6
  MB per ANLI-R3 request) is shown before the run and checked against the
  free space. The folder is removed when the run ends, also on errors and
  Ctrl-C. The joint head trains on those states in float32: AdamW 5e-5,
  weight decay 0.01, 5% warm-up and a linear decay, 16 requests a step.
- **`finetune`** (Laya, Julia): AdamW 2e-5 (head alone 1e-4) with the
  same schedule. Batches hold up to 4,096 tokens and 32 rows. Compute is
  in bf16 with float32 master weights and moments, and activations are
  recomputed per layer (gradient checkpointing). The fitted weights are
  stored in bf16.

The trained kinds use cross-entropy with label smoothing 0.1 plus a Brier
term; the readout fit uses the cross-entropy alone. `head` and
`finetune` train for 3 epochs (`--epochs`). After each epoch they score
the dev part, keep the epoch with the lowest dev loss, and stop when an
epoch is worse than the best one before it. `--no-early-stop` trains
every epoch (the best is still kept).

A family's other kinds are refused with a message naming the ones it
takes. With `--seed` fixed, `calibrate` and `readout` give the same
fitted model again. Fine-tuning and the head can differ by a point or so
between runs.

### Calibration needs data like the target

Every fit ends with a calibration, fitted on the dev part, a slice of
your training data. It can only correct the model's confidence as it is
on that data. If the model finds the training data much easier (or
harder) than what it will be asked later, the calibration corrects in
the wrong direction.

ANLI-R3 shows this. Clef answers 87% of its train split right but only
50% of its test split. A calibration fitted on the train split made Clef
more confident (temperature 0.89), and on the test split ECE rose from
20.9 to 23.0. Fitted on 500 rows of the `dev_r3` split, which is like
the test split, it fell to 2.8. A head fit on train rows shows the same
effect: its calibration, fitted on train-like dev rows, raises ECE on
the test split (table above).

So use data drawn from the same source as the requests the model will
answer. When a public dataset's train split differs from your use, fit
the calibration (`--method calibrate`) on a sample that does not.

### Memory and time

Before loading, `training fit` estimates the peak memory: the model's
weights plus what the kind keeps beside them (optimizer state,
activations, kept hidden states). It shows the estimate and checks it
against this Mac's GPU working set, the same check the other commands
make, which is about two thirds of the memory on a 16 GB Mac. A run over
it is refused, and the message names smaller runs that would fit
(`--head-only`, `--method calibrate`) or a quantized copy to convert
first. A head fit also shows the disk its stored states need.

Peak memory of the `auto` runs, estimated by `training fit` for these
datasets (measured, where the run above was made), and the smallest Mac
whose working set holds it:

| Model | Peak | Mac |
|---|---|---|
| clef-flash, 4-bit copy | about 10 GB (`calibrate` 7 GB) | 16 GB (at the edge) |
| clef-flash, 8-bit copy | 15.1 GB (`calibrate` 11.9 GB) | 24 GB |
| clef-flash, bf16 | about 24 GB (`calibrate` 21 GB) | 48 GB (`calibrate` 32 GB) |
| pplx, 4-bit copy | about 17 GB | 32 GB |
| pplx, 8-bit copy | about 31 GB | 48 GB |
| H2O-Lightning-4B, 8-bit / 4-bit copy | about 6 / 3.5 GB | 16 GB |
| H2O-Lightning-4B, bf16 | 10.6 GB | 16 GB (at the edge) |
| d1-3B (bf16) | 6.1 GB | 16 GB |
| laya, laya-typed-decisions | 8.9 GB (`--head-only` 1.4 GB) | 16 GB |
| laya-multilingual | about 10.2 GB (`--head-only` 1 GB) | 16 GB (at the edge) |
| Julia-1 | 4.8 GB (`--head-only` 0.4 GB) | 16 GB |

The clef-flash 4-bit and the pplx fits have not been measured. Clef 27B
takes the same kinds as clef-flash; on its 8-bit copy the head needs
about 36 GB (a 64 GB Mac), also unmeasured.

Time depends mostly on the Mac's chip, so `training fit` does not
estimate it before the run. During the run, the progress bar shows the
time left, measured from the steps so far. As a guide, on the M5 Pro
above (see the table at the top):

- Scoring: about 0.04 s per example for d1, 0.14 s for H2O-Lightning,
  0.26 s for clef-flash (8-bit) on ANLI-R3's long premises. A readout fit
  scores every example once and then trains in a few seconds.
- Head: about 40 s per epoch over 1,440 ANLI-R3 requests, after the
  scoring pass that stores their states.
- Fine-tuning: Julia about 40 s per epoch over 3,900 TREC questions,
  laya about 5 minutes per epoch over 8,600 tweets. `--head-only` is
  several times faster but gains less.
- Calibration alone: one scoring pass.

A Mac with fewer GPU cores, or an older chip, takes longer: on a base M1,
expect several times these figures.

### Progress, events, cancel

The progress bar runs on stderr. `--events` prints the progress as JSON
lines on stdout instead, one per event, for other programs to follow:

```json
{"stage": "check", "metrics": {"examples": 600, "holdout": 120, "fit": 430, "dev": 50, "labels": {...}, "kinds": ["calibrate"], "estimate": {"peak_gb": 0.9}}, "message": "loading laya"}
{"stage": "before", "step": 3, "total": 120, "eta_s": 9.5}
```

The stages are `check`, `before`, `features`, `train`, `calibrate`,
`after`, `write` and `done`; a run goes through those its kinds need. A
run that fails ends with `{"stage": "error", "message": ...}`, a
cancelled one with `{"stage": "cancelled"}`. Ctrl-C stops the run at its
next step: the next example scored, the next training batch, or the next
20 steps of a search. A second Ctrl-C stops it at once. A stopped run
writes nothing: there is no half-written folder.

### Menus

Run in a terminal without `--method` (or without the data, base or
output), `training fit` asks for what is missing. It offers the kinds
the family takes, each with its estimated peak memory (marked when it
does not fit), then the holdout share or a validation file, the epochs
and the output folder. It then shows a summary with the equivalent
command line and asks before starting. `--yes` takes the defaults, and
`--events` never asks.

## Fitted models

A fitted model is a folder:

- `mlx_decision.json`: the family, the base (a Hugging Face repo id at
  its commit, or an absolute path), a fingerprint of the base, the kinds,
  the questions with their options, the metrics, the seed and the data
  file's hash.
- `fit.safetensors`: the trained tensors only. The base's weights are
  not copied.
- `training.json`: the full report and the options of the run.

It loads anywhere a model does, and its name is the folder's name:

```bash
mlx-decision run -m julia-trec -s "Who wrote Hamlet?" -q questions.json
mlx-decision chat -m julia-trec
mlx-decision server -m ~/Models          # lists fitted models beside their bases
mlx-decision mcp -m julia-trec
```

```python
import mlx_decision

model = mlx_decision.load("julia-trec")
model.info()["fit"]      # kinds, questions, metrics, seed
```

Through the server, the Jev SDK picks a fitted model with
`model="julia-trec"`.

**The base must not change.** Loading resolves the base (a repo id from
the Hugging Face cache, downloaded if needed, at the recorded commit)
and compares its fingerprint: its config files and the names, shapes and
types of its weights. A missing, replaced or re-quantized base is
refused with a message naming it. A fit made on a quantized copy needs
that copy; fit again to use another.

**Fits on unchanged weights share their base.** A calibration or readout
fit changes only probabilities and readout rows, so the server loads the
base once for all its fitted models of these kinds. Each such fit adds a
few kilobytes. A head or fine-tune fit loads its own copy of the base,
with its trained weights.

**Which questions a fit applies to.** Calibrations and readout rows
apply to a question with the same id, type and options as in the
training data. Other questions are answered without them: by the base
for a calibration or readout fit, and by the trained weights without
their calibration for a head fit or fine-tune. `run` and `chat` warn
once per such question, the server and MCP log it, and the MCP tool
description names the fitted questions. A fine-tuned encoder or a
trained Clef head applies to every question. A fine-tune on one task
can make the model worse at others, so check the questions you rely on
with `training eval`.

H2O-Lightning's yes/no floor still applies after a fit, and every
family keeps its input limits and checks.

## Evaluating

```bash
mlx-decision training eval -m julia-trec --data trec-test.jsonl --compare SupersonicLabs/Julia-1
```

```
answer_type (choice, 500 examples)
             julia-trec     Julia-1
  accuracy        97.4%       16.0%
  macro-F1        96.0%       14.0%
  ECE              1.9%       63.1%
  confusion, julia-trec (rows: gold, columns: predicted)
  ...
```

For each labelled question, `training eval` shows accuracy, macro-F1
and ECE (`score` questions also mean absolute error), refused and cut
examples when there are any, and a confusion matrix. `--compare` scores
a second model after the first; only one is loaded at a time. `--json`
prints the results as JSON. Evaluation needs no extra.

Scores from `training eval` are measured on data the fit has not seen
only if you give it such data: use a test split or your own held-back
examples, not the training file.

## Python

```python
import threading

from mlx_decision.training import evaluate, fit

stop = threading.Event()            # stop.set() from another thread cancels

result = fit(
    "SupersonicLabs/Julia-1",
    "trec-train.jsonl",            # or a list of data.format.Example
    output="julia-trec",
    method="auto",                 # calibrate, readout, head, finetune
    holdout=0.2,
    seed=0,
    epochs=3,
    on_event=lambda event: print(event.as_json()),
    cancel=stop.is_set,
)
print(result.metrics["answer_type"]["after"])

report = evaluate("julia-trec", "trec-test.jsonl", compare="SupersonicLabs/Julia-1")
print(report.as_json())
```

`fit` raises `Cancelled` when cancelled (nothing written),
`mlx_decision.data.format.DataError` for a bad training file (with its
`line`), and `mlx_decision.ModelTooLargeError` when the run does not
fit. Further
options: `validation=`, `head_only=`, `early_stop=`, `overwrite=` and
the loader's options (`max_input_tokens=`).

## Not in this version

- LoRA or full fine-tuning of Clef, pplx, H2O-Lightning or d1. Their
  networks stay frozen, and only the head or the readout rows train.
- Images in training data.
- Multi-label tasks (several correct options for one question).
- Generating or labelling training data.
- Publishing fitted models to the Hugging Face Hub. A fitted folder
  refers to its base, so share it together with the base's repo id.
