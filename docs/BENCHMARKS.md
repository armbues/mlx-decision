# Benchmarks

The full measurements behind the numbers in the README. All were taken on an
Apple M5 Pro (20-core GPU) with 64 GB, macOS 26, torch 2.14.1 and
transformers 5.18.0 for the PyTorch side, except Clef 27B in bf16, which
needs more memory and ran on an Apple M2 Ultra with 192 GB (macOS 26.7,
torch 2.14.1, transformers 5.19.0). Times are medians per request,
end to end from the request to the probabilities (input encoding and image
preprocessing included), after a warm-up.

- [Speed compared with PyTorch](#speed-compared-with-pytorch)
- [Latency: `benchmark`](#latency-benchmark)
- [Images](#images)
- [Quantization](#quantization)
- [Agreement with the reference code](#agreement-with-the-reference-code)
- [Accuracy](#accuracy)

## Speed compared with PyTorch

### clef-flash

Cloudflare's reference code (`joint_schema_model.py` from the release) with
transformers and PyTorch on MPS, in bf16 as released, against mlx-decision
in bf16 and as an 8-bit copy. Median of 3 runs:

| Input tokens | Questions | Image | PyTorch bf16 | mlx-decision bf16 | mlx-decision 8-bit |
|---|---|---|---|---|---|
| 394 | 1 | - | 0.88 s | 0.25 s (3.5x) | 0.28 s (3.2x) |
| 801 | 5 | - | 1.70 s | 0.47 s (3.6x) | 0.57 s (3.0x) |
| 2,348 | 20 | - | 4.87 s | 1.29 s (3.8x) | 1.68 s (2.9x) |
| 1,144 | 1 | - | 2.35 s | 0.64 s (3.7x) | 0.82 s (2.9x) |
| 1,551 | 5 | - | 3.24 s | 0.89 s (3.6x) | 1.11 s (2.9x) |
| 3,098 | 20 | - | 6.49 s | 1.83 s (3.6x) | 2.25 s (2.9x) |
| 4,145 | 1 | - | 8.65 s | 2.54 s (3.4x) | 3.00 s (2.9x) |
| 4,552 | 5 | - | 9.63 s | 2.81 s (3.4x) | 3.32 s (2.9x) |
| 6,099 | 20 | - | 12.96 s | 3.80 s (3.4x) | 4.46 s (2.9x) |
| 15,151 | 1 | - | 33.34 s | 9.86 s (3.4x) | 11.53 s (2.9x) |
| 15,558 | 5 | - | 34.48 s | 10.47 s (3.3x) | 11.89 s (2.9x) |
| 16,384 | 20 | - | 36.12 s | 11.39 s (3.2x) | 12.47 s (2.9x) |
| 682 | 3 | 640x480 | 1.58 s | 0.55 s (2.9x) | 0.60 s (2.6x) |
| 1,150 | 3 | 1024x768 | 2.73 s | 1.01 s (2.7x) | 1.08 s (2.5x) |
| 2,422 | 3 | 1920x1080 | 6.71 s | 2.52 s (2.7x) | 2.89 s (2.3x) |
| Peak memory | | | 23.9 GB | 19.0 GB | 11.2 GB |

The largest difference of any probability to the reference is 0.013 in
bf16 and 0.035 for the 8-bit copy. The reference runs with PyTorch's SDPA
attention. On a Mac the optional fused kernels for its linear-attention
layers (flash-linear-attention, causal-conv1d) are not available, so
transformers uses its plain PyTorch code for them, as it does for anyone
running the reference there.

```bash
python scripts/reference_speed.py torch --model PATH --out torch.json
python scripts/reference_speed.py mlx --model PATH --out mlx.json
python scripts/reference_speed.py report torch.json mlx.json
```

### Clef 27B

In bf16 on the M2 Ultra, over the 63 requests of the parity set (text up
to 16,384 tokens, 13 with images): Cloudflare's reference took 40.0 min,
mlx-decision 10.8 min, 3.7x faster overall (2.2x to 4.1x per request).
Peak memory 64.2 GB against 54.2 GB. This ran with the bundle from
`scripts/clef_parity_bundle.py`, which runs both sides on a Mac that has
the memory.

### pplx

Not measured: the release's code loads the 27B in float32 on a Mac (about
104 GB). `scripts/pplx_parity_bundle.py` writes the same kind of bundle as
for Clef 27B, for a Mac that has the memory.

### Laya and Julia

Each family's own code on MPS, run as it runs there: Laya through the `laya`
package (0.3.28; float32, with float16 autocast from 5 questions per
request), Julia through its `TransformerEngine` (float32; it autocasts only
on CUDA). mlx-decision runs them in float16. State tokens x questions, on
the `benchmark` grid for each model; median of 3 runs:

| Model | State x questions | PyTorch (MPS) | mlx-decision | Speedup | Max probability difference |
|---|---|---|---|---|---|
| laya | 250 x 1 | 46 ms | 15 ms | 3.0x | 0.001 |
| laya | 250 x 5 | 87 ms | 60 ms | 1.5x | 0.001 |
| laya | 250 x 20 | 324 ms | 234 ms | 1.4x | 0.001 |
| laya | 450 x 1 | 72 ms | 21 ms | 3.4x | 0.001 |
| laya | 450 x 5 | 143 ms | 97 ms | 1.5x | 0.001 |
| laya | 450 x 20 | 543 ms | 386 ms | 1.4x | 0.001 |
| laya-typed-decisions | 250 x 1 | 46 ms | 16 ms | 2.9x | 0.000 |
| laya-typed-decisions | 250 x 5 | 88 ms | 60 ms | 1.5x | 0.000 |
| laya-typed-decisions | 250 x 20 | 324 ms | 235 ms | 1.4x | 0.000 |
| laya-typed-decisions | 1,000 x 1 | 151 ms | 41 ms | 3.7x | 0.001 |
| laya-typed-decisions | 1,000 x 5 | 294 ms | 199 ms | 1.5x | 0.001 |
| laya-typed-decisions | 1,000 x 20 | 1,167 ms | 792 ms | 1.5x | 0.001 |
| laya-multilingual | 250 x 1 | 20 ms | 8 ms | 2.5x | 0.000 |
| laya-multilingual | 250 x 5 | 43 ms | 27 ms | 1.6x | 0.000 |
| laya-multilingual | 250 x 20 | 154 ms | 103 ms | 1.5x | 0.000 |
| laya-multilingual | 1,000 x 1 | 63 ms | 18 ms | 3.6x | 0.000 |
| laya-multilingual | 1,000 x 5 | 141 ms | 86 ms | 1.6x | 0.000 |
| laya-multilingual | 1,000 x 20 | 558 ms | 337 ms | 1.7x | 0.000 |
| Julia-1 | 250 x 1 | 11 ms | 5 ms | 2.0x | 0.002 |
| Julia-1 | 250 x 5 | 34 ms | 15 ms | 2.3x | 0.003 |
| Julia-1 | 250 x 20 | 128 ms | 55 ms | 2.3x | 0.003 |
| Julia-1 | 1,000 x 1 | 30 ms | 13 ms | 2.3x | 0.000 |
| Julia-1 | 1,000 x 5 | 141 ms | 48 ms | 3.0x | 0.000 |
| Julia-1 | 1,000 x 20 | 557 ms | 182 ms | 3.1x | 0.000 |
| Julia-1 | 4,000 x 1 | 202 ms | 43 ms | 4.7x | 0.000 |
| Julia-1 | 4,000 x 5 | 990 ms | 225 ms | 4.4x | 0.000 |
| Julia-1 | 4,000 x 20 | 4,360 ms | 966 ms | 4.5x | 0.000 |
| Julia-1 | 7,350 x 1 | 552 ms | 105 ms | 5.3x | 0.001 |
| Julia-1 | 7,350 x 5 | 2,989 ms | 516 ms | 5.8x | 0.007 |
| Julia-1 | 7,350 x 20 | 12,366 ms | 2,035 ms | 6.1x | 0.007 |

```bash
python scripts/marker_reference_speed.py torch --model PATH --out torch.json
python scripts/marker_reference_speed.py mlx --model PATH --out mlx.json
python scripts/marker_reference_speed.py report torch.json mlx.json
```

## Latency: `benchmark`

`mlx-decision benchmark -m MODEL`, median of 5 runs per row.

clef-flash, in bf16, as an 8-bit copy and as a uniform 4-bit copy:

| Input tokens | Questions | bf16 | 8-bit | uniform 4-bit |
|---|---|---|---|---|
| 394 | 1 | 0.25 s | 0.27 s | 0.25 s |
| 1,552 | 5 | 0.84 s | 1.07 s | 1.03 s |
| 4,149 | 1 | 2.41 s | 2.89 s | 2.76 s |
| 6,103 | 20 | 3.63 s | 4.33 s | 4.13 s |
| 16,384 | 20 | 10.7 s | 12.1 s | 11.4 s |
| Peak memory | | 19.0 GB | 11.2 GB | 7.1 GB |

Clef 27B as an 8-bit and a 4-bit copy (bf16 does not fit on the M5 Pro):

| Input tokens | Questions | 8-bit | 4-bit |
|---|---|---|---|
| 394 | 1 | 1.10 s | 1.12 s |
| 1,552 | 5 | 4.17 s | 3.88 s |
| 4,149 | 1 | 10.6 s | 9.96 s |
| 6,103 | 20 | 15.9 s | 14.7 s |
| 16,384 | 20 | 43.8 s | 40.7 s |
| Peak memory | | 31.8 GB | 19.3 GB |

pplx as an 8-bit and a 4-bit copy, median of 3 runs per row (`--questions
1,5 --repeats 3`; the time grows linearly with the questions, since each
is its own pass over the state and its question). Input tokens count the
state once per question:

| State tokens | Questions | Input tokens | 8-bit | 4-bit |
|---|---|---|---|---|
| 250 | 1 | 336 | 0.83 s | 0.89 s |
| 250 | 5 | 1,721 | 4.19 s | 4.57 s |
| 1,000 | 1 | 1,085 | 2.35 s | 2.58 s |
| 1,000 | 5 | 5,466 | 12.9 s | 13.2 s |
| 4,000 | 1 | 4,084 | 9.52 s | 9.97 s |
| 4,000 | 5 | 20,461 | 49.3 s | 50.5 s |
| 7,350 | 1 | 7,432 | 19.0 s | 18.8 s |
| 7,350 | 5 | 37,201 | 101 s | 94.4 s |
| Peak memory | | | 29.1 GB | 17.1 GB |

Laya and Julia in float16. They read each question as its own sequence with
the whole state, so the input tokens count the state once per question:

| Model | Input tokens x questions: median | Peak memory |
|---|---|---|
| laya | 308 x 1: 16 ms; 1,556 x 5: 63 ms; 6,246 x 20: 243 ms; 512 x 1: 22 ms; 2,560 x 5: 98 ms; 10,240 x 20: 387 ms | 1.8 GB |
| laya-typed-decisions | 283 x 1: 16 ms; 1,431 x 5: 60 ms; 5,746 x 20: 238 ms; 1,024 x 1: 43 ms; 5,120 x 5: 207 ms; 20,480 x 20: 842 ms | 1.7 GB |
| laya-multilingual | 283 x 1: 8 ms; 1,428 x 5: 26 ms; 5,738 x 20: 104 ms; 1,024 x 1: 18 ms; 5,120 x 5: 86 ms; 20,480 x 20: 338 ms | 1.6 GB |
| Julia-1 | 269 x 1: 6 ms; 1,364 x 5: 15 ms; 5,464 x 20: 59 ms; 1,020 x 1: 11 ms; 5,119 x 5: 50 ms; 20,484 x 20: 196 ms; 4,024 x 1: 43 ms; 20,139 x 5: 202 ms; 80,564 x 20: 800 ms; 7,379 x 1: 89 ms; 36,914 x 5: 421 ms; 147,664 x 20: 1,666 ms | 1.4 GB |

Throughput: clef-flash about 1,550-1,850 tokens per second, Clef 27B
(8-bit) about 360-400, pplx about 370-460 (8-bit) and 380-420 (4-bit),
laya and
laya-typed-decisions about 24,000-26,000, laya-multilingual about 60,000,
Julia-1 about 100,000.

### Prefix reuse

Clef keeps the computed state of recent requests (2 GB by default,
`--prefix-cache GB`), so a later request on the same state and images
computes only its questions. `benchmark` times this as "warm" columns: each
row's request again with its state kept. clef-flash with a 16,384-token
input (the state is cut to fit the questions), median of 5 runs:

| Questions | bf16 first request | bf16 same state | 8-bit first request | 8-bit same state |
|---|---|---|---|---|
| 1 | 9.91 s | 0.20 s | 11.30 s | 0.17 s |
| 5 | 10.41 s | 0.48 s | 11.83 s | 0.53 s |
| 20 | 10.37 s | 1.53 s | 12.72 s | 1.82 s |
| Peak memory | 20.4 GB | | 12.5 GB | |

Clef 27B as an 8-bit copy, with a state of 15,000 tokens: 40.4 s the
first time, 0.48 s with one question on the kept state, 1.89 s with five
and 6.05 s with twenty.

A warm request takes about as long as the same questions on a short state
(20 questions on a 250-token state: 1.27 s in bf16); the questions still
attend over the whole state, which adds a little. Keeping the 16,384-token
state adds about 1.3 GB to the peak memory. pplx, Laya and Julia read
each question together with the state, so they have nothing to reuse.

```bash
mlx-decision benchmark -m Cloudflare/clef-flash --lengths 16384
```

## Images

clef-flash with a short state and three questions (text only: 0.23 s):

| Image | Tokens | bf16 | 8-bit |
|---|---|---|---|
| 256x256 or smaller | 64 | 0.27 s | 0.29 s |
| 640x480 | 300 | 0.47 s | 0.51 s |
| 1024x768 | 768 | 0.84 s | 1.03 s |
| 1920x1080, or larger (shrunk to 2 MP) | about 2,048 | 2.3 s | 3.0 s |
| 4000x3000 with `--max-image-mp 0` | 11,750 | 36 s | 39 s |

Shrinking images above 2 MP changed no answer on test pictures with small
text (a log window, an A4 invoice scan, a settings page) and photos of up
to 18 MP; at 0.5 MP and below small text was misread.

## Quantization

clef-flash copies written by `convert`, on the 50 text requests of the
parity set (139 questions), against the unquantized model:

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| bf16 (as released) | 17.8 GB | 19.0 GB | - | - |
| 8-bit (`-q`) | 9.1 GB | 11.2 GB | 0.001 / 0.014 | 0 |
| mixed 5 (`--target-bits 5`) | 6.0 GB | 8.1 GB | 0.003 / 0.070 | 0 |
| mixed 4 (`--target-bits 4`) | 4.9 GB | 7.1 GB | 0.006 / 0.084 | 1 |
| uniform 4-bit (`-q --bits 4`) | 4.9 GB | 7.1 GB | 0.014 / 0.212 | 9 |

Clef 27B copies on the whole parity set (63 requests, 177 questions),
against Cloudflare's reference in bf16 (the unquantized model does not
fit here; mlx-decision in bf16 is within 0.031 of the reference, below).
Changed answers count the top option; in brackets those where the
reference's top two are more than 0.04 apart:

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| 8-bit (`-q`) | 27.7 GB | 32.0 GB | 0.002 / 0.113 | 1 (1) |
| uniform 4-bit (`-q --bits 4`) | 15.2 GB | 19.5 GB | 0.011 / 0.290 | 3 (2) |

pplx copies have not been compared yet: the unquantized model does not fit
here, and there is no bf16 run of it from a larger Mac.

## Agreement with the reference code

The parity tests (`tests/parity/`) compare mlx-decision with each family's
own code on fixed request sets, token for token and probability by
probability.

- clef-flash: 63 requests (13 with one to three images, states up to beyond
  the 16,384-token limit) against Cloudflare's reference in bf16 on MPS.
  Largest difference 0.035 per probability (mean 0.001; with images 0.015,
  mean 0.002), no changed answers. Also the 34 requests of the Laya and
  Julia set below (22 languages, long options; 89 questions): same token
  ids, largest difference 0.009, mean 0.001, no changed answers.
- Clef 27B: the same 63 requests in bf16 on the M2 Ultra against
  Cloudflare's reference. Same token ids, largest difference 0.031 per
  probability (mean 0.001), no changed answers.
- pplx: 97 requests (the 63 above plus the 34 below; 266 questions)
  against the release's own code. With the release's tokenizer the token
  ids are the same for every question, and the same 19 are refused as
  over 8,192 tokens. Probabilities were compared on a small random model
  in the release's layout (8 layers, real vocabulary), on the CPU: within
  0.00001. The 27B weights were not run against the release's code.
- Laya and Julia: 84 text requests (clef-flash's 50 plus 34 in 22
  languages, around the input limits, and requests the models cut or
  refuse) against their own code in float32 on the CPU. Same token ids and
  the same refusals; probabilities in float16:

| Model | Answered / refused | Largest difference | Mean difference |
|---|---|---|---|
| laya | 83 / 1 | 0.004 | 0.0003 |
| laya-typed-decisions | 83 / 1 | 0.004 | 0.0002 |
| laya-multilingual | 83 / 1 | 0.004 | 0.0003 |
| Julia-1 | 66 / 18 | 0.018 | 0.0020 |

float16 is the default for Laya and Julia because it was the closest of the
GPU precisions: float32 on the GPU runs its matrix products as TF32 (Julia:
largest difference 0.059), and bfloat16 was furthest off (Julia: 0.148).

## Accuracy

Five public benchmarks that together cover the three question types, each
with its complete test split (5,270 examples per model), with the same
questions and option descriptions for every model
(`scripts/accuracy_benchmark.py`, suite `coverage`). Each split is checked
against the Hub commit it was saved from (row count and a digest of the
rows), so later changes on the Hub cannot change the numbers silently.

| Benchmark | Split | Commit | Examples | Question |
|---|---|---|---|---|
| TREC (`CogComp/trec`) | test | `65752bf` | 500 | `choice`: kind of answer the question asks for, 6 coarse labels in words |
| TweetEval Offensive (`cardiffnlp/tweet_eval`, `offensive`) | test | `b3a375b` | 860 | `noul`: is this tweet offensive? |
| ANLI round 3 (`facebook/anli`) | test_r3 | `8e4813d` | 1,200 | `choice`: entailment, neutral or contradiction |
| OpenBookQA (`allenai/openbookqa`, `main`) | test | `388097e` | 500 | `choice`: 4 answers, without the supporting fact |
| SST-5 (`SetFit/sst5`) | test | `e51bdcd` | 2,210 | `score`: 5 levels from very negative to very positive |

Accuracy / macro-F1 / ECE (expected calibration error of the top
probability, lower is better), in percent; for SST-5 also the mean
absolute error of the expected level, in levels. A `noul` answer counts
as yes at a probability of 0.5 or more; a `score` answer's most likely
level is its prediction. clef-flash ran in bf16, Clef 27B and pplx as
8-bit copies; Jev is TypeSafe AI's hosted model (`typesafe/jev`, run on
2026-10-08), and its column needs a Jev account.

| Benchmark | clef-flash | clef (8-bit) | pplx (8-bit) | laya | laya-typed-decisions | laya-multilingual | Julia-1 | Jev |
|---|---|---|---|---|---|---|---|---|
| TREC | 96.8 / 95.5 / 4.5 | 96.4 / 94.5 / 2.8 | 94.0 / 93.7 / 3.3 | 79.0 / 76.9 / 2.9 | 78.8 / 77.2 / 14.3 | 89.8 / 87.2 / 4.0 | 16.0 / 14.0 / 63.1 | 92.8 / 92.8 / 4.3 |
| TweetEval Offensive | 84.0 / 78.1 / 9.5 | 84.8 / 81.0 / 8.0 | 84.9 / 79.6 / 4.0 | 77.8 / 61.9 / 7.7 | 80.1 / 68.4 / 7.3 | 79.2 / 71.9 / 4.1 | 38.8 / 37.9 / 43.0 | 80.5 / 77.7 / 3.6 |
| ANLI round 3 | 50.8 / 50.5 / 20.7 | 55.2 / 54.7 / 16.8 | 66.8 / 67.0 / 9.4 | 39.5 / 39.4 / 42.0 | 38.5 / 37.8 / 32.2 | 35.5 / 35.3 / 50.8 | 33.2 / 31.3 / 51.3 | 69.2 / 69.4 / 13.0 |
| OpenBookQA | 95.4 / 95.4 / 3.8 | 95.4 / 95.4 / 2.9 | 96.8 / 96.7 / 1.4 | 39.2 / 38.3 / 11.5 | 41.4 / 40.9 / 3.2 | 29.2 / 28.8 / 19.8 | 33.2 / 32.8 / 48.2 | 96.0 / 95.9 / 3.1 |
| SST-5 | 57.6 / 55.1 / 10.9 / 0.51 | 57.6 / 53.3 / 3.5 / 0.51 | 58.0 / 52.9 / 5.2 / 0.47 | 35.1 / 27.6 / 31.2 / 0.99 | 44.3 / 38.4 / 7.0 / 0.68 | 27.7 / 16.8 / 58.4 / 1.40 | 36.7 / 32.2 / 41.7 / 0.82 | 57.7 / 55.3 / 17.2 / 0.49 |

Notes:
- No model refused an example or had its input cut: every state fits
  every model's limit, and every option list fits Julia's 2 to 20 options.
- Clef 27B gains on clef-flash mainly on entailment and calibration; pplx
  is the only local model close to Jev on ANLI round 3.
- About 28% of the Offensive tweets are offensive. Laya marks too few of
  them (laya: 65 of 860), Julia too many (726 of 860). On TREC, Julia
  answers 453 of 500 questions with its first two options.
- Run times on the M5 Pro, for the whole suite: Laya and Julia under a
  minute, clef-flash 22 minutes, pplx (8-bit) 44, Clef 27B (8-bit) 74.

```bash
python scripts/accuracy_benchmark.py local --datasets DIR --model MODEL --out results.json
python scripts/accuracy_benchmark.py report results.json [more.json ...]
```

How to save the datasets at the checked commits is in the script's
docstring. `local` writes the result file as it goes and resumes from it.
