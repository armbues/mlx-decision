# Benchmarks

The full measurements behind the numbers in the README. All were taken on an
Apple M5 Pro (20-core GPU) with 64 GB, macOS 26, torch 2.14.1 and
transformers 5.18.0 for the PyTorch side. Times are medians per request,
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

Laya and Julia in float16. They read each question as its own sequence with
the whole state, so the input tokens count the state once per question:

| Model | Input tokens x questions: median | Peak memory |
|---|---|---|
| laya | 308 x 1: 16 ms; 1,556 x 5: 63 ms; 6,246 x 20: 243 ms; 512 x 1: 22 ms; 2,560 x 5: 98 ms; 10,240 x 20: 387 ms | 1.8 GB |
| laya-typed-decisions | 283 x 1: 16 ms; 1,431 x 5: 60 ms; 5,746 x 20: 238 ms; 1,024 x 1: 43 ms; 5,120 x 5: 207 ms; 20,480 x 20: 842 ms | 1.7 GB |
| laya-multilingual | 283 x 1: 8 ms; 1,428 x 5: 26 ms; 5,738 x 20: 104 ms; 1,024 x 1: 18 ms; 5,120 x 5: 86 ms; 20,480 x 20: 338 ms | 1.6 GB |
| Julia-1 | 269 x 1: 6 ms; 1,364 x 5: 15 ms; 5,464 x 20: 59 ms; 1,020 x 1: 11 ms; 5,119 x 5: 50 ms; 20,484 x 20: 196 ms; 4,024 x 1: 43 ms; 20,139 x 5: 202 ms; 80,564 x 20: 800 ms; 7,379 x 1: 89 ms; 36,914 x 5: 421 ms; 147,664 x 20: 1,666 ms | 1.4 GB |

Throughput: clef-flash about 1,550-1,850 tokens per second, laya and
laya-typed-decisions about 24,000-26,000, laya-multilingual about 60,000,
Julia-1 about 100,000.

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

## Agreement with the reference code

The parity tests (`tests/parity/`) compare mlx-decision with each family's
own code on fixed request sets, token for token and probability by
probability.

- clef-flash: 63 requests (13 with one to three images, states up to beyond
  the 16,384-token limit) against Cloudflare's reference in bf16 on MPS.
  Largest difference 0.035 per probability (mean 0.001; with images 0.015,
  mean 0.002), no changed answers.
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

Five public benchmarks, 500 test examples each (a fixed sample, seed 1234),
with the same questions and option descriptions for every model
(`scripts/accuracy_benchmark.py`). Accuracy / macro-F1 / ECE (expected
calibration error of the top probability, lower is better), in percent.
Jev is TypeSafe AI's hosted model; its column needs a Jev account.

| Benchmark | Options | clef-flash | laya | laya-typed-decisions | laya-multilingual | Julia-1 | Jev |
|---|---|---|---|---|---|---|---|
| AG News (topic) | 4 | 91.4 / 91.5 / 2.5 | 94.6 / 94.7 / 6.8 | 94.6 / 94.6 / 18.2 | 93.8 / 93.8 / 3.0 | 83.0 / 83.4 / 6.3 | 86.4 / 86.4 / 9.5 |
| DAIR Emotion | 6 | 60.0 / 54.6 / 19.4 | 59.8 / 50.5 / 25.0 | 61.2 / 52.5 / 10.8 | 49.0 / 40.9 / 29.8 | 73.8 / 74.4 / 20.3 | 62.2 / 56.2 / 26.0 |
| ANLI r1-r3 (entailment) | 3 | 58.2 / 57.9 / 14.1 | 48.6 / 48.5 / 34.6 | 47.4 / 46.9 / 25.2 | 39.2 / 38.7 / 48.7 | 33.0 / 31.0 / 52.2 | 71.6 / 71.9 / 11.5 |
| BANKING77 (intent) | 77 | 96.0 / 95.8 / 3.5 | 36.0 / 31.5 / 51.9 | 36.2 / 31.7 / 15.3 | 35.0 / 32.7 / 44.7 | n/a | 80.2 / 79.4 / 8.2 |
| MMLU | 4 | 93.0 / 93.0 / 6.0 | 35.2 / 34.5 / 11.1 | 37.6 / 37.5 / 2.7 | 30.2 / 29.9 / 16.2 | 32.2 / 32.2 / 51.2 | 91.6 / 91.6 / 3.7 |

Notes:
- Cloudflare publishes macro-F1 for clef-flash / Jev from its own prompts:
  ANLI 59.1 / 74.8, BANKING77 90.9 / 79.7, MMLU (accuracy) 91.8 / 91.7.
- Julia answers 2 to 20 options, so BANKING77 does not apply; 4 MMLU
  examples have an option over its 48-token limit and count as wrong.
- Laya cuts BANKING77's 77 options to a few tokens each to fit its question
  budget, which costs most of the accuracy.

```bash
python scripts/accuracy_benchmark.py local --datasets DIR --model MODEL --out results.json
python scripts/accuracy_benchmark.py report results.json [more.json ...]
```
