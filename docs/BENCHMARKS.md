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

Median time per request, end to end, against each model's own PyTorch code
on the same Mac (MPS; Apple M5 Pro with 64 GB); short requests are in the
[model table](MODELS.md):

| Model | Request | PyTorch | mlx-decision | Speedup |
|---|---|---|---|---|
| clef-flash (bf16) | 6,099 tokens, 20 questions | 12.96 s | 3.80 s | 3.4x |
| clef-flash (bf16) | 16,384 tokens, 20 questions | 36.12 s | 11.39 s | 3.2x |
| clef-flash (bf16) | 1024x768 image, 3 questions | 2.73 s | 1.01 s | 2.7x |
| laya | 450-token state, 20 questions | 543 ms | 386 ms | 1.4x |
| laya-multilingual | 1,000-token state, 20 questions | 558 ms | 337 ms | 1.7x |
| d1-2B (bf16) | 5,000-token state, 1 question | 957 ms | 649 ms | 1.5x |
| d1-2B (bf16) | 15,000-token state, 20 questions | 7.68 s | 5.25 s | 1.5x |
| d1-3B (bf16) | 3,400-token state, 1 question | 809 ms | 676 ms | 1.2x |
| d1-3B (bf16) | 15,000-token state, 20 questions | 4.74 s | 4.01 s | 1.2x |
| Julia-1 | 7,350-token state, 20 questions | 12.4 s | 2.0 s | 6.1x |

clef-flash peaks at 19.0 GB against the reference's 23.9 GB.
H2O-Lightning is not in the table: its release runs on vLLM, which needs
CUDA. The full tables per model follow.

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

Not measured for v1.1: the release's code loads the 27B in float32 on a
Mac (about 104 GB). `scripts/pplx_parity_bundle.py` writes the same kind
of bundle as for Clef 27B, for a Mac that has the memory. v1.2 and flash
have not been measured yet.

### d3-flash

Compared over the whole parity set (276 answered questions, up to 40,000
tokens; each question alone, as both read them), not per request:
the release's runtime (`d3_runtime.D3`, bf16 on MPS, its readout in
float32; transformers 5.19.0) took 26.5 minutes, mlx-decision in bf16 about 9 minutes with
loading, 2.9x. Its optional GPU kernels for the linear-attention layers
need CUDA, so on a Mac transformers runs its plain PyTorch code for
them, as for Clef. On one request with three questions of about 16,000
tokens each: 113 s against 31 s.

### Base Labs

Compared over the whole parity set (68 text requests, 166 questions, up
to 32,768 tokens per question): the release's engine (OneJev's `qev` at
the pinned commit with Base Labs' `serve.patch`, elephant-1 in bf16 on
MPS, the LM head in float32; transformers 5.19.0) took 6.6 minutes and
46.5 GB at its peak, mlx-decision in bf16 112 s and 23.4 GB, 3.6x. As
for Clef and d3, the release's linear-attention kernels need CUDA, so
on a Mac transformers runs its plain PyTorch code for them. joey-1 the
same way: the release's engine about 2 minutes, mlx-decision in bf16 27
s. mammoth-1 was not run against the release.

### H2O-Lightning

Not compared: the release runs on vLLM, which needs CUDA. For the parity
tests below its shim ran with transformers in vLLM's place, in bf16 on
MPS, one pass per question as the shim sends them; for v1.2.3 that took
20.5 minutes for the 271 questions of the parity set, mlx-decision in bf16
94 s. That run is not how the release is meant to be served, so it is no
speedup.

### d1-2B

The release's own code (`system_one`, loaded with `trust_remote_code`, as
its model card shows, not compiled) in bf16 on MPS, against mlx-decision in
bf16. The card's cases (the picture: 384 x 384 pixels), then the
`benchmark` grid; median of 3 runs. Both read the questions in sets of 16
with the state; the release peaks at 12.8 GB, mlx-decision at 11.8 GB
(process footprints):

| Request | PyTorch bf16 | mlx-decision bf16 | Speedup | Max probability difference |
|---|---|---|---|---|
| card: one question | 28 ms | 14 ms | 1.9x | 0.000 |
| card: three questions | 35 ms | 24 ms | 1.4x | 0.004 |
| card: 5,000-token state | 957 ms | 649 ms | 1.5x | 0.001 |
| card: 384 px picture | 63 ms | 44 ms | 1.4x | 0.000 |
| 250 x 1 | 59 ms | 42 ms | 1.4x | 0.002 |
| 250 x 5 | 85 ms | 58 ms | 1.5x | 0.003 |
| 250 x 20 | 305 ms | 202 ms | 1.5x | 0.008 |
| 1,000 x 1 | 187 ms | 128 ms | 1.5x | 0.001 |
| 1,000 x 5 | 212 ms | 143 ms | 1.5x | 0.005 |
| 1,000 x 20 | 586 ms | 381 ms | 1.5x | 0.008 |
| 4,000 x 1 | 745 ms | 519 ms | 1.4x | 0.001 |
| 4,000 x 5 | 782 ms | 538 ms | 1.5x | 0.003 |
| 4,000 x 20 | 1,815 ms | 1,189 ms | 1.5x | 0.009 |
| 15,000 x 1 | 3,539 ms | 2,428 ms | 1.5x | 0.003 |
| 15,000 x 5 | 3,617 ms | 2,500 ms | 1.4x | 0.002 |
| 15,000 x 20 | 7,677 ms | 5,248 ms | 1.5x | 0.010 |

The model card's own M5 Pro times (18.6 ms for one question, 45.2 ms for
the picture) are for compiled warm calls.

### d1-3B

The release's own code (`system_one`, loaded with `trust_remote_code`, as
its model card shows) in bf16 on MPS, against mlx-decision in bf16 (kept
states cleared before every request). The card's text cases, then the
`benchmark` grid; median of 3 runs. The release already reads the state
once for all questions (its "trunk and branches"), so the two are about
level:

| Request | PyTorch bf16 | mlx-decision bf16 | Speedup | Max probability difference |
|---|---|---|---|---|
| card: one question | 28 ms | 29 ms | 0.9x | 0.001 |
| card: three questions | 40 ms | 45 ms | 0.9x | 0.005 |
| card: 3,400-token state | 809 ms | 676 ms | 1.2x | 0.001 |
| 250 x 1 | 71 ms | 63 ms | 1.1x | 0.002 |
| 250 x 5 | 108 ms | 126 ms | 0.9x | 0.012 |
| 250 x 20 | 277 ms | 252 ms | 1.1x | 0.012 |
| 1,000 x 1 | 232 ms | 227 ms | 1.0x | 0.001 |
| 1,000 x 5 | 281 ms | 253 ms | 1.1x | 0.024 |
| 1,000 x 20 | 476 ms | 401 ms | 1.2x | 0.024 |
| 4,000 x 1 | 999 ms | 820 ms | 1.2x | 0.001 |
| 4,000 x 5 | 1,058 ms | 879 ms | 1.2x | 0.006 |
| 4,000 x 20 | 1,312 ms | 1,068 ms | 1.2x | 0.018 |
| 15,000 x 1 | 3,841 ms | 3,206 ms | 1.2x | 0.003 |
| 15,000 x 5 | 4,078 ms | 3,390 ms | 1.2x | 0.005 |
| 15,000 x 20 | 4,740 ms | 4,011 ms | 1.2x | 0.022 |

```bash
python scripts/d1_reference_speed.py torch --model PATH --out torch.json
python scripts/d1_reference_speed.py mlx --model PATH --out mlx.json
python scripts/d1_reference_speed.py report torch.json mlx.json
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

Clef 27B as an 8-bit and a 4-bit copy (bf16 does not fit on the M5 Pro):

| Input tokens | Questions | 8-bit | 4-bit |
|---|---|---|---|
| 394 | 1 | 1.10 s | 1.12 s |
| 1,552 | 5 | 4.17 s | 3.88 s |
| 4,149 | 1 | 10.6 s | 9.96 s |
| 6,103 | 20 | 15.9 s | 14.7 s |
| 16,384 | 20 | 43.8 s | 40.7 s |
| Peak memory | | 31.8 GB | 19.3 GB |

pplx v1.1 as an 8-bit and a 4-bit copy, median of 3 runs per row (`--questions
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

d3-flash in bf16 and as an 8-bit copy, and d3-mini in bf16, the same
way (`--question-counts 1,5 --repeats 3`; the longest length, 29,450, is 90% of
the default 32,768-token limit):

| State tokens | Questions | Input tokens | d3-flash bf16 | d3-flash 8-bit | d3-mini bf16 |
|---|---|---|---|---|---|
| 250 | 1 | 342 | 0.225 s | 0.234 s | 0.121 s |
| 250 | 5 | 1,751 | 1.14 s | 1.37 s | 0.664 s |
| 1,000 | 1 | 1,095 | 0.606 s | 0.793 s | 0.488 s |
| 1,000 | 5 | 5,516 | 3.03 s | 3.95 s | 2.07 s |
| 4,000 | 1 | 4,112 | 2.33 s | 2.99 s | 1.62 s |
| 4,000 | 5 | 20,601 | 12.6 s | 15.1 s | 8.44 s |
| 15,000 | 1 | 15,170 | 10.3 s | 12.3 s | 7.84 s |
| 15,000 | 5 | 75,891 | 52.8 s | 62.0 s | 37.6 s |
| 29,450 | 1 | 29,697 | 25.7 s | 28.1 s | 18.6 s |
| 29,450 | 5 | 148,526 | 130 s | 140 s | 89.9 s |
| Peak memory | | | 18.7 GB | 11.7 GB | 11.1 GB |

Longer input, d3-flash in bf16, one question (`--lengths
8192,32768,131072`; peak memory measured per length): 8,326 tokens 4.8 s
and 16.2 GB, 33,032 tokens 27.6 s and 19.1 GB, 131,855 tokens 317 s and
31.9 GB. Above the weights (14.8 GB) memory grows by about 136 KB a
token.

elephant-1 in bf16 and as an 8-bit copy (`--question-counts 1,5`; the
longest length, 15,000, stays under the default 16,384-token limit). The
state is computed once per request and its questions run together off
it, so input tokens count the state once; warm: the same request again
with the state kept:

| State tokens | Questions | Input tokens | bf16 | 8-bit | bf16 warm |
|---|---|---|---|---|---|
| 250 | 1 | 354 | 0.293 s | 0.314 s | 0.068 s |
| 250 | 5 | 619 | 0.488 s | 0.535 s | 0.269 s |
| 1,000 | 1 | 1,103 | 0.686 s | 0.841 s | 0.070 s |
| 1,000 | 5 | 1,368 | 0.872 s | 1.05 s | 0.272 s |
| 4,000 | 1 | 4,102 | 2.37 s | 2.93 s | 0.080 s |
| 4,000 | 5 | 4,367 | 2.71 s | 3.11 s | 0.305 s |
| 15,000 | 1 | 15,098 | 10.6 s | 11.7 s | 0.123 s |
| 15,000 | 5 | 15,363 | 11.4 s | 12.5 s | 0.567 s |
| Peak memory | | | 17.6 GB | 10.7 GB | |

The 8-bit copy is 7-24% slower; its warm times are within 0.05 s of
bf16's.

joey-1 the same way:

| State tokens | Questions | Input tokens | bf16 | 8-bit | bf16 warm |
|---|---|---|---|---|---|
| 250 | 1 | 354 | 0.072 s | 0.085 s | 0.024 s |
| 250 | 5 | 619 | 0.112 s | 0.125 s | 0.064 s |
| 1,000 | 1 | 1,103 | 0.158 s | 0.177 s | 0.026 s |
| 1,000 | 5 | 1,368 | 0.202 s | 0.224 s | 0.069 s |
| 4,000 | 1 | 4,102 | 0.541 s | 0.621 s | 0.033 s |
| 4,000 | 5 | 4,367 | 0.604 s | 0.694 s | 0.095 s |
| 15,000 | 1 | 15,098 | 2.42 s | 2.80 s | 0.065 s |
| 15,000 | 5 | 15,363 | 2.63 s | 2.99 s | 0.246 s |
| Peak memory | | | 5.0 GB | 3.4 GB | |

joey-1 is about four times as fast as elephant-1; its 8-bit copy is
11-18% slower.

H2O-Lightning v1.4.2 in bf16, read two-way, its default (`--reading
two-way --question-counts 1,5`). Each question is its own sequence and
is read whole, the full-attention layers looking both ways over it, so
nothing is shared between questions or kept between requests; input
tokens count the state once per question. A state is read whole while
the prompt fits the 40,960-token context:

| State tokens | Questions | Input tokens | bf16 two-way |
|---|---|---|---|
| 250 | 1 | 328 | 0.122 s |
| 250 | 5 | 1,684 | 0.690 s |
| 1,000 | 1 | 1,079 | 0.402 s |
| 1,000 | 5 | 5,439 | 1.93 s |
| 4,000 | 1 | 4,083 | 1.57 s |
| 4,000 | 5 | 20,459 | 7.72 s |
| 15,000 | 1 | 15,099 | 7.01 s |
| 15,000 | 5 | 75,539 | 36.7 s |
| 28,800 | 1 | 28,920 | 17.7 s |
| 28,800 | 5 | 144,644 | 90.9 s |
| Peak memory | | | 11.2 GB |

Each question costs about one pass over the whole state. For many
questions on one long state, read it one-way (`--reading one-way`): the
questions then share the system prompt and the state, computed once per
request. The table below was measured one-way on v1.2.3, in bf16, as an
8-bit and as a uniform 4-bit copy. Measured back to back on v1.4.2 in
bf16, one question at 4,000 / 15,000 / 28,800 tokens took 1.37 / 6.24 /
16.4 s two-way and 1.50 / 5.84 / 13.6 s one-way: a single two-way
question costs about the same at 4,000 tokens, about 7% more at 15,000
and about 20% more at 28,800 (runs on one Mac vary by up to 10%). Warm:
the same request again with the state kept:

| State tokens | Questions | Input tokens | bf16 | 8-bit | 4-bit | bf16 warm |
|---|---|---|---|---|---|---|
| 250 | 1 | 328 | 0.175 s | 0.185 s | 0.173 s | 0.057 s |
| 250 | 5 | 1,684 | 0.406 s | 0.437 s | 0.412 s | 0.289 s |
| 1,000 | 1 | 1,079 | 0.381 s | 0.430 s | 0.403 s | 0.060 s |
| 1,000 | 5 | 5,439 | 0.617 s | 0.687 s | 0.646 s | 0.296 s |
| 4,000 | 1 | 4,083 | 1.31 s | 1.54 s | 1.50 s | 0.069 s |
| 4,000 | 5 | 20,459 | 1.58 s | 1.82 s | 1.78 s | 0.329 s |
| 15,000 | 1 | 15,099 | 5.61 s | 6.20 s | 6.04 s | 0.127 s |
| 15,000 | 5 | 75,539 | 6.01 s | 6.57 s | 6.38 s | 0.488 s |
| 28,800 | 1 | 28,920 | 12.2 s | 13.3 s | 13.0 s | 0.197 s |
| 28,800 | 5 | 144,644 | 12.7 s | 14.0 s | 13.5 s | 0.687 s |
| Peak memory | | | 12.6 GB | 8.9 GB | 7.0 GB | |

The quantized copies are 5-17% slower from 1,000 tokens on; their warm
times are within 0.03 s of bf16's.

d1-2B in bf16 (the default grid; the longest length, 29,450, is 90% of its
default 32,768-token limit per set). The questions go in sets of 16, each
set one pass with the whole state, so 20 questions read the state twice;
the input tokens count both sets. Nothing is kept between requests (the
state's hidden states depend on its set's questions), so there is no warm
column:

| State tokens | Questions | Input tokens | Median |
|---|---|---|---|
| 250 | 1 | 278 | 43 ms |
| 250 | 5 | 432 | 58 ms |
| 250 | 20 | 1,252 | 202 ms |
| 1,000 | 1 | 1,027 | 127 ms |
| 1,000 | 5 | 1,181 | 143 ms |
| 1,000 | 20 | 2,750 | 378 ms |
| 4,000 | 1 | 4,026 | 513 ms |
| 4,000 | 5 | 4,180 | 531 ms |
| 4,000 | 20 | 8,748 | 1.15 s |
| 15,000 | 1 | 15,022 | 2.33 s |
| 15,000 | 5 | 15,176 | 2.39 s |
| 15,000 | 20 | 30,740 | 5.14 s |
| 29,450 | 1 | 29,465 | 6.19 s |
| 29,450 | 5 | 29,619 | 6.25 s |
| 29,450 | 20 | 59,626 | 12.7 s |
| Peak memory | | | 5.3 GB |

With `--max-input-tokens 128000`, one question on a 127,774-token state
takes 123 s at a peak of 11.8 GB.

d1-3B in bf16 (the default grid; the longest length, 29,450, is 90% of its
32,768-token limit). A request's questions run in one pass; a state shared
by enough questions is computed once, and kept (warm: the same request
again with the state kept; - where a 276-token state is not worth keeping
for one question):

| State tokens | Questions | Input tokens | Median | Warm |
|---|---|---|---|---|
| 250 | 1 | 276 | 62 ms | - |
| 250 | 5 | 1,442 | 125 ms | 56 ms |
| 250 | 20 | 5,802 | 245 ms | 178 ms |
| 1,000 | 1 | 1,025 | 220 ms | 35 ms |
| 1,000 | 5 | 5,187 | 248 ms | 58 ms |
| 1,000 | 20 | 20,782 | 386 ms | 202 ms |
| 4,000 | 1 | 4,024 | 769 ms | 42 ms |
| 4,000 | 5 | 20,182 | 823 ms | 96 ms |
| 4,000 | 20 | 80,762 | 1.01 s | 283 ms |
| 15,000 | 1 | 15,020 | 3.13 s | 70 ms |
| 15,000 | 5 | 75,162 | 3.36 s | 223 ms |
| 15,000 | 20 | 300,682 | 3.95 s | 844 ms |
| 29,450 | 1 | 29,463 | 6.74 s | 118 ms |
| 29,450 | 5 | 147,377 | 6.99 s | 273 ms |
| 29,450 | 20 | 589,542 | 7.76 s | 1.00 s |
| Peak memory | | | 11.6 GB | |

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
d3-flash about 1,150-1,800 (bf16) and 1,050-1,450 (8-bit), d3-mini
about 1,600-2,800,
elephant-1 about 1,200-1,750 and joey-1 about 4,900-7,600 (bf16; the
questions share the state), H2O-Lightning about 1,600-2,800 two-way and
2,400-3,100 one-way for one question (bf16; one-way, more questions share
the state), d1-2B about 4,700-8,200 (a set's questions share the
state), d1-3B about 4,400-5,200 (likewise), laya and
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

elephant-1 keeps states the same way (2 GB by default): with a
15,000-token state, 10.6 s the first time and 0.12 s for one question on
the kept state, 0.57 s for five (bf16, table above). joey-1: 2.42 s the
first time, 0.065 s for one question on the kept state, 0.25 s for five.

H2O-Lightning read one-way keeps states the same way (2 GB by default):
with a 28,800-token state, 12.2 s the first time and 0.20 s for one
question on the kept state, 0.69 s for five (v1.2.3 in bf16, table
above). Read two-way, its default, it keeps nothing.

d1-3B keeps long states the same way: with a 29,450-token state, 6.74 s
the first time, 0.12 s for one question on the kept state and 1.00 s for
twenty. A state short enough that repeating it in every question's row
costs less than a second pass is not kept.

A warm request takes about as long as the same questions on a short state
(20 questions on a 250-token state: 1.27 s in bf16); the questions still
attend over the whole state, which adds a little. Keeping the 16,384-token
state adds about 1.3 GB to the peak memory. pplx, d3, H2O-Lightning read
two-way, Laya and Julia read the state together with each question, so
they have nothing to reuse.

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

pplx v1.1 copies have not been compared yet: the unquantized model does not fit
here, and there is no bf16 run of it from a larger Mac.

d3-flash's copies have not been run over the whole parity set. On the
five questions of two requests (one with an image) the 8-bit copy is
within 0.003 of the release in bf16, the 4-bit copy within 0.037. The
smallest model, d3-edge, over its whole parity set against its own bf16:
uniform 4-bit moves probabilities by up to 0.567 (mean 0.046) and
changes 35 of 276 top answers, 30 of them clear ones; keep the smallest
models at full size or 8-bit.

elephant-1 copies on its parity set (the 67 answered requests below, 166
questions, up to 32,768 tokens), against the release's engine in bf16
on MPS; brackets as above. mlx-decision in bf16 is the first row:

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| bf16 (as released) | 17.5 GB | 20.0 GB | 0.001 / 0.039 | 0 (0) |
| 8-bit (`-q`) | 8.7 GB | 13.3 GB | 0.002 / 0.025 | 0 (0) |
| uniform 4-bit (`-q --bits 4`) | 5.0 GB | 9.6 GB | 0.016 / 0.336 | 9 (8) |

joey-1 copies on the same set, against the release's engine in bf16 on
MPS:

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| bf16 (as released) | 5.1 GB | 6.3 GB | 0.001 / 0.010 | 0 (0) |
| 8-bit (`-q`) | 2.5 GB | 4.8 GB | 0.001 / 0.019 | 0 (0) |
| uniform 4-bit (`-q --bits 4`) | 1.6 GB | 3.9 GB | 0.010 / 0.176 | 5 (3) |

On the image requests (22 requests, 53 questions), against the releases'
own scoring in bf16 (below), the 8-bit copies stay within 0.029
(elephant-1) and 0.023 (joey-1), with one changed answer, a near tie;
the 4-bit copies move answers by up to 0.30 and 0.10, two changed each
(elephant-1's both clear ones). The vision tower is kept as released in
every copy.

mammoth-1 was stored as an 8-bit copy by `download --bits 8` (26.2 GB,
33.3 GB at its peak on the same set): its top answers agree with
elephant-1 in bf16 on 153 of the 166 questions; the other 13 are
judgement calls (sentiment, adjacent levels, urgency, the language of
a mixed thread). Making the copy took about 57 GB of memory, since the
release's first weight file (49.8 GB) is read at once.

H2O-Lightning v1.4.2 copies on its parity set (the 102 answered requests
below, 276 questions), read two-way, against the release's code in bf16
on MPS; brackets as above. mlx-decision in bf16 is the first row:

| Model | Disk | Peak memory | Mean / max probability difference | Changed answers |
|---|---|---|---|---|
| bf16 (as released) | 8.7 GB | 13.2 GB | 0.002 / 0.031 | 4 (0) |
| 8-bit (`-q`) | 4.8 GB | 9.5 GB | 0.002 / 0.047 | 1 (0) |
| uniform 4-bit (`-q --bits 4`) | 2.8 GB | 7.5 GB | 0.024 / 0.491 | 29 (23) |

Uniform 4-bit moves this model's answers far more than clef-flash's
(largest difference 0.49 against 0.21, 23 clear answers changed), and
more than it moved v1.2.3's (12); keep it at bf16 or 8-bit. Mixed
precision was not measured on v1.4.2 (on v1.2.3 it halved uniform
4-bit's difference).

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
- pplx v1.2 and flash: 113 requests (the 97 above, each sent whole, plus
  16 of its own: several questions, special tokens in every field, 255
  options beside other questions, requests split under a lowered limit)
  against the release's own code at 32,768 tokens or the request's
  limit. With the release's tokenizer the parts, token ids and marker
  positions are the same for every request, and the same 2 are refused
  (a question over the limit alone). Probabilities were compared on small
  random models in both releases' layouts (8 layers, real vocabulary),
  345 questions each, on the CPU: within 0.00001. Flash on its weights:
  the 102 requests whose parts have at most 4,096 tokens (299 questions;
  longer ones need more memory than the 16 GB M2 it ran on), the release's
  code in bf16 on MPS against mlx-decision on the same Mac. bf16: largest
  difference 0.058 per probability (on a 1.6 MP image), mean 0.0023, the
  top answer the same for 296 questions (the other three near ties,
  margins 0.03, 0 and 0); the release reports up to 0.059 between two
  runs of its own reference code. 8-bit copy: largest 0.142, mean
  0.0034, the same top answer for 296 (margins 0.02, 0 and 0).
- d3-flash: 108 requests (the 97 above plus 11 of its own: null, empty
  and whitespace states, empty instructions and option descriptions,
  score levels of mixed types, duplicate and empty levels, 11 levels, an
  image with no state) against the release's runtime. The prompt, token
  ids and image grids are the same for every question; the same
  questions are refused (duplicate or empty score levels, more than 10
  levels). The release reads every question whole; mlx-decision's
  default limit is 32,768 tokens, raised for the test. In bf16 on MPS,
  every question alone: 276 questions, largest difference 0.038 (a near
  tie, where the release itself moves 0.010 between MPS and CPU), mean
  0.0016, no changed answer where the release's top two are more than
  0.04 apart. On a small random model in the release's layout with its
  files, on the CPU: within 0.00001. The other sizes against their own
  releases on 26 questions of 12 requests: d3-mini within 0.011,
  d3-lite 0.027, d3-edge 0.030, d3-nano 0.058 (one 255-option question
  where its release moves 0.026 between MPS and CPU; elsewhere within
  0.025). d3 27B was not run.
- Base Labs: 71 requests (the 50 text requests of the 63 above plus 21
  of its own: every question type twice in one request, no
  instructions, empty and partial yes/no criteria, 1, 26, 27 and 256
  options, 11 levels, states ending in a space, newlines, `<` or
  `</state`, chat markup in the state) against the release's engine.
  The prompt, token ids and the shared prefix's length are the same for
  every question; the release refuses the same three requests (over
  32,768 tokens, 256 options, 11 levels), and mlx-decision's wire format
  also a one-level score, which the release takes. In bf16 on MPS, as
  the release batches them: 166 questions, largest difference 0.039 (a
  near tie), mean 0.0010, no changed answer; joey-1 within 0.010, mean
  0.0007, no changed answer. On small random models in the three
  releases' shapes with their files, on the CPU: within 0.000001.
  Images: the releases serve text only, so 22 requests with images (the
  13 of the set above and 9 of its own: images as JSON values, between
  texts and two in a list, an image-only state, top-level images beside
  several kinds of state, a very small image) were compared with the
  releases' own training-side scoring, which answers text as their
  engine does. The prompts and token ids are the same for every
  question; in bf16 on MPS, 53 questions: elephant-1 within 0.017,
  joey-1 within 0.021, mean 0.0018 each, no changed answer. mammoth-1
  was not run against the release.
- H2O-Lightning v1.4.2: 104 requests (the 97 above plus 7 of its own:
  empty instructions, empty option descriptions, images inside the
  state, five images, a yes/no question with `yes` / `no` option ids,
  and a text and a JSON state over the 40,960-token context) against
  the release's shim, with transformers in vLLM's place, in both
  readings. The token ids and image grids are the same for every
  question; over the context, the parts of the state the release keeps
  for each question are the same, token for token, and so are the
  prompts with the context lowered to 256, 1,024 and 8,192 tokens. Two
  requests are refused: five images (the release refuses them too) and
  the `yes` / `no` ids (the release takes them, the Jev request format
  does not). On the other 102 (276 questions), in bf16 on MPS with
  float32 logits, as the release computes them: two-way, largest
  difference 0.031, mean 0.0016, 4 changed answers, all near ties (top
  two within 0.04); one-way 0.030, mean 0.0015. The release's three
  two-way check requests are answered within its own tolerance (0.05).
  On a small random model in the release's layout, on the CPU, both
  readings: within 0.00001. v1.2.3 keeps its own small-model check.
- d1-2B: 118 requests (the 63 of the set above, 13 with pictures; the 34
  text requests below; 21 of its own: 16, 17 and 33 questions, the same
  questions in two orders, 26 / 27 / 54 / 255 options, single-letter and
  digit keys, chat markup in states and questions, a null state, two
  pictures without state text, empty instructions and others) against the
  release's own code. Prompt texts, the packed sets (token ids, where each
  question starts and ends, where its answer is read) and the tokens each
  option is scored by are the same for all 110 requests the release
  takes; pictures are cut into the same patches (within one 8-bit level).
  The release refuses 8: 7 for missing or empty instructions or score
  levels that are not text, which mlx-decision fills in as for d1-3B
  (instructions from the question id, levels as text), and one with a
  single score level, which mlx-decision refuses too. On a small random
  model in the release's layout with its tokenizer, on the CPU: within
  0.00001 for 107 requests (343 questions, text and pictures). In bf16,
  every request as served (sets of 16) against the release in bf16 on
  MPS: 343 questions, largest difference 0.035, mean 0.0030, no changed
  answer where the release's top two are more than 0.04 apart.
- d1-3B: 110 text requests (the 50 text requests of the set above, the
  34 below, and 26 of its own: letter keys, 26 / 27 / 100 / 101 options, missing
  descriptions, criteria on one side, a null state, missing instructions
  and others) against the release's own code. The prompt text and token
  ids are the same for every question (257), and so are the tokens each
  option is scored by. The release refuses 5 requests with questions
  without instructions (mlx-decision fills in the question id) and
  answers score questions with 11 and 12 levels, which mlx-decision
  refuses (the model card's range is 2 to 10). On a small random model
  in the release's layout with its tokenizer, on the CPU: within 0.00001
  for 102 requests. In bf16, every question alone against the release in
  bf16 on MPS: 253 questions, largest difference 0.030, mean 0.0054, no
  changed answer where the release's top two are more than 0.04 apart.
  The release's own two paths (all questions together, or each alone)
  differ by up to 0.061 in bf16. The 40,000-token request is cut by
  mlx-decision and left out (the release reads it whole, past its
  32,768-token context).
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
A sixth, [typed-decisions](#typed-decisions), asks several questions per
request and scores the answers against gold distributions.

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
level is its prediction. clef-flash, pplx-decider-flash-v1.2, d3-flash,
elephant-1, joey-1, H2O-Lightning, d1-2B and d1-3B ran in bf16, Clef 27B
and pplx v1.1 as 8-bit copies; H2O-Lightning is its release v1.4.2, read
two-way, its default; Jev is TypeSafe AI's hosted model
(`typesafe/jev-1.14.0`, run on 2026-10-20), and its column needs a Jev
account.

| Benchmark | clef-flash | clef (8-bit) | pplx v1.1 (8-bit) | pplx-decider-flash-v1.2 | d3-flash | elephant-1 | joey-1 | H2O-Lightning-4B v1.4.2 | d1-2B | d1-3B | laya | laya-typed-decisions | laya-multilingual | Julia-1 | Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TREC | 96.8 / 95.5 / 4.5 | 96.4 / 94.5 / 2.8 | 94.0 / 93.7 / 3.3 | 93.6 / 92.3 / 5.1 | 85.8 / 87.3 / 3.5 | 89.6 / 89.3 / 3.3 | 80.0 / 73.4 / 16.3 | 89.0 / 88.5 / 4.9 | 88.2 / 87.9 / 5.9 | 86.6 / 86.8 / 5.5 | 79.0 / 76.9 / 2.9 | 78.8 / 77.2 / 14.3 | 89.8 / 87.2 / 4.0 | 16.0 / 14.0 / 63.1 | 92.8 / 92.6 / 3.6 |
| TweetEval Offensive | 84.0 / 78.1 / 9.5 | 84.8 / 81.0 / 8.0 | 84.9 / 79.6 / 4.0 | 83.7 / 77.9 / 3.5 | 83.8 / 77.7 / 3.6 | 82.4 / 75.9 / 5.0 | 85.6 / 80.5 / 2.7 | 85.1 / 80.0 / 5.1 | 84.9 / 80.7 / 3.5 | 83.1 / 78.7 / 5.3 | 77.8 / 61.9 / 7.7 | 80.1 / 68.4 / 7.3 | 79.2 / 71.9 / 4.1 | 38.8 / 37.9 / 43.0 | 80.0 / 77.3 / 4.0 |
| ANLI round 3 | 50.8 / 50.5 / 20.7 | 55.2 / 54.7 / 16.8 | 66.8 / 67.0 / 9.4 | 61.8 / 61.8 / 15.4 | 64.3 / 64.5 / 11.4 | 74.5 / 74.4 / 5.9 | 56.5 / 56.2 / 17.7 | 60.4 / 60.5 / 16.7 | 52.2 / 52.4 / 12.5 | 55.7 / 55.8 / 16.6 | 39.5 / 39.4 / 42.0 | 38.5 / 37.8 / 32.2 | 35.5 / 35.3 / 50.8 | 33.2 / 31.3 / 51.3 | 70.2 / 70.5 / 11.2 |
| OpenBookQA | 95.4 / 95.4 / 3.8 | 95.4 / 95.4 / 2.9 | 96.8 / 96.7 / 1.4 | 93.8 / 93.7 / 2.5 | 95.2 / 95.2 / 2.4 | 96.8 / 96.8 / 1.7 | 83.0 / 82.9 / 3.3 | 93.0 / 92.9 / 3.5 | 86.2 / 86.3 / 6.6 | 86.2 / 85.9 / 4.5 | 39.2 / 38.3 / 11.5 | 41.4 / 40.9 / 3.2 | 29.2 / 28.8 / 19.8 | 33.2 / 32.8 / 48.2 | 96.0 / 95.9 / 2.3 |
| SST-5 | 57.6 / 55.1 / 10.9 / 0.51 | 57.6 / 53.3 / 3.5 / 0.51 | 58.0 / 52.9 / 5.2 / 0.47 | 60.6 / 59.7 / 13.5 / 0.50 | 57.2 / 52.4 / 2.5 / 0.50 | 53.9 / 51.2 / 10.0 / 0.52 | 43.7 / 43.9 / 5.8 / 0.67 | 61.4 / 59.6 / 3.0 / 0.48 | 56.3 / 52.8 / 2.3 / 0.54 | 56.8 / 54.3 / 1.6 / 0.54 | 35.1 / 27.6 / 31.2 / 0.99 | 44.3 / 38.4 / 7.0 / 0.68 | 27.7 / 16.8 / 58.4 / 1.40 | 36.7 / 32.2 / 41.7 / 0.82 | 57.2 / 55.1 / 16.2 / 0.50 |

Notes:
- No model refused an example or had its input cut: every state fits
  every model's limit, and every option list fits Julia's 2 to 20 options.
- Clef 27B gains on clef-flash mainly on entailment and calibration; pplx
  v1.1 is close to Jev on ANLI round 3, with d3-flash, at 9B, 2.5 points
  behind it there (and 1.6 on OpenBookQA). H2O-Lightning v1.4.2, at 4B,
  is the best of all on SST-5 (61.4, macro-F1 59.6), ahead of clef-flash
  on ANLI round 3 and level with pplx-decider-flash-v1.2, the other 4B,
  on OpenBookQA (93.0 against 93.8); its
  previous release, v1.2.3, scored lower on every benchmark (TREC 85.2,
  Offensive 82.1, ANLI round 3 59.9, OpenBookQA 84.6, SST-5 57.3).
  d1-3B, at 3B, trails it everywhere, with the lowest calibration error
  on SST-5 (1.6).
- d1-2B, at 1.9B, is level with d1-3B: ahead on TREC (88.2 against 86.6)
  and Offensive (84.9, with pplx v1.1 behind only joey-1 and
  H2O-Lightning), level on OpenBookQA and SST-5, behind on ANLI
  round 3 (52.2 against 55.7). Its answers depend on which questions
  share a set of 16 and their order (each question here was asked
  alone).
- pplx-decider-flash-v1.2, at 4B, is close to the 27B pplx v1.1 on TREC
  (93.6 against 94.0) and second on SST-5 (60.6, macro-F1 59.7), though
  less well calibrated there (13.5); it trails the 27B by 5 points
  on ANLI round 3 and 3 on OpenBookQA. Each example was asked as a request
  with one question, so its joint reading of several questions plays no
  part here.
- elephant-1, at 9B, is the only local model ahead of Jev on ANLI round 3
  (74.5 against 70.2, with the best calibration there, 5.9), level with
  pplx v1.1 on OpenBookQA and between d3-flash and the Clef models on
  TREC; it trails all the large models on SST-5 (53.9). Its model card
  lists public decision data among the training data without naming
  the datasets, so whether ANLI's training split was among them is not
  known.
- joey-1, Base Labs' 2B model, has the best Offensive accuracy of all
  (85.6, macro-F1 80.5, calibration error 2.7). Against d1-2B, the other
  model of its size, it is ahead on Offensive and ANLI round 3 (56.5
  against 52.2) and behind on TREC (80.0 against 88.2, with a
  calibration error of 16.3), OpenBookQA (83.0 against 86.2) and SST-5
  (43.7 against 56.3).
- About 28% of the Offensive tweets are offensive. Laya marks too few of
  them (laya: 65 of 860), Julia too many (726 of 860). On TREC, Julia
  answers 453 of 500 questions with its first two options.
- Run times on the M5 Pro, for the whole suite: Laya and Julia under a
  minute, d1-2B 2 minutes, d1-3B 3, joey-1 5, pplx-decider-flash-v1.2 6,
  H2O-Lightning 6, d3-flash 13, elephant-1 19,
  clef-flash 22,
  pplx v1.1 (8-bit) 44,
  Clef 27B (8-bit) 74.

d3's smaller sizes, in bf16 as released, beside d3-flash (same metrics;
whole suite in 6 minutes for d3-mini, 3 for d3-nano, 1.3 for d3-lite,
under a minute for d3-edge):

| Benchmark | d3-flash (9B) | d3-mini (4B) | d3-nano (2B) | d3-lite (0.8B) | d3-edge (0.6B) |
|---|---|---|---|---|---|
| TREC | 85.8 / 87.3 / 3.5 | 86.2 / 86.3 / 3.2 | 80.4 / 82.4 / 6.4 | 67.4 / 72.0 / 17.8 | 73.8 / 72.3 / 8.0 |
| TweetEval Offensive | 83.8 / 77.7 / 3.6 | 84.4 / 78.5 / 2.9 | 83.1 / 76.3 / 5.3 | 82.7 / 77.3 / 1.9 | 77.2 / 73.8 / 3.2 |
| ANLI round 3 | 64.3 / 64.5 / 11.4 | 59.2 / 59.5 / 16.3 | 48.1 / 48.1 / 28.8 | 41.1 / 41.0 / 31.9 | 37.9 / 36.4 / 35.2 |
| OpenBookQA | 95.2 / 95.2 / 2.4 | 93.8 / 93.7 / 2.5 | 88.2 / 88.0 / 3.1 | 73.8 / 73.5 / 8.0 | 57.8 / 57.6 / 17.3 |
| SST-5 | 57.2 / 52.4 / 2.5 / 0.50 | 58.5 / 56.4 / 4.5 / 0.50 | 52.1 / 47.5 / 3.9 / 0.56 | 47.6 / 40.7 / 4.5 / 0.60 | 42.5 / 34.7 / 8.3 / 0.69 |

d3-mini, at under half of d3-flash's size, is level with it: ahead on
answer type, offensive language and sentiment, behind on entailment (5
points) and science questions. Below it, entailment and science questions
fall away first; offensive language holds up down to d3-lite.

The three large models are level with Jev or ahead of it on four of the
five benchmarks; on entailment pplx comes close to Jev, Clef does not.
d3-flash, at 9B, is close behind pplx on entailment and science
questions and level on sentiment, but trails the Clef models and pplx on
answer type.
H2O-Lightning, at 4B about a seventh of the size of the 27B models, trails
them on answer type and science questions, is ahead of clef-flash on
entailment and the best of all on sentiment. elephant-1, at 9B, is ahead
of every local model on entailment and level with the best on science
questions; joey-1, its 2B sibling, leads on offensive language. d1-3B,
smaller than H2O-Lightning, trails it on every benchmark. d1-2B, under
two thirds of d1-3B's size, is level with it: ahead on answer type and
offensive language, behind on entailment.
The small models are near chance on entailment and science questions,
as their model cards lead one to expect; Julia, made for emotion and
routing, answers TREC almost only with its first two options and calls
most tweets offensive. Each model matches its own
reference code token for token on a fixed test set, with probabilities
within 0.035 (clef-flash), 0.031 (clef), 0.038 (d3-flash), 0.039
(elephant-1), 0.010 (joey-1), 0.031 (H2O-Lightning), 0.035 (d1-2B), 0.030
(d1-3B), 0.004 (Laya), 0.018 (Julia) and 0.058
(pplx-decider-flash-v1.2, on requests up to 4,096 tokens); pplx v1.1 and
the v1.2 27B were compared on small random models in their layouts
(within 0.00001), not on the full weights. pplx v1.2 27B is not in the
table yet.

### typed-decisions

The typed-decisions benchmark (`LocalLLaMA/typed-decisions`, its `test`
split at `7189f46`): 400 cases from four workflows (agent traces,
customer service, invoices, security incidents), each a state with five
questions of the three types, 2,000 decisions in all, each with a gold
probability distribution. Each case is one request with its five
questions, as the benchmark's leaderboard sends them (suite
`typed-decisions`), scored with the leaderboard's published scorer
(`bench/score_with_jobs.py` in the dataset's repository, at `fa01e86`):
accuracy (the most likely option against the gold's), KL divergence from
the gold (predictions floored at 1e-9), Brier score (squared error summed
over the options) and ECE (pooled over 10 bins of the top probability);
lower is better for all but accuracy. Fractions, as the leaderboard
gives them; each model as released, no quantized copies. Accuracy by
question type beside it:

| Model | Accuracy / KL / Brier / ECE | choice | noul | score |
|---|---|---|---|---|
| clef-flash | 0.699 / 0.210 / 0.111 / 0.011 | 0.695 | 0.800 | 0.625 |
| pplx-decider-flash-v1.2 | 0.718 / 0.268 / 0.115 / 0.035 | 0.700 | 0.812 | 0.661 |
| d3-flash | 0.706 / 0.332 / 0.131 / 0.040 | 0.732 | 0.757 | 0.647 |
| d3-mini | 0.730 / 0.318 / 0.123 / 0.045 | 0.743 | 0.798 | 0.667 |
| d3-nano | 0.619 / 0.575 / 0.211 / 0.122 | 0.638 | 0.738 | 0.515 |
| d3-lite | 0.524 / 0.519 / 0.244 / 0.136 | 0.515 | 0.657 | 0.431 |
| d3-edge | 0.421 / 1.179 / 0.457 / 0.347 | 0.373 | 0.678 | 0.265 |
| elephant-1 | 0.732 / 0.151 / 0.076 / 0.089 | 0.713 | 0.832 | 0.670 |
| joey-1 | 0.618 / 0.234 / 0.127 / 0.054 | 0.542 | 0.768 | 0.562 |
| H2O-Lightning-4B v1.4.2 | 0.694 / 0.349 / 0.173 / 0.091 | 0.683 | 0.803 | 0.620 |
| H2O-Lightning-4B v1.2.3 | 0.689 / 0.247 / 0.132 / 0.021 | 0.673 | 0.797 | 0.620 |
| d1-2B | 0.628 / 0.281 / 0.137 / 0.045 | 0.608 | 0.765 | 0.541 |
| d1-3B | 0.648 / 0.286 / 0.144 / 0.038 | 0.610 | 0.795 | 0.568 |
| laya | 0.363 / 0.572 / 0.315 / 0.173 | 0.292 | 0.487 | 0.325 |
| laya-typed-decisions | 0.766 / 0.117 / 0.061 / 0.213 | 0.733 | 0.855 | 0.724 |
| laya-multilingual | 0.347 / 1.113 / 0.463 / 0.317 | 0.290 | 0.485 | 0.286 |
| Julia-1 | 0.723 / 4.212 / 0.337 / 0.235 | 0.707 | 0.802 | 0.675 |
| the training split's frequencies | 0.4815 / 0.4174 / 0.1973 / 0.0367 | | | |
| uniform answers | 0.2765 / 0.4445 / 0.2382 / 0.045 | | | |

Notes:
- laya-typed-decisions was fine-tuned on the benchmark's four workflows;
  it is the most accurate here, with the lowest KL and Brier score, and
  scores the 0.766 its model card reports. laya, its base, scores the
  card's 0.362 within a rounding (0.363); the untuned Laya models fall
  below the training split's frequencies on requests of this shape. laya
  cut 7 cases at its 512-token limit; no model refused a case.
- Of the general models, elephant-1 is the most accurate (0.732), level
  with d3-mini (0.730), and the best on KL and Brier; clef-flash is by
  far the best calibrated (ECE 0.011). Julia-1 is accurate but nearly
  certain of every answer, so its KL divergence is large (4.21).
- H2O-Lightning v1.4.2, read two-way, is as accurate as v1.2.3 but less
  well calibrated (ECE 0.091 against 0.021). The weights cause this, not
  the reading or the yes/no floor: read one-way, v1.4.2 scores 0.688 /
  0.369 / 0.181 / 0.088, and v1.2.3's floor (0.801) would raise its ECE
  to 0.101. v1.4.2 keeps v1.2.3's temperatures, but its probabilities
  are sharper: on choice and score questions its confidence is 9 to 13
  points above its accuracy (v1.2.3's: under 1).
- The leaderboard's rows for the same models (third-party runs on vLLM,
  counting a decision whose gold has a tie as right for either option;
  35 of the 2,000 have one) agree with these within 0.003 under that
  rule: clef-flash 0.7100 (leaderboard 0.710), d3-flash 0.7190 (0.717),
  d3-mini 0.7405 (0.743), d3-nano 0.6290 (0.629), d3-lite 0.5335
  (0.531), d3-edge 0.4260 (0.425), d1-3B 0.6555 (0.656), with KL and
  Brier score equal to the third decimal.
- Run times for the whole suite: from 7 s (Julia-1) to 8 minutes
  (d3-flash); clef-flash 4.1 minutes, elephant-1 4.3, H2O-Lightning 4.2.

```bash
python scripts/accuracy_benchmark.py local --datasets DIR --model MODEL --out results.json
python scripts/accuracy_benchmark.py local --datasets DIR --suite typed-decisions --model MODEL --out typed.json
python scripts/accuracy_benchmark.py report results.json [more.json ...]
```

How to save the datasets at the checked commits is in the script's
docstring. `local` writes the result file as it goes and resumes from it.
