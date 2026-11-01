# Models

The model families mlx-decision runs, what each reads and how much
memory it needs. The README has a short version of this table.

- [Which model to start with](#which-model-to-start-with)
- [How they read a request](#how-they-read-a-request)
- [Memory](#memory)
- [Clef](#clef), [pplx](#pplx), [d3](#d3), [Base Labs](#base-labs),
  [H2O-Lightning](#h2o-lightning), [d1 LFM3](#d1-lfm3), [d1](#d1), [Laya](#laya),
  [Julia](#julia)

| Model | Size | Input | Limit | Memory | Short request | Speedup | Good at |
|---|---|---|---|---|---|---|---|
| Cloudflare's [clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 9B | text, images | 16,384 tokens | 19 GB (8-bit: 11 GB) | 0.25 s | 3.5x | knowledge, classification, many options, images |
| Cloudflare's [clef](https://huggingface.co/Cloudflare/clef) | 27B | text, images | 16,384 tokens | 54 GB (8-bit: 32 GB, 4-bit: 19 GB) | 1.1 s (8-bit) | 3.7x (bf16) | as clef-flash, a little better on entailment (8-bit) |
| Perplexity's [pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | 4B | text, images | 32,768 tokens per request (up to 262,144) | 9.1 GB of weights (8-bit: 5.1 GB, 4-bit: 3.0 GB) | - | - | all questions of a request in one pass, images up to 1.6 MP; distilled from the 27B |
| Perplexity's [pplx-decider-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) | 27B | text, images | 32,768 tokens per request (up to 262,144) | 52 GB (8-bit: 29 GB, 4-bit: 17 GB) | - | - | as flash, and more accurate (Perplexity's Decision Index 68.1 against 58.9) |
| vLLM Semantic Router's [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | 9B | text, images | 32,768 tokens per question (up to 262,144) | 19 GB (8-bit: 12 GB) | 0.23 s | - | entailment, knowledge, images, calibrated scores; five more sizes from 0.6B to 27B |
| Base Labs' [elephant-1](https://huggingface.co/baselabs/elephant-1) | 9B | text | 16,384 tokens per question (up to 262,144) | 18 GB (8-bit: 11 GB) | 0.29 s | - | entailment and science questions, calibrated; a request's questions share one pass over the state |
| Base Labs' [mammoth-1](https://huggingface.co/baselabs/mammoth-1) | 27B | text | 16,384 tokens per question (up to 262,144) | 8-bit: 33 GB | - | - | as elephant-1; checked only as an 8-bit copy |
| H2O.ai's [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b) | 4B | text, images | 40,960 tokens per question | 13 GB (8-bit: 9 GB, 4-bit: 7 GB) | 0.18 s | - | fast with long states and many questions, images, calibrated scores |
| LiquidAI's [d1-2B](https://huggingface.co/LiquidAI/d1-2B) | 1.9B | text, images, many languages | 32,768 tokens per set of 16 questions (up to 128,000) | 5.3 GB | 0.04 s | 1.4x | small and fast, images; answers depend on which questions are asked together; commercial use limited by its licence |
| LiquidAI's [d1-3B](https://huggingface.co/LiquidAI/d1-3B) | 3B | text, many languages | 32,768 tokens per question | 12 GB | 0.06 s | 1.1x | small and fast, many questions on one state; commercial use limited by its licence |
| [laya](https://huggingface.co/convaiinnovations/laya) | 421M | English text | 512 tokens per question | 1.8 GB | 16 ms | 3.0x | topic classification of English text |
| [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 421M | English text | 1,024 tokens per question | 1.7 GB | 16 ms | 2.9x | as laya, tuned for typed decision workflows |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | 322M | text, 100+ languages | 1,024 tokens per question | 1.6 GB | 8 ms | 2.5x | topic classification in many languages |
| Supersonic Labs' [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) | 144M | text, many languages | 8,192 tokens per question | 1.4 GB | 6 ms | 2.0x | emotion and routing with 2 to 20 options, long states |

"Short request": one question about a state of about 250 tokens, on an
Apple M5 Pro. "Speedup": the same request against the model's own PyTorch
code on the same Mac (MPS), with the same answers; across all measured
requests it ranges from 1.4 to 6 times ([Speed](BENCHMARKS.md#speed-compared-with-pytorch)), except for
d1-3B, whose own code is about as fast (0.9 to 1.2 times). Clef 27B's
reference does not fit that Mac in bf16, so its speedup was measured on
an M2 Ultra; pplx's was not measured (v1.2 not yet on speed either), nor
H2O-Lightning's, whose release runs on vLLM with CUDA. d3-flash's own runtime was compared over a whole
test set, not per request: it took 2.9 times as long; Base Labs'
engine, with elephant-1, 3.6 times as long. How the models
differ is [below](#how-they-read-a-request).

## Which model to start with

By your Mac's memory, among the models measured on the
[benchmarks](BENCHMARKS.md#accuracy):

- **16 GB**: d1-3B, small and fast (0.06 s a question), about level
  with H2O-Lightning-4B on the benchmarks, or d1-2B, smaller and faster
  still (0.04 s), with pictures, level with d1-3B; check their licence
  before commercial use. H2O-Lightning-4B for long states with many questions
  (its 8-bit copy when the states are long).
- **32 GB or more**: clef-flash, the most accurate on classification
  (TREC 96.8%), with images; elephant-1 when entailment matters (ANLI
  round 3: 74.5% against clef-flash's 50.8%, ahead of Jev) and for
  science questions (OpenBookQA 96.8%), text only; d3-flash for
  entailment with images (64.3%).
- **48 GB or more**: an 8-bit copy of Clef 27B (a little ahead of
  clef-flash on entailment and calibration) or of pplx-decider-v1.2
  (not measured yet; its v1.1 was the only local model close to Jev on
  entailment).
- **Any Mac, many short texts**: laya for English topics,
  laya-multilingual for other languages (89.8% on TREC, in
  milliseconds); both are weak at entailment and science questions.

pplx-decider-flash-v1.2, the model in the Studio's screenshots, fits
16 GB Macs (9.1 GB) and asks all questions of a request in one pass; on
the benchmarks it is close to the 27B pplx v1.1 on answer type (TREC
93.6%) and the best of all on sentiment (SST-5 60.6%). d3-mini (4B, 9.1
GB) is level with d3-flash at under half its size (ahead on answer type
and sentiment, 5 points behind on entailment). Compare
in the Studio, on your own examples, says more than any benchmark.

## How they read a request

The models differ in size and in how they read a request. Clef reads the
state once and answers all questions together; pplx v1.2 reads the state
and all questions as one sequence, so the questions see each other and
its input limit applies to the whole request; d3, Laya, Julia and pplx
v1.1 read each question on its own, with the whole state, so their input
limit applies per question and their time grows with the number of
questions.
H2O-Lightning and d1 also read each question on its own, but after the
state, which they compute once per request, so more questions cost
little. Base Labs computes the state once too, reading it both ways, and
then runs the questions together in one batch: each sees the state and
itself, never the other questions. d1 LFM3 reads the questions in sets of
16, each set one sequence with the whole state, in which the state's
tokens also see the set's questions; so its answers depend on which
questions share a set and in what order.

## Memory

Before reading the weights, loading checks that the model fits: its
weights plus 10% must be within the GPU's recommended working set
(macOS swaps beyond it). A model that does not fit is refused with both
sizes and, for Clef, pplx, d3, Base Labs and H2O-Lightning, the `convert` command for a
smaller copy that fits.
`--no-memory-check` loads it anyway.

## Clef

Cloudflare's clef-flash (9B) and clef (27B) take text and images (videos
are rejected) up to 16,384 tokens; a longer state is cut to fit.

### Which copy fits

Each Clef model runs in bf16 as released or as a quantized copy. The fit
check asks for the weights plus 10% within the GPU's working set, which
is about three quarters of a Mac's memory on a 16 GB Mac (12.7 GB) and
four fifths on a 64 GB Mac (55.7 GB) (`mlx-decision benchmark` prints
it):

| Copy | clef-flash needs | Clef 27B needs |
|---|---|---|
| bf16 (as released) | 21.0 GB: 32 GB Macs and up | 60.5 GB: 96 GB Macs and up |
| 8-bit | 11.7 GB: 16 GB Macs and up | 32.7 GB: 48 GB Macs and up |
| 4-bit | 6.8 GB: 16 GB Macs and up | 17.9 GB: 32 GB Macs and up |

Peak memory at the full 16,384 tokens: clef-flash 19.0 GB in bf16 and
11.2 GB in 8-bit; Clef 27B 32 GB in 8-bit and 19.5 GB in 4-bit.
[`download --bits`](CLI.md#getting-a-model-download) stores a quantized copy directly;
[convert](CLI.md#quantizing-convert) makes one from a full download and
offers mixed precision, which changes fewer answers than uniform 4-bit.

Clef 27B is three to four times slower than clef-flash and only a little
more accurate on our benchmarks, mainly on entailment, with better
calibrated probabilities (8-bit, see the table above). In bf16 it
matches Cloudflare's reference within 0.031 per probability (measured
on an M2 Ultra, where it is 3.7 times faster than the reference); the
8-bit copy changed one of 177 answers, the 4-bit copy three.

### Images

The Python API takes images as file paths,
bytes, PIL images or data URLs; `run` and `chat` take files (`--image`,
`/image`); the server takes data URLs only. They must be PNG, JPEG or WebP,
at most about 89 million pixels. Each image is resized the way Cloudflare's
processor does it and costs one token per 32x32 pixels. Images larger than
2 megapixels (about 2,048 tokens) are shrunk to that size, which changed no
answer in our tests, even on screenshots with small text; `--max-image-mp`
(Python: `max_image_pixels`) sets another cap, and `0` lifts it to the
processor's own maximum (16.7 MP), as Cloudflare's reference does. A
1024x768 image (768 tokens) with a short state and three questions takes
0.84 s on clef-flash ([more sizes](BENCHMARKS.md#images)).

Images count against the input limit together with the questions: the
state is cut first, and a request whose images and questions alone do not
fit fails with an error on `images`. The vision tower (0.85 GB) loads with
the first image request.

### Reusing a state

Clef keeps what it computed for recent states (up
to 2 GB per model), so a request with the same state and images as a
recent one, and other questions, computes only the new questions: on
clef-flash, a 16,384-token state takes about 10 s the first time and
0.2 s for one new question after that (1.5 s for twenty). The answers
are the same as without reuse, within the parity tolerance.
`--prefix-cache GB` on `run`, `chat`, `server` and `mcp` (Python:
`prefix_cache_gb`) changes the amount; 0 turns it off. Base Labs,
H2O-Lightning and d1 keep recent states the same way (d1 only states long enough to be
worth it). pplx, d3, d1 LFM3, Laya and Julia read the state together with
the questions, so they have nothing to reuse.

## pplx

Perplexity's pplx-decider-v1.2 (27B, fine-tuned from Qwen3.8-27B, the
same architecture as Clef 27B) and pplx-decider-flash-v1.2 (4B, on
Qwen3.5-4B, distilled from the 27B) answer a whole request in one
sequence: the state, then every question with its options, each followed
by a marker where its answer is read out. Asking five questions costs
about as much as asking one. Both take text and images and up to 255
options or score levels.

Because they share one sequence, the questions see each other, and an
answer can change when other questions are in the same request.
Perplexity's cards say this helps when the questions are facets of one
decision (select-all-that-apply options, steps of one routing decision)
and hurts when they are independent candidates (documents, tools,
function calls), more so for flash, which scores 60.1 on their Decision
Index with one question per request and 58.9 with joint requests (the
27B: 68.1 joint). For independent answers, send one question per request.

The input limit applies to the sequence of a request: 32,768 tokens by
default, the length the models were trained on, and up to 262,144 (the
release's default, the backbone's context) with `--max-input-tokens`
(Python: `max_input_tokens`). A request over the limit is answered in
parts, as the release does: its questions are halved until each part
fits, every part with the whole state and images. A question that does
not fit alone is refused, naming it; input is never cut. Text in the
request that looks like the prompt's special tokens is neutralized, as
in the release, so a state cannot fake a marker.

| Copy | Disk | The fit check needs |
|---|---|---|
| flash, bf16 (as released) | 9.1 GB | 10.0 GB: 16 GB Macs and up |
| flash, 8-bit | 5.1 GB | 5.7 GB: 16 GB Macs and up |
| flash, 4-bit | 3.0 GB | 3.4 GB: 16 GB Macs and up |
| 27B, bf16 (as released) | 52.2 GB | 57.4 GB: 96 GB Macs and up |
| 27B, 8-bit | 28.2 GB | 31.0 GB: 48 GB Macs and up |
| 27B, 4-bit | 15.3 GB | 16.9 GB: 32 GB Macs and up |

`download --bits 8` or `--bits 4` stores a copy without keeping the
full-size files, and [convert](CLI.md#quantizing-convert) writes one from a
full download (also mixed precision); the decision head and the vision
tower stay as released. Each image is scaled
to at most 1.6 megapixels (about 1,600 tokens), the release's maximum,
which `--max-image-mp` lowers. The vision tower loads with the first
image request.

Both match the release's code token for token, request parts included,
on a fixed test set; their probabilities were compared with that code on
small random models in their layouts, and flash also on its weights:
in bf16 against the release's code in bf16, the largest difference was
0.058 per probability (mean 0.002), about what the release's own code
varies between two runs, and the 8-bit copy chose the same answer for
296 of 299 questions (the other three near ties). On the
[accuracy benchmarks](BENCHMARKS.md#accuracy) flash (bf16) is close to the 27B
pplx v1.1 on answer type (TREC 93.6% against 94.0%) and the best of all
models on sentiment (SST-5 60.6%), behind it on entailment (61.8% against
66.8%) and science questions (93.8% against 96.8%). Speed against the
release, and the v1.2 27B on the benchmarks, have not been measured yet.

### pplx v1.1

pplx-decider-v1.1-27b, the earlier version, still loads (by repo id,
or a folder or converted copy) and answers with its own prompt: every
question as its own sequence with the whole state and its images, so its
input limit, 8,192 tokens, applies per question (it can be lowered, not
raised); images take 64 to 256 tokens each. Its copies have the 27B's
sizes above (8-bit: 29.1 GB peak memory at 7,350 tokens). One question
about a 250-token state takes 0.83 s (8-bit); five questions take five
times as long (4,000-token state: 9.5 s for one question, 49 s for
five). On the [accuracy benchmarks](BENCHMARKS.md#accuracy) its 8-bit copy is the most
accurate local model overall, well ahead on entailment (ANLI round 3:
66.8% against 55.2% for Clef 27B; Jev 70.2%). Perplexity reports a
Decision Index of 61.6 for it (Jev: 57.9).

## d3

vLLM Semantic Router's Decision 3.0 models come in six sizes, all
built on Qwen3.5 with text and images, with pplx v1.1's design and their own
prompt: each question is its own sequence with the whole state and its
images, and the answer is read out under letter codes. They answer up to
255 options and 2 to 10 score levels. The input limit applies per
question: 32,768 tokens by default. The release sets no limit, so
`--max-input-tokens` (Python: `max_input_tokens`) can raise it up to
262,144, the backbone's context. Longer input is refused, never cut.
Memory grows with the input: d3-flash needs about 4.3 GB more at 32,768
tokens and 17 GB more at 131,072, where one question takes about 5
minutes.

| Model | Disk (bf16) | The fit check needs | One question, 250-token state |
|---|---|---|---|
| [d3](https://huggingface.co/vllm-sr/d3) (27B) | 52 GB | as pplx: 96 GB Macs in bf16, 48 GB in 8-bit, 32 GB in 4-bit | - |
| [d3-flash](https://huggingface.co/vllm-sr/d3-flash) (9B) | 16.8 GB | 18.5 GB: 32 GB Macs and up (8-bit 10.3 GB, 4-bit 5.9 GB: 16 GB Macs) | 0.23 s |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) (4B) | 9.1 GB | 10.0 GB: 16 GB Macs and up | 0.12 s |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) (2B) | 4.5 GB | 4.9 GB: 16 GB Macs and up | - |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) (0.8B) | 1.7 GB | 1.9 GB: 16 GB Macs and up | - |
| [d3-edge](https://huggingface.co/vllm-sr/d3-edge) (0.6B) | 1.2 GB | 1.3 GB: 16 GB Macs and up | - |

d3-flash peaks at 18.7 GB in bf16 and 11.7 GB in 8-bit on the default
`benchmark` lengths (up to 29,450 tokens). The fit check counts the
weights, not the input: on a 16 GB Mac the 8-bit copy passes it but
long inputs take it close to the GPU's working set (12.7 GB there), so
the 4-bit copy is the safer choice for long inputs (about 10 GB at the
full 32,768 tokens). Five questions take five
times as long as one: 12.6 s for five about a 4,000-token state. The
8-bit copy saves memory, not time (up to 30% slower). `download --bits`
and [convert](CLI.md#quantizing-convert) write quantized copies, mixed
precision included; prefer 8-bit for the smallest models.

Images, as for Clef: any number per request,
each scaled to between 65,536 and 1,638,400 pixels (64 to 1,600 tokens)
as in the release; `--max-image-mp` (Python: `max_image_pixels`) lowers
the maximum. Videos, which the release also reads, are refused. The
vision tower loads with the first image request.

On the [accuracy benchmarks](BENCHMARKS.md#accuracy) d3-flash comes close to pplx
v1.1 on entailment (ANLI round 3: 64.3% against 66.8%) and science
questions, at a third of its size, with well calibrated probabilities.
d3-mini (4B) is level with d3-flash (TREC 86.2%, ANLI round 3 59.2%,
OpenBookQA 93.8%, SST-5 58.5%); d3-nano (2B) keeps most of it on answer
type and science questions (80.4%, 88.2%) but not on entailment (48.1%);
d3-lite and d3-edge are near chance on entailment
([all sizes](BENCHMARKS.md#accuracy)).
vLLM Semantic Router reports a Jev Decision Index of 55.8 for d3-flash
(text board 0.3.2). Its prompt and token ids match the release's runtime
for every question of the test set; in bf16 its probabilities are within
0.038 of that runtime run in bf16 on the same Mac (mean 0.0016), with no
changed answer where the release's top two are more than 0.04 apart. The
smaller sizes agree with their releases on a sample of 26 questions
(within 0.011 to 0.030; d3-nano 0.058 on one 255-option question, where
its own release moves 0.026 between the Mac's GPU and CPU). d3 27B was
not run.

## Base Labs

Base Labs (Baseten) post-trained two Qwen3.5 models for decisions, text
only, under the MIT licence; their makers serve them with
[OneJev](https://github.com/OmniJev/OneJev)'s runtime, whose prompt
mlx-decision uses. A request's prompt starts with the state, which the
model reads once, its full-attention layers looking both ways over it;
the questions then run together off that state, each reading the state
and itself both ways but never the other questions, and the answer is
read out under letter codes. They answer up to 255 options and 2 to 10
score levels. The input limit applies per question, the state included:
16,384 tokens by default, the length the models were trained on;
`--max-input-tokens` (Python: `max_input_tokens`) raises it up to
262,144. Longer input is refused, never cut. Images are refused (the
release reads them, but not with the attention it was trained with);
an image inside the state is read as text.

| Model | Disk (bf16) | The fit check needs | One question, 250-token state |
|---|---|---|---|
| [elephant-1](https://huggingface.co/baselabs/elephant-1) (9B) | 18.8 GB | 20.7 GB: 32 GB Macs and up (8-bit 10.3 GB: 16 GB Macs) | 0.29 s |
| [mammoth-1](https://huggingface.co/baselabs/mammoth-1) (27B) | 54.7 GB | as pplx: 96 GB Macs in bf16, 48 GB in 8-bit | - |

elephant-1 peaks at 17.6 GB in bf16 and 10.7 GB in 8-bit on the default
`benchmark` lengths (up to 15,000 tokens), 20.0 and 13.3 GB at 32,768.
More questions cost little: five about a 15,000-token state take 11.4 s,
one 10.6 s, and with the state kept from an earlier request 0.57 s. On
a 16 GB Mac the 8-bit copy fits the GPU's working set (12.7 GB there)
up to the default limit, not at 32,768 tokens. The 8-bit copy saves
memory, not time (7 to 24% slower); it answers as bf16 does (no changed
answer on the test set), the 4-bit copy changed 9 of 166, so prefer
8-bit. `download --bits` and [convert](CLI.md#quantizing-convert) write
quantized copies; the answer codes' rows of the LM head (all the
readout uses of it) stay in float32. Storing mammoth-1 quantized reads
its 49.8 GB first weight file at once, about 57 GB of memory, so on
Macs with less it does not finish; its 8-bit copy itself needs 33 GB.

On the [accuracy benchmarks](BENCHMARKS.md#accuracy) elephant-1 is the
best of the local models on entailment, ahead of Jev (ANLI round 3:
74.5% against 70.2%), and level with pplx v1.1 27B on science questions
(OpenBookQA 96.8%), with well calibrated probabilities; it is weaker on
sentiment levels (SST-5: 53.9%). Its model card lists public decision
data among the training data without naming the datasets. Its prompt
and token ids match the release's runtime for every question of the
test set; in bf16 its probabilities are within 0.039 of that runtime
run in bf16 on the same Mac (mean 0.0010), with no changed answer, and
mlx-decision answers the test set 3.6 times as fast. mammoth-1 was only
checked as an 8-bit copy: it agrees with elephant-1 on 153 of the 166
top answers; it is not on the benchmarks.

## H2O-Lightning

H2O.ai's H2O-Lightning-4B, built on Qwen3.5-4B, takes text and images and
answers 1 to 255 options. Each question is its own sequence, but the
system prompt and the state come first: they are computed once per
request and shared by its questions, and kept for later requests on the
same state as for Clef ([reusing a state](#reusing-a-state)). The input
limit, 40,960 tokens, applies per question. As in the release, a state
over 32,000 tokens keeps its first and last 16,000 around a `...` line
(`truncated` is set), and a question still over the limit is refused.
`--max-input-tokens` lowers the limit, not raises it; the state is then
cut to leave room for the request's longest question.

As released, a yes/no answer less sure than 0.801 is reported at 0.801 on
its side: the answer stays, its probability moves. `--noul-floor 0`
(Python: `noul_floor=0`, also for `server` and `mcp`) reports the model's
own probability, for routing uncertain answers elsewhere; a value
between 0.5 and 1 sets another floor. Other families refuse the option.

| Copy | Disk | The fit check needs | Peak memory (longest states) |
|---|---|---|---|
| bf16 (as released) | 9.1 GB | 10.0 GB: 16 GB Macs and up | 12.6 GB |
| 8-bit | 5.1 GB | 5.7 GB: 16 GB Macs and up | 8.9 GB |
| mixed 4.5 | 3.3 GB | 3.6 GB: 16 GB Macs and up | 7.2 GB |
| 4-bit | 3.0 GB | 3.3 GB: 16 GB Macs and up | 7.0 GB |

On a 16 GB Mac the full-size model fits, but long states take it to the
edge of the GPU's working set (12.6 of 12.7 GB); use the 8-bit copy
there for long states. `download --bits` and
[convert](CLI.md#quantizing-convert) write the copies; the vision tower stays as
released.

One question about a 250-token state takes 0.18 s, five take 0.41 s;
a 28,800-token state takes 12.2 s with one question and 12.7 s with five,
and 0.69 s for five new questions once the state is kept. As for Clef,
the quantized copies save memory, not time: from 1,000 tokens on they
are 5-17% slower.

Images, as for Clef: up to 4 per request, each
scaled to at most 1,048,576 pixels as in the release (`--max-image-mp`
does not apply), placed ahead of the text. The vision tower loads with
the first image request.

On the [accuracy benchmarks](BENCHMARKS.md#accuracy) it is behind the 27B models, ahead
of clef-flash on entailment (ANLI round 3: 59.9% against 50.8%) and has the
best calibrated sentiment scores. H2O.ai reports 205 of the 231 public
JevBench items for the release. Its prompt and token ids match the
release's code (its shim, with transformers in place of vLLM) for every
question of the test set; in bf16 its probabilities are within 0.020 of
that code run in bf16 on the same Mac (mean 0.0014), and the 3 answers
that changed of 271 were near ties.

## d1 LFM3

LiquidAI's d1-2B (1.9B), built on LFM3-VL-2B (LFM3: gated short
convolutions with sliding-window and full attention layers), reads text
in many languages and pictures (videos are refused). It answers up to 255
options and 2 to 10 score levels.

It reads a request's questions in their order in sets of 16 (the size it
was trained with; the last set may be smaller). Each set is one sequence:
the pictures and the state, then the set's questions. On the full
attention layers the state's tokens see every question of the set, and
each question sees the state and itself, not the other questions. So:

- answers depend on which questions share a set and on their order: the
  same question asked alone or after other questions can get a somewhat
  different answer (in LiquidAI's release too: on our test set, alone
  against within a set, by 0.07 per probability on average);
- with more than 16 questions the state is read once per set (twenty
  questions read it twice);
- nothing is kept between requests: the state's computed form depends on
  its set's questions, so there is no state reuse (`--prefix-cache` is
  refused).

The input limit applies per set (the state, its pictures and the set's
questions): 32,768 tokens by default, up to 128,000 (the model's context)
with `--max-input-tokens`. A state too long for the request's largest
set is cut, keeping its start and end around a `...` line (`truncated` is
set); a set whose questions and pictures alone do not fit is refused,
naming its first question. LiquidAI's card finds answers less reliable
past about 64,000 tokens of state. A question without instructions (or
with empty ones) uses its id, and score levels given as objects are read
as their text, as for d1; the release itself refuses both.

Pictures (in `images` or inside the state) larger than 1.6 megapixels
are shrunk to that size, as the release does (`--max-image-mp`, Python:
`max_image_pixels`, lowers the cap); each costs one token per 32x32
pixels, so a 1 MP picture is about 1,024 tokens. The vision tower (0.2B)
loads with the first picture and runs once per request; every set reads
the pictures as part of the state. A 384 x 384 picture with one question
takes 44 ms.

Its weights are under the LFM Open License v1.0, as d1-3B's, which
licenses commercial use only to organisations with less than $10 million
in annual revenue; read it before using the model. `download` shows this
and asks before downloading (`--yes` skips the question), and loading by
repo id notes it on stderr when it downloads.

It runs in bf16 as released (3.8 GB): the fit check needs 4.2 GB, so 16
GB Macs and up; there are no quantized copies. Its peak is 5.3 GB at
29,450 tokens with twenty questions, and 11.8 GB for one question on a
128,000-token state (123 s). One question about a 250-token state takes
43 ms, twenty 202 ms; a 4,000-token state takes 0.51 s with one question
and 1.15 s with twenty. That is 1.4 to 1.9 times as fast as its release's
own PyTorch code on the same Mac (not compiled), with probabilities
within 0.010.

On the [accuracy benchmarks](BENCHMARKS.md#accuracy) it is level with d1-3B at under
two thirds of its size: ahead on answer type (TREC 88.2% against 86.6%)
and offensive language, level on science questions and sentiment, behind
on entailment (ANLI round 3: 52.2% against 55.7%). Its prompts, the
packed sets and token ids match the release's code for every request of
the test set (pictures included); in bf16 its probabilities are within
0.035 of that code run in bf16 on the same Mac (mean 0.0030), with no
clear answer changed. The prompt and answer reading were written in a
clean room from the release's documented and observed behaviour; the
release's own code (under the LFM licence) is not part of mlx-decision.
The readout fit and calibration (`training fit`) work as for d1, on text.

## d1

LiquidAI's d1-3B, built on LFM2.5-VL-3B (LFM2: mostly short convolutions
with a few attention layers), reads text in many languages; the release
also reads images, which mlx-decision does not support for it yet
(`images` are refused). It answers up to 255 options and 2 to 10 score
levels (the model card's range). Each question is its own prompt with the
state first: a request's questions run together in one pass, and a long
state is computed once and kept for later requests, as for Clef
([reusing a state](#reusing-a-state)). The input limit, 32,768 tokens
(the model's context), applies per question; a longer state is cut,
keeping its start and end around a `...` line (`truncated` is set), and
only a question too long on its own is refused. `--max-input-tokens`
lowers the limit. A question without instructions uses its id, as for
Laya and Julia.

Its weights are not under Apache-2.0 but the LFM Open License v1.0,
which licenses commercial use only to organisations with less than $10
million in annual revenue; read it before using the model. `download`
shows this and asks before downloading (`--yes` skips the question), and
loading by repo id notes it on stderr when it downloads.

It runs in bf16 as released (6.2 GB): the fit check needs 6.9 GB, so 16
GB Macs and up, though its peak of 11.6 GB (29,450 tokens, twenty
questions) comes close to a 16 GB Mac's working set (12.7 GB). There are no quantized copies: they would save memory but no
time. One question about a 250-token state takes 62 ms, five 125 ms,
twenty 245 ms; a 4,000-token state takes 0.77 s with one question and
1.0 s with twenty, and twenty new questions on a kept state take 0.2 to
1.0 s. Its release's own PyTorch code is about as fast on the same Mac
(mlx-decision 0.9 to 1.2 times as fast): it reads the state once for all
questions too.

On the [accuracy benchmarks](BENCHMARKS.md#accuracy) it is level with H2O-Lightning-4B
in a third of the time, ahead on answer type, offensive language and
science questions and behind on entailment (ANLI round 3: 55.7% against
59.9%). LiquidAI reports it as the best decision model under 10B on the
Decision Index 0.2.1 (scored by themselves). Its prompt and token ids
match the release's code for every question of the test set (22
languages included); in bf16 its probabilities are within 0.030 of that
code run in bf16 on the same Mac (mean 0.0054), with no clear answer
changed. The prompt and answer reading were written from the release's
documented and observed behaviour; the release's own code (under the LFM
licence) is not part of mlx-decision.

## Laya

Encoder models from Convai Innovations: laya and laya-typed-decisions
(ModernBERT-large, English; non-Latin scripts fail, per Laya's README) and
laya-multilingual (mmBERT-base). A state that does not fit is cut, keeping
the end of a list such as a conversation, and long option lists are
shortened to fit the model's question budget (which costs accuracy when
there are many options). `--max-input-tokens` raises the limit up to 8,192.
Probabilities use Laya's calibrated temperatures, as its own package
applies them.

## Julia

Supersonic Labs' encoder model (mmBERT-small). It answers 2 to 20 options
per question and, as its release recommends, refuses what it would have to
cut: a state that does not fit, an option over 48 tokens, its marker token
`<mask>` in the text. In Python, `strict_encoding=False` cuts instead.

Laya and Julia need instructions for every question, and Julia a
description for every option. Where a question leaves them out (as the
short forms `--choice` and `--noul` do), the question id stands in for the
instructions and the option id for the description. Both run in float16.

Julia reads an option only through its description, and how that is
worded matters a lot. Describe each option in a short phrase of what it
covers: on 11 support tickets routed to billing, technical or sales,
Julia got 8 to 11 right with phrases such as "Billing and payment
disputes", but 4 to 6 with the bare ids and 1 to 3 with one-word labels
such as "Billing" (Laya: 9 to 10 with any of them). Check the wording on
a few of your own cases before relying on it. `run` and `chat` print a
hint when Julia gets choice options without descriptions.
