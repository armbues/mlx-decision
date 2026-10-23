# Command line

The `mlx-decision` command. `server` and `mcp` are in [SERVER.md](SERVER.md),
`web` (the Studio) in [STUDIO.md](STUDIO.md), `data` and `training` in
[TRAINING.md](TRAINING.md). `mlx-decision COMMAND --help` lists every flag.

- [One state: `run`](#one-state-run)
- [Many states](#many-states)
- [Defaults in a config file](#defaults-in-a-config-file)
- [Interactive: `chat`](#interactive-chat)
- [Getting a model: `download`](#getting-a-model-download)
- [Quantizing: `convert`](#quantizing-convert)
- [Measuring speed: `benchmark`](#measuring-speed-benchmark)

## One state: `run`

Ask questions about a state:

```bash
mlx-decision run -m Cloudflare/clef-flash \
  -s "My card was charged twice this month." \
  --noul "refund=The customer wants money back" \
  --choice "team=billing,technical,sales" \
  --score "anger=calm,annoyed,angry"
```

```
refund (noul): no, p(yes) 0.110
  ███

team (choice): billing, confidence 0.98
  billing    0.989  ████████████████████████
  technical  0.007
  sales      0.004

anger (score): 2.05 on 1-3, confidence 0.54
   1  0.128  ███                       calm
   2  0.695  █████████████████         annoyed
   3  0.177  ████                      angry

clef-flash · 291 input tokens
```

A score's answer is the expected level, weighted by the probabilities:
1 × 0.128 + 2 × 0.695 + 3 × 0.177 = 2.05 on a scale of 1 to 3. The text
counts levels from 1, as people rate; `--json` and the API give Jev's
value, which counts from 0 (here 1.05, levels `"0"` to `"2"`).

The state can also come from `--state-file` or stdin (`--state-json` parses
it as JSON), and questions from a JSON file with `-q questions.json`
(a questions map or a whole request body). `--json` prints the exact
response body. `--image FILE` (repeatable) sends images along with the
state (Clef, pplx, d3, H2O-Lightning). `--max-input-tokens N` (also on `chat`,
`server` and `mcp`) changes the model's input limit, e.g. to give Laya up
to 8,192 tokens or d3 and pplx v1.2 up to 262,144; `--noul-floor` changes H2O-Lightning's
[yes/no floor](MODELS.md#h2o-lightning).

`mlx-decision --version` prints the installed version.

## Many states

`--states FILE` answers a JSON lines file with one request body per line
and prints one response body per line, in the same order. `-` reads
stdin. A line is the same body the server takes, so it can bring its own
questions; questions from `-q` or the shorthand flags apply to lines
without them:

```bash
mlx-decision run -m Cloudflare/clef-flash -q questions.json --states tickets.jsonl > answers.jsonl
```

```
{"state": "My card was charged twice this month."}
{"state": {"ticket": 4711, "text": "The app crashes on login."}}
{"state": "Is this spam?", "questions": {"spam": {"type": "noul", "instructions": "Is it spam?"}}}
```

Lines can carry images as file paths or data URLs
(`{"state": "...", "images": ["shots/4711.png"]}`); `--image` flags apply
to lines without their own. A line that cannot be answered yields an
error body with its line number, and the run goes on:

```
{"error": {"message": "state is required", "type": "invalid_request_error", "param": "state"}, "line": 3}
```

## Defaults in a config file

`~/.mlx-decision/config.toml`, if it exists, holds defaults for some
flags. A flag on the command line wins over the file, the file over the
built-in default:

```toml
models_dir = "~/Models"       # where models live: -m takes their names; download stores there
models = ["Cloudflare/clef-flash"]  # more models for server, web and benchmark without -m
default_model = "clef-flash"  # run, chat, mcp and training without -m; server's and web's --default-model
memory_budget_gb = 30         # server's and web's --memory-budget
host = "127.0.0.1"            # server's, mcp's and web's --host
port = 8000                   # server's, mcp's and web's --port
workspace = "~/Decisions"     # web's --workspace: where the Studio keeps its files
```

`models_dir` is a folder of models (each subfolder a model named after
it) or `"hf-cache"` for the Hugging Face cache. Its models are served
with `models` where `-m` is absent, `-m laya` finds `~/Models/laya`, and
`download` puts a model into it. When it is not set, the first
`download` in a terminal asks for a folder (or the cache) and adds the
key to the file, as the Studio's first start does; besides that, only
the Studio's [Settings](STUDIO.md#settings) page writes the file. `run` and
`chat` without `-m` or a `default_model` show a menu of the models in
`models_dir`, in `models` and in the Hugging Face cache.

`models` takes what `-m` takes; `default_model` is the name of one of
those models, or a model folder or repo id of its own. Paths start with
`/`, `~` or `./` (relative to the file's folder). An unknown key is a
warning, a value of the wrong type an error naming the key. The API key
is not read from the file (`--api-key` or `MLX_DECISION_API_KEY`).
`MLX_DECISION_HOME` moves the folder.

## Interactive: `chat`

`mlx-decision chat` loads the model once and answers one state after
another: type or paste a state, finish it with a blank line, and quit with
Ctrl-D. Lines can be edited, Up recalls earlier states, and a toolbar at
the bottom shows the keys. Questions come from the same flags and files as
in `run`:

```bash
mlx-decision chat -m Cloudflare/clef-flash --choice "team=billing,technical,sales"
```

Without any questions, `chat` first asks for them: the type from a menu
(arrow keys or the number, then Enter), an id (Enter takes the one
suggested), the instructions, and then the options of a choice (`option`
or `option: description`) or the levels of a score (lowest first), one per
line, ending with a blank line. Each question is checked against the model
as soon as it is complete, and shown as understood. `/save FILE` keeps the
questions for the next time:

```
$ mlx-decision chat -m Cloudflare/clef-flash -q questions.json
clef-flash loaded, 2 questions.
Type a state and finish it with a blank line. /help lists the commands, Ctrl-D quits.
state> My card was charged twice this month.

team (choice): billing, confidence 0.93
  billing    0.951  ███████████████████████
  technical  0.028  █
  sales      0.021  █

refund (noul): no, p(yes) 0.030
  █

clef-flash · 225 input tokens
```

Between states, commands change the questions, whether they came from the
builder, flags or a file. `/edit` and `/remove` without an id offer a menu
of the questions.

| Command | |
|---|---|
| `/list` | show the questions |
| `/add` | build another question |
| `/edit [ID]` | change a question; the current values are filled in |
| `/remove [ID]` | remove a question |
| `/save FILE` | write the questions as a file that `-q` reads |
| `/image FILE`, `/image clear` | attach an image to the following states, or drop them (`--image` attaches at start) |
| `/help`, `/quit` | |

A state that itself starts with `/` is typed as `//`. `chat` needs a
terminal; scripts use `run --states`. With `--json`, the response bodies
go to stdout and can be redirected while the prompts stay on screen.

## Getting a model: `download`

Models load from a local folder or a Hugging Face repo id; a repo id is
downloaded into the Hugging Face cache on first use. This is enough to
start:

```bash
mlx-decision run -m SupersonicLabs/Julia-1 -s "My card was charged twice." --noul "Is it a billing problem?"
```

`mlx-decision download` fetches a model ahead of time, from a menu or by
repo id. The menu lists the families first (maker, a short line, how many
models and their sizes), then the picked family's models; Escape or
"Back" returns to the families, and "Other..." takes any repo id. In a
terminal it asks for a folder to
download into (the model goes into a folder named after the repo inside
it, which you then pass to `-m`; an empty answer keeps the Hugging Face
cache); `--local-dir` names the folder directly. It warns before
downloading a repository no supported model family can load, and shows a
model's own terms of use where it has them (d1-3B: commercial use only
below $10 million annual revenue), asking before it downloads; `--yes`
skips both questions. Private or gated repos need `HF_TOKEN`. Ctrl-C
stops the download and removes the model's new folder, so no half model
is left (in the Hugging Face cache, the Hub keeps its unfinished files
and the next download of the model resumes them). `--events` prints the
progress as JSON lines for other programs, as [`training fit`](TRAINING.md#progress-events-cancel)
does, and asks nothing.

Clef, pplx, d3 and H2O-Lightning models can also be stored quantized. After the model, the menu asks
for full size, 8-bit or 4-bit, with the size of each and which of them fit
in the Mac's memory (the largest that fits is the default); `--bits 8` or
`--bits 4` chooses without asking:

```bash
mlx-decision download Cloudflare/clef --bits 8
mlx-decision run -m Cloudflare/clef -s "My card was charged twice." --noul "Is it a billing problem?"
```

The full-size weight files are then fetched one at a time, quantized and
deleted, so the full-size model never has to fit in memory or on disk:
Clef 27B in 8-bit takes about 30 GB on disk, plus one 5 GB file while
the download runs. The result is the same as running
[convert](#quantizing-convert) on a full download. In the Hugging
Face cache it takes the place of the repo's files, so `-m
Cloudflare/clef` loads it without fetching anything. `hf cache ls` and
`hf cache rm` treat it like any download; `hf cache verify` reports it
as changed, since it no longer holds the repo's files.

## Quantizing: `convert`

Laya, Julia and d1 load as released. Clef, pplx, d3 and H2O-Lightning can be
written as a smaller copy:

```bash
# 8-bit: half the memory, answers within the parity tolerance
mlx-decision convert -m Cloudflare/clef-flash -q                 # -> clef-flash-q8

# mixed precision: bits go where the answers are most sensitive
mlx-decision convert -m Cloudflare/clef-flash --target-bits 5    # -> clef-flash-mq5
```

Without `-o` the folder is named after the model plus `-q<bits>`,
`-mq<target>` or `-mlx` (no quantization). A converted folder loads like
the original (`mlx_decision.load("clef-flash-q8")`). Only the text backbone
is quantized; the decision head and the vision tower stay as released.
With `--target-bits` the converter first measures, on a built-in
calibration set, how far the answers move when each block of the model is
quantized alone (about 6 minutes), then gives the sensitive blocks more
bits within the average you asked for.

clef-flash's copies, against the released model:

| Copy | Disk | Peak memory | Changed answers (of 139) |
|---|---|---|---|
| bf16 (as released) | 17.8 GB | 19.0 GB | - |
| 8-bit (`-q`) | 9.1 GB | 11.2 GB | 0 |
| mixed 5 (`--target-bits 5`) | 6.0 GB | 8.1 GB | 0 |
| mixed 4 (`--target-bits 4`) | 4.9 GB | 7.1 GB | 1 |
| uniform 4-bit (`-q --bits 4`) | 4.9 GB | 7.1 GB | 9 |

Use the released model if it fits ([which copy fits](MODELS.md#which-copy-fits)),
otherwise 8-bit, then a 4-bit copy. Quantization saves memory, not time:
quantized copies are 10-20% slower.

pplx takes `-q` (8-bit) and `-q --bits 4`, v1.2 and flash also mixed
precision (flash: the full-size model, 9.1 GB, is loaded to measure it;
the 27B needs a 96 GB Mac for it, and v1.1 is not offered it).

d3 takes all of these, mixed precision included (d3-lite: about a
minute to measure; d3 27B, like Clef 27B, needs a 96 GB Mac for it).
d3-flash's 8-bit copy stays within 0.003 of the release on the questions
checked, its 4-bit copy within 0.037. The smallest models lose more:
d3-edge in uniform 4-bit changed 30 clear answers of 276 against its own
bf16. Use them at full size or 8-bit.

H2O-Lightning takes all of these, mixed precision included (about 4
minutes to measure). Against the release's code on its test set (271
questions), the 8-bit copy changed 6 answers, all near ties (largest
difference 0.035); mixed 4.5 changed 11, 4 of them clear ones (0.16);
uniform 4-bit 19, 12 of them clear (0.33). For this model 4-bit costs
more than for clef-flash: prefer 8-bit, or mixed precision on a small Mac.

`convert` reads one weight file at a time, so a model larger than the
Mac's memory can be converted: Clef 27B to 8-bit takes about 10 s and
9 GB of memory. Mixed precision runs the model to measure it, so making
such a copy needs the full-size model to fit (for Clef 27B, a 96 GB
Mac); the copy then runs on smaller Macs.
[`download --bits`](#getting-a-model-download) stores a uniform copy without the
full-size download.

Ctrl-C stops `convert` at the next tensor; nothing is written. `--events`
prints the progress as JSON lines (stages `load`, `convert` with tensors
as steps, `write` and `done`).

## Measuring speed: `benchmark`

To measure a model on your machine:

```bash
mlx-decision benchmark -m Cloudflare/clef-flash
```

It reports load time, peak memory and median / p95 time per request
across state lengths (those that fit the model's limit) and numbers of
questions; for Clef, H2O-Lightning and d1 also each request again on a
state it has kept. As a rule of thumb, clef-flash reads about 1,700 tokens per
second and Clef 27B (8-bit) about 380; pplx v1.1 (8-bit) about 420;
d3-flash about 1,150 to 1,800 and d3-mini about 1,600 to 2,800, counting
the state once per question;
H2O-Lightning about 2,400 to 3,100 for one question, and more questions
add little since they share the state; d1-3B about 4,400 to 5,200 for
one question, likewise; laya about 25,000,
laya-multilingual about 60,000 and Julia-1 about 100,000, counting the
state once per question. `-o FILE` (`--output`) also
writes the results as JSON, with the Mac's chip, CPU and GPU cores,
memory, macOS and the versions used, so runs on different Macs can be
compared.

Given a folder of models, `benchmark` measures each in turn, each in a
process of its own, and ends with a summary table; folders no family
loads and models that do not fit are listed as skipped. `-o DIR`
then writes one file per model:

```bash
mlx-decision benchmark -m ~/Models -o results/
```
