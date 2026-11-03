# Python

The Python API: `load` a model, then `decide`. Fitting a model from
Python is in [TRAINING.md](TRAINING.md#python).

- [Deciding](#deciding)
- [Results and errors](#results-and-errors)
- [Loading options](#loading-options)
- [Images](#images)

## Deciding

```python
import mlx_decision
from mlx_decision import Choice, Noul, Score

model = mlx_decision.load("Cloudflare/clef-flash")

ticket = (
    "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP."
)
result = model.decide(
    ticket,
    {
        "department": Choice(
            instructions="Which team should handle this",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": Score(
            instructions="How frustrated the customer appears",
            criteria=[
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        ),
        "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity"),
    },
)

print(result.answers["department"].choice)      # technical
print(result.answers["department"].confidence)  # 0.90
print(result.answers["frustration"].score)      # 1.29 (expected level, 0-based)
print(result.answers["is_urgent"].noul)         # 0.94 (probability of yes)
print(result.to_wire())                         # the response body, as JSON data
```

`confidence` follows the definitions of the Jev API (choice: how far the top
probability sits above an even split; score: how concentrated the
probability is around the most likely level), whatever the model's own
definition.

Questions can also be plain dicts in the API's wire format
(`{"type": "choice", "instructions": ..., "criteria": {...}}`), and
`model.decide_request(body)` takes a whole request body;
`model.check(body)` raises the errors answering it would, without
answering (it does not read images or count tokens). Invalid requests
raise `mlx_decision.DecisionError`, whose `param` names the offending field
(building an invalid `Choice`, `Score` or `Noul` object directly raises
pydantic's `ValidationError` instead). The state, instructions and criteria
take any JSON data nested up to 100 levels deep, with finite numbers
(no `NaN` or infinity).

## Results and errors

`decide` and `decide_request` return a `Result`: `model` (the model's
name, as `model.name`), `answers` (by question id), `usage`
(`input_tokens`; `output_tokens` is always 0, as nothing is generated)
and `truncated` (whether the state was cut to fit the model's input
limit). `to_wire()` gives the response body as JSON data, the
`Response` the server sends. Each answer has its question's `type`:

- `ChoiceAnswer`: `choice` (the most likely option), `confidence` and
  `probabilities` (option -> probability);
- `ScoreAnswer`: `score` (the expected level, counted from 0),
  `confidence`, `legend` (level -> its text, keys `"0"`, `"1"`, ...) and
  `probabilities` (level -> probability);
- `NoulAnswer`: `noul` (the probability of yes).

These types, `Usage`, `Request` (what `check` returns) and `Response`
can be imported from `mlx_decision`.

An invalid request raises `DecisionError` (a `ValueError`) with
`message`, `param` (the field at fault, such as
`questions.team.criteria`, or `None`) and `code` (only for some errors,
such as `unsupported_media`). `load` raises `FileNotFoundError` for a
path that is neither a folder nor a model on the Hugging Face Hub,
`ValueError` for an option the model's family does not take, and the
errors below for memory and platform.

## Loading options

`load` takes options per model: `max_input_tokens` changes the input limit
(Laya and Julia up to 8,192, d1-2B up to 128,000 per set of questions,
pplx v1.2, d3 and Base Labs up to 262,144; pplx v1.1, H2O-Lightning and
d1-3B only lower),
`strict_encoding=False` lets Julia cut what it would otherwise refuse,
`dtype` sets Laya's and Julia's precision (default float16),
`max_image_pixels` caps image size for Clef, pplx, d3, Base Labs and d1
LFM3 (in pixels, at least 1,024; `None` keeps only the release's own
maximum), `noul_floor` sets H2O-Lightning's
[yes/no floor](MODELS.md#h2o-lightning), `reading` sets how
H2O-Lightning from v1.4 reads a prompt (`"two-way"`, the release's
default, or `"one-way"`, faster for many questions about one long state;
a v1.2.3 folder reads one-way only), and `prefix_cache_gb` sets how much
memory Clef, Base Labs, H2O-Lightning and d1 keep for
[recent states](MODELS.md#reusing-a-state) (default 2, 0 turns it off).
`vision=True` loads the vision tower at once rather than with the first
image request (Clef, pplx, d3, Base Labs, H2O-Lightning, d1 LFM3).
`check_memory=False` (by keyword) skips the memory check; a model that
does not fit raises `mlx_decision.ModelTooLargeError` otherwise, a
`DecisionError` whose `required` and `available` are in bytes. Without Apple Silicon (no Metal GPU), loading raises
`mlx_decision.PlatformError`.
`model.info()` describes the loaded model as JSON data: family,
precision, H2O-Lightning's `reading` (`None` for other families), input
limit, option limits and whether it takes images.

## Images

Clef, pplx, d3, Base Labs, H2O-Lightning and d1 LFM3 take images as
file paths, bytes, PIL images or data URLs:

```python
result = model.decide(
    "Customer says checkout fails; screenshot attached.",
    {"shows_error": Noul(instructions="The screenshot shows an error message")},
    images=["screenshot.png"],
)
```

Images can also come inside the state, as H2O-Lightning's release takes
them: a string value that is a data URL (`data:image/png;base64,...`) or
bare base64 of an image, or a chat-style image part such as
`{"type": "image_url", "image_url": {"url": "data:..."}}`, anywhere in a
string, object or list state. Each is taken out of the state, replaced
by `[image N]` (numbered after the images in `images`) and read as if it
had been given in `images`; errors name its place in the state
(`state.photos.1`). Image URLs are refused, not fetched. This works the
same in `run`, `server` and `mcp`; models without images read such a
state as text.
