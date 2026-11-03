# Development

From source: clone the [repository](https://github.com/armbues/mlx-decision)
and run `pip install -e ".[dev]"` in it (what `make setup` does: the
package with every extra and the tools for tests, lint and builds).

```bash
make setup         # editable install with dev tools
make test          # unit tests; parity tests run when the weights are found
make test-slow     # five large models against their references (about 25 min)
make test-browser  # the Studio in Chrome (Playwright drives the installed Chrome; skipped without it)
make lint          # ruff check and format check
make build         # sdist and wheel in dist/, checked with twine
make clean         # caches and build output
```

Tests that need weights look for them in `$MLX_DECISION_MODELS/<name>`
and are skipped when they are missing:

- `clef-flash`, `laya`, `laya-typed-decisions`, `laya-multilingual` and
  `Julia-1` (the parity tests), also found in the Hugging Face cache by
  repo id;
- `clef` (only its `tokenizer.json`, for Clef 27B's token test), also
  found as `Cloudflare/clef`'s small files in the cache;
- for `make test-slow`, in `$MLX_DECISION_MODELS` only: `clef-8bit`
  (Clef 27B 8-bit, about 8 min), `d3-flash` (about 9 min),
  `pplx-decider-flash-v1.2` (about 4 min), `h2o-lightning-4b` (about
  2 min) and `d1-3B` (about 1.5 min), each against its release's
  reference in bf16.

The token tests of pplx, d3, Base Labs, H2O-Lightning and d1 need only their
release's small files (tokenizer and configs) in the Hugging Face cache.
The LFM2 backbone (d1's) is compared with `mlx-lm`'s when it is
installed (`pip install mlx-lm`; not a dependency).

The parity fixtures are rebuilt with `scripts/make_parity_set.py` and
`scripts/make_parity_reference.py` (runs Cloudflare's reference with
PyTorch), and for Laya and Julia with `scripts/make_marker_parity_set.py`
and `scripts/make_marker_reference.py` (runs the `laya` package or Julia's
code from its release folder). pplx's fixtures come from
`scripts/make_pplx_reference.py` (v1.1) and
`scripts/make_pplx_joint_reference.py` (v1.2 and flash), which run the
release's code on small random models with the release's tokenizer; its tests need only the
release's small files in the Hugging Face cache. H2O-Lightning's comes
from `scripts/make_h2o_reference.py` in the same way (the release's own
shim, with transformers in place of vLLM, in both reading modes;
`--revision v1.2.3` for the older release's fixture); with `--model` it
runs the real weights in bf16 for the slow test, which takes the release
from `$MLX_DECISION_MODELS/h2o-lightning-4b`. d1-3B's come from
`scripts/make_d1_reference.py` (its release's code, which runs only
there; `--model` for the slow test, from `$MLX_DECISION_MODELS/d1-3B`).
d3's comes from `scripts/make_d3_reference.py`, which runs the release's
runtime on a small random model with the release's files; with `--model`
it runs d3-flash's weights in bf16 for the slow test, which takes the
release from `$MLX_DECISION_MODELS/d3-flash`. Base Labs' comes from
`scripts/make_baselabs_reference.py`, which runs the releases' engine
(OneJev's `qev`, from a checkout at the pinned commit with the releases'
`serve.patch` applied: `--onejev DIR`; the script checks both) on small
random models in the three releases' shapes; with `--model` it runs a
release's weights in bf16 for the slow test (elephant-1 or joey-1, told
apart by their `config.json`), which takes the release from
`$MLX_DECISION_MODELS/elephant-1` and `$MLX_DECISION_MODELS/joey-1`.

The reference scripts run the families' own code, so they need packages
that `make setup` does not install: `torch` and `transformers` for all
of them, `torchvision` and `Pillow` for Clef's image requests,
`accelerate` for d3, and the `laya` package for Laya (Julia's code is
imported from its release folder).
