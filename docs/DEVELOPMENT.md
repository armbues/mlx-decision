# Development

From source: clone the [repository](https://github.com/armbues/mlx-decision)
and run `pip install -e ".[server,images,mcp]"` in it.

```bash
make setup   # editable install with dev tools
make test       # unit tests; parity tests run when the weights are found
make test-slow  # Clef 27B 8-bit, d3-flash, H2O-Lightning and d1-3B against their references (about 21 min)
make test-browser  # the Studio in Chrome (Playwright drives the installed Chrome; skipped without it)
make lint
```

Tests that need weights look for them in `$MLX_DECISION_MODELS/<name>`
(`clef-flash`, `clef-8bit`, `laya`, `laya-typed-decisions`,
`laya-multilingual`, `Julia-1`), then in the Hugging Face cache, and are
skipped otherwise. The
parity fixtures are rebuilt with `scripts/make_parity_set.py` and
`scripts/make_parity_reference.py` (runs Cloudflare's reference with
PyTorch), and for Laya and Julia with `scripts/make_marker_parity_set.py`
and `scripts/make_marker_reference.py` (runs the `laya` package or Julia's
code from its release folder). pplx's fixtures come from
`scripts/make_pplx_reference.py` (v1.1) and
`scripts/make_pplx_joint_reference.py` (v1.2 and flash), which run the
release's code on small random models with the release's tokenizer; its tests need only the
release's small files in the Hugging Face cache. H2O-Lightning's comes
from `scripts/make_h2o_reference.py` in the same way (the release's shim,
with transformers in place of vLLM); with `--model` it runs the real
weights in bf16 for the slow test, which takes the release from
`$MLX_DECISION_MODELS/h2o-lightning-4b`. d1-3B's come from
`scripts/make_d1_reference.py` (its release's code, which runs only
there; `--model` for the slow test, from `$MLX_DECISION_MODELS/d1-3B`).
d3's comes from `scripts/make_d3_reference.py`, which runs the release's
runtime on a small random model with the release's files; with `--model`
it runs d3-flash's weights in bf16 for the slow test, which takes the
release from `$MLX_DECISION_MODELS/d3-flash`.
