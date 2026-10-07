"""Write a folder that runs Clef parity on another Mac, without this repository.

usage: python scripts/clef_parity_bundle.py --out DIR [--model Cloudflare/clef]

The folder holds ``run_parity.py`` (``scripts/clef_parity_runner.py``), the
parity set (``requests.json``), the images it uses and ``README.txt`` with
the steps. Run there, it writes one results file with Cloudflare's reference
and mlx-decision side by side, for models too large for this Mac.
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
README = """\
Clef parity bundle: Cloudflare's PyTorch reference and mlx-decision, both in bf16,
on the parity set of mlx-decision ({count} requests, {images} with images).

1. A Mac with enough memory: the reference peaks at about 1.35x the weights
   (Clef 27B: about 75 GB; it runs first, mlx-decision after it, never together).
2. A Python 3.12 environment, then:

     pip install "mlx-decision[images]=={version}" torch torchvision \\
         "transformers>=5.18" safetensors

3. In this folder:

     python run_parity.py --model {model} --out {result}

   It downloads the model (all of the repository, including Cloudflare's
   reference code) unless --model is a local folder. Progress is printed per
   request. If it stops, run the same command again: finished requests are
   kept in {stem}.reference.json and {stem}.mlx.json.

4. Send back {result} (the two stage files are not needed).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Cloudflare/clef")
    parser.add_argument("--version", default="0.5.0", help="mlx-decision to install there")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output folder is not empty: {args.out}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    cases = json.loads((PARITY / "requests.json").read_text())
    files = {
        spec if isinstance(spec, str) else spec["file"]
        for case in cases
        for spec in case.get("images") or []
    }
    for name in sorted(files) + ["SOURCES.md"]:
        shutil.copy2(PARITY / "images" / name, args.out / "images" / name)
    shutil.copy2(PARITY / "requests.json", args.out / "requests.json")
    shutil.copy2(ROOT / "scripts" / "clef_parity_runner.py", args.out / "run_parity.py")
    stem = args.model.rstrip("/").rsplit("/", 1)[-1] + "-parity"
    (args.out / "README.txt").write_text(
        README.format(
            count=len(cases),
            images=sum(1 for case in cases if case.get("images")),
            version=args.version,
            model=args.model,
            result=f"{stem}.json",
            stem=stem,
        )
    )
    print(f"wrote {args.out}: {len(cases)} requests, {len(files)} images")


if __name__ == "__main__":
    main()
