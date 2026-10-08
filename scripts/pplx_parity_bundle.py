"""Write a folder that runs pplx-decider parity on another Mac, without this repository.

usage: python scripts/pplx_parity_bundle.py --out DIR [--model REPO_OR_FOLDER] [--wheel FILE]

The folder holds ``run_parity.py`` (``scripts/pplx_parity_runner.py``), the
parity set (``requests.json``: tests/parity/requests.json plus
tests/parity/marker/requests.json), the images it uses, a wheel of
mlx-decision (built from this tree unless ``--wheel`` names one) and
``README.txt`` with the steps. Run there, it writes one results file with the
release's own code and mlx-decision side by side, for a model too large for
this Mac. Building the wheel needs ``build`` (installed by ``make setup``).
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "tests" / "parity"
# The versions the committed tiny-model reference was made with; the release
# itself pins transformers 5.17.0, whose tokenizer gives the same token ids.
REVISION = "3b45dead91dfa6d95aad6b95764a606fab2bf7a6"  # as make_pplx_reference.py
REQUIREMENTS = 'torch==2.14.1 torchvision==0.29.1 "transformers==5.18.0" safetensors'
README = """\
pplx-decider parity bundle: the release's own code (its DecisionModel, from the
repository's source/src) and mlx-decision, both in bf16, on the parity set of
mlx-decision ({count} requests, {questions} questions, {images} requests with
images). Every question is one pass on both sides.

1. A Mac with enough memory: the reference holds the bf16 weights (52 GB) plus
   its activations, about 70 GB at the peak; it runs first, mlx-decision after
   it, never together. Expect one to two hours for the reference and about 20
   minutes for mlx-decision on an M2 Ultra.
2. A Python 3.12 environment, then in this folder:

     pip install "./{wheel}[images]" {requirements}

   The wheel is mlx-decision {version} as built for this check (pplx support
   is not on PyPI yet).

3. In this folder:

     python run_parity.py --model {model}{revision} --out {result}

   It downloads the model (the whole repository at the commit mlx-decision's
   fixtures were made from, about 53 GB, including the release's code) unless
   --model is a local folder of the release with its source/ folder.
   Progress is printed per request. If it stops, run the same command
   again: finished requests are kept in {stem}.reference.json and
   {stem}.mlx.json. transformers warns that flash-linear-attention is not
   installed: expected on a Mac (it needs CUDA); its fallback gives the same
   results, only slower.

4. Send back {result} (the two stage files are not needed).
"""


def build_wheel(folder: Path) -> Path:
    with tempfile.TemporaryDirectory() as temporary:
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", temporary, str(ROOT)]
        subprocess.run(command, check=True, capture_output=True)
        (wheel,) = Path(temporary).glob("*.whl")
        return Path(shutil.copy2(wheel, folder / wheel.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="perplexity-ai/pplx-decider-v1.1-27b")
    parser.add_argument("--wheel", type=Path, help="mlx-decision wheel (default: build one)")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output folder is not empty: {args.out}")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    cases = json.loads((PARITY / "requests.json").read_text())
    cases += json.loads((PARITY / "marker" / "requests.json").read_text())
    if len({case["id"] for case in cases}) != len(cases):
        raise SystemExit("the two request sets share an id")
    files = {
        spec if isinstance(spec, str) else spec["file"]
        for case in cases
        for spec in case.get("images") or []
    }
    for name in sorted(files) + ["SOURCES.md"]:
        shutil.copy2(PARITY / "images" / name, args.out / "images" / name)
    (args.out / "requests.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "scripts" / "pplx_parity_runner.py", args.out / "run_parity.py")
    if args.wheel:
        wheel = Path(shutil.copy2(args.wheel, args.out / args.wheel.name))
    else:
        wheel = build_wheel(args.out)
    version = wheel.name.split("-")[1]
    stem = args.model.rstrip("/").rsplit("/", 1)[-1] + "-parity"
    (args.out / "README.txt").write_text(
        README.format(
            count=len(cases),
            questions=sum(len(case["questions"]) for case in cases),
            images=sum(1 for case in cases if case.get("images")),
            wheel=wheel.name,
            requirements=REQUIREMENTS,
            version=version,
            model=args.model,
            revision="" if Path(args.model).is_dir() else f" \\\n         --revision {REVISION}",
            result=f"{stem}.json",
            stem=stem,
        )
    )
    print(f"wrote {args.out}: {len(cases)} requests, {len(files)} images, {wheel.name}")


if __name__ == "__main__":
    main()
