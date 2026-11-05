"""A Studio on fake models (``fake_backend``) with two sample sets.

``write_studio`` writes the models and sets under a folder and builds the app;
``serve`` runs it on a thread. The browser tests use both; so does
``_helpers/studio_fake.py``, which serves one to drive by hand.

Models: ``fake`` (at most 4 options, 50 input tokens), ``fake-vision`` (reads
images: at most 4, shrunk to 2.1 MP), ``fake-strict`` (needs instructions;
choice and noul only). With ``sizes`` they count made-up sizes against an
8 GB budget, and a fitted ``fake-fit`` and an over-budget ``fake-big`` are
added (the Models page's memory card); with ``slow`` also ``fake-slow`` (1.5 s an answer, other
probabilities; Compare's Stop and its notes where models differ). With
``empty`` it serves no models at all (a first start); with ``convertible``
also ``tiny-clef``, the tiny Clef of ``tests/tiny_models.py``, which
Convert offers; with ``adaptable`` also ``tiny-d3``, the tiny d3 of
``tests/tiny_models.py``, for which New Fit offers LoRA;
``max_body_bytes`` sets the most a request (an upload) holds.
With ``downloads`` it lists downloaded models too (Delete for them): the
Hugging Face cache this process uses (``huggingface_hub.constants.HF_HUB_CACHE``,
which the caller points at a scratch folder) gains ``org/fake-cached`` (two
revisions), and ``home/models-dir``, the config's ``models_dir``, holds
``fake-local`` (a folder), ``fake-linked`` (a link to a folder elsewhere) and
``fake-named`` (named in the config's ``models``, so it stays).

Jobs run ``tests/fake_job.py`` in the CLI's place (``fake_jobs`` sets how
many steps and how fast); ``data/tickets.jsonl`` is there for a fit.
"""

import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from fake_backend import write_cached, write_fitted, write_model
from mlx_decision.config import Config
from mlx_decision.memory import ModelTooLargeError
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio, server
from mlx_decision.studio.catalog import Catalog, studio_models

GB = 10**9
FAKE_JOB = Path(__file__).resolve().parents[1] / "fake_job.py"
SIZES = {
    "fake": 1.8 * GB,
    "fake-fit": 1.85 * GB,
    "fake-vision": 3.2 * GB,
    "fake-strict": 4.6 * GB,
    "fake-big": 19 * GB,
    "fake-slow": 2.5 * GB,
}
TICKET_TRIAGE = {
    "state": "Hi, I've been trying to connect my Stripe account for 3 days and the "
    "integration keeps failing. I'm losing sales. Please help ASAP.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    },
}
ORDER_JSON = {
    "state": {"order": 1234, "items": ["mug", "plate"], "total": 31.5},
    "questions": {
        "refund": {
            "type": "noul",
            "criteria": {"true": "the customer asks for money back", "false": "anything else"},
        }
    },
}


class SizedPool(ModelPool):
    """Made-up sizes, so the memory card shows something like a real Mac."""

    def size(self, name):
        if name not in SIZES:
            return super().size(name)
        if SIZES[name] > self.budget:
            raise ModelTooLargeError(
                f"{name} needs more than the budget", int(SIZES[name]), self.budget
            )
        return int(SIZES[name])


def write_studio(
    home: Path,
    port: int,
    sizes: bool = False,
    slow: bool = False,
    dev: bool = False,
    empty: bool = False,
    convertible: bool = False,
    max_body_bytes: int | None = None,
    downloads: bool = False,
    adaptable: bool = False,
) -> FastAPI:
    """Write the fake models under ``home/fake-models`` and the sample sets in
    ``home/ws/sets`` (kept when there), and build the Studio for ``port``."""
    models = home / "fake-models"
    write_model(models / "fake", max_choice_options=4, max_input_tokens=50)
    write_model(
        models / "fake-vision",
        supports_images=True,
        open_images=True,
        max_image_pixels=2**21,
        max_images=4,
    )
    write_model(
        models / "fake-strict", requires_instructions=True, question_types=["choice", "noul"]
    )
    if slow:
        fixed = {
            "department": {"billing": 0.2, "technical": 0.7, "sales": 0.1},
            "frustration": {"0": 0.1, "1": 0.3, "2": 0.6},
            "is_urgent": {"true": 0.4, "false": 0.6},
        }
        write_model(models / "fake-slow", delay=1.5, fixed=fixed)
    sets = home / "ws" / "sets"
    sets.mkdir(parents=True, exist_ok=True)
    if not (sets / "ticket-triage.json").exists():
        (sets / "ticket-triage.json").write_text(json.dumps(TICKET_TRIAGE, indent=2))
        (sets / "order-json.json").write_text(json.dumps(ORDER_JSON, indent=2))
    if convertible:  # a real model that Convert offers
        from tiny_models import write_clef

        if not (models / "tiny-clef").exists():
            write_clef(models / "tiny-clef")
    if adaptable:  # a real model whose family takes a lora fit
        from tiny_models import write_d3

        if not (models / "tiny-d3").exists():
            write_d3(models / "tiny-d3")
    if empty:  # a first start: no models, a budget like a 16 GB Mac's
        pool = ModelPool([], budget=int(12.7 * GB))
    elif sizes:
        write_fitted(models / "fake-fit", models / "fake")
        write_model(models / "fake-big")
        pool = SizedPool(discover([models]), budget=8 * GB)
    else:
        pool = ModelPool(discover([models]), budget=GB)
    models_dir, named = home / "models-dir", ()
    if downloads:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
        if not (cache / "models--org--fake-cached").exists():
            write_cached(cache, "org/fake-cached", "b" * 40, main=False, weights=4096)
            write_cached(cache, "org/fake-cached", "a" * 40)
        write_model(models_dir / "fake-local")
        write_model(models_dir / "fake-named")
        if not (models_dir / "fake-linked").is_symlink():
            (models_dir / "fake-linked").symlink_to(write_model(home / "elsewhere" / "linked"))
        named = (str(models_dir / "fake-named"),)
        # Named as web names them: the cached repo after its repo id.
        found = studio_models([str(models_dir)], None, include_cache=True)
        pool = ModelPool([*pool.specs.values(), *found], budget=pool.budget)
    data = home / "ws" / "data"
    data.mkdir(parents=True, exist_ok=True)
    line = {**TICKET_TRIAGE, "labels": {"department": "technical", "is_urgent": "true"}}
    (data / "tickets.jsonl").write_text(json.dumps(line) + "\n")
    # Its own config file (none), never the one in the user's home.
    config = Config(home / "config.toml")
    # Where the models come from, as web passes it (Where models live, rescans).
    catalog = Catalog([] if empty else [str(models)], home / "ws" / "models", include_cache=False)
    if downloads:  # as web without -m: models_dir scanned, the cache too
        config = Config(home / "config.toml", models_dir=models_dir, models=named)
        catalog = Catalog([str(models), *named], home / "ws" / "models", models_dir=models_dir)
    app = create_studio(
        pool,
        port=port,
        workspace=home / "ws",
        config=config,
        load_default=False,
        dev=dev,
        catalog=catalog,
        **({"max_body_bytes": max_body_bytes} if max_body_bytes else {}),
    )
    fake_jobs(app)
    return app


def fake_jobs(app: FastAPI, steps: int = 200, every: float = 0.1, fits: bool = False) -> None:
    """Run ``tests/fake_job.py`` for the Studio's jobs: ``steps`` work events
    ``every`` seconds apart (20 s by default); Stop takes effect at the next.
    With ``fits``, a fit job acts as ``training fit`` does (its events, then a
    fitted model with its report)."""
    jobs = app.state.jobs
    jobs.command = [sys.executable, str(FAKE_JOB), "--steps", str(steps), "--every", str(every)]
    if fits:
        jobs.command.append("--fit-events")
    jobs.stop_grace = jobs.kill_grace = 2.0


@dataclass
class Served:
    app: FastAPI
    base: str  # http://127.0.0.1:PORT
    url: str  # the address with the session token, which logs a tab in
    home: Path

    @property
    def workspace(self) -> Path:
        return self.home / "ws"

    @property
    def sets(self) -> Path:
        return self.home / "ws" / "sets"

    @property
    def history(self) -> Path:
        return self.home / "ws" / "history"

    def set_file(self, name: str) -> dict:
        return json.loads((self.sets / f"{name}.json").read_text())

    def history_lines(self, name: str) -> list[dict]:
        path = self.history / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def serve(home: Path, port: int = 0, **options):
    """Serve a fake Studio on a free port (or ``port``: a Studio started again
    where one ran), on a thread; a context manager that yields ``Served`` and
    stops the server on exit."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen()
    port = sock.getsockname()[1]
    app = write_studio(home, port, **options)
    studio = server(app, host="127.0.0.1", port=port, log_level="warning")
    thread = threading.Thread(target=studio.run, kwargs={"sockets": [sock]}, daemon=True)

    class Running:
        def __enter__(self) -> Served:
            thread.start()
            deadline = time.monotonic() + 10
            while not studio.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("the fake Studio did not start")
                time.sleep(0.02)
            base = f"http://127.0.0.1:{port}"
            return Served(app, base, f"{base}/?token={app.state.session.token}", home)

        def __exit__(self, *exc):
            studio.should_exit = True
            thread.join(10)
            sock.close()

    return Running()
