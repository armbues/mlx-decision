"""The ``data`` and ``training fit`` menus in a real terminal (a pty, rendered by pyte)."""

import shlex
import subprocess
import sys

from test_readout_fit import labelled

from mlx_decision.data.format import write_examples
from terminal import DOWN, ENTER, Terminal
from tiny_models import write_d1


def test_data_menus_end_with_a_command_that_writes_the_same_file(tmp_path):
    (tmp_path / "tickets.csv").write_text(
        "text,label\nrefund please,billing\napp crashes,tech\nprice?,sales\nrefund,billing\n"
    )
    term = Terminal(["data", "tickets.csv"], tmp_path)
    for prompt in ("Shape:", "Text column:", "Label column:", "id:"):
        term.expect(prompt)
        term.send(ENTER)
    term.expect("instructions:")
    term.send("Which team handles it?" + ENTER)
    term.expect("blank line ends")
    term.send(ENTER)
    term.expect("Write to:")
    term.send(ENTER)
    screen = term.expect("Write 4 examples to tickets.jsonl?")
    assert "labels: tickets: billing 2, tech 1, sales 1" in screen
    term.send("y")
    assert term.wait() == 0
    screen = term.text()
    assert "wrote tickets.jsonl" in screen
    command = screen.split("The same without menus:")[1].strip().splitlines()[0]
    words = shlex.split(command)
    assert words[:2] == ["mlx-decision", "data"] and "--question" in words
    words[words.index("-o") + 1] = "again.jsonl"
    rerun = subprocess.run(
        [sys.executable, "-m", "mlx_decision", *words[1:]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert rerun.returncode == 0, rerun.stderr
    assert (tmp_path / "again.jsonl").read_bytes() == (tmp_path / "tickets.jsonl").read_bytes()


def test_fit_menus_offer_the_kinds_with_their_memory_and_run(tmp_path):
    write_d1(tmp_path / "tiny-d1")
    write_examples(tmp_path / "tickets.jsonl", labelled(80))
    term = Terminal(["training", "fit", "-m", "tiny-d1", "--data", "tickets.jsonl"], tmp_path)
    screen = term.expect("Method:")
    term.expect("calibrate")
    screen = term.text()
    assert "readout, then calibrate  [auto]" in screen and "peak" in screen
    assert "M-series" not in screen
    term.send(ENTER)
    term.expect("Report on:")
    term.send(ENTER)
    term.expect("Write the fitted model to:")
    term.send(ENTER)
    screen = term.expect("Start?")
    assert "mlx-decision training fit -m tiny-d1 --data tickets.jsonl -o tiny-d1-tickets " \
           "--method readout" in screen  # fmt: skip
    term.send("y")
    assert term.wait(300) == 0, term.text()
    assert "wrote tiny-d1-tickets" in term.text()
    assert (tmp_path / "tiny-d1-tickets" / "mlx_decision.json").exists()


def test_no_input_never_asks(tmp_path):
    (tmp_path / "tickets.csv").write_text("text,label\nrefund,billing\ncrash,tech\n")
    term = Terminal(["data", "tickets.csv", "--no-input", "-o", "t.jsonl"], tmp_path)
    assert term.wait() == 1
    assert "the question's wording is open: pass --question TEXT" in term.text()


ESCAPE = "\x1b"


def test_download_menu_goes_from_family_to_model_and_back(tmp_path, monkeypatch):
    # Offline: the walk ends at the Hub check of the model picked, naming it.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    term = Terminal(["download", "--local-dir", str(tmp_path / "models")], tmp_path)
    screen = term.expect("Other...")
    assert "d3             vLLM Semantic Router" in screen and "6 models, 1.2-52 GB" in screen
    term.send(DOWN * 2 + ENTER)  # d3
    screen = term.expect("Back")
    assert "d3 model:" in screen and "vllm-sr/d3-edge" in screen
    term.send(ESCAPE, pause=1)  # back to the families, d3 still selected
    assert term.text().count("Model family:") == 2
    term.send(ENTER)
    assert term.text().count("d3 model:") == 2
    term.send(DOWN * 5 + ENTER)  # d3-edge
    assert term.wait() == 1
    assert "error: vllm-sr/d3-edge: " in term.text()


def test_escape_in_the_family_menu_cancels(tmp_path):
    term = Terminal(["download", "--local-dir", str(tmp_path / "models")], tmp_path)
    term.expect("Other...")
    term.send(ESCAPE, pause=1)
    assert term.wait() == 1
    assert "error" not in term.text()
