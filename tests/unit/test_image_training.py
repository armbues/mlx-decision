"""The fit kinds and ``evaluate`` on image data, on the tiny models of the
families that read images; the head fit's estimate counting image tokens."""

import json

import pytest
from test_image_data import COLOURS, PET, image_folder, png

import mlx_decision
from mlx_decision.data.dataset import read_for
from mlx_decision.training import evaluate, fit, plan
from tiny_models import write_baselabs, write_clef, write_d1lfm3, write_d3

WRITERS = {
    "d3": write_d3,
    "d1lfm3": write_d1lfm3,
    "baselabs": write_baselabs,
    "clef": lambda folder: write_clef(folder, vision=True),
}
MAIN = {"d3": "readout", "d1lfm3": "readout", "baselabs": "readout", "clef": "head"}


@pytest.fixture
def short_training(monkeypatch):
    monkeypatch.setattr("mlx_decision.training.readout.STEPS", 50)


@pytest.mark.parametrize("family", list(WRITERS))
def test_each_kind_fits_and_evaluates_on_images(tmp_path, family, short_training):
    base = WRITERS[family](tmp_path / f"tiny-{family}")
    folder = image_folder(tmp_path / "pets", 12)
    for method, kinds in (("auto", [MAIN[family], "calibrate"]), ("calibrate", ["calibrate"])):
        result = fit(base, folder, output=tmp_path / method, method=method, epochs=1)
        assert result.kinds == kinds
        assert result.report["data"]["examples"] == 12
        assert result.metrics["pet"]["after"]["accuracy"] is not None
    evaluated = evaluate(tmp_path / "auto", folder, compare=base).as_json()
    assert evaluated["data"]["examples"] == 12
    assert [m["model"] for m in evaluated["models"]] == [str(tmp_path / "auto"), str(base)]
    for scored in evaluated["models"]:
        assert scored["questions"]["pet"]["accuracy"] is not None
    # The fitted model reads the picture: red and blue answer differently.
    model = mlx_decision.load(tmp_path / "auto")
    red, blue = (
        model.decide("a picture", {"pet": PET}, images=[str(folder / "images" / name)])
        .answers["pet"]
        .probabilities
        for name in ("0.png", "1.png")
    )
    assert red != blue


def test_the_head_estimate_counts_image_tokens(tmp_path):
    base = write_clef(tmp_path / "tiny-clef", vision=True)
    (tmp_path / "small.png").write_bytes(png(COLOURS["red"], (16, 16)))
    (tmp_path / "large.png").write_bytes(png(COLOURS["blue"], (640, 480)))
    rows = [
        {"id": "text", "state": "a picture", "questions": {"pet": PET}, "labels": {"pet": "red"}}
    ]
    rows += [{**rows[0], "id": name, "images": [f"{name}.png"]} for name in ("small", "large")]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    examples = read_for(path).examples
    guessed = plan.head_tokens(examples, base, {})
    assert guessed[2] > guessed[1] > guessed[0]

    model = mlx_decision.load(base)
    backend = model.backend
    norm = backend.backbone.language_model.model.norm.weight
    stored = [
        backend.head_size(model.check(example.request)) // (norm.size * norm.itemsize)
        for example in examples
    ]
    # The text is guessed from its length; the images' tokens are exact.
    assert [g - guessed[0] for g in guessed] == [s - stored[0] for s in stored]
    # A smaller image limit, as the loader would take it, counts fewer.
    capped = plan.head_tokens(examples, base, {"max_image_pixels": 1024})
    assert capped[2] < guessed[2]


def test_request_tokens_leave_data_urls_out_of_the_text():
    from test_image_data import data_url, line

    from mlx_decision.data.format import parse_examples

    url = data_url(png(size=(64, 64)))
    plain, given, inside = parse_examples([
        line(),
        line(images=[url]),
        line(state={"text": "a picture", "photo": url}),
    ])  # fmt: skip
    text = plan.request_tokens(plain)
    assert plan.request_tokens(given) == text + plan.IMAGE_TOKENS
    assert abs(plan.request_tokens(inside) - text - plan.IMAGE_TOKENS) < 10
