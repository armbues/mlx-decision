"""Templates: question sets that ship with the Studio, each showing what
decision models can do (the question types, many options, entailment,
languages, a JSON state, images).

Each ``<name>.json`` here holds ``title``, ``description`` and ``set``: the
set's ``questions`` and its named ``examples``. An example's ``images`` name
PNG files in this folder; a set made from a template carries them as data
URLs, like any set saved in the Studio.
"""

import base64
import json
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent

# Question types in the order the Studio lists them.
TYPES = ("choice", "score", "noul")

# The templates, in the order the New sheet lists them.
ORDER = (
    "support-triage",
    "review-rating",
    "content-moderation",
    "fact-check",
    "intent-routing",
    "many-languages",
    "conversation",
    "screenshot-check",
)


class UnknownTemplate(KeyError):
    pass


@cache
def _read(name: str) -> dict[str, Any]:
    if name not in ORDER:
        raise UnknownTemplate(name)
    return json.loads((HERE / f"{name}.json").read_text())


def list_templates() -> list[dict[str, Any]]:
    """Each template's name, title, description, question types (choice,
    score, noul, those it uses), and its numbers of questions and examples;
    ``images`` when its examples have images."""
    entries = []
    for name in ORDER:
        template = _read(name)
        questions = template["set"]["questions"]
        examples = template["set"]["examples"]
        used = {question["type"] for question in questions.values()}
        types = [kind for kind in TYPES if kind in used]
        entries.append(
            {
                "name": name,
                "title": template["title"],
                "description": template["description"],
                "types": types,
                "questions": len(questions),
                "examples": len(examples),
                "images": any(example.get("images") for example in examples),
            }
        )
    return entries


def template_set(name: str) -> dict[str, Any]:
    """The template ``name`` as a set file, laid out as the Studio saves one:
    the first example's state and images at the top level, then every
    example. ``UnknownTemplate`` for a name not in ``ORDER``."""
    template = _read(name)
    examples = []
    for example in template["set"]["examples"]:
        item = {"name": example["name"], "state": example["state"]}
        if example.get("images"):
            item["images"] = [_data_url(file) for file in example["images"]]
        examples.append(item)
    first = examples[0]
    body: dict[str, Any] = {"state": first["state"], "questions": template["set"]["questions"]}
    if "images" in first:
        body["images"] = first["images"]
    body["examples"] = examples
    return json.loads(json.dumps(body))  # a copy the caller may change


def _data_url(file: str) -> str:
    data = (HERE / file).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()
