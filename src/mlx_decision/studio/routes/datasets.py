"""Public datasets: the ``data`` command's presets, for "Get a Dataset…" (TRS-6).

``GET /studio/api/datasets`` lists them: each with its title and what it
asks, the dataset on the Hub (repo, config, the pinned commit), its
question as the training data will have it (built by the preset's own
code) and the splits it can be read from with their rows, the coverage
suite's evaluation split marked. Getting one is a ``data`` job
(``POST /studio/api/jobs``).
"""

from fastapi import APIRouter

from ...data.presets import PRESETS, SPLITS, describe

# How the sheet names each preset, and what it is for.
TITLES = {
    "trec": ("TREC", "Question classification: which kind of answer a question asks for (6 types)"),
    "offensive": ("TweetEval Offensive", "Whether a tweet is offensive (yes or no)"),
    "sst5": ("SST-5", "Movie review sentiment on five levels, very negative to very positive"),
    "openbookqa": ("OpenBookQA", "Elementary science questions with four answers to pick from"),
    "anli-r3": (
        "ANLI round 3",
        "Whether a premise entails, contradicts or is neutral to a hypothesis",
    ),
}
# The split a training file is read from by default.
TRAIN_SPLITS = {"anli-r3": "train_r3"}


def datasets_router() -> APIRouter:
    router = APIRouter(prefix="/studio/api/datasets")

    @router.get("")
    async def presets():
        """``{presets}``: ``name``, ``title``, ``summary``, ``repo``, ``config``,
        ``revision``, ``question`` (``id``, ``type``, ``instructions``,
        ``options`` as ``[id, description]``), ``splits`` (``name``, ``rows``,
        ``evaluation``) and ``default_split``."""
        listed = []
        for name in ("trec", "offensive", "sst5", "openbookqa", "anli-r3"):
            preset = PRESETS[name]
            question = describe(preset)
            criteria = question.get("criteria")
            if isinstance(criteria, dict):
                options = [[option, text] for option, text in criteria.items()]
            else:  # a score's levels, lowest first
                options = [[str(level), text] for level, text in enumerate(criteria or [])]
            title, summary = TITLES[name]
            listed.append(
                {
                    "name": name,
                    "title": title,
                    "summary": summary,
                    "repo": preset.source.repo,
                    "config": preset.source.config,
                    "revision": preset.source.revision,
                    "question": {
                        "id": question["id"],
                        "type": question["type"],
                        "instructions": question.get("instructions"),
                        "options": options,
                    },
                    "splits": [
                        {"name": split, "rows": rows, "evaluation": split == preset.source.split}
                        for split, rows in SPLITS[name].items()
                    ],
                    "default_split": TRAIN_SPLITS.get(name, "train"),
                }
            )
        return {"presets": listed}

    return router
