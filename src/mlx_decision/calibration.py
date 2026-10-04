"""Calibration requests for measuring quantization sensitivity.

Written for this package and kept apart from the parity set the tests use,
so that tuning on these does not flatter the parity numbers. Deterministic:
the same list every time. Requests are in the wire format, so they serve
any model family.
"""

import random

_HR = [
    "I would like to request two weeks of leave in August for a family wedding.",
    "My manager has not approved my expense report from March and finance keeps asking.",
    "Could someone explain how the new parental leave policy applies to part-time staff?",
    "I feel the feedback in my review was unfair and would like to discuss it.",
    "The onboarding laptop still has not arrived and my start date is Monday.",
    "Ich habe eine Frage zur Gleitzeitregelung im Homeoffice.",
]
_RETURNS = [
    "The jacket arrived in the wrong size, I ordered a medium and got an XL.",
    "One of the plates in the set was cracked when I opened the box.",
    "I changed my mind about the blender, it is still sealed in the original packaging.",
    "The headphones stopped charging after nine days; I'd like a replacement.",
    "El paquete llegó abierto y falta uno de los cables.",
    "I never received the parcel although the tracking says it was delivered.",
]
_BUGS = [
    "Clicking save twice creates a duplicate record in the projects table.",
    "Since version 3.2 the CSV import ignores the header row on Windows.",
    "The dark theme makes the error banner unreadable, white text on yellow.",
    "Search results are empty when the query contains an apostrophe.",
    "The mobile app crashes on launch when the device language is Turkish.",
    "Timeouts appear when more than 200 users are invited at once.",
]
_REVIEWS = [
    "The pasta was cooked perfectly but the sauce was far too salty.",
    "Service was slow, we waited forty minutes for drinks.",
    "Lovely terrace, friendly staff, and the desserts are worth the trip.",
    "Prices went up again while the portions got smaller.",
    "We came for a birthday and they surprised us with a candle and a song.",
    "Die Suppe war kalt, aber der Kellner hat sie sofort ersetzt.",
]
_SENSORS = ["boiler", "pump-2", "freezer", "server-rack", "greenhouse"]

_INTENT = {
    "type": "choice",
    "instructions": "What does the writer want?",
    "criteria": {
        "info": "Information or an explanation",
        "action": "Someone to do something",
        "complaint": "To express dissatisfaction",
        "praise": "To express satisfaction",
    },
}
_MOOD = {
    "type": "score",
    "instructions": "How negative is the tone?",
    "criteria": ["Positive", "Neutral", "Slightly negative", "Negative", "Hostile"],
}


def _text(rng: random.Random, sentences: list[str], count: int) -> str:
    return " ".join(rng.choice(sentences) for _ in range(count))


def calibration_requests() -> list[dict]:
    rng = random.Random(20261004)
    readings = [
        {
            "sensor": rng.choice(_SENSORS),
            "minute": minute,
            "celsius": round(rng.gauss(21, 6), 1),
            "status": rng.choice(["ok", "ok", "ok", "warn", "fault"]),
        }
        for minute in range(0, 600, 5)
    ]
    return [
        {
            "state": _text(rng, _HR, 4),
            "questions": {
                "intent": _INTENT,
                "hr_topic": {
                    "type": "choice",
                    "instructions": "Which HR topic is this about?",
                    "criteria": {
                        "leave": "Holidays and leave",
                        "pay": "Pay and expenses",
                        "review": "Performance reviews",
                        "it": "Equipment",
                        "policy": "Policies",
                    },
                },
            },
        },
        {
            "state": _text(rng, _RETURNS, 3),
            "questions": {
                "refund": {"type": "noul", "instructions": "The customer wants money back."},
                "damaged": {"type": "noul", "instructions": "The item arrived damaged."},
                "mood": _MOOD,
            },
        },
        {
            "state": _text(rng, _BUGS, 6),
            "questions": {
                "severity": {
                    "type": "score",
                    "instructions": "How severe is the worst bug described?",
                    "criteria": ["Cosmetic", "Minor", "Major", "Critical"],
                },
                "platform": {
                    "type": "choice",
                    "instructions": "Which platform is affected?",
                    "criteria": {"web": None, "windows": None, "mobile": None, "all": None},
                },
            },
        },
        {
            "state": [_text(rng, _REVIEWS, 2) for _ in range(5)],
            "questions": {
                "stars": {
                    "type": "score",
                    "instructions": "Overall rating implied by the reviews",
                    "criteria": ["1", "2", "3", "4", "5"],
                },
                "recommend": {"type": "noul", "instructions": "Most reviewers would return."},
            },
        },
        {
            "state": {"readings": readings[:40]},
            "questions": {
                "fault": {"type": "noul", "instructions": "Some sensor reported a fault."},
                "hottest": {
                    "type": "choice",
                    "instructions": "Which sensor shows the highest temperature?",
                    "criteria": {sensor: None for sensor in _SENSORS},
                },
            },
        },
        {
            "state": "\n\n".join(_text(rng, _BUGS + _HR, 5) for _ in range(6)),
            "questions": {"intent": _INTENT, "mood": _MOOD},
        },
        {
            "state": {"readings": readings},
            "questions": {
                "trend": {
                    "type": "choice",
                    "instructions": "How do temperatures develop over the log?",
                    "criteria": {"rising": None, "falling": None, "stable": None},
                },
                "action_needed": {
                    "type": "noul",
                    "instructions": "An operator should act on this log.",
                    "criteria": {"true": "Faults or warnings that persist", "false": "Normal"},
                },
            },
        },
        {
            "state": "\n".join(
                f"{'Guest' if i % 2 else 'Host'}: {rng.choice(_REVIEWS + _RETURNS)}"
                for i in range(40)
            ),
            "questions": {
                "resolved": {"type": "noul", "instructions": "The conversation ends amicably."},
                "mood": _MOOD,
                "intent": _INTENT,
            },
        },
        {
            "state": "ok thanks",
            "questions": {
                "praise": {"type": "noul"},
                "tone": {"type": "score", "criteria": ["cold", "neutral", "warm"]},
            },
        },
        {
            "state": _text(rng, _RETURNS + _REVIEWS, 30),
            "questions": {
                "language": {
                    "type": "choice",
                    "instructions": "Which languages appear besides English?",
                    "criteria": {"de": "German", "es": "Spanish", "none": "Only English"},
                },
                **{
                    f"mentions_{word}": {
                        "type": "noul",
                        "instructions": f"The text mentions {word}.",
                    }
                    for word in ("pasta", "headphones", "terrace", "cables")
                },
            },
        },
    ]
