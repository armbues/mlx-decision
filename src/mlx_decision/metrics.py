"""Accuracy metrics over answered examples: accuracy, macro-F1, calibration, MAE.

An answer becomes a record with :func:`record`: the gold option id, the
predicted one and the top probability; a refused example is a record with
``predicted`` None and a ``refused`` reason. :func:`metrics` summarizes a list
of records of one question.
"""

import statistics


def record(answer: dict, gold: str) -> dict:
    """What the metrics need of an answer (in the wire format): ``predicted`` is an
    option id, a level (as its string key) or "true" / "false" (noul, at 0.5)."""
    if answer["type"] == "noul":
        return {
            "gold": gold,
            "predicted": "true" if answer["noul"] >= 0.5 else "false",
            "p_max": max(answer["noul"], 1 - answer["noul"]),
            "noul": answer["noul"],
        }
    probabilities = answer["probabilities"]
    result = {
        "gold": gold,
        "predicted": max(probabilities, key=probabilities.get),
        "p_max": max(probabilities.values()),
        "confidence": answer["confidence"],
        "probabilities": probabilities,
    }
    if answer["type"] == "score":
        result["score"] = answer["score"]
    else:
        result["predicted"] = answer["choice"]
    return result


def macro_f1(records: list[dict]) -> float:
    """F1 averaged over the gold labels present."""
    labels = {r["gold"] for r in records}
    scores = []
    for label in labels:
        tp = sum(r["predicted"] == label and r["gold"] == label for r in records)
        fp = sum(r["predicted"] == label and r["gold"] != label for r in records)
        fn = sum(r["predicted"] != label and r["gold"] == label for r in records)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return statistics.mean(scores)


def ece(records: list[dict], bins: int = 10) -> float:
    """Expected calibration error of the top probability."""
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        group = [
            r
            for r in records
            if "p_max" in r and (low < r["p_max"] <= high or (b == 0 and r["p_max"] == 0))
        ]
        if group:
            accuracy = sum(r["predicted"] == r["gold"] for r in group) / len(group)
            mean_p = statistics.mean(r["p_max"] for r in group)
            answered = sum("p_max" in r for r in records)
            total += len(group) / answered * abs(accuracy - mean_p)
    return total


def mae(records: list[dict]) -> float | None:
    """Mean absolute error of the expected level against the gold level (score)."""
    errors = [abs(r["score"] - int(r["gold"])) for r in records if "score" in r]
    return statistics.mean(errors) if errors else None


def metrics(records: list[dict]) -> dict:
    """Accuracy and macro-F1 over all records (refused ones are wrong); ECE and MAE
    over the answered ones. ``noul`` predictions are yes at 0.5 or more."""
    result = {
        "n": len(records),
        "refused": sum("refused" in r for r in records),
        "truncated": sum(bool(r.get("truncated")) for r in records),
        "accuracy": sum(r["predicted"] == r["gold"] for r in records) / len(records),
        "macro_f1": macro_f1(records),
        "ece": ece(records),
    }
    if any("score" in r for r in records):
        result["mae"] = mae(records)
    return result
