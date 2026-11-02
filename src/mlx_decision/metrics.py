"""Accuracy metrics over answered examples: accuracy, macro-F1, calibration, MAE, log loss.

An answer becomes a record with :func:`record`: the gold option id, the
predicted one, the top probability and the log loss (against a soft label's
distribution when there is one); a refused example is a record with
``predicted`` None and a ``refused`` reason. :func:`metrics` summarizes a list
of records of one question; :func:`confusion` counts gold against predicted.
"""

import math
import statistics

# log(0) is replaced by log of this in the log loss.
FLOOR = 1e-12


def log_loss(probabilities: dict[str, float], gold: str, target: dict | None = None) -> float:
    """The negative log-probability of ``gold``, or with ``target`` (a soft label:
    option id -> probability) the cross-entropy against it."""
    if target is None:
        return -math.log(max(probabilities.get(gold, 0.0), FLOOR))
    return -sum(
        weight * math.log(max(probabilities.get(option, 0.0), FLOOR))
        for option, weight in target.items()
        if weight > 0
    )


def record(answer: dict, gold: str, target: dict | None = None) -> dict:
    """What the metrics need of an answer (in the wire format): ``predicted`` is an
    option id, a level (as its string key) or "true" / "false" (noul, at 0.5);
    ``gold`` the label (a soft label's most likely answer), ``target`` a soft
    label's distribution, which the log loss is against."""
    if answer["type"] == "noul":
        noul = answer["noul"]
        return {
            "gold": gold,
            "predicted": "true" if noul >= 0.5 else "false",
            "p_max": max(noul, 1 - noul),
            "noul": noul,
            "log_loss": log_loss({"true": noul, "false": 1 - noul}, gold, target),
        }
    probabilities = answer["probabilities"]
    result = {
        "gold": gold,
        "predicted": max(probabilities, key=probabilities.get),
        "p_max": max(probabilities.values()),
        "confidence": answer["confidence"],
        "probabilities": probabilities,
        "log_loss": log_loss(probabilities, gold, target),
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
    """Accuracy and macro-F1 over all records (refused ones are wrong); ECE, MAE
    and the mean log loss over the answered ones (records written before the log
    loss have none: then it is left out). ``noul`` predictions are yes at 0.5 or
    more."""
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
    losses = [r["log_loss"] for r in records if "log_loss" in r]
    if losses:
        result["log_loss"] = statistics.mean(losses)
    return result


REFUSED = "refused"


def confusion(records: list[dict], labels: list[str]) -> dict[str, dict[str, int]]:
    """Counts by gold label (rows, in ``labels``' order) and predicted label
    (columns: ``labels``, then "refused" when any example was refused). Labels
    not in ``labels`` are added after them."""
    seen = list(labels)
    for r in records:
        for label in (r["gold"], r["predicted"]):
            if label is not None and label not in seen:
                seen.append(label)
    columns = seen + ([REFUSED] if any(r["predicted"] is None for r in records) else [])
    table = {gold: dict.fromkeys(columns, 0) for gold in seen}
    for r in records:
        table[r["gold"]][r["predicted"] if r["predicted"] is not None else REFUSED] += 1
    return table
