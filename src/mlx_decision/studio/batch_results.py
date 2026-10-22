"""A batch's results read back: per question its answers' counts and, where the
rows have labels, the metrics ``training eval`` reports, a confusion matrix
and a reliability diagram's bins; the rows as table pages, filtered on the
server; the export; the answers saved as a labelled dataset.

``Results`` reads a batch's rows once and the new tail of ``results.jsonl``
as it grows, so a page of a large batch does not read them again. The
response is kept unrounded (the metrics are ``training eval``'s); the JSON
lines export rounds it as the wire does. The methods block: call them off
the loop.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.dataset import Dataset, SplitOut, write_dataset
from ..data.format import Example
from ..metrics import confusion, metrics, record
from ..types import _round_floats as round_floats
from .batches import BATCH_FILE, QUESTIONS_FILE, RESULTS_FILE, ROWS_FILE, BatchError, read_lines
from .workspace import NotFound

# The state as the table shows it, cut to this many characters.
STATE_CUT = 300
# Rows on a page, by default and at most.
PAGE = 100
MAX_PAGE = 500
# An answer below this confidence is Unsure, unless the page says otherwise.
UNSURE = 0.6
# The reliability diagram's bins.
BINS = 10


@dataclass
class Entry:
    """A row with its answer, as the table and the metrics read it."""

    row: int
    id: str
    state: Any
    labels: dict[str, str]
    request: dict[str, Any]
    result: dict[str, Any] | None  # None: not answered yet
    picked: dict[str, tuple[str, float, float]]  # id -> (answer key, confidence, probability)

    @property
    def error(self) -> dict[str, Any] | None:
        return None if self.result is None else self.result.get("error")


def answer_key(answer: dict[str, Any]) -> tuple[str, float, float]:
    """The answer as the metrics name it (an option id, "true" / "false", a
    level's key), its confidence and its probability (as the Playground shows
    it next to the answer)."""
    item = record(answer, "")
    probability = (item.get("probabilities") or {}).get(item["predicted"], item["p_max"])
    return item["predicted"], answer.get("confidence", item["p_max"]), probability


def label_order(question: dict[str, Any]) -> list[str]:
    """A question's answers in their order: options, levels, yes then no."""
    kind = question.get("type", "noul")
    if kind == "choice":
        return list(question.get("criteria") or {})
    if kind == "score":
        return [str(level) for level in range(len(question.get("criteria") or []))]
    return ["true", "false"]


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


@dataclass
class _Joined:
    """A batch's rows with the results read so far, kept between requests."""

    rows_stamp: tuple
    shared: dict[str, Any]  # questions.json's
    questions: dict[str, Any]  # those and the rows' own, by id, in order
    entries: list[Entry]
    by_row: dict[int, Entry]
    offset: int = 0  # bytes of results.jsonl read (whole lines)
    analysed: dict[float, dict[str, Any]] = field(default_factory=dict)  # by Unsure's bound


class Results:
    """The batches' results. A batch's rows are read once (``rows.jsonl`` and
    ``questions.json`` do not change); ``results.jsonl`` grows as rows are
    answered, and each read takes only its new whole lines. The latest few
    batches are kept (``KEEP``); one lock, so two tabs do not read a batch twice."""

    KEEP = 4

    def __init__(self, root: Path):
        self.root = root
        self._cache: OrderedDict[str, _Joined] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _stat(path: Path, name: str):
        try:
            return path.stat()
        except FileNotFoundError:
            raise NotFound(f"no batch named {name!r}") from None

    def _joined(self, name: str) -> _Joined:
        folder = self.root / name
        rows_stat = self._stat(folder / ROWS_FILE, name)
        questions_stat = self._stat(folder / QUESTIONS_FILE, name)
        results_size = self._stat(folder / RESULTS_FILE, name).st_size
        stamp = (rows_stat.st_size, rows_stat.st_mtime_ns, questions_stat.st_mtime_ns)
        with self._lock:
            joined = self._cache.get(name)
            if joined is None or joined.rows_stamp != stamp or results_size < joined.offset:
                joined = self._read_rows(folder, stamp)
            if results_size > joined.offset:
                self._read_results(folder / RESULTS_FILE, joined)
            self._cache[name] = joined
            self._cache.move_to_end(name)
            while len(self._cache) > self.KEEP:
                self._cache.popitem(last=False)
            return joined

    @staticmethod
    def _read_rows(folder: Path, stamp: tuple) -> _Joined:
        shared = json.loads((folder / QUESTIONS_FILE).read_bytes())["questions"]
        questions = dict(shared)
        entries, by_row = [], {}
        for row in read_lines(folder / ROWS_FILE):
            request = {k: v for k, v in row.items() if k not in ("row", "id", "labels")}
            for qid, question in (request.get("questions") or {}).items():
                questions.setdefault(qid, question)  # a row's own question
            request.setdefault("questions", shared)
            entry = Entry(
                row["row"], row.get("id", str(row["row"])), row.get("state"),
                row.get("labels") or {}, request, None, {},
            )  # fmt: skip
            entries.append(entry)
            by_row[entry.row] = entry
        return _Joined(stamp, shared, questions, entries, by_row)

    @staticmethod
    def _read_results(path: Path, joined: _Joined) -> None:
        """The whole lines added to ``results.jsonl`` since the last read."""
        with open(path, "rb") as file:
            file.seek(joined.offset)
            data = file.read()
        end = data.rfind(b"\n") + 1  # a torn last line waits for its newline
        for text in data[:end].split(b"\n"):
            if not text.strip():
                continue
            try:
                item = json.loads(text)
            except ValueError:
                continue
            entry = joined.by_row.get(item.get("row")) if isinstance(item, dict) else None
            if entry is None:
                continue
            entry.result = item
            entry.picked = {
                qid: answer_key(answer)
                for qid, answer in (item.get("response") or {}).get("answers", {}).items()
            }
        joined.offset += end
        joined.analysed.clear()

    def load(self, name: str) -> tuple[dict[str, Any], list[Entry]]:
        """The batch's questions (the shared ones, then the rows' own) and its
        entries in row order."""
        joined = self._joined(name)
        return joined.questions, joined.entries

    def forget(self, name: str) -> None:
        with self._lock:
            self._cache.pop(name, None)

    # The summary: answers, metrics

    def analyse(self, name: str, unsure: float = UNSURE) -> dict[str, Any]:
        """Per question: its type, the count of each answer, Unsure's count, and
        with labels the metrics, confusion matrix, wrong count and reliability
        bins (``low``, ``high``, ``n``, ``accuracy``, ``confidence``). Kept until
        more rows are answered."""
        joined = self._joined(name)
        with self._lock:
            cached = joined.analysed.get(unsure)
        if cached is not None:
            return cached
        result = self._analyse(joined.questions, joined.entries, unsure)
        with self._lock:
            joined.analysed[unsure] = result
        return result

    @staticmethod
    def _analyse(questions, entries, unsure: float) -> dict[str, Any]:
        answered = [e for e in entries if e.result is not None]
        out: dict[str, Any] = {}
        for qid, question in questions.items():
            order = label_order(question)
            counts = dict.fromkeys(order, 0)
            records, unsure_count = [], 0
            for entry in answered:
                if qid not in entry.request["questions"]:
                    continue  # another row's own question
                picked = entry.picked.get(qid)
                if picked is not None:
                    counts[picked[0]] = counts.get(picked[0], 0) + 1
                    unsure_count += picked[1] < unsure
                gold = entry.labels.get(qid)
                if gold is None:
                    continue
                if picked is None:
                    message = (entry.error or {}).get("message", "no answer")
                    records.append({"gold": gold, "predicted": None, "refused": message})
                else:
                    answer = entry.result["response"]["answers"][qid]
                    item = record(answer, gold)
                    item["truncated"] = entry.result.get("truncated", False)
                    records.append(item)
            # Labelled when its rows carry labels, answered yet or not (the
            # metrics come with the answers).
            labelled = bool(records) or any(
                qid in e.request["questions"] and e.labels.get(qid) is not None for e in entries
            )
            summary: dict[str, Any] = {
                "type": question.get("type", "noul"),
                "answers": counts,
                "unsure": unsure_count,
                "labelled": labelled,
                "wrong": 0,
            }
            if records:
                summary["metrics"] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics(records).items()
                }
                summary["wrong"] = sum(r["predicted"] != r["gold"] for r in records)
                summary["labels"] = order
                summary["confusion"] = confusion(records, order)
                summary["reliability"] = reliability(records)
            out[qid] = summary
        errors = sum(e.error is not None for e in answered)
        return {
            "questions": out,
            "answered": len(answered),
            "errors": errors,
            "unsure_below": unsure,
        }

    # The table

    def page(
        self,
        name: str,
        offset: int = 0,
        limit: int = PAGE,
        question: str | None = None,
        answer: str | None = None,
        wrong: bool = False,
        unsure: float | None = None,
        search: str | None = None,
        errors: bool = False,
        gold: str | None = None,
    ) -> dict[str, Any]:
        """A page of the answered rows that pass the filters (``answer``,
        ``gold`` (the expected answer), ``wrong`` and ``unsure`` on
        ``question``, the first one by default; ``search`` in the state,
        ignoring case), with the number that pass."""
        questions, entries = self.load(name)
        if question is None:
            question = next(iter(questions), None)
        elif question not in questions:
            raise BatchError(f"the batch asks no question {question!r}", param="question")
        needle = search.casefold() if search else None
        found = []
        for entry in entries:
            if entry.result is None:
                continue
            if errors and entry.error is None:
                continue
            picked = entry.picked.get(question) if question else None
            if answer is not None and (picked is None or picked[0] != answer):
                continue
            if gold is not None and entry.labels.get(question) != gold:
                continue
            if wrong:
                expected = entry.labels.get(question)
                if expected is None or (picked is not None and picked[0] == expected):
                    continue
            if unsure is not None and (picked is None or picked[1] >= unsure):
                continue
            if needle and needle not in state_text(entry.state).casefold():
                continue
            found.append(entry)
        limit = max(1, min(limit, MAX_PAGE))
        offset = max(0, offset)
        return {
            "matching": len(found),
            "offset": offset,
            "rows": [self._row(entry, cut=True) for entry in found[offset : offset + limit]],
        }

    def row(self, name: str, row: int) -> dict[str, Any]:
        """One row whole, with each answer's probabilities."""
        _, entries = self.load(name)
        for entry in entries:
            if entry.row == row:
                return self._row(entry, cut=False)
        raise NotFound(f"the batch has no row {row}")

    @staticmethod
    def _row(entry: Entry, cut: bool) -> dict[str, Any]:
        text = state_text(entry.state)
        out: dict[str, Any] = {
            "row": entry.row,
            "id": entry.id,
            "state": text[:STATE_CUT] + "…" if cut and len(text) > STATE_CUT else text,
            "labels": entry.labels,
            "answers": {
                qid: {"answer": key, "confidence": round(c, 4), "probability": round(p, 4)}
                for qid, (key, c, p) in entry.picked.items()
            },
            "error": entry.error,
        }
        if not cut:
            out["state"] = entry.state
            out["request"] = entry.request
            if entry.result is not None:
                out["response"] = entry.result.get("response")
                for key in ("latency_ms", "input_tokens", "truncated"):
                    out[key] = entry.result.get(key)
        else:
            out["truncated"] = bool(entry.result and entry.result.get("truncated"))
        return out

    # Export

    def export_csv(self, name: str) -> str:
        """A row per answered state: id, state, then per question its answer,
        confidence and (with labels) expected answer; an error column."""
        questions, entries = self.load(name)
        labelled = {qid for e in entries for qid in e.labels}
        header = ["id", "state"]
        for qid in questions:
            header += [qid, f"{qid}_confidence"]
            if qid in labelled:
                header.append(f"{qid}_expected")
        header.append("error")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(header)
        for entry in entries:
            if entry.result is None:
                continue
            line = [entry.id, state_text(entry.state)]
            for qid in questions:
                picked = entry.picked.get(qid)
                line += [picked[0], round(picked[1], 4)] if picked else ["", ""]
                if qid in labelled:
                    line.append(entry.labels.get(qid, ""))
            line.append((entry.error or {}).get("message", ""))
            writer.writerow(line)
        return buffer.getvalue()

    def export_jsonl(self, name: str) -> str:
        """The lines ``run --states --json`` prints for the rows answered (the
        response body, or an error body with the row's ``line``), each with the
        row's ``id``."""
        _, entries = self.load(name)
        lines = []
        for number, entry in enumerate(entries, 1):
            if entry.result is None:
                continue
            if entry.error is not None:
                body = {"error": {**entry.error, "type": "invalid_request_error"}, "line": number}
            else:
                body = round_floats(entry.result["response"])  # as the wire rounds it
            lines.append(json.dumps({"id": entry.id, **body}, ensure_ascii=False))
        return "".join(line + "\n" for line in lines)

    # Save Answers as a Dataset

    def save_dataset(
        self, name: str, folder: Path, threshold: float | None = None
    ) -> dict[str, Any]:
        """``folder`` (a new dataset folder): the batch's answered rows with the
        top answers as labels, by question only where the confidence is at least
        ``threshold``; a row left with no label is left out. One split, ``train``
        with the role train, from the batch's split (its ``split``); ``source``
        names the batch."""
        info = json.loads((self.root / name / BATCH_FILE).read_bytes())
        joined = self._joined(name)
        shared, entries = joined.shared, joined.entries
        examples, left_out = [], 0
        for entry in entries:
            if entry.result is None or entry.error is not None:
                continue
            labels = {
                qid: key
                for qid, (key, confidence, _) in entry.picked.items()
                if threshold is None or confidence >= threshold
            }
            if not labels:
                left_out += 1
                continue
            request = dict(entry.request)
            if request.get("questions") == shared:
                request["questions"] = shared  # the same object: shared in dataset.json
            examples.append(Example(entry.id, request, labels))
        if not examples:
            raise BatchError("no row has an answer to keep as a label", param="threshold")
        split = "train"
        source = {
            "source": f"batch:{name}",
            "batch": name,
            "dataset": info.get("dataset"),
            "split": info.get("split"),
            "model": info.get("model"),
            "fingerprint": info.get("fingerprint"),
            "threshold": threshold,
        }
        dataset = Dataset(folder, source=source)
        meta = {"role": "train", "examples": len(examples), "split": info.get("split")}
        try:
            write_dataset(folder, dataset, [SplitOut(split, examples, meta)])
        except FileExistsError:
            raise BatchError(
                f"a dataset named {folder.name} exists already", status=409, param="name"
            ) from None
        return {
            "name": folder.name,
            "examples": len(examples),
            "left_out": left_out,
            "split": split,
        }


def reliability(records: list[dict[str, Any]], bins: int = BINS) -> list[dict[str, Any]]:
    """The answered records by top probability in ``bins`` equal bins: each
    bin's count, accuracy and mean top probability (as ``metrics.ece`` bins)."""
    out = []
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        group = [
            r
            for r in records
            if "p_max" in r and (low < r["p_max"] <= high or (b == 0 and r["p_max"] == 0))
        ]
        item: dict[str, Any] = {"low": low, "high": high, "n": len(group)}
        if group:
            item["accuracy"] = round(
                sum(r["predicted"] == r["gold"] for r in group) / len(group), 4
            )
            item["confidence"] = round(sum(r["p_max"] for r in group) / len(group), 4)
        out.append(item)
    return out
