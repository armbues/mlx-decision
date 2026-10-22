"""Batches at their edges: names, a killed or cancelled run, datasets whose
questions are on their lines or that have no labels, large results read as
they grow, unrounded answers, a row's own questions."""

import json
import threading
import time

import pytest
from test_studio_batches import (  # noqa: F401  (fixtures)
    TOPIC,
    models,
    start,
    stop,
    studio,
    wait,
    write_tweets,
)

from mlx_decision.config import Config
from mlx_decision.pool import ModelPool, discover
from mlx_decision.studio import create_studio
from mlx_decision.studio.batch_results import Results
from mlx_decision.studio.batches import BatchError

API = "/studio/api/batches"
JSON = {"Content-Type": "application/json"}


def write_lines(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


# Models and names


def test_an_unknown_model_is_refused_even_with_a_default(tmp_path, models):  # noqa: F811
    pool = ModelPool(discover([models]), default="fake", budget=10**9)
    app = create_studio(pool, workspace=tmp_path / "ws", config=Config(tmp_path / "c.toml"))
    write_tweets((tmp_path / "ws" / "data").resolve())
    with pytest.raises(BatchError) as error:
        app.state.batches.prepare("tweets", "test", "no-such-model")
    assert error.value.param == "model" and "no-such-model" in str(error.value)
    assert app.state.batches.list() == []


def test_a_typed_name_that_is_taken_is_refused_a_default_one_numbered(studio):  # noqa: F811
    studio.batches.prepare("tweets", "test", "fake", name="mine")
    with pytest.raises(BatchError) as error:
        studio.batches.prepare("tweets", "test", "fake", name="mine")
    assert (error.value.status, error.value.param) == (409, "name")
    names = [studio.batches.prepare("tweets", "test", "fake").name for _ in range(2)]
    assert names == ["tweets-fake", "tweets-fake-2"]


def test_two_starts_at_once_get_two_names(studio):  # noqa: F811
    barrier, names, errors = threading.Barrier(4), [], []

    def one():
        barrier.wait()
        try:
            names.append(studio.batches.prepare("tweets", "test", "fake").name)
        except Exception as error:  # pragma: no cover - the failure this guards against
            errors.append(error)

    threads = [threading.Thread(target=one) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(set(names)) == 4


def test_run_again_keeps_a_name_that_only_ends_in_digits(studio):  # noqa: F811
    numbered = studio.batches.prepare("tweets", "test", "fake").name
    studio.batches.prepare("tweets", "test", "fake")  # tweets-fake-2
    assert studio.batches.again("tweets-fake-2").name == "tweets-fake-3"
    assert studio.batches.again(numbered).name == "tweets-fake-4"
    own = studio.batches.prepare("tweets", "test", "fake", name="run-v-1").name
    assert studio.batches.again(own).name == "run-v-1-2"


def test_a_folder_that_is_not_a_batch_name_does_not_break_the_list(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    root = studio.app_state.workspace.root / "batches"
    copy = root / "tweets-fake copy"  # as Finder names a copy
    copy.mkdir()
    (copy / "batch.json").write_text((root / name / "batch.json").read_text())
    valid = root / "kept"
    valid.mkdir()
    for file in ("batch.json", "questions.json", "rows.jsonl", "results.jsonl"):
        (valid / file).write_bytes((root / name / file).read_bytes())
    listed = studio.get(API).json()["batches"]
    assert sorted(item["name"] for item in listed) == ["kept", "tweets-fake"]
    assert studio.get(f"{API}/kept").json()["batch"]["name"] == "kept"


# A killed or cancelled run


def test_a_restart_counts_what_a_killed_run_answered(tmp_path, models):  # noqa: F811
    config = Config(tmp_path / "c.toml")
    app = create_studio(
        ModelPool(discover([models]), budget=10**9), workspace=tmp_path / "ws", config=config
    )
    write_tweets((tmp_path / "ws" / "data").resolve())
    name = app.state.batches.prepare("tweets", "train", "fake").name
    folder = tmp_path / "ws" / "batches" / name
    info = json.loads((folder / "batch.json").read_text())
    (folder / "batch.json").write_text(json.dumps({**info, "state": "running"}))
    lines = [
        {"row": 0, "response": {}},
        {"row": 1, "error": {"message": "x"}},
        {"row": 2, "response": {}},
    ]
    write_lines(folder / "results.jsonl", lines)
    again = create_studio(
        ModelPool(discover([models]), budget=10**9), workspace=tmp_path / "ws", config=config
    )
    info = again.state.batches.summary(name)
    assert (info["state"], info["answered"], info["errors"]) == ("stopped", 3, 1)


def test_the_rest_counts_the_errors_already_answered(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    folder = studio.app_state.workspace.root / "batches" / name
    write_lines(folder / "results.jsonl", [{"row": 0, "error": {"message": "x", "param": None}}])
    start(studio, name)
    info = wait(studio, name)
    assert (info["state"], info["answered"], info["errors"]) == ("done", 6, 1)


def test_a_cancelled_run_is_stopped_and_lets_go_of_the_model(studio):  # noqa: F811
    write_tweets(studio.data, "many", count=200)
    name = studio.batches.prepare("many", "train", "slow").name
    start(studio, name)
    deadline = time.time() + 10
    while studio.batches.summary(name)["answered"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    studio.portal.call(lambda: studio.batches.running.task.cancel())
    deadline = time.time() + 10
    while studio.batches.running is not None and time.time() < deadline:
        time.sleep(0.01)
    info = studio.batches.summary(name)
    assert info["state"] == "stopped" and info["answered"] < 200
    assert studio.app_state.gpu.pool.pinned_by("slow") is None


def test_stop_is_in_the_summary_until_the_row_ends(studio):  # noqa: F811
    write_tweets(studio.data, "many", count=200)
    name = studio.batches.prepare("many", "train", "slow").name
    assert start(studio, name)["stop_requested"] is False
    assert stop(studio, name)["stop_requested"] is True
    wait(studio, name)


# Datasets


def test_a_folder_whose_lines_ask_their_questions_runs(studio):  # noqa: F811
    folder = studio.data / "own"
    folder.mkdir()
    asked = {"questions": {"topic": TOPIC}, "labels": {"topic": "fed"}}
    lines = [{"id": str(i), "state": f"s {i}", **asked} for i in range(4)]
    write_lines(folder / "train.jsonl", lines)
    (folder / "dataset.json").write_text(json.dumps({"splits": {"train": {"role": "train"}}}))
    listed = {item["name"]: item for item in studio.get("/studio/api/data").json()["files"]}
    assert listed["own"]["questions"] == ["topic"] and listed["own"]["missing"] == []
    name = studio.batches.prepare("own", "train", "fake").name
    start(studio, name)
    info = wait(studio, name)
    assert (info["answered"], info["labelled"]) == (4, ["topic"])


def test_a_plain_file_of_states_runs_with_a_sets_questions(studio):  # noqa: F811
    write_lines(studio.data / "states.jsonl", [{"state": f"s {i}"} for i in range(3)])
    listed = {item["name"]: item for item in studio.get("/studio/api/data").json()["files"]}
    assert listed["states"]["questions"] == []
    with pytest.raises(BatchError, match="pick a question set"):
        studio.batches.prepare("states", None, "fake")
    studio.app_state.workspace.save_set("topics", {"state": "", "questions": {"topic": TOPIC}})
    name = studio.batches.prepare("states", None, "fake", questions_from="topics").name
    start(studio, name)
    assert wait(studio, name)["answered"] == 3
    # Its own questions on every line, no labels: answers only.
    write_lines(
        studio.data / "asked.jsonl",
        [{"state": f"s {i}", "questions": {"topic": TOPIC}} for i in range(3)],
    )
    listed = {item["name"]: item for item in studio.get("/studio/api/data").json()["files"]}
    assert listed["asked"]["questions"] == ["topic"]
    name = studio.batches.prepare("asked", None, "fake").name
    start(studio, name)
    info = wait(studio, name)
    assert (info["answered"], info["labelled"]) == (3, [])


# Results


def test_the_metrics_read_unrounded_answers_and_the_export_rounds(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    folder = studio.app_state.workspace.root / "batches" / name
    answer = {"type": "noul", "noul": 0.499996}
    response = {"model": "fake", "answers": {"topic": answer}, "usage": {"input_tokens": 3}}
    write_lines(folder / "results.jsonl", [{"row": 0, "response": response}])
    # As written by the runner: unrounded (0.499996 is no; rounded to 0.5 it would be yes).
    rows = studio.get(f"{API}/{name}/rows").json()["rows"]
    assert rows[0]["answers"]["topic"]["answer"] == "false"
    line = json.loads(
        studio.get(f"{API}/{name}/export", params={"format": "jsonl"}).text.splitlines()[0]
    )
    assert line["answers"]["topic"]["noul"] == 0.5  # the wire's four places


def test_the_runner_keeps_answers_unrounded_and_error_codes(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    folder = studio.app_state.workspace.root / "batches" / name
    rows = [json.loads(text) for text in (folder / "rows.jsonl").read_text().splitlines()]
    rows[1]["questions"] = {"topic": {"type": "choice", "criteria": {}}}
    write_lines(folder / "rows.jsonl", rows)
    start(studio, name)
    wait(studio, name)
    results = studio.batches.results(name)
    probabilities = results[0]["response"]["answers"]["topic"]["probabilities"]
    assert any(len(repr(value)) > 6 for value in probabilities.values())  # not cut to 4 places
    assert "message" in results[1]["error"]


def test_a_rows_own_question_is_counted_and_exported(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    folder = studio.app_state.workspace.root / "batches" / name
    rows = [json.loads(text) for text in (folder / "rows.jsonl").read_text().splitlines()]
    rows[0]["questions"] = {"topic": TOPIC, "urgent": {"type": "noul", "instructions": "Urgent?"}}
    write_lines(folder / "rows.jsonl", rows)
    start(studio, name)
    wait(studio, name)
    questions = studio.get(f"{API}/{name}").json()["results"]["questions"]
    assert sum(questions["urgent"]["answers"].values()) == 1
    assert sum(questions["topic"]["answers"].values()) == 6
    header = studio.get(f"{API}/{name}/export", params={"format": "csv"}).text.splitlines()[0]
    assert "urgent,urgent_confidence" in header


def test_results_are_read_as_they_grow_and_a_torn_line_when_whole(tmp_path):
    folder = tmp_path / "b"
    folder.mkdir()
    (folder / "questions.json").write_text(json.dumps({"questions": {"topic": TOPIC}}))
    write_lines(
        folder / "rows.jsonl",
        [{"row": i, "id": str(i), "state": "s", "labels": {}} for i in range(3)],
    )
    answer = {
        "type": "choice",
        "choice": "fed",
        "confidence": 0.9,
        "probabilities": {"analyst": 0.05, "fed": 0.9, "other": 0.05},
    }
    response = {"model": "m", "answers": {"topic": answer}, "usage": {"input_tokens": 1}}
    results = folder / "results.jsonl"
    results.write_text(json.dumps({"row": 0, "response": response}) + "\n" + '{"row": 1, "resp')
    reader = Results(tmp_path)
    assert reader.analyse("b")["answered"] == 1
    with open(results, "w") as file:  # the torn line finished, and one more
        file.write(json.dumps({"row": 0, "response": response}) + "\n")
        file.write(json.dumps({"row": 1, "response": response}) + "\n")
    assert reader.analyse("b")["answered"] == 2
    with open(results, "a") as file:
        file.write(json.dumps({"row": 2, "response": response}) + "\n")
    assert reader.analyse("b")["answered"] == 3
    assert reader.analyse("b")["questions"]["topic"]["answers"]["fed"] == 3


def test_the_results_cache_keeps_a_few_batches(tmp_path):
    for index in range(Results.KEEP + 2):
        folder = tmp_path / f"b{index}"
        folder.mkdir()
        (folder / "questions.json").write_text(json.dumps({"questions": {"topic": TOPIC}}))
        write_lines(folder / "rows.jsonl", [{"row": 0, "id": "0", "state": "s", "labels": {}}])
        (folder / "results.jsonl").write_text("")
    reader = Results(tmp_path)
    for index in range(Results.KEEP + 2):
        reader.analyse(f"b{index}")
    assert list(reader._cache) == [f"b{index}" for index in range(2, Results.KEEP + 2)]


def test_a_delete_is_announced_to_every_tab(studio):  # noqa: F811
    name = studio.batches.prepare("tweets", "test", "fake").name
    assert studio.delete(f"{API}/{name}", headers=JSON).status_code == 200
    deadline = time.time() + 5
    while time.time() < deadline:
        changes = [
            item for _, item in studio.app_state.hub.recent if item[1]["type"] == "workspace"
        ]
        if any(c[1]["folder"] == "batches" and c[1]["name"] == name for c in changes):
            break
        time.sleep(0.02)
    else:
        raise AssertionError("no workspace message for the delete")


def test_a_labelled_batch_says_so_before_its_first_answer(studio):  # noqa: F811
    # Nothing answered yet: the question is labelled (its rows carry labels),
    # with no wrong answers and no metrics until answers come.
    folder = studio.batches.prepare("tweets", "test", "fake", name="waiting")
    analysed = Results(studio.batches.root).analyse(folder.name)["questions"]["topic"]
    assert analysed["labelled"] is True and analysed["wrong"] == 0
    assert "metrics" not in analysed
