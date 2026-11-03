"""The Studio's workspace: layout, names, atomic writes, sets, history, speed."""

import base64
import io
import json
import os
import re

import pytest
from typer.testing import CliRunner

from fake_backend import write_model
from mlx_decision.cli import app as cli_app
from mlx_decision.config import Config
from mlx_decision.studio import create_studio, workspace
from mlx_decision.studio.workspace import (
    FOLDERS,
    README,
    SPEED_KEEP,
    Conflict,
    NotFound,
    Workspace,
    WorkspaceError,
    root_folder,
    sha256,
)

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"technical": "Bugs", "billing": "Payments", "sales": "Pricing"},
    },
    "frustration": {"type": "score", "criteria": ["calm", "upset", "angry"]},
    "is_urgent": {"type": "noul", "instructions": "Urgent?"},
}
BODY = {"state": "Stripe is down and we cannot bill anyone", "questions": QUESTIONS}


def png(color: tuple[int, int, int] = (255, 0, 0)) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def space(tmp_path) -> Workspace:
    files = Workspace(tmp_path / "ws")
    files.open()
    return files


# Layout


def test_open_creates_the_layout_and_the_readme(tmp_path):
    files = Workspace(tmp_path / "new" / "ws")
    assert files.open() == []
    for folder in FOLDERS:
        assert (files.root / folder).is_dir()
    assert (files.root / "history" / "images").is_dir()
    assert (files.root / "README.md").read_text() == README
    for folder in FOLDERS:
        assert f"`{folder}/`" in README
    assert json.loads((files.root / "workspace.json").read_text()) == {"format": 1}
    assert "`workspace.json`" in README


def test_a_workspace_without_a_marker_gets_one_and_a_marker_is_kept(space):
    marker = space.root / "workspace.json"
    marker.unlink()  # as a workspace from before the marker
    space.open()
    assert json.loads(marker.read_text()) == {"format": 1}
    marker.write_text('{"format": 1, "migrated": "later"}')
    space.open()
    assert json.loads(marker.read_text()) == {"format": 1, "migrated": "later"}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('{"format": 2}', "the workspace's format is 2, .*a newer mlx-decision"),
        ('{"format": "1"}', "must hold"),
        ("not json", "cannot read"),
    ],
)
def test_a_marker_it_does_not_read_is_refused_before_anything_changes(space, body, message):
    (space.root / "workspace.json").write_text(body)
    (space.root / "README.md").write_text("from a later version")
    (space.root / "sets" / ".a.json.tmp").write_text("{}")
    with pytest.raises(ValueError, match=message):
        space.open()
    assert (space.root / "README.md").read_text() == "from a later version"
    assert (space.root / "sets" / ".a.json.tmp").exists()  # not cleaned up either


def test_the_readme_is_rewritten_but_other_files_are_kept(space):
    (space.root / "README.md").write_text("old")
    (space.root / "config.toml").write_text("port = 9000\n")
    space.open()
    assert (space.root / "README.md").read_text() == README
    assert (space.root / "config.toml").read_text() == "port = 9000\n"


def test_the_root_follows_flag_then_config_then_home(tmp_path, mlx_decision_home):
    config = Config(path=tmp_path / "config.toml", workspace=tmp_path / "from-config")
    assert root_folder(config, tmp_path / "flag") == tmp_path / "flag"
    assert root_folder(config) == tmp_path / "from-config"
    assert root_folder(Config(path=tmp_path / "c.toml")) == mlx_decision_home
    assert root_folder() == mlx_decision_home


def test_create_studio_opens_the_workspace(tmp_path, mlx_decision_home):
    from mlx_decision.pool import ModelPool

    app = create_studio(ModelPool([]), workspace=tmp_path / "ws")
    assert app.state.workspace.root == (tmp_path / "ws").resolve()
    assert (tmp_path / "ws" / "sets").is_dir()
    app = create_studio(ModelPool([]))
    assert app.state.workspace.root == mlx_decision_home.resolve()
    assert (mlx_decision_home / "README.md").is_file()


# Names and paths


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        ".hidden",
        "../escape",
        "a/b",
        "a\\b",
        "/etc/passwd",
        "~root",
        "with space",
        "x" * 81,
        "naïve",
        "a\x00b",
        None,
        3,
    ],
)
def test_bad_names_are_refused(space, name):
    with pytest.raises(WorkspaceError, match="1 to 80 letters"):
        space.set_path(name)
    with pytest.raises(WorkspaceError):
        space.save_set(name, BODY)
    with pytest.raises(WorkspaceError):
        space.load_set(name)


@pytest.mark.parametrize("name", ["a", "tickets", "Tickets_v2.final", "x" * 80, "-dash", "9"])
def test_good_names(space, name):
    space.save_set(name, BODY)
    assert space.load_set(name)[0] == BODY


def test_a_symlink_out_of_the_workspace_is_refused(space, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(BODY))
    (space.root / "sets" / "link.json").symlink_to(outside)
    with pytest.raises(WorkspaceError, match="not inside the workspace"):
        space.load_set("link")
    with pytest.raises(WorkspaceError, match="not inside the workspace"):
        space.save_set("link", BODY, sha256(outside.read_bytes()))
    assert json.loads(outside.read_text()) == BODY


def test_a_symlinked_folder_out_of_the_workspace_is_refused(space, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (space.root / "history").rename(space.root / "history-old")
    (space.root / "history").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(WorkspaceError, match="not inside the workspace"):
        space.append_history("tickets", {"request": BODY})
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_a_workspace_reached_through_a_symlink_works(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    files = Workspace(tmp_path / "link")
    files.open()
    files.save_set("tickets", BODY)
    assert (tmp_path / "real" / "sets" / "tickets.json").is_file()


# Atomic writes


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary(space, monkeypatch):
    loaded = space.save_set("tickets", BODY)

    def broken_replace(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(workspace.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        space.save_set("tickets", {**BODY, "state": "changed"}, loaded)
    assert space.load_set("tickets") == (BODY, loaded)
    assert [p.name for p in (space.root / "sets").iterdir()] == ["tickets.json"]


def test_writes_go_through_a_hidden_temporary_file(space, monkeypatch):
    seen = []
    real_replace = os.replace

    def replace(source, target):
        seen.append((os.path.basename(source), os.path.basename(target)))
        real_replace(source, target)

    monkeypatch.setattr(workspace.os, "replace", replace)
    space.save_set("tickets", BODY)
    assert seen == [(".tickets.json.tmp", "tickets.json")]


# Start-up clean-up


def test_clean_up_removes_stale_temporaries_and_partials(space):
    stale = [
        space.root / "sets" / ".tickets.json.tmp",
        space.root / "speed" / ".abc.jsonl.tmp",
        space.root / ".README.md.tmp",
        space.root / "history" / "images" / ".x.png.tmp",
    ]
    for path in stale:
        path.write_text("half")
    partial = space.root / "models" / ".fitted.partial"
    (partial / "sub").mkdir(parents=True)
    (partial / "sub" / "weights.safetensors").write_text("x")
    kept = [
        space.root / "sets" / "tickets.json",
        space.root / "models" / "fitted" / ".inner.tmp",  # inside a model: not touched
        space.root / "sets" / "visible.tmp",
    ]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    removed = space.open()
    assert sorted(removed) == sorted([*stale, partial])
    assert not any(path.exists() for path in [*stale, partial])
    assert all(path.exists() for path in kept)


def test_clean_up_finishes_a_fitted_model_whose_replacement_was_stopped(space, tmp_path):
    from mlx_decision import fitted
    from mlx_decision.fitted import Calibration, Fit, FitQuestion, base_reference, fingerprint

    base = write_model(tmp_path / "base")
    question = FitQuestion("team", "choice", ["billing", "tech"])
    calibration = {0: Calibration(1.0, [0.0, 0.0])}

    def fit(seed):
        return Fit(["calibrate"], [question], calibration, meta={"seed": seed})

    def write(folder, seed):
        reference = base_reference(base, base)
        return fitted.write(
            folder, family="fake", base=reference, base_fingerprint=fingerprint(base), fit=fit(seed)
        )

    models = space.root / "models"
    write(models / ".support.new", 2)  # stopped between the renames: both aside
    write(models / ".support.old", 1)
    backup = models / ".other.old"  # not a fit: left alone
    backup.mkdir()
    removed = space.open()
    assert sorted(removed) == [models / ".support.new", models / ".support.old"]
    assert sorted(p.name for p in models.iterdir()) == [".other.old", "support"]
    marker = json.loads((models / "support" / "mlx_decision.json").read_text())
    assert marker["fit"]["seed"] == 2


def test_clean_up_marks_jobs_without_an_outcome_interrupted(space):
    jobs = space.root / "jobs"
    for job_id, job in {
        "running": {"kind": "fit", "command": ["training", "fit"]},
        "done": {"kind": "fit", "outcome": "done", "ended": "2026-10-18T10:00:00Z"},
        "empty-outcome": {"kind": "download", "outcome": None},
    }.items():
        (jobs / job_id).mkdir()
        (jobs / job_id / "job.json").write_text(json.dumps(job))
    (jobs / "broken").mkdir()
    (jobs / "broken" / "job.json").write_text("{not json")
    (jobs / "no-file").mkdir()
    changed = space.clean_up()
    assert sorted(path.parent.name for path in changed) == ["empty-outcome", "running"]
    read = lambda job_id: json.loads((jobs / job_id / "job.json").read_text())  # noqa: E731
    assert read("running") == {
        "kind": "fit",
        "command": ["training", "fit"],
        "outcome": "interrupted",
        "state": "ended",
        "ended": None,
    }
    assert read("done")["outcome"] == "done"
    assert read("empty-outcome")["outcome"] == "interrupted"
    assert (jobs / "broken" / "job.json").read_text() == "{not json"


# Sets


def test_save_load_and_list(space):
    digest = space.save_set("tickets", BODY)
    body, loaded = space.load_set("tickets")
    assert body == BODY and loaded == digest
    assert loaded == sha256((space.root / "sets" / "tickets.json").read_bytes())
    space.save_set("Alpha", {"questions": {}})
    space.save_set("beta", {"questions": {}})
    assert [entry["name"] for entry in space.list_sets()] == ["Alpha", "beta", "tickets"]
    entry = space.list_sets()[-1]
    assert entry["bytes"] == (space.root / "sets" / "tickets.json").stat().st_size
    assert isinstance(entry["modified"], float)


def test_list_skips_other_files(space):
    space.save_set("tickets", BODY)
    for name in [".hidden.json", "notes.txt", "bad name.json"]:
        (space.root / "sets" / name).write_text("{}")
    (space.root / "sets" / "folder.json").mkdir()
    assert [entry["name"] for entry in space.list_sets()] == ["tickets"]


def test_the_file_is_readable_json(space):
    space.save_set("tickets", {**BODY, "state": "naïve ✓"})
    text = (space.root / "sets" / "tickets.json").read_text(encoding="utf-8")
    assert "naïve ✓" in text and text.endswith("\n") and text.startswith("{\n  ")


def test_saving_over_the_loaded_version(space):
    first = space.save_set("tickets", BODY)
    second = space.save_set("tickets", {**BODY, "state": "v2"}, first)
    assert second != first
    assert space.load_set("tickets") == ({**BODY, "state": "v2"}, second)


def test_a_set_changed_on_disk_is_not_overwritten(space):
    loaded = space.save_set("tickets", BODY)
    path = space.root / "sets" / "tickets.json"
    path.write_text(json.dumps({**BODY, "state": "edited in an editor"}))
    with pytest.raises(Conflict, match="changed on disk") as caught:
        space.save_set("tickets", {**BODY, "state": "from the Studio"}, loaded)
    assert caught.value.status == 409
    assert caught.value.current == sha256(path.read_bytes())
    assert json.loads(path.read_text())["state"] == "edited in an editor"
    # Saving over the version now on disk works.
    space.save_set("tickets", {**BODY, "state": "after reload"}, caught.value.current)


def test_a_set_deleted_on_disk_is_a_conflict(space):
    loaded = space.save_set("tickets", BODY)
    (space.root / "sets" / "tickets.json").unlink()
    with pytest.raises(Conflict, match="was deleted") as caught:
        space.save_set("tickets", BODY, loaded)
    assert caught.value.current is None


def test_a_new_set_needs_a_free_name(space):
    digest = space.save_set("tickets", BODY)
    with pytest.raises(Conflict, match="exists already") as caught:
        space.save_set("tickets", {"questions": {}})
    assert caught.value.current == digest


def test_load_errors(space):
    with pytest.raises(NotFound) as caught:
        space.load_set("missing")
    assert caught.value.status == 404
    (space.root / "sets" / "broken.json").write_text("{nope")
    with pytest.raises(WorkspaceError, match="not valid JSON"):
        space.load_set("broken")


@pytest.mark.parametrize(
    "body, message",
    [
        ([], "JSON object"),
        ("text", "JSON object"),
        ({"questions": {}, "images": "x"}, "list of data URLs"),
        ({"questions": {}, "images": ["/etc/passwd"]}, "images.0"),
    ],
)
def test_a_set_must_be_a_request_body(space, body, message):
    with pytest.raises(WorkspaceError, match=message):
        space.save_set("tickets", body)
    assert not (space.root / "sets" / "tickets.json").exists()


def test_a_set_with_problems_can_still_be_saved(space):
    # The live check reports these; a set being edited is saved as it is.
    body = {"state": "", "questions": {"q": {"type": "choice", "criteria": {}}}}
    space.save_set("draft", body)
    assert space.load_set("draft")[0] == body


def test_rename_moves_the_set_and_its_history(space):
    space.save_set("tickets", BODY)
    space.append_history("tickets", {"request": BODY})
    space.rename_set("tickets", "support")
    assert [entry["name"] for entry in space.list_sets()] == ["support"]
    assert len(space.read_history("support")) == 1
    assert space.read_history("tickets") == []
    assert not (space.root / "history" / "tickets.jsonl").exists()


def test_rename_without_history_and_errors(space):
    space.save_set("tickets", BODY)
    space.save_set("taken", BODY)
    with pytest.raises(Conflict):
        space.rename_set("tickets", "taken")
    (space.root / "history" / "orphan.jsonl").write_text("")
    with pytest.raises(Conflict):
        space.rename_set("tickets", "orphan")  # its history name is taken
    with pytest.raises(NotFound):
        space.rename_set("missing", "other")
    with pytest.raises(WorkspaceError):
        space.rename_set("tickets", "../x")
    space.rename_set("tickets", "tickets")
    space.rename_set("tickets", "renamed")
    assert space.load_set("renamed")[0] == BODY


def test_rename_changing_case_only(space):
    space.save_set("tickets", BODY)
    space.rename_set("tickets", "Tickets")
    assert [entry["name"] for entry in space.list_sets()] == ["Tickets"]


def test_delete_removes_the_set_its_history_and_unused_images(space):
    image = png()
    space.save_set("tickets", BODY)
    space.save_set("other", BODY)
    space.append_history("tickets", {"request": {**BODY, "images": [image, png((0, 0, 255))]}})
    space.append_history("other", {"request": {**BODY, "images": [image]}})
    images = space.root / "history" / "images"
    assert len(list(images.iterdir())) == 2
    space.delete_set("tickets")
    assert [entry["name"] for entry in space.list_sets()] == ["other"]
    assert not (space.root / "history" / "tickets.jsonl").exists()
    # The image the other set's history still uses stays.
    assert [path.stem for path in images.iterdir()] == [
        space.read_history("other")[0]["request"]["images"][0]["sha256"]
    ]
    with pytest.raises(NotFound):
        space.delete_set("tickets")


# History


def test_history_lines_with_images_by_hash(space):
    space.save_set("tickets", BODY)
    image = png()
    run = {
        "model": "fake",
        "fingerprint": "f" * 64,
        "request": {**BODY, "images": [image, image]},
        "response": {"answers": {}},
        "latency_ms": 12.5,
        "load_ms": None,
        "truncated": False,
    }
    first = space.append_history("tickets", run)
    second = space.append_history("tickets", {**run, "model": "other"})
    raw = base64.b64decode(image.split(",", 1)[1])
    reference = {"sha256": sha256(raw), "type": "image/png"}
    assert first["request"]["images"] == [reference, reference]
    assert first["id"] != second["id"]
    assert first["time"].endswith("Z")
    stored = space.root / "history" / "images" / f"{sha256(raw)}.png"
    assert stored.read_bytes() == raw
    assert len(list(stored.parent.iterdir())) == 1  # stored once
    lines = (space.root / "history" / "tickets.jsonl").read_text().splitlines()
    assert [json.loads(text) for text in lines] == [first, second]
    assert space.read_history("tickets") == [first, second]
    assert image not in lines[0]
    assert run["request"]["images"] == [image, image]  # the caller's run is not changed


def test_restore_images(space):
    space.save_set("tickets", BODY)
    image = png()
    line = space.append_history("tickets", {"request": {**BODY, "images": [image]}})
    assert space.restore_images(line["request"]) == {**BODY, "images": [image]}
    assert space.restore_images(BODY) == BODY
    space.image_file(line["request"]["images"][0]["sha256"]).unlink()
    assert space.restore_images(line["request"]) == line["request"]
    assert space.image_file("../../etc") is None


def test_a_line_cut_short_by_a_crash_is_skipped(space):
    space.save_set("tickets", BODY)
    space.append_history("tickets", {"request": BODY, "model": "a"})
    path = space.root / "history" / "tickets.jsonl"
    with open(path, "ab") as handle:
        handle.write(b'{"id": "half", "requ')
    space.append_history("tickets", {"request": BODY, "model": "b"})
    assert [entry["model"] for entry in space.read_history("tickets")] == ["a", "b"]


def test_no_history_for_a_set_that_is_gone(space):
    """A run answered after its set was deleted leaves no history behind."""
    run = {"model": "fake", "request": BODY, "response": {"answers": {}}}
    assert space.append_history("gone", run) is None
    assert not space.history_path("gone").exists()
    space.save_set("gone", BODY)
    assert space.append_history("gone", run) is not None


@pytest.mark.parametrize(
    ("body", "where"),
    [
        ({**BODY, "images": "a.png"}, "images"),
        ({**BODY, "images": ["/etc/hosts"]}, "images.0"),
        ({**BODY, "examples": {"a": 1}}, "examples"),
        ({**BODY, "examples": [{"name": "A", "images": "a.png"}]}, "examples.0.images"),
        (
            {**BODY, "examples": [{"name": "A"}, {"name": "B", "images": ["/etc/hosts"]}]},
            "examples.1.images.0",
        ),
    ],
)
def test_a_sets_images_and_its_examples_images_must_be_data_urls(space, body, where):
    with pytest.raises(WorkspaceError, match=re.escape(where)):
        space.save_set("tickets", body)


def test_clear_history(space):
    space.append_history("tickets", {"request": {**BODY, "images": [png()]}})
    space.clear_history("tickets")
    space.clear_history("tickets")  # nothing to clear
    assert space.read_history("tickets") == []
    assert list((space.root / "history" / "images").iterdir()) == []


# Speed


def test_speed_records_keep_the_latest(space, monkeypatch):
    fingerprint = "a" * 64
    for index in range(SPEED_KEEP + 150):
        chip = "Apple M2" if index % 2 else "Apple M5 Pro"
        space.record_speed(fingerprint, {"chip": chip, "latency_ms": index, "input_tokens": 3})
    lines = (space.root / "speed" / f"{fingerprint}.jsonl").read_text().splitlines()
    assert SPEED_KEEP <= len(lines) <= SPEED_KEEP + SPEED_KEEP // 10
    records = space.read_speed(fingerprint)
    assert len(records) == SPEED_KEEP
    assert records[-1]["latency_ms"] == SPEED_KEEP + 149
    assert records[0]["latency_ms"] == 150
    on_m2 = space.read_speed(fingerprint, chip="Apple M2")
    assert {record["chip"] for record in on_m2} == {"Apple M2"}
    assert space.read_speed("b" * 64) == []
    with pytest.raises(WorkspaceError):
        space.record_speed("../x", {})


# The set's file is what run -q answers


def test_a_saved_set_answers_the_same_with_run_q(space, fake_model_path, fake_model):
    space.save_set("tickets", BODY)
    path = space.root / "sets" / "tickets.json"
    result = CliRunner().invoke(
        cli_app, ["run", "-m", str(fake_model_path), "-q", str(path), "--json"]
    )
    assert result.exit_code == 0, result.output
    body, _ = space.load_set("tickets")
    assert json.loads(result.stdout) == fake_model.decide_request(body).to_wire()


def test_a_saved_set_with_images_answers_the_same_with_run_q(space, tmp_path):
    import mlx_decision

    model_path = write_model(tmp_path / "seeing", supports_images=True)
    body = {**BODY, "images": [png()]}
    space.save_set("pictures", body)
    path = space.root / "sets" / "pictures.json"
    result = CliRunner().invoke(cli_app, ["run", "-m", str(model_path), "-q", str(path), "--json"])
    assert result.exit_code == 0, result.output
    expected = mlx_decision.load(model_path).decide_request(space.load_set("pictures")[0])
    assert json.loads(result.stdout) == expected.to_wire()


def test_two_saves_over_one_version_one_wins(space):
    from concurrent.futures import ThreadPoolExecutor

    loaded = space.save_set("tickets", BODY)

    def save(index):
        try:
            return space.save_set("tickets", {**BODY, "state": f"tab {index}"}, loaded)
        except Conflict:
            return None

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(save, range(16)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert space.load_set("tickets")[1] == winners[0]


def test_changes_are_reported(space):
    seen = []
    space.on_change = seen.append
    space.save_set("tickets", BODY)
    line = space.append_history("tickets", {"request": BODY})
    space.rename_set("tickets", "support")
    space.clear_history("support")
    space.delete_set("support")
    with pytest.raises(NotFound):
        space.delete_set("support")  # nothing changed: nothing reported
    assert seen == [
        {"folder": "sets", "name": "tickets", "action": "saved"},
        {"folder": "history", "name": "tickets", "action": "added", "id": line["id"]},
        {"folder": "sets", "name": "support", "action": "renamed", "old_name": "tickets"},
        {"folder": "history", "name": "support", "action": "cleared"},
        {"folder": "sets", "name": "support", "action": "deleted"},
    ]


def test_a_failing_change_callback_does_not_fail_the_change(space):
    def broken(change):
        raise RuntimeError("boom")

    space.on_change = broken
    space.save_set("tickets", BODY)
    assert space.load_set("tickets")[0] == BODY


def test_a_json_file_that_is_not_a_set_does_not_open(tmp_path):
    from mlx_decision.studio.workspace import Workspace, WorkspaceError

    files = Workspace(tmp_path / "ws")
    files.open()
    (tmp_path / "ws" / "sets" / "notaset.json").write_text("[1, 2, 3]")
    with pytest.raises(WorkspaceError, match="notaset.json is not a question set"):
        files.load_set("notaset")


# A folder that is not a workspace; a second Studio


def test_a_folder_of_other_files_is_refused_before_anything_changes(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "README.md").write_text("# My project\n")
    (project / ".draft.md.tmp").write_text("an editor's draft")
    with pytest.raises(ValueError, match="is not an mlx-decision workspace .*README.md"):
        Workspace(project).open()
    assert (project / "README.md").read_text() == "# My project\n"
    assert (project / ".draft.md.tmp").exists()
    assert sorted(p.name for p in project.iterdir()) == [".draft.md.tmp", "README.md", "src"]


def test_the_home_and_the_workspaces_own_folders_open(tmp_path):
    home = tmp_path / "home"  # the default workspace: config file, HF cache, models
    (home / "hf-cache").mkdir(parents=True)
    (home / "config.toml").write_text("port = 9000\n")
    (home / "models" / "converted").mkdir(parents=True)
    Workspace(home).open()
    assert (home / "workspace.json").is_file()
    older = tmp_path / "older"  # a Studio's workspace from before workspace.json
    older.mkdir()
    (older / "README.md").write_text(README)
    (older / "notes.txt").write_text("mine")
    Workspace(older).open()
    assert (older / "notes.txt").read_text() == "mine"


def test_a_second_studio_on_the_workspace_refuses_naming_the_first(tmp_path):
    import subprocess
    import sys
    import textwrap

    root = tmp_path / "ws"
    Workspace(root).open()
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sys, time
            from mlx_decision.studio.workspace import Workspace
            held = Workspace({str(root)!r})  # kept, as the Studio keeps its own
            held.open(holder={{"address": "http://127.0.0.1:8123"}})
            print("held", flush=True)
            time.sleep(60)
        """)],
        stdout=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        assert holder.stdout.readline().strip() == "held"
        leftover = root / "models" / ".fit.partial"  # the first one's work in progress
        leftover.mkdir()
        with pytest.raises(ValueError, match=r"in use by the Studio at http://127.0.0.1:8123 \("):
            Workspace(root).open(holder={"address": "http://127.0.0.1:8124"})
        assert leftover.exists()  # nothing cleaned up
    finally:
        holder.kill()
        holder.wait()
    Workspace(root).open(holder={"address": "http://127.0.0.1:8124"})  # free again
    assert "8124" in (root / "studio.lock").read_text()


def test_one_process_may_open_its_workspace_again(tmp_path):
    first = Workspace(tmp_path / "ws")
    first.open(holder={"address": "a"})
    second = Workspace(tmp_path / "ws")
    second.open(holder={"address": "b"})  # takes the lock over
    second.close()
    first.close()
