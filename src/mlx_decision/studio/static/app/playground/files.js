// Question sets as files in the workspace: New, Save, Save As, Duplicate,
// Rename, Delete, Import, Export, Show in Finder. A save names the version it was loaded
// from, so a file changed on disk since is never overwritten: the Studio
// says so and offers Reload or Save As.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { navigate } from "../router.js";
import { reloadSets, sets } from "../store/sets.js";
import { toast } from "../store/ui.js";
import { toFile } from "./draft.js";
import { plural } from "../format.js";
import {
  drafts,
  editedNames,
  forgetRuns,
  loadSet,
  loadedEntry,
  moveRuns,
  setEntry,
  setName,
} from "./session.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;
const EMPTY_SET = { state: "", questions: {} };

const takenNames = () => new Set((sets.value ?? []).map((item) => item.name));
const fileOf = (name) => `sets/${name}.json`;

export function freeName(base) {
  const taken = takenNames();
  const stem = base.slice(0, 76);
  if (!taken.has(stem)) return stem;
  let n = 2;
  while (taken.has(`${stem}-${n}`)) n += 1;
  return `${stem}-${n}`;
}

// A file name as a set's name: letters, digits, ".", "_" and "-" only.
export function nameFromFile(fileName) {
  const stem = fileName
    .replace(/\.json$/i, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^\.+/, "")
    .slice(0, 80);
  return stem || "imported";
}

function nameProblem(name) {
  if (name === "") return "Give the set a name";
  if (!NAME.test(name)) return "Use letters, digits, '.', '_' and '-' (not starting with '.'), at most 80";
  if (takenNames().has(name)) return `A set named ${name} exists already`;
  return null;
}

// ---------- the name sheet ----------

const nameText = signal("");
const nameError = signal(null);

function NameField() {
  const problem = nameError.value ?? (nameText.value ? nameProblem(nameText.value) : null);
  return html`
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" value=${nameText.value} spellcheck=${false} aria-label="Name"
             onInput=${(event) => {
               nameText.value = event.target.value;
               nameError.value = null;
             }} />
      <span class="dim">.json</span>
    </label>
    ${problem && html`<div class="field-problem sheet-problem" role="alert">${problem}</div>`}
  `;
}

function askName({ title, purpose, initial, action, onName }) {
  nameText.value = initial;
  nameError.value = null;
  const submit = async () => {
    const name = nameText.value.trim();
    const problem = nameProblem(name);
    if (problem) {
      nameError.value = problem;
      return;
    }
    try {
      await onName(name);
      closeSheet();
    } catch (error) {
      nameError.value = error.message;
    }
  };
  openSheet({
    title,
    purpose,
    Body: NameField,
    note: "Sets live in the workspace's sets/ folder; mlx-decision run -q reads them.",
    actions: [{ label: action, primary: true, onSelect: submit }],
  });
}

// ---------- the actions ----------

// The draft's file (every example), unless it has problems that would lose
// part of it (two questions, options or examples with one name).
function bodyToSave(name) {
  const value = drafts.value[name];
  if (!value?.draft) return null;
  const { body, problems } = toFile(value.draft);
  if (problems.length) {
    toast(`${problems[0].message}: fix it before saving`);
    return null;
  }
  return body;
}

async function write(name, body, loaded) {
  const result = await api(`/sets/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: { body, loaded },
  });
  return result.sha256;
}

// The new file `name` holds `body`; its draft starts from it.
function adopt(name, body, sha256, draft = null) {
  const value = loadedEntry({ body, sha256 });
  if (draft) value.draft = draft;
  setEntry(name, value);
}

export async function save() {
  const name = setName.value;
  const value = drafts.value[name];
  const body = bodyToSave(name);
  if (!body) return;
  try {
    const sha256 = await write(name, body, value.sha256);
    const current = drafts.value[name];
    setEntry(name, { ...current, sha256, baseline: JSON.stringify(body), onDisk: null });
    toast(`Saved ${fileOf(name)}`);
  } catch (error) {
    if (error.status === 409) conflict(name, error.current);
    else toast(`Not saved: ${error.message}`);
  }
}

// The file changed (or went) since it was loaded: Reload, Save As, or for a
// deleted file, Save Again.
function conflict(name, current) {
  const gone = current === null;
  const actions = [{ label: "Save As…", onSelect: () => saveAs() }];
  if (gone)
    actions.push({
      label: "Save Again",
      primary: true,
      onSelect: async () => {
        closeSheet();
        const body = bodyToSave(name);
        if (!body) return;
        try {
          const sha256 = await write(name, body, null);
          adopt(name, body, sha256, drafts.value[name].draft);
          toast(`Saved ${fileOf(name)}`);
        } catch (error) {
          toast(`Not saved: ${error.message}`);
        }
      },
    });
  else
    actions.unshift({
      label: "Reload",
      onSelect: () => {
        closeSheet();
        loadSet(name);
        toast(`Reloaded ${fileOf(name)}; your edits are gone`);
      },
    });
  openSheet({
    title: gone ? `${fileOf(name)} was deleted` : `${fileOf(name)} changed on disk`,
    purpose: gone
      ? "It was deleted after you opened it. Your edits are not saved yet."
      : "Something changed the file after you opened it, so saving would overwrite that. Your edits are not saved yet.",
    note: gone ? "" : "Reload shows the file as it is now and drops your edits.",
    actions,
  });
}

export function saveAs() {
  const from = setName.value;
  if (!drafts.value[from]?.draft) return;
  askName({
    title: "Save As",
    purpose: "Save the questions and state as a new set. The open set stays as it was saved.",
    initial: freeName(`${from}-copy`),
    action: "Save",
    onName: async (name) => {
      const body = bodyToSave(from);
      if (!body) return;
      const draft = drafts.value[from].draft;
      const sha256 = await write(name, body, null);
      adopt(name, body, sha256, draft);
      if (drafts.value[from]?.onDisk !== "deleted") loadSet(from); // drop the edits there
      else setEntry(from, null);
      navigate("playground", name);
      toast(`Saved ${fileOf(name)}`);
    },
  });
}

export function newSet() {
  askName({
    title: "New Question Set",
    purpose: "Start an empty set: a state and the questions to ask about it.",
    initial: freeName("untitled"),
    action: "Create",
    onName: async (name) => {
      const sha256 = await write(name, EMPTY_SET, null);
      adopt(name, EMPTY_SET, sha256);
      navigate("playground", name);
    },
  });
}

// A copy of the open set, edits included, under a free name; the original
// keeps its edits.
export async function duplicate() {
  const from = setName.value;
  const body = bodyToSave(from);
  if (!body) return;
  const name = freeName(`${from}-copy`);
  try {
    const sha256 = await write(name, body, null);
    adopt(name, body, sha256);
    navigate("playground", name);
    toast(`Duplicated as ${fileOf(name)}`);
  } catch (error) {
    toast(`Not duplicated: ${error.message}`);
  }
}

// Rename the open set; its history and this tab's edits and answers go with it.
export function renameSet() {
  const from = setName.value;
  if (!from) return;
  askName({
    title: "Rename",
    purpose: `A new name for ${from}. Its history moves with it; unsaved edits stay unsaved.`,
    initial: from,
    action: "Rename",
    onName: async (name) => {
      const value = drafts.value[from] ?? null;
      await api(`/sets/${encodeURIComponent(from)}/rename`, { method: "POST", body: { name } });
      moveRuns(from, name);
      setEntry(name, value);
      setEntry(from, null);
      reloadSets();
      navigate("playground", name);
      toast(`Renamed to ${fileOf(name)}`);
    },
  });
}

// Delete the open set and its history, after asking; the next set opens.
export async function deleteSet() {
  const name = setName.value;
  if (!name) return;
  let count = 0;
  try {
    count = (await api(`/history/${encodeURIComponent(name)}`)).runs.length;
  } catch {
    // No history to count: the question still says what goes.
  }
  const edits = editedNames.value.has(name) ? " Your unsaved edits go too." : "";
  openSheet({
    title: `Delete ${name}?`,
    purpose: count
      ? `The set's file and its history (${plural(count, "run")}) are deleted. This cannot be undone.${edits}`
      : `The set's file is deleted. This cannot be undone.${edits}`,
    note: "Export keeps a copy of the file first.",
    actions: [
      {
        label: "Delete",
        primary: true,
        danger: true,
        onSelect: async () => {
          try {
            await api(`/sets/${encodeURIComponent(name)}`, { method: "DELETE" });
          } catch (error) {
            if (error.status !== 404) {
              toast(`Not deleted: ${error.message}`);
              return;
            }
          }
          closeSheet();
          forgetRuns(name);
          setEntry(name, null);
          reloadSets();
          const next = (sets.value ?? []).find((item) => item.name !== name)?.name ?? null;
          navigate("playground", next ?? undefined);
          toast(`Deleted ${fileOf(name)}`);
        },
      },
    ],
  });
}

// Pick a JSON file (a -q questions map or a whole request body) and add it as
// a set.
export function importSet() {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ".json,application/json";
  picker.addEventListener("change", async () => {
    const file = picker.files?.[0];
    if (file) await importFile(file);
  });
  picker.click();
}

export async function importFile(file) {
  try {
    const result = await api("/sets/import", {
      method: "POST",
      body: { name: nameFromFile(file.name), text: await file.text() },
    });
    adopt(result.name, result.body, result.sha256);
    navigate("playground", result.name);
    toast(`Imported ${file.name} as ${fileOf(result.name)}`);
  } catch (error) {
    toast(`Not imported: ${error.message}`);
  }
}

export function reloadFromDisk() {
  const name = setName.value;
  loadSet(name);
  toast(`Reloaded ${fileOf(name)}; your edits are gone`);
}

export function exportSet() {
  const name = setName.value;
  const link = document.createElement("a");
  link.href = `/studio/api/sets/${encodeURIComponent(name)}/export`;
  link.download = `${name}.json`;
  link.click();
}

export async function reveal() {
  try {
    await api(`/sets/${encodeURIComponent(setName.value)}/reveal`, { method: "POST" });
  } catch (error) {
    toast(error.message);
  }
}

// Leaving the page with unsaved edits asks first.
addEventListener("beforeunload", (event) => {
  if (editedNames.value.size === 0) return;
  event.preventDefault();
  event.returnValue = "";
});
