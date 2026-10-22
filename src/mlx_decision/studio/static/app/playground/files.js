// Question sets as files in the workspace: New (empty or from a template),
// Save, Save As, Rename, Delete, Import, Export, Show in Finder. A save names the version it was loaded
// from, so a file changed on disk since is never overwritten: the Studio
// says so and offers Reload or Save As.
import { effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet, sheet, sheetError } from "../components/sheet.js";
import { navigate } from "../router.js";
import { reloadSets, sets } from "../store/sets.js";
import { toast } from "../store/ui.js";
import { TYPE_LABELS, toFile } from "./draft.js";
import { plural } from "../format.js";
import { saveDataset } from "./dataset.js";
import { workspaceChange } from "../stream.js";
import {
  datasetOfDocument,
  drafts,
  editedNames,
  forgetRuns,
  isDataset,
  isEdited,
  loadSet,
  loadedEntry,
  moveRuns,
  setEntry,
  setName,
} from "./session.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

// Changes from another tab (or the CLI): a set renamed there moves here too,
// edits and all; a dataset deleted while its questions are edited here says
// so, with Save as Question Set to keep them.
effect(() => {
  const change = workspaceChange.value;
  // Out of the effect: what follows reads and writes the drafts.
  if (change) queueMicrotask(() => followChange(change));
});

function followChange(change) {
  const all = drafts.peek();
  if (change.folder === "sets" && change.action === "renamed" && change.old_name) {
    const from = change.old_name;
    const value = all[from];
    if (!value || all[change.name]) return; // renamed here, or nothing open under the old name
    moveRuns(from, change.name);
    setEntry(change.name, { ...value, onDisk: null });
    setEntry(from, null);
    if (setName.peek() === from) {
      navigate("playground", change.name);
      toast(`Renamed to ${change.name} in another tab`);
    }
  }
  if (change.folder === "data" && change.action === "deleted") {
    const document = `data:${change.name}`;
    const value = all[document];
    if (!value?.draft) return;
    if (isEdited(value)) setEntry(document, { ...value, onDisk: "deleted" });
    else loadSet(document); // shows that it is gone
  }
}
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

// `Extra` is shown under the name field (New's template list).
function askName({ title, purpose, initial, action, onName, Extra = null }) {
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
    Body: Extra ? () => html`<${NameField} /><${Extra} />` : NameField,
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
  if (isDataset(setName.value)) return saveDataset();
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
  if (isDataset(from)) return saveAsSet(from);
  askName({
    title: "Save As",
    purpose: "The open set stays as it was saved.",
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

// Save as Question Set: a dataset's questions and the rows shown, as a new set.
// The dataset keeps its edits.
function saveAsSet(from) {
  const dataset = datasetOfDocument(from);
  askName({
    title: "Save as Question Set",
    purpose: `A new set with ${dataset}'s questions and the rows shown as its examples. The dataset stays as it is.`,
    initial: freeName(dataset),
    action: "Save",
    onName: async (name) => {
      const { body, problems } = toFile(drafts.value[from].draft);
      const own = problems.filter((problem) => !problem.where?.example);
      if (own.length) {
        toast(`${own[0].message}: fix it before saving`);
        return;
      }
      const sha256 = await write(name, body, null);
      adopt(name, body, sha256);
      navigate("playground", name);
      toast(`Saved ${fileOf(name)}`);
    },
  });
}

// ---------- New, from a template ----------

// The templates the Studio ships (fetched once), the one picked (null: an
// empty set) and the name it suggested, which a pick replaces only while
// nobody has typed over it.
const templates = signal(null);
const picked = signal(null);
let suggested = "";

function pick(name) {
  picked.value = name;
  if (nameText.value === suggested) {
    suggested = freeName(name ?? "untitled");
    nameText.value = suggested;
    nameError.value = null;
  }
}

// A double-click on a row creates the set, as the primary button does.
function create() {
  sheet.value?.actions?.find((action) => action.primary)?.onSelect();
}

function TemplateRow({ name, title, description, tags }) {
  const on = picked.value === name;
  return html`
    <button type="button" role="radio" aria-checked=${on} tabindex=${on ? 0 : -1}
      class=${`tpl${on ? " on" : ""}`} data-template=${name ?? ""}
      onClick=${() => pick(name)} onDblClick=${() => (pick(name), create())}>
      <span class="tpl-title">${title}</span>
      <span class="tpl-tags">${tags.map((tag) => html`<span class="chip">${tag}</span>`)}</span>
      <span class="tpl-desc">${description}</span>
    </button>
  `;
}

function TemplateList() {
  const listed = templates.value;
  const rows = [
    { name: null, title: "Empty set", description: "A blank state and no questions yet.", tags: [] },
    ...(listed ?? []).map((item) => ({
      name: item.name,
      title: item.title,
      description: item.description,
      tags: [
        ...item.types.map((type) => TYPE_LABELS[type]),
        ...(item.images ? ["Images"] : []),
        `${item.examples} examples`,
      ],
    })),
  ];
  // Arrow keys move the pick, as in any radio group; Enter creates.
  const onKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      create();
      return;
    }
    const step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const group = event.currentTarget;
    const at = rows.findIndex((row) => row.name === picked.value);
    const next = rows[Math.min(rows.length - 1, Math.max(0, at + step))];
    pick(next.name);
    requestAnimationFrame(() => group.querySelector(`[data-template="${next.name ?? ""}"]`)?.focus());
  };
  return html`
    <div class="tpl-list" role="radiogroup" aria-label="Start from" onKeyDown=${onKeyDown}>
      <h3>Start from</h3>
      ${rows.map((row) => html`<${TemplateRow} key=${row.name ?? ""} ...${row} />`)}
      ${listed === null && html`<div class="dim tpl-desc">Loading templates…</div>`}
    </div>
  `;
}

export function newSet() {
  picked.value = null;
  suggested = freeName("untitled");
  if (templates.value === null)
    api("/templates")
      .then((data) => (templates.value = data.templates))
      .catch(() => (templates.value = []));
  askName({
    title: "New Question Set",
    purpose: "Start empty or from a template.",
    initial: suggested,
    action: "Create",
    Extra: TemplateList,
    onName: async (name) => {
      const template = picked.value;
      if (template) {
        const made = await api(`/sets/${encodeURIComponent(name)}/from-template`, {
          method: "POST",
          body: { template },
        });
        adopt(name, made.body, made.sha256);
      } else {
        const sha256 = await write(name, EMPTY_SET, null);
        adopt(name, EMPTY_SET, sha256);
      }
      navigate("playground", name);
    },
  });
}

// Rename the open set; its history and this tab's edits and answers go with it.
export function renameSet() {
  const from = setName.value;
  if (!from) return;
  askName({
    title: "Rename Question Set",
    purpose: "Its history moves with it; unsaved edits stay unsaved.",
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
// Delete a set (the open one by default) and its history, after asking.
export async function deleteSet(name = setName.value) {
  if (!name) return;
  let count = 0;
  try {
    count = (await api(`/history/${encodeURIComponent(name)}`)).runs.length;
  } catch {
    // No history to count: the question still says what goes.
  }
  const edits = editedNames.value.has(name) ? " Your unsaved edits go too." : "";
  openSheet({
    title: `Delete the question set ${name}?`,
    purpose: count
      ? `The set's file and its history (${plural(count, "run")}) are deleted. This cannot be undone.${edits}`
      : `The set's file is deleted. This cannot be undone.${edits}`,
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
              sheetError(`Not deleted: ${error.message}`);
              return;
            }
          }
          closeSheet();
          forgetRuns(name);
          setEntry(name, null);
          reloadSets();
          if (name === setName.value) {
            const next = (sets.value ?? []).find((item) => item.name !== name)?.name ?? null;
            navigate("playground", next ?? undefined);
          }
          toast(`Deleted the question set ${name}`);
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
  const where = isDataset(name) ? `data/${datasetOfDocument(name)}/dataset.json` : fileOf(name);
  toast(`Reloaded ${where}; your edits are gone`);
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
    toast(`Could not show it in Finder: ${error.message}`);
  }
}

// Leaving the page with unsaved edits asks first.
addEventListener("beforeunload", (event) => {
  if (editedNames.value.size === 0) return;
  event.preventDefault();
  event.returnValue = "";
});
