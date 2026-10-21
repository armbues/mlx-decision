// Datasets as files: the sidebar's list with Download and Add (upload a
// JSON-lines file, checked as training fit checks it), Delete and Show in Finder.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import * as icons from "../components/icons.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { fileSize, plural } from "../format.js";
import { href, navigate, route } from "../router.js";
import { reveal } from "../settings/places.js";
import { dataFiles, dataFolder, described, forget, nameTaken } from "../store/data.js";
import { toast } from "../store/ui.js";
import { openGetDataset } from "./get.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

// A file's name as a workspace name: its stem, other characters as "-".
export function dataName(fileName) {
  const stem = fileName
    .replace(/\.(jsonl|json|ndjson|txt)$/i, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^\.+/, "")
    .slice(0, 80);
  return stem || "data";
}

// The dataset the Datasets page shows (#/datasets/NAME).
export const pickedDataset = () => (route.value.page === "datasets" ? route.value.item : null);

// ---------- add ----------

export function uploadData() {
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ".jsonl,.ndjson,.json,.txt,application/jsonl,application/json";
  picker.addEventListener("change", async () => {
    const file = picker.files?.[0];
    if (!file) return;
    const limit = dataFolder.value?.upload_limit;
    if (limit && file.size > limit) tooLarge(file);
    else await send(file, dataName(file.name), await file.text());
  });
  picker.click();
}

async function send(file, name, text) {
  try {
    const result = await api("/data", { method: "POST", body: { name, text } });
    closeSheet();
    navigate("datasets", result.name);
    toast(`Added ${file.name} as data/${result.name}.jsonl · ${plural(result.examples, "example")}`);
  } catch (error) {
    if (error.status === 409) askName(file, name, text, error.message);
    else if (error.status === 413) tooLarge(file);
    else if (error.status === 400 && error.body?.error?.errors) showErrors(file, error.body.error.errors);
    else toast(`Not added: ${error.message}`);
  }
}

const newName = signal("");

// A name is taken: ask for another one.
function askName(file, name, text, message) {
  let n = 2;
  while (nameTaken(`${name}-${n}`)) n += 1;
  newName.value = `${name}-${n}`;
  const ok = () => NAME.test(newName.value.trim()) && !nameTaken(newName.value.trim());
  openSheet({
    title: "Add a Dataset",
    purpose: `${message}. Add ${file.name} under another name:`,
    Body: () => html`
      <label class="name-field">
        <span>Name</span>
        <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${newName.value}
               onInput=${(event) => (newName.value = event.target.value)} />
        <span class="suffix">.jsonl</span>
      </label>`,
    note: "Datasets live in the workspace's data/ folder.",
    actions: () => [{ label: "Add", primary: true, disabled: !ok(), onSelect: () => send(file, newName.value.trim(), text) }],
  });
}

// The file does not check: its errors by line, as training fit would name them.
function showErrors(file, errors) {
  openSheet({
    title: `${file.name} cannot be used`,
    purpose: "A dataset is JSON lines: each a request body (state, questions) plus labels, an object from question id to the correct answer. Fix these lines and add the file again.",
    Body: () => html`
      <ul class="data-errors">
        ${errors.map(
          (error) => html`<li><span class="num">${error.line ? `Line ${error.line}` : "The file"}</span>
            <span>${error.message}${error.param ? html` <code>${error.param}</code>` : ""}</span></li>`,
        )}
      </ul>
      ${errors.length >= 20 && html`<p class="note">The first 20 problems; there may be more.</p>`}`,
    done: "OK",
    wide: true,
  });
}

// Too large for an upload: copy it into data/ instead, where the list finds it.
function tooLarge(file) {
  const folder = dataFolder.value;
  const limit = folder?.upload_limit;
  openSheet({
    title: `${file.name} is too large to add here`,
    purpose: `It is ${fileSize(file.size)}; the Studio takes files up to ${limit ? fileSize(limit) : "its upload limit"} this way. Copy it into the workspace's data folder instead, as ${dataName(file.name)}.jsonl: it joins Datasets under that name, and it is checked when it is read.`,
    Body: () => html`
      <div class="name-field too-large">
        <span>Folder</span>
        <code class="mono" title=${folder?.folder ?? ""}>${folder?.display ?? "data/"}</code>
        <button class="btn sm" onClick=${() => reveal("data")}>Show in Finder</button>
      </div>`,
    done: "OK",
  });
}

// ---------- delete, Show in Finder ----------

// Delete a dataset, after asking; the page goes back to its start when it showed it.
export function deleteData(name) {
  const summary = described.value[name];
  const count = summary?.examples != null ? ` (${plural(summary.examples, "example")})` : "";
  openSheet({
    title: `Delete ${name}?`,
    purpose: `The dataset data/${name}.jsonl${count} is deleted. This cannot be undone; models fitted on it stay.`,
    actions: [
      {
        label: "Delete",
        primary: true,
        danger: true,
        onSelect: async () => {
          try {
            await api(`/data/${encodeURIComponent(name)}`, { method: "DELETE" });
          } catch (error) {
            if (error.status !== 404) {
              toast(`Not deleted: ${error.message}`);
              return;
            }
          }
          closeSheet();
          forget(name);
          dataFiles.value = (dataFiles.value ?? []).filter((item) => item.name !== name);
          if (pickedDataset() === name) navigate("datasets");
          toast(`Deleted data/${name}.jsonl`);
        },
      },
    ],
  });
}

export async function revealData(name) {
  try {
    await api(`/data/${encodeURIComponent(name)}/reveal`, { method: "POST" });
  } catch (error) {
    toast(`Could not show it in Finder: ${error.message}`);
  }
}

// ---------- the sidebar's list ----------

// The list as rows: the files of one Get (several splits) under its name, in
// the Get's order, the rest alone; both by name.
const rows = computed(() => {
  const list = dataFiles.value ?? [];
  const groups = new Map();
  const result = [];
  for (const item of list) {
    const get = item.record?.get;
    const key = get && get.files.length > 1 ? JSON.stringify(get.files.map((file) => file.name)) : null;
    if (!key) {
      result.push({ name: item.name, file: item });
      continue;
    }
    if (!groups.has(key)) {
      const group = { name: get.name, key, order: get.files.map((file) => file.name), source: item.record.source, files: [] };
      groups.set(key, group);
      result.push({ name: get.name, group });
    }
    groups.get(key).files.push(item);
  }
  for (const group of groups.values()) group.files.sort((a, b) => group.order.indexOf(a.name) - group.order.indexOf(b.name));
  // A Get with one file left is a file like the others.
  const shown = result.map((row) => (row.group && row.group.files.length === 1 ? { name: row.group.files[0].name, file: row.group.files[0] } : row));
  return shown.sort((a, b) => a.name.localeCompare(b.name, "en", { sensitivity: "base" }) || (a.group ? 1 : -1));
});

function FileRow({ item, picked, nested = false }) {
  const role = item.record?.role;
  return html`
    <li key=${item.name} class=${[item.name === picked ? "open" : "", nested ? "nested" : ""].filter(Boolean).join(" ")}>
      <a class="sb-item" href=${href("datasets", item.name)} aria-current=${item.name === picked ? "true" : undefined}>
        <${icons.File} /><span class="label">${item.name}</span>
        ${role && html`<span class="count role">${role}</span>`}
      </a>
      ${item.name === picked &&
      html`<button class="sb-delete" aria-label=${`Delete ${item.name}…`} title="Delete…"
        onClick=${() => deleteData(item.name)}><${icons.Trash} /></button>`}
    </li>
  `;
}

export function DatasetList() {
  const list = dataFiles.value;
  const picked = pickedDataset();
  return html`
    <div class="sb-sec">
      <h3>
        Datasets
        <span class="sb-actions">
          <button aria-label="Download a dataset" title="Download (a public dataset from the Hugging Face Hub)" onClick=${openGetDataset}>
            <${icons.Download} />
          </button>
          <button aria-label="Add a dataset" title="Add (a JSON-lines file of labelled examples)" onClick=${uploadData}>
            <${icons.Plus} />
          </button>
        </span>
      </h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No datasets yet</p>`}
      <ul class="sb-list data-list">
        ${rows.value.map((row) =>
          row.file
            ? html`<${FileRow} key=${row.name} item=${row.file} picked=${picked} />`
            : html`<li key=${row.group.key} class="sb-group" title=${`Got together from ${row.group.source}`}>
                <span class="sb-group-name">${row.group.name}</span>
                <ul class="sb-list">${row.group.files.map((item) => html`<${FileRow} key=${item.name} item=${item} picked=${picked} nested />`)}</ul>
              </li>`,
        )}
      </ul>
    </div>
  `;
}
