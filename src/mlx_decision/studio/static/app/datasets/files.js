// Datasets as files: the sidebar's list with Get a Dataset… and Add… (upload a
// JSON-lines file, checked as training fit checks it), Delete and Show in Finder.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import * as icons from "../components/icons.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { href, navigate, route } from "../router.js";
import { dataFiles, described, forget } from "../store/data.js";
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
    if (file) await send(file, dataName(file.name), await file.text());
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
    else if (error.status === 400 && error.body?.error?.errors) showErrors(file, error.body.error.errors);
    else toast(`Not added: ${error.message}`);
  }
}

const newName = signal("");

// A name is taken: ask for another one.
function askName(file, name, text, message) {
  const taken = new Set((dataFiles.value ?? []).map((item) => item.name));
  let free = `${name}-2`;
  for (let n = 3; taken.has(free); n += 1) free = `${name}-${n}`;
  newName.value = free;
  const ok = () => NAME.test(newName.value.trim()) && !taken.has(newName.value.trim());
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

export function DatasetList() {
  const list = dataFiles.value;
  const picked = pickedDataset();
  return html`
    <div class="sb-sec">
      <h3>
        Datasets
        <span class="sb-actions">
          <button aria-label="Get a dataset" title="Get a Dataset… (a public dataset from the Hugging Face Hub)" onClick=${openGetDataset}>
            <${icons.Download} />
          </button>
          <button aria-label="Add a dataset" title="Add… (a JSON-lines file of labelled examples)" onClick=${uploadData}>
            <${icons.Plus} />
          </button>
        </span>
      </h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No datasets yet</p>`}
      <ul class="sb-list data-list">
        ${(list ?? []).map(
          (item) => html`
            <li key=${item.name} class=${item.name === picked ? "open" : ""}>
              <a class="sb-item" href=${href("datasets", item.name)} aria-current=${item.name === picked ? "true" : undefined}>
                <${icons.File} /><span class="label">${item.name}</span>
              </a>
              ${item.name === picked &&
              html`<button class="sb-delete" aria-label=${`Delete ${item.name}…`} title="Delete…"
                onClick=${() => deleteData(item.name)}><${icons.Trash} /></button>`}
            </li>
          `,
        )}
      </ul>
    </div>
  `;
}
