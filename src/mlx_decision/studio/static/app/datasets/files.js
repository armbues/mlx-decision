// The datasets: the sidebar's list (one row per dataset, marked when it is not
// ready for a fit) with Download and Add (a JSON-lines file checked as training
// fit checks it, or a CSV / JSON-lines file of columns kept as a dataset
// without questions), Delete and Show in Finder.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import * as icons from "../components/icons.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { fileSize, plural } from "../format.js";
import { href, navigate, route } from "../router.js";
import { reveal } from "../settings/places.js";
import { dataFiles, dataFolder, datasetOf, forget, nameTaken, reloadData } from "../store/data.js";
import { toast } from "../store/ui.js";
import { openGetDataset } from "./get.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

// A file's name as a workspace name: its stem, other characters as "-".
export function dataName(fileName) {
  const stem = fileName
    .replace(/\.(jsonl|json|ndjson|txt|csv)$/i, "")
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
  picker.accept = ".jsonl,.ndjson,.json,.txt,.csv,application/jsonl,application/json,text/csv";
  picker.addEventListener("change", async () => {
    const file = picker.files?.[0];
    if (!file) return;
    if (!dataFolder.value) await reloadData(); // the upload limit and the folder, if not read yet
    const limit = dataFolder.value?.upload_limit;
    if (limit && file.size > limit) tooLarge(file);
    else if (/\.csv$/i.test(file.name)) await askColumns(file, await file.text(), "csv");
    else await send(file, dataName(file.name), await file.text());
  });
  picker.click();
}

// ``columns``: {format, text_column, label_column} for rows of columns.
async function send(file, name, text, columns = null) {
  try {
    const result = await api("/data", { method: "POST", body: { name, text, ...(columns ?? {}) } });
    closeSheet();
    navigate("datasets", result.name);
    const where = result.kind === "folder" ? `data/${result.name}/` : `data/${result.name}.jsonl`;
    const more = result.missing?.length ? " · no questions yet" : "";
    toast(`Added ${file.name} as ${where} · ${plural(result.examples, "example")}${more}`);
  } catch (error) {
    if (error.status === 409) askName(file, name, text, error.message, columns);
    else if (error.status === 413) tooLarge(file);
    else if (error.status === 400 && error.body?.error?.errors) showErrors(file, text, error.body.error.errors);
    else toast(`Not added: ${error.message}`);
  }
}

// Rows of columns (a CSV, or JSON lines that are not training lines): which
// column is the state and which holds the labels; added without questions.
const described = signal(null); // POST /data/columns
const stateColumn = signal(null);
const valueColumn = signal(null);
const rowsName = signal("");

async function askColumns(file, text, format) {
  let found;
  try {
    found = await api("/data/columns", { method: "POST", body: { text, format } });
  } catch (error) {
    toast(`Cannot read ${file.name}: ${error.message}`);
    return;
  }
  described.value = found;
  stateColumn.value = found.text ?? found.text_columns[0] ?? null;
  valueColumn.value = found.label ?? found.label_columns[0]?.name ?? null;
  let name = dataName(file.name);
  for (let n = 2; nameTaken(name); n += 1) name = `${dataName(file.name)}-${n}`;
  rowsName.value = name;
  const problem = () => {
    if (!stateColumn.value) return "No column holds text for the state";
    if (!valueColumn.value) return "No column has a few values to use as labels";
    if (stateColumn.value === valueColumn.value) return "The state and the labels are different columns";
    const value = rowsName.value.trim();
    if (!NAME.test(value)) return "Name it: letters, digits, '.', '_' or '-'";
    return nameTaken(value);
  };
  openSheet({
    title: `Add ${file.name}`,
    purpose: `${plural(found.rows, "row")}: one column becomes the state, another holds the labels. The dataset arrives without questions: you write them afterwards, and say what each label value means.`,
    Body: () => {
      const info = described.value;
      const values = info.label_columns.find((column) => column.name === valueColumn.value);
      return html`
        <div class="opts">
          <label class="any-row"><span>State</span>
            <select aria-label="State column" value=${stateColumn.value ?? ""} onChange=${(event) => (stateColumn.value = event.target.value)}>
              ${info.text_columns.map((item) => html`<option value=${item}>${item}</option>`)}
            </select><small>the text the model reads</small></label>
          <label class="any-row"><span>Labels</span>
            <select aria-label="Label column" value=${valueColumn.value ?? ""} onChange=${(event) => (valueColumn.value = event.target.value)}>
              ${info.label_columns.map((item) => html`<option value=${item.name}>${item.name} (${plural(item.classes.length, "value")})</option>`)}
            </select><small>its values become the answers</small></label>
          ${values && html`<p class="hint">${values.classes.map(([value, count]) => `${value} (${count})`).join(", ")}</p>`}
          <label class="name-field">
            <span>Name</span>
            <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${rowsName.value}
                   onInput=${(event) => (rowsName.value = event.target.value)} />
            <span class="suffix">/</span>
          </label>
          ${problem() && html`<p class="warn-text">${problem()}</p>`}
        </div>`;
    },
    wide: true,
    note: "Datasets live in the workspace's data/ folder.",
    actions: () => [
      {
        label: "Add",
        primary: true,
        disabled: Boolean(problem()),
        onSelect: () =>
          send(file, rowsName.value.trim(), text, { format, text_column: stateColumn.value, label_column: valueColumn.value }),
      },
    ],
  });
}

const newName = signal("");

// A name is taken: ask for another one.
function askName(file, name, text, message, columns = null) {
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
        <span class="suffix">${columns ? "/" : ".jsonl"}</span>
      </label>`,
    note: "Datasets live in the workspace's data/ folder.",
    actions: () => [{ label: "Add", primary: true, disabled: !ok(), onSelect: () => send(file, newName.value.trim(), text, columns) }],
  });
}

// The file does not check: its errors by line, as training fit would name them;
// or it is rows of columns, to add as such.
function showErrors(file, text, errors) {
  openSheet({
    title: `${file.name} is not training data`,
    purpose: "Training data is JSON lines: each a request body (state, questions) plus labels, an object from question id to the correct answer. Fix these lines and add the file again, or add its lines as rows of columns: a column for the state, one for the labels, the questions written afterwards.",
    Body: () => html`
      <ul class="data-errors">
        ${errors.map(
          (error) => html`<li><span class="num">${error.line ? `Line ${error.line}` : "The file"}</span>
            <span>${error.message}${error.param ? html` <code>${error.param}</code>` : ""}</span></li>`,
        )}
      </ul>
      ${errors.length >= 20 && html`<p class="note">The first 20 problems; there may be more.</p>`}`,
    wide: true,
    actions: [{ label: "Add as Rows…", onSelect: () => askColumns(file, text, "jsonl") }],
  });
}

// Too large for an upload: copy it into data/ instead, where the list finds it.
function tooLarge(file) {
  const folder = dataFolder.value;
  const limit = folder?.upload_limit;
  openSheet({
    title: `${file.name} is too large to add here`,
    purpose: `It is ${fileSize(file.size)}; the Studio takes files up to ${limit ? fileSize(limit) : "its upload limit"} this way. Copy it into the workspace's data folder instead, as ${dataName(file.name)}.jsonl (training data: JSON lines with questions and labels): it joins Datasets under that name, and it is checked when it is read.`,
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
  const item = datasetOf(name);
  const folder = item?.kind === "folder";
  const where = folder ? `data/${name}/` : `data/${name}.jsonl`;
  const splits = folder ? ` (${plural(item.splits.length, "split")})` : "";
  openSheet({
    title: `Delete ${name}?`,
    purpose: `The dataset ${where}${splits} is deleted${folder ? ", and the history of its questions tried in the Playground" : ""}. This cannot be undone; models fitted on it stay.`,
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
          toast(`Deleted ${where}`);
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

// What keeps a dataset from a fit, said on its row; null when it is ready.
export const notReadyText = (item) => (item?.error ? item.error : item?.missing?.length ? `Not ready for a fit: ${item.missing.join("; ")}` : null);
// A plain file of a folder's name, which the list cannot show beside it.
const hiddenText = (item) => (item?.hides ? `data/${item.hides} has this name too and is not listed: rename it to see it` : null);

function Row({ item, picked }) {
  const open = item.name === picked;
  const problem = notReadyText(item);
  const splits = item.kind === "folder" ? item.splits.length : 0;
  return html`
    <li key=${item.name} class=${open ? "open" : ""}>
      <a class="sb-item" href=${href("datasets", item.name)} aria-current=${open ? "true" : undefined}
         title=${[problem ?? (splits > 1 ? `${item.name}: ${item.splits.map((split) => split.name).join(", ")}` : item.name), hiddenText(item)].filter(Boolean).join(" · ")}>
        <${icons.File} /><span class="label">${item.name}</span>
        ${problem
          ? html`<span class="mark" aria-label="Not ready for a fit"></span>`
          : splits > 1 && html`<span class="count">${splits}</span>`}
      </a>
      <button class="sb-delete" aria-label=${`Delete ${item.name}…`} title="Delete…"
        onClick=${() => deleteData(item.name)}><${icons.Trash} /></button>
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
          <button aria-label="Download a dataset" title="Download a Dataset… (a public dataset from the Hugging Face Hub)" onClick=${openGetDataset}>
            <${icons.Download} />
          </button>
          <button aria-label="Add a dataset" title="Add a Dataset… (training data as JSON lines, or rows of columns as CSV or JSON lines)" onClick=${uploadData}>
            <${icons.Plus} />
          </button>
        </span>
      </h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No datasets yet</p>`}
      <ul class="sb-list data-list">
        ${(list ?? []).map((item) => html`<${Row} key=${item.name} item=${item} picked=${picked} />`)}
      </ul>
    </div>
  `;
}
