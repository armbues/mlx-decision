// Training data: the sidebar's list with "+" (upload a JSON-lines file, checked
// as training fit checks it) and the inspector's Data for the file picked.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { api } from "../api.js";
import * as icons from "../components/icons.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { href, navigate, route } from "../router.js";
import { dataFiles } from "../store/data.js";
import { toast } from "../store/ui.js";
import { openDatasets } from "./datasets.js";

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

// The data file the page shows (#/training/NAME).
export const pickedData = () => (route.value.page === "training" ? route.value.item : null);

// ---------- upload ----------

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
    navigate("training", result.name);
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
    title: "Add Training Data",
    purpose: `${message}. Add ${file.name} under another name:`,
    Body: () => html`
      <label class="name-field">
        <span>Name</span>
        <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${newName.value}
               onInput=${(event) => (newName.value = event.target.value)} />
        <span class="suffix">.jsonl</span>
      </label>`,
    note: "Training data lives in the workspace's data/ folder.",
    actions: () => [{ label: "Add", primary: true, disabled: !ok(), onSelect: () => send(file, newName.value.trim(), text) }],
  });
}

// The file does not check: its errors by line, as training fit would name them.
function showErrors(file, errors) {
  openSheet({
    title: `${file.name} cannot be used`,
    purpose: "Training data is JSON lines: each a request body (state, questions) plus labels, an object from question id to the correct answer. Fix these lines and add the file again.",
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

// ---------- the sidebar's list ----------

export function DataList() {
  const list = dataFiles.value;
  const picked = pickedData();
  return html`
    <div class="sb-sec">
      <h3>
        Training Data
        <span class="sb-actions">
          <button aria-label="Get a dataset" title="Get a Dataset… (a public dataset, as training data)" onClick=${openDatasets}>
            <${icons.Download} />
          </button>
          <button aria-label="Add training data" title="Add Training Data (a JSON-lines file of labelled examples)" onClick=${uploadData}>
            <${icons.Plus} />
          </button>
        </span>
      </h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No data yet</p>`}
      <ul class="sb-list data-list">
        ${(list ?? []).map(
          (item) => html`
            <li key=${item.name}>
              <a class="sb-item" href=${href("training", item.name)} aria-current=${item.name === picked ? "true" : undefined}>
                <${icons.File} /><span class="label">${item.name}</span>
              </a>
            </li>
          `,
        )}
      </ul>
    </div>
  `;
}

// ---------- the inspector's Data ----------

const described = signal({}); // name -> summary or {error}

async function describe(name) {
  try {
    const summary = await api(`/data/${encodeURIComponent(name)}`);
    described.value = { ...described.value, [name]: summary };
  } catch (error) {
    described.value = { ...described.value, [name]: { error: error.message } };
  }
}

function Labels({ question, total }) {
  const most = Math.max(1, ...question.labels.map(([, count]) => count));
  const counts = new Map(question.labels);
  const options = question.options ?? question.labels.map(([label]) => label);
  return html`
    <ul class="label-counts">
      ${options.map((label) => {
        const count = counts.get(label) ?? 0;
        return html`<li key=${label} class=${count ? "" : "none"}>
          <code>${label}</code>
          <span class="bar"><i style=${`width:${((count / most) * 100).toFixed(1)}%`}></i></span>
          <span class="num">${count}</span>
        </li>`;
      })}
    </ul>
    <small>${question.labelled} of ${plural(total, "example")} labelled</small>
  `;
}

export function DataInspector() {
  const name = pickedData();
  const files = dataFiles.value ?? [];
  const modified = files.find((item) => item.name === name)?.modified;
  useEffect(() => {
    if (name) describe(name);
  }, [name, modified]);
  if (!name)
    return html`<div class="empty"><b>No data picked</b><span>Pick training data on the left, or add a JSON-lines file with +.</span></div>`;
  const summary = described.value[name];
  if (!summary) return null;
  if (summary.error) return html`<p class="warn-text">${summary.error}</p>`;
  return html`
    <div class="idet data-inspector">
      <div class="idet-head"><div><h2 class="mono">${name}</h2><p>${plural(summary.examples, "example")} · ${plural(summary.questions.length, "question")}</p></div></div>
      ${summary.questions.map(
        (question) => html`<section class="igroup" key=${question.id}><h4>${question.id} · ${question.type}</h4><dl>
          ${question.instructions && html`<div class="kv"><dt>Asks</dt><dd>${question.instructions}</dd></div>`}
          <div class="kv"><dt>Labels</dt><dd><${Labels} question=${question} total=${summary.examples} /></dd></div>
        </dl></section>`,
      )}
      <section class="igroup"><h4>Examples</h4><dl>
        ${summary.sample.map(
          (example) => html`<div class="kv" key=${example.id}><dt class="mono">${example.id}</dt>
            <dd>${example.state}<small>${Object.entries(example.labels).map(([id, label]) => `${id}: ${label}`).join(" · ") || "no labels"}</small></dd></div>`,
        )}
        ${summary.examples > summary.sample.length && html`<p class="note sample-note">The first ${summary.sample.length} of ${summary.examples}.</p>`}
      </dl></section>
      <section class="igroup"><h4>File</h4><dl>
        <div class="kv"><dt>Path</dt><dd><code>${summary.path}</code></dd></div>
        <div class="kv"><dt>Used by</dt><dd><code>mlx-decision training fit --data ${summary.path}</code></dd></div>
      </dl></section>
    </div>
  `;
}
