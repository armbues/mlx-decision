// Get a Dataset: the `data` command's presets, each with its question, options
// and splits; the split picked (train by default), an optional random sample
// (1,000 rows by default, or the split's rows when fewer) and the file's name
// (the first one free: not a file there, not one a Get is writing). It runs
// `data` as a job; the file joins the Datasets list when it is done. The
// sheet opens again on the entry last used.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { freeName, nameTaken } from "../store/data.js";
import { toast } from "../store/ui.js";
import { AnyDataset, anyJob, anyProblem, defaultSample, resetAny } from "./any-dataset.js";

const ANY = "\u0000any"; // the list's last entry: any Hugging Face dataset

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

const presets = signal(null);
const picked = signal(null); // preset name
const split = signal(null);
const sampling = signal(false);
const sample = signal("1000");
const sampleTyped = signal(false); // typed: another split keeps it
const seed = signal("0");
const name = signal("");
const named = signal(false);
const starting = signal(false);

const preset = computed(() => presets.value?.find((item) => item.name === picked.value) ?? null);
const splitRows = computed(() => preset.value?.splits.find((item) => item.name === split.value)?.rows ?? 0);

// The name and the sample for the split picked, unless typed.
function suggest() {
  if (!preset.value) return;
  if (!named.value) name.value = freeName(`${preset.value.name}-${split.value}`);
  if (!sampleTyped.value) sample.value = defaultSample(splitRows.value);
}

function pickSplit(splitName) {
  split.value = splitName;
  suggest();
}

// A preset on its default split; ``keep`` keeps the split picked before, if it has it.
function pick(presetName, keep = false) {
  picked.value = presetName;
  if (presetName === ANY) return;
  if (!(keep && preset.value.splits.some((item) => item.name === split.value))) split.value = preset.value.default_split;
  suggest();
}

const problem = computed(() => {
  if (picked.value === ANY) return anyProblem.value;
  if (!preset.value) return "Pick a dataset";
  const value = name.value.trim();
  if (!NAME.test(value)) return "Name the file: letters, digits, '.', '_' or '-'";
  const taken = nameTaken(value);
  if (taken) return taken;
  if (sampling.value) {
    const count = Number(sample.value);
    if (!Number.isInteger(count) || count < 1 || count > splitRows.value) return `A sample is 1 to ${splitRows.value} rows`;
  }
  return null;
});

async function start() {
  if (problem.value || starting.value) return;
  let body;
  if (picked.value === ANY) body = anyJob();
  else {
    body = { kind: "data", preset: preset.value.name, split: split.value, name: name.value.trim() };
    if (sampling.value) {
      body.sample = Number(sample.value);
      body.seed = Number(seed.value) || 0;
    }
  }
  starting.value = true;
  try {
    const { job } = await api("/jobs", { method: "POST", body });
    closeSheet();
    // The next Get starts afresh: "Another dataset…" empty, a preset's name free again.
    if (picked.value === ANY) resetAny(true);
    named.value = false;
    toast(`${job.title}: it joins Datasets when it is done`);
  } catch (error) {
    toast(`Could not get the dataset: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

function Presets() {
  return html`
    <div class="fam-list" role="listbox" aria-label="Datasets">
      ${presets.value.map(
        (item) => html`<button class="fam ds" role="option" aria-selected=${item.name === picked.value ? "true" : "false"}
          data-preset=${item.name} onClick=${() => pick(item.name)}>
          <span><b>${item.title}</b><span>${item.question.type} · ${plural(item.question.options.length, "option")}</span></span>
        </button>`,
      )}
      <button class="fam ds" role="option" aria-selected=${picked.value === ANY ? "true" : "false"} data-preset="any" onClick=${() => pick(ANY)}>
        <span><b>Another dataset…</b><span>any on the Hub, with your question set</span></span>
      </button>
    </div>
  `;
}

function Picked() {
  if (picked.value === ANY) return html`<${AnyDataset} />`;
  const item = preset.value;
  if (!item) return null;
  const question = item.question;
  return html`
    <p>${item.summary}. From <code>${item.repo}</code>${item.config ? html` (${item.config})` : ""} on the Hugging Face Hub, at a pinned commit.</p>
    <div class="opts">
      <h4>Question <code>${question.id}</code></h4>
      <p class="ds-instructions">${question.instructions}</p>
      <ul class="ds-options">
        ${question.options.map(([option, text]) => html`<li><code>${option}</code><span>${text}</span></li>`)}
      </ul>
    </div>
    <div class="opts">
      <h4>Split</h4>
      <div class="cat" role="radiogroup" aria-label="Split">
        ${item.splits.map(
          (entry) => html`<button class="crow" role="radio" aria-checked=${entry.name === split.value ? "true" : "false"}
            data-split=${entry.name} onClick=${() => pickSplit(entry.name)}>
            <span class="radio"></span>
            <span><code>${entry.name}</code>${entry.evaluation && html`<small>The coverage suite evaluates on it: keep it for training eval</small>`}</span>
            <span class="meta num">${entry.rows.toLocaleString("en")} rows</span>
          </button>`,
        )}
      </div>
      <label class="check ds-sample">
        <input type="checkbox" checked=${sampling.value} onChange=${(event) => (sampling.value = event.target.checked)} />
        A random sample of
        <input class="setting-input mono" aria-label="Rows" type="number" min="1" max=${splitRows.value} value=${sample.value}
               onFocus=${() => (sampling.value = true)} onInput=${(event) => {
                 sample.value = event.target.value;
                 sampleTyped.value = true;
               }} />
        rows, seed
        <input class="setting-input mono" aria-label="Seed" type="number" min="0" value=${seed.value}
               onInput=${(event) => (seed.value = event.target.value)} />
      </label>
    </div>
    <label class="name-field">
      <span>Name</span>
      <input class="sheet-input mono" aria-label="Name" spellcheck=${false} value=${name.value}
             onInput=${(event) => {
               name.value = event.target.value;
               named.value = true;
             }} />
      <span class="suffix">.jsonl</span>
    </label>
    ${problem.value && html`<p class="warn-text">${problem.value}</p>`}
  `;
}

function Body() {
  if (!presets.value) return null;
  return html`<div class="dl-panes"><${Presets} /><div class="fam-pane"><${Picked} /></div></div>`;
}

export async function openGetDataset() {
  try {
    presets.value = (await api("/datasets")).presets;
  } catch (error) {
    toast(`Could not list the datasets: ${error.message}`);
    return;
  }
  // The entry last used ("Another dataset…" as it was left), else the first.
  const known = picked.value === ANY || presets.value.some((item) => item.name === picked.value);
  if (picked.value !== ANY) named.value = false;
  pick(known ? picked.value : presets.value[0].name, true);
  openSheet({
    title: "Get a Dataset",
    purpose: "A public dataset turned into a dataset of labelled examples in the workspace's data/ folder, with the question and wording of the data command's presets. It downloads from the Hugging Face Hub as a job.",
    Body,
    wide: true,
    note: "In a terminal, mlx-decision data does the same.",
    actions: () => [{ label: "Get", primary: true, disabled: Boolean(problem.value) || starting.value, onSelect: start }],
  });
}
