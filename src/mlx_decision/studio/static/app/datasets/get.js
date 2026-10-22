// Download: the `data` command's presets, each with its question, options and
// splits, or any Hugging Face dataset ("Another dataset…", its rows without a
// question). The table of splits gives each its role (train, report, evaluate
// or left out) or its parts, and its own random sample (1,000 rows by
// default, or the split's rows when fewer); the dataset's name (the first
// free one: not there, not written by a download that is running) names its
// folder in data/. It runs `data` as one job; the dataset joins the Datasets
// list when it is done. The sheet opens again on the entry last used.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { toast } from "../store/ui.js";
import { AnyDataset, anyJob, anyProblem, resetAny } from "./any-dataset.js";
import { NameAndSeed, SplitTable, freeBase, seeded, setSplits, splitState, splitsBody, splitsProblem } from "./splits.js";

const ANY = "\u0000any"; // the list's last entry: any Hugging Face dataset

const presets = signal(null);
const picked = signal(null); // preset name
const table = splitState();
const tableOf = signal(null); // the preset the table is for
const seed = signal("0");
const name = signal("");
const named = signal(false);
const starting = signal(false);

const preset = computed(() => presets.value?.find((item) => item.name === picked.value) ?? null);
const splits = computed(() =>
  (preset.value?.splits ?? []).map((split) => ({
    name: split.name,
    rows: split.rows,
    note: split.evaluation ? "the coverage suite evaluates on it" : null,
  })),
);
const suggested = computed(() => (preset.value ? freeBase(preset.value.name) : ""));
const base = computed(() => (named.value ? name.value.trim() : suggested.value));

// A preset, its table kept when it is the one shown before (``keep``).
function pick(presetName, keep = false) {
  picked.value = presetName;
  if (presetName === ANY) return;
  if (!(keep && tableOf.value === presetName)) {
    setSplits(table, splits.value, preset.value.default_split, true);
    tableOf.value = presetName;
  }
}

const problem = computed(() => {
  if (picked.value === ANY) return anyProblem.value;
  if (!preset.value) return "Pick a dataset";
  return splitsProblem(table, splits.value, base.value);
});

async function start() {
  if (problem.value || starting.value) return;
  let body;
  if (picked.value === ANY) body = anyJob();
  else {
    body = { kind: "data", preset: preset.value.name, splits: splitsBody(table, splits.value), name: base.value };
    if (seeded(table, splits.value)) body.seed = Number(seed.value) || 0;
  }
  starting.value = true;
  try {
    const { job } = await api("/jobs", { method: "POST", body });
    closeSheet();
    // The next Get starts afresh: "Another dataset…" empty, a preset's name free again.
    if (picked.value === ANY) resetAny(true);
    named.value = false;
    toast(`Started ${job.title}: ${job.details?.name ?? "the dataset"} shows up in Datasets when it is done`);
  } catch (error) {
    toast(`Could not start the download: ${error.message}`);
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
          <span><b>${item.title}</b><span>${item.summary}</span></span>
        </button>`,
      )}
      <button class="fam ds" role="option" aria-selected=${picked.value === ANY ? "true" : "false"} data-preset="any" onClick=${() => pick(ANY)}>
        <span><b>Another dataset…</b><span>any on the Hub, in the shape you pick</span></span>
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
    <p>From <code>${item.repo}</code>${item.config ? html` (${item.config})` : ""} on the Hugging Face Hub, at a pinned commit.</p>
    <div class="opts">
      <h4>Question <code>${question.id}</code></h4>
      <p class="ds-instructions">${question.instructions}</p>
      <ul class="ds-options">
        ${question.options.map(([option, text]) => html`<li><code>${option}</code><span>${text}</span></li>`)}
      </ul>
    </div>
    <div class="opts">
      <h4>Splits</h4>
      <${SplitTable} state=${table} splits=${splits.value} />
    </div>
    <${NameAndSeed} name=${name} named=${named} suggested=${suggested.value} seed=${seed} sampled=${seeded(table, splits.value)} />
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
    title: "Download a Dataset",
    purpose: "A public dataset from the Hugging Face Hub, as a dataset in the workspace's data/ folder: its splits, each with its role. A preset brings its question; another dataset brings its rows and label values, and you write its questions afterwards (multiple choice asks its question here). It downloads as a job.",
    Body,
    wide: true,
    note: "In a terminal, mlx-decision data does the same.",
    actions: () => [{ label: "Download", primary: true, disabled: Boolean(problem.value) || starting.value, onSelect: start }],
  });
}
