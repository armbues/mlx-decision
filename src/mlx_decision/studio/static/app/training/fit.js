// The fit form: the dataset picked in its own field (or by the Datasets page's
// Fit on It…), what the report is scored on (a share of it, or another dataset:
// the same Get's report file by default, from the records), its split and labels, the base model, the method cards the
// base's family offers (each with its memory estimate and the fit check, as
// `training fit`'s menu has them), the result's name and the options. Start Fit (⌘↩) starts `training fit` as a job;
// the Studio builds its command line.
import { computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { useSummary } from "../datasets/summary.js";
import { plural, size } from "../format.js";
import { gpuJob } from "../store/jobs.js";
import { remembered } from "../store/prefs.js";
import { href, navigate } from "../router.js";
import { dataFiles } from "../store/data.js";
import { loadedNames, models, status } from "../store/status.js";
import { toast } from "../store/ui.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

export const base = remembered("training.base", null);
const chosenData = remembered("training.data", null);
const holdout = signal(20); // percent kept aside for the report
const seed = signal(0);
const choice = signal(null); // "method" or "method+head" (the card picked)
const epochs = signal(3);
const earlyStop = signal(true);
const name = signal("");
const named = signal(false); // typed: a new plan keeps it
const starting = signal(false);

export const planned = signal(null); // POST /fit/plan
const planError = signal(null);

const key = (method) => (method.head_only ? `${method.method}+head` : method.method);
const picked = computed(() => planned.value?.methods.find((method) => key(method) === choice.value) ?? null);

// The base: the remembered one while it is listed, else the default, else the first.
const baseName = computed(() => {
  const list = (models.value ?? []).map((model) => model.name);
  if (base.value && list.includes(base.value)) return base.value;
  return status.value?.default_model && list.includes(status.value.default_model) ? status.value.default_model : (list[0] ?? null);
});

// The dataset: the one chosen while it is listed, else the first.
export const fitData = computed(() => {
  const list = (dataFiles.value ?? []).map((item) => item.name);
  return chosenData.value && list.includes(chosenData.value) ? chosenData.value : (list[0] ?? null);
});

// Fit on It… (the Datasets page): the form, with the dataset picked.
export function fitOn(name) {
  chosenData.value = name;
  navigate("training");
}

// What the report is scored on: "" (a share of the dataset) or another
// dataset's name. Picked for one dataset; another dataset starts from its
// default: the same Get's report file, when its record names one.
const reportChoice = signal(null); // {data, file}

const recordOf = (name) => (dataFiles.value ?? []).find((item) => item.name === name)?.record ?? null;

// The datasets a report can be scored on: not the dataset itself, not one
// kept for Evaluate.
export const reportFiles = computed(() =>
  (dataFiles.value ?? []).filter((item) => item.name !== fitData.value && item.record?.role !== "evaluate").map((item) => item.name),
);

function defaultReport(data) {
  const sibling = recordOf(data)?.get?.files?.find((file) => file.role === "report" && file.name !== data);
  return sibling && reportFiles.value.includes(sibling.name) ? sibling.name : "";
}

export const reportFile = computed(() => {
  const data = fitData.value;
  const chosen = reportChoice.value;
  if (chosen?.data === data && (chosen.file === "" || reportFiles.value.includes(chosen.file))) return chosen.file;
  return data ? defaultReport(data) : "";
});

let timer = null;
let asked = 0;

// The plan again whenever the data, the report, the base, the share kept aside or the seed change.
effect(() => {
  const data = fitData.value;
  const report = reportFile.value;
  const model = baseName.value;
  const share = holdout.value;
  const order = seed.value;
  clearTimeout(timer);
  if (!data || !model) return;
  timer = setTimeout(() => plan(model, data, report, share, order), 150);
});

async function plan(model, data, report, share, order) {
  const ask = ++asked;
  const body = { model, data, seed: Number(order) || 0 };
  if (report) body.report = report;
  else body.holdout = share / 100;
  try {
    const result = await api("/fit/plan", { method: "POST", body });
    if (ask !== asked) return; // a newer question is out
    planned.value = result;
    planError.value = null;
    if (!result.methods.some((method) => key(method) === choice.value)) choice.value = key(result.methods[0]);
    if (!named.value) name.value = result.name;
  } catch (error) {
    if (ask !== asked) return;
    planned.value = null;
    planError.value = error.message;
  }
}

// Why Start Fit is off, or null.
export const notReady = computed(() => {
  const job = gpuJob.value;
  if (job) return `${job.title} is running; one job at a time can use the GPU`;
  if (!fitData.value) return "Get or add a dataset first, on the Datasets page";
  if (planError.value) return planError.value;
  const current = planned.value;
  if (!current || !picked.value || current.data !== fitData.value || (current.report ?? "") !== reportFile.value)
    return "Reading the data…";
  if (!picked.value.fits) return "The method needs more memory than this Mac's GPU working set";
  const value = name.value.trim();
  if (!NAME.test(value)) return "Give the fitted model a name: letters, digits, '.', '_' or '-'";
  if ((models.value ?? []).some((model) => model.name === value)) return `A model named ${value} exists already`;
  return null;
});

export async function startFit() {
  if (notReady.value || starting.value) return;
  const method = picked.value;
  const body = {
    kind: "fit",
    model: planned.value.model,
    data: planned.value.data,
    name: name.value.trim(),
    method: method.method,
    seed: Number(seed.value) || 0,
  };
  if (planned.value.report) body.report = planned.value.report;
  else body.holdout = holdout.value / 100;
  if (method.head_only) body.head_only = true;
  if (method.epochs) {
    body.epochs = Number(epochs.value) || 3;
    if (!earlyStop.value) body.early_stop = false;
  }
  starting.value = true;
  try {
    const { job } = await api("/jobs", { method: "POST", body });
    named.value = false;
    navigate("training", job.id); // its progress, then its report
  } catch (error) {
    toast(`Could not start the fit: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

// ---------- the form ----------

const TITLES = { head: "Head", readout: "Readout", finetune: "Fine-tune", calibrate: "Calibrate" };

// The examples to train on and to tune, and those the report is scored on:
// kept aside from the dataset, or the report file's (``report``).
function Split({ splits, report }) {
  const total = Math.max(1, splits.fit + splits.dev + splits.holdout);
  const share = (count) => `${((count / total) * 100).toFixed(2)}%`;
  const held = report ? `${splits.holdout} in ${report} for the report` : `${splits.holdout} kept aside for the report`;
  return html`
    <div class="split-field">
      <div class=${`split${report ? " other-file" : ""}`} role="img" aria-label=${`${splits.fit} to train on, ${splits.dev} to tune, ${held}`}>
        <i class="fit" style=${`width:${share(splits.fit)}`}></i><i class="dev" style=${`width:${share(splits.dev)}`}></i><i class="held" style=${`width:${share(splits.holdout)}`}></i>
      </div>
      <div class=${`split-legend num${report ? " other-file" : ""}`}>
        <span><s class="fit"></s>${splits.fit} to train on</span>
        <span><s class="dev"></s>${splits.dev} to tune</span>
        <span><s class="held"></s>${held}</span>
      </div>
    </div>
  `;
}

function ReportOn({ data }) {
  const files = reportFiles.value;
  const current = reportFile.value;
  const roleOf = (name) => recordOf(name)?.role;
  return html`
    <div class="row"><label for="fit-report">Report on</label><div class="line">
      <select id="fit-report" value=${current} onChange=${(event) => (reportChoice.value = { data, file: event.target.value })}>
        <option value="">A share of this dataset</option>
        ${files.length > 0 &&
        html`<optgroup label="Another dataset">
          ${files.map((file) => html`<option value=${file}>${file}${roleOf(file) ? ` (${roleOf(file)})` : ""}</option>`)}
        </optgroup>`}
      </select>
      ${current === ""
        ? html`<input id="fit-holdout" type="number" min="5" max="90" aria-label="Share kept aside" value=${holdout.value}
                 onChange=${(event) => (holdout.value = Math.min(90, Math.max(5, Number(event.target.value) || 20)))} /> %
            <span class="hint">kept aside, by label</span>`
        : html`<span class="hint">${defaultReport(data) === current ? "got with it as its report split" : "it must ask the same questions"}</span>`}
    </div></div>
  `;
}

// Per question, its labels in the question's own option order (from the
// dataset's summary), options without an example to train on included.
function Labels({ labels, summary }) {
  const optionsOf = new Map((summary?.questions ?? []).map((question) => [question.id, question.options]));
  return html`
    <div class="fit-labels">
      ${Object.entries(labels).map(([question, counts]) => {
        const options = optionsOf.get(question) ?? [];
        const order = [...options, ...Object.keys(counts).filter((label) => !options.includes(label))];
        return html`<div class="line num" key=${question}>
          <code>${question}</code>
          ${order.map((label) => html`<span class=${counts[label] ? "" : "none"}><b>${label}</b> ${counts[label] ?? 0}</span>`)}
        </div>`;
      })}
    </div>
  `;
}

function MethodCard({ method }) {
  const checked = key(method) === choice.value;
  const title = `${TITLES[method.method] ?? method.method}${method.head_only ? ", head only" : ""}`;
  return html`
    <button class="method" role="radio" aria-checked=${checked ? "true" : "false"} data-method=${key(method)}
            onClick=${() => (choice.value = key(method))}>
      <span class="mt"><b>${title}</b>${method.auto && html`<span class="pill-tag">auto</span>`}</span>
      <p>${method.description}</p>
      <span class="est num">
        <span>Memory at the peak <b>about ${size(method.peak_bytes)}</b></span>
        ${method.disk_bytes > 0 && html`<span>On disk while it runs <b>about ${size(method.disk_bytes)}</b></span>`}
        <span class=${`fit ${method.fits ? "ok" : "no"}`}>${method.fits ? "Fits this Mac" : "More than this Mac's GPU working set"}</span>
        <span class="est-note">Estimated from the model and the data</span>
      </span>
    </button>
  `;
}

function gpuNote() {
  const loaded = [...loadedNames.value];
  const unloaded = loaded.length ? `${loaded.join(", ")} ${loaded.length === 1 ? "is" : "are"} unloaded, and ` : "";
  return `While it runs, the fit has the GPU to itself: ${unloaded}the Playground and the API answer busy. Downloads keep going.`;
}

export function FitForm() {
  const data = fitData.value;
  const summary = useSummary(data);
  const files = dataFiles.value ?? [];
  const result = planned.value;
  const list = models.value ?? [];
  const method = picked.value;
  return html`
    <div class="fit-form">
      <section class="gsec"><div class="gsec-head"><h2>Data</h2><span class="tag">Labelled examples, as JSON lines</span></div>
        <div class="group form">
          <div class="row"><label for="fit-data">Dataset</label><div class="line">
            <select id="fit-data" value=${data ?? ""} onChange=${(event) => (chosenData.value = event.target.value)}>
              ${files.map((item) => html`<option value=${item.name}>${item.name}</option>`)}
            </select>
            ${result && result.data === data && html`<span class="hint num">${plural(result.splits.examples, "example")}</span>`}
            ${data && html`<a class="hint-link" href=${href("datasets", data)}>Show in Datasets</a>`}
          </div></div>
          ${data && html`<${ReportOn} data=${data} />`}
          ${result && html`<div class="row"><label>Split</label><${Split} splits=${result.splits} report=${result.report} /></div>`}
          ${result && html`<div class="row"><label>Labels</label><div><${Labels} labels=${result.labels} summary=${summary?.name === data ? summary : null} /><span class="hint">to train on and to tune</span></div></div>`}
          ${result?.warnings.map((warning) => html`<p class="warn-text">${warning}</p>`)}
          ${planError.value && html`<p class="warn-text">${planError.value}</p>`}
        </div></section>

      <section class="gsec"><div class="gsec-head"><h2>Base model</h2></div>
        <div class="group form">
          <div class="row"><label for="fit-base">Model</label><div class="line">
            <select id="fit-base" value=${baseName.value ?? ""} onChange=${(event) => (base.value = event.target.value)}>
              ${list.map((model) => html`<option value=${model.name}>${model.name}${model.fitted ? ` (fitted on ${model.base})` : ""}</option>`)}
            </select>
            ${result && html`<span class="hint">${result.family}${result.base !== baseName.value ? ` · its base ${result.base}; the fit is replaced` : ""}</span>`}
          </div></div>
        </div></section>

      ${result &&
      html`<section class="gsec"><div class="gsec-head"><h2>Method</h2><span class="tag">What ${result.base} can learn</span></div>
        <div class="group">
          <div class="methods" role="radiogroup" aria-label="Method">${result.methods.map((m) => html`<${MethodCard} key=${key(m)} method=${m} />`)}</div>
          <p class="method-foot">Memory is an estimate, checked against this Mac before the fit starts. How long it takes shows once it runs, measured on its first steps.</p>
        </div></section>`}

      <section class="gsec"><div class="gsec-head"><h2>Result</h2></div>
        <div class="group form">
          <div class="row"><label for="fit-name">Save as</label><div class="line">
            <input id="fit-name" class="mono" value=${name.value} spellcheck=${false}
                   onInput=${(event) => {
                     name.value = event.target.value;
                     named.value = true;
                   }} />
            <span class="hint">in the workspace's models/ folder</span>
          </div></div>
          <details class="adv"><summary>Options</summary>
            ${method?.epochs &&
            html`<div class="row"><label for="fit-epochs">Epochs</label><div class="line">
              <input id="fit-epochs" type="number" min="1" value=${epochs.value} onInput=${(event) => (epochs.value = event.target.value)} />
              <label class="check"><input type="checkbox" checked=${earlyStop.value} onChange=${(event) => (earlyStop.value = event.target.checked)} /> stop early when the tuning examples get worse</label>
              <span class="hint">the best epoch is kept</span>
            </div></div>`}
            <div class="row"><label for="fit-seed">Seed</label><div class="line">
              <input id="fit-seed" type="number" min="0" value=${seed.value} onChange=${(event) => (seed.value = Number(event.target.value) || 0)} />
              <span class="hint">same seed, same split</span>
            </div></div>
          </details>
        </div></section>

      <div class="start-bar">
        <button class="btn primary" disabled=${Boolean(notReady.value) || starting.value} title=${notReady.value ?? "Start Fit (⌘↩)"} onClick=${startFit}>Start Fit</button>
        <p class="note">${notReady.value && !gpuJob.value ? notReady.value : gpuNote()}</p>
      </div>
    </div>
  `;
}
