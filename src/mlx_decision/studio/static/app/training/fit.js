// The fit form: the dataset picked in its own field (or by the Datasets page's
// Fit on It…), its split and labels, the base model, the method cards the
// base's family offers (each with its memory estimate and the fit check, as
// `training fit`'s menu has them), the result's name and the options. Start Fit (⌘↩) starts `training fit` as a job;
// the Studio builds its command line.
import { computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
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

let timer = null;
let asked = 0;

// The plan again whenever the data, the base, the share kept aside or the seed change.
effect(() => {
  const data = fitData.value;
  const model = baseName.value;
  const share = holdout.value;
  const order = seed.value;
  clearTimeout(timer);
  if (!data || !model) return;
  timer = setTimeout(() => plan(model, data, share, order), 150);
});

async function plan(model, data, share, order) {
  const ask = ++asked;
  try {
    const result = await api("/fit/plan", {
      method: "POST",
      body: { model, data, holdout: share / 100, seed: Number(order) || 0 },
    });
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
  if (!planned.value || !picked.value || planned.value.data !== fitData.value) return "Reading the data…";
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
    holdout: holdout.value / 100,
    seed: Number(seed.value) || 0,
  };
  if (method.head_only) body.head_only = true;
  if (method.epochs) {
    body.epochs = Number(epochs.value) || 3;
    if (!earlyStop.value) body.early_stop = false;
  }
  starting.value = true;
  try {
    const { job } = await api("/jobs", { method: "POST", body });
    toast(`${job.title} started: the fitted model shows up in Models when it is done`);
    named.value = false;
  } catch (error) {
    toast(`Could not start the fit: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

// ---------- the form ----------

const TITLES = { head: "Head", readout: "Readout", finetune: "Fine-tune", calibrate: "Calibrate" };

function Split({ splits }) {
  const share = (count) => `${((count / Math.max(1, splits.examples)) * 100).toFixed(2)}%`;
  return html`
    <div class="split-field">
      <div class="split" role="img" aria-label=${`${splits.fit} to train on, ${splits.dev} to tune, ${splits.holdout} kept aside`}>
        <i class="fit" style=${`width:${share(splits.fit)}`}></i><i class="dev" style=${`width:${share(splits.dev)}`}></i><i class="held" style=${`width:${share(splits.holdout)}`}></i>
      </div>
      <div class="split-legend num">
        <span><s class="fit"></s>${splits.fit} to train on</span>
        <span><s class="dev"></s>${splits.dev} to tune</span>
        <span><s class="held"></s>${splits.holdout} kept aside for the report</span>
      </div>
    </div>
  `;
}

function Labels({ labels }) {
  return html`
    <div class="fit-labels">
      ${Object.entries(labels).map(
        ([question, counts]) => html`<div class="line num" key=${question}>
          <code>${question}</code>
          ${Object.entries(counts).map(([label, count]) => html`<span><b>${label}</b> ${count}</span>`)}
        </div>`,
      )}
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
          ${result && html`<div class="row"><label>Split</label><${Split} splits=${result.splits} /></div>`}
          ${result && html`<div class="row"><label>Labels</label><div><${Labels} labels=${result.labels} /><span class="hint">to train on and to tune</span></div></div>`}
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
            <div class="row"><label for="fit-holdout">Kept aside</label><div class="line">
              <input id="fit-holdout" type="number" min="5" max="90" value=${holdout.value}
                     onChange=${(event) => (holdout.value = Math.min(90, Math.max(5, Number(event.target.value) || 20)))} /> %
              <span class="hint">for the report, by label</span>
            </div></div>
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
