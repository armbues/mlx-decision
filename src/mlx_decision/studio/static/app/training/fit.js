// The fit form: the dataset picked in its own field (or by the Datasets page's
// Fit on It…; it trains on its train split), what the report is scored on (its
// report split by default, a share kept aside, or another dataset), the split
// and labels, the base model, the method cards the
// base's family offers (each with its memory estimate and the fit check, as
// `training fit`'s menu has them), LoRA beside them where the family takes it
// (its own panel, options and Start LoRA Fit), the result's name and the
// options. Start Fit (⌘↩) starts `training fit` as a job;
// the Studio builds its command line.
import { computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { splitOption } from "../datasets/splits.js";
import { api } from "../api.js";
import { useSummary } from "../datasets/summary.js";
import { plural, size } from "../format.js";
import { gpuJob } from "../store/jobs.js";
import { remembered } from "../store/prefs.js";
import { href, navigate } from "../router.js";
import { dataFiles, datasetOf, splitFor } from "../store/data.js";
import { NO_MODELS, NoModels } from "../components/no-models.js";
import { batches } from "../store/batches.js";
import { loadedNames, models, status } from "../store/status.js";
import { toast } from "../store/ui.js";
import { openMix } from "./mix.js";

const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;

export const base = remembered("training.base", null);
const chosenData = remembered("training.data", null);
const holdout = signal(20); // percent kept aside for the report
const seed = signal(0);
const choice = signal(null); // "method" or "method+head" (the card picked)
const epochs = signal(3);
const earlyStop = signal(true);
const calibration = signal("auto"); // the closing calibration
const CALIBRATIONS = {
  auto: "temperature after training; a calibration alone adds a bias per option from 10 training rows of each",
  full: "a temperature and a bias per option",
  temperature: "a temperature only",
  none: "the answers as the method gives them, e.g. for thresholds tuned on the base",
};
// LoRA's own options (its panel beside the methods), as typed.
const loraEpochs = signal(2);
const loraRate = signal("1e-4");
const loraRank = signal(8);
const RANKS = [2, 4, 8, 16, 32];
const name = signal("");
const named = signal(false); // typed: a new plan keeps it
const starting = signal(false);

export const planned = signal(null); // POST /fit/plan
const planError = signal(null);

const key = (method) => (method.head_only ? `${method.method}+head` : method.method);
const picked = computed(() => planned.value?.methods.find((method) => key(method) === choice.value) ?? null);

// The base: the remembered one while it is listed, else the model of the
// newest batch on the dataset, else a loaded one, else the default, else the first.
const baseName = computed(() => {
  const list = (models.value ?? []).map((model) => model.name);
  if (base.value && list.includes(base.value)) return base.value;
  const batched = (batches.value ?? []).find((batch) => batch.dataset === fitData.value)?.model;
  if (batched && list.includes(batched)) return batched;
  const loaded = list.find((name) => loadedNames.value.has(name));
  if (loaded) return loaded;
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

// The dataset's split to train on: its train split, else its first.
const dataItem = computed(() => datasetOf(fitData.value));
// Picked for a dataset ("Train on"), else its train split, else its first.
const trainChoice = signal(null); // {data, split}
export const trainSplit = computed(() => {
  const names = (dataItem.value?.splits ?? []).map((split) => split.name);
  const chosen = trainChoice.value;
  if (chosen?.data === fitData.value && names.includes(chosen.split)) return chosen.split;
  return splitFor(dataItem.value, "train") ?? names[0] ?? null;
});
// What keeps the dataset from a fit, or null.
const notReadyData = computed(() => {
  const item = dataItem.value;
  if (!item) return null;
  if (item.error) return `${item.name} cannot be read: ${item.error}`;
  return item.missing?.length ? `${item.name} is not ready for a fit: ${item.missing.join("; ")}` : null;
});

// What the report is scored on: "" (a share of the train split kept aside),
// "split:NAME" (another split of the dataset) or "data:NAME" (another
// dataset). Picked for one dataset; another dataset starts from its report
// split when it has one.
const reportChoice = signal(null); // {data, value}

// This dataset's other splits a report can be scored on (not its evaluate split).
export const reportSplits = computed(() =>
  (dataItem.value?.splits ?? []).filter((split) => split.name !== trainSplit.value && split.role !== "evaluate").map((split) => split.name),
);
// The other datasets ready for a run, each by a split: its report split (or
// its only one), else every split but an evaluate one. Each [value, label].
export const reportData = computed(() =>
  (dataFiles.value ?? [])
    .filter((item) => item.name !== fitData.value && !item.error && !item.missing?.length)
    .flatMap((item) => {
      const own = splitFor(item, "report");
      const splits = own ? [own] : item.splits.filter((split) => split.role !== "evaluate").map((split) => split.name);
      const many = item.kind === "folder";
      return splits.map((split) => [`data:${item.name}/${split}`, many && !own ? `${item.name} (${split})` : item.name]);
    }),
);

function defaultReport() {
  const report = (dataItem.value?.splits ?? []).find((split) => split.role === "report" && split.name !== trainSplit.value);
  return report ? `split:${report.name}` : "";
}

export const reportValue = computed(() => {
  const data = fitData.value;
  const chosen = reportChoice.value;
  if (chosen?.data === data) {
    const [kind, which] = chosen.value.split(/:(.*)/s);
    const others = reportData.value.map(([value]) => value);
    if (chosen.value === "" || (kind === "split" && reportSplits.value.includes(which)) || (kind === "data" && others.includes(chosen.value)))
      return chosen.value;
  }
  return data ? defaultReport() : "";
});

// A plan's report as reportValue says it.
const reportOf = (result) =>
  result.report == null
    ? ""
    : result.report === result.data && result.report_split
      ? `split:${result.report_split}`
      : `data:${result.report}/${result.report_split}`;

let timer = null;
let asked = 0;

// The plan again whenever the data, the report, the base, the share kept aside or the seed change.
effect(() => {
  const data = notReadyData.value ? null : fitData.value;
  const split = trainSplit.value;
  const report = reportValue.value;
  const model = baseName.value;
  const share = holdout.value;
  const order = seed.value;
  clearTimeout(timer);
  if (!data || !model) return;
  timer = setTimeout(() => plan(model, data, split, report, share, order), 150);
});

async function plan(model, data, split, report, share, order) {
  const ask = ++asked;
  const body = { model, data, seed: Number(order) || 0 };
  if (split) body.split = split;
  const [kind, which] = report.split(/:(.*)/s);
  if (kind === "split") Object.assign(body, { report: data, report_split: which });
  else if (kind === "data") {
    const [other, otherSplit] = which.split("/");
    Object.assign(body, { report: other, report_split: otherSplit });
  } else body.holdout = share / 100;
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

// Why a fit with ``entry`` (a method card, or the plan's LoRA) cannot start
// (``button`` names its button), or null.
function blocked(entry, button) {
  const job = gpuJob.value;
  if (job) return `${job.title} has the GPU: ${button} is off until it is done`;
  if (models.value && models.value.length === 0) return NO_MODELS;
  if (!fitData.value) return "Download or add a dataset first, on the Datasets page";
  if (notReadyData.value) return notReadyData.value;
  if (planError.value) return planError.value;
  const current = planned.value;
  if (!current || !entry || current.data !== fitData.value || reportOf(current) !== reportValue.value || (trainSplit.value && current.split !== trainSplit.value))
    return "Reading the data…";
  if (!entry.fits) return "The method needs more memory than this Mac's GPU working set";
  const value = name.value.trim();
  if (!NAME.test(value)) return "Give the fitted model a name: letters, digits, '.', '_' or '-'";
  if ((models.value ?? []).some((model) => model.name === value)) return `A model named ${value} exists already`;
  return null;
}

// What LoRA's own options say is wrong, or null.
const loraOptions = computed(() => {
  const rate = Number(loraRate.value);
  if (!(rate > 0 && rate < 1)) return "Give LoRA a learning rate between 0 and 1, such as 1e-4";
  if (!(Number(loraEpochs.value) >= 1)) return "Give LoRA 1 or more epochs";
  return null;
});

// Why Start Fit is off, or null.
export const notReady = computed(() => blocked(picked.value, "Start Fit"));
// Why Start LoRA Fit is off, or null.
export const notReadyLora = computed(() => blocked(planned.value?.lora, "Start LoRA Fit") ?? loraOptions.value);

export async function startFit() {
  if (notReady.value || starting.value) return;
  const method = picked.value;
  const extra = {};
  if (method.head_only) extra.head_only = true;
  if (method.epochs) extra.epochs = Number(epochs.value) || 3;
  await start(method.method, extra, method.epochs);
}

export async function startLoraFit() {
  if (notReadyLora.value || starting.value) return;
  const extra = { epochs: Number(loraEpochs.value) || 2 };
  const lora = planned.value.lora;
  if (Number(loraRate.value) !== lora.learning_rate) extra.learning_rate = Number(loraRate.value);
  if (Number(loraRank.value) !== lora.rank) extra.rank = Number(loraRank.value);
  await start("lora", extra, true);
}

// Start ``training fit`` as a job with ``method``; ``epochs``: it trains in epochs.
async function start(method, extra, inEpochs) {
  const body = {
    kind: "fit",
    model: planned.value.model,
    data: planned.value.data,
    split: planned.value.split,
    name: name.value.trim(),
    method,
    seed: Number(seed.value) || 0,
    ...extra,
  };
  if (planned.value.report) {
    body.report = planned.value.report;
    if (planned.value.report_split) body.report_split = planned.value.report_split;
  } else body.holdout = holdout.value / 100;
  if (calibration.value !== "auto" && !(calibration.value === "none" && method === "calibrate")) body.calibration = calibration.value;
  if (inEpochs && !earlyStop.value) body.early_stop = false;
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
// kept aside from the train split, or another split's or dataset's (``report``).
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
  const splits = reportSplits.value;
  const others = reportData.value;
  const current = reportValue.value;
  const splitOf = (name) => dataItem.value?.splits?.find((split) => split.name === name) ?? { name };
  return html`
    <div class="row"><label for="fit-report">Report on</label><div class="line">
      <select id="fit-report" value=${current} onChange=${(event) => (reportChoice.value = { data, value: event.target.value })}>
        ${splits.length > 0 &&
        html`<optgroup label=${`A split of ${data}`}>
          ${splits.map((name) => html`<option value=${`split:${name}`}>${splitOption(splitOf(name))}</option>`)}
        </optgroup>`}
        <option value="">A share of ${trainSplit.value && dataItem.value?.kind === "folder" ? `its ${trainSplit.value} split` : "this dataset"}</option>
        ${others.length > 0 &&
        html`<optgroup label="Another dataset">${others.map(([value, label]) => html`<option value=${value}>${label}</option>`)}</optgroup>`}
      </select>
      ${current === ""
        ? html`<input id="fit-holdout" type="number" min="5" max="90" aria-label="Share kept aside" value=${holdout.value}
                 onChange=${(event) => (holdout.value = Math.min(90, Math.max(5, Number(event.target.value) || 20)))} /> %
            <span class="hint">kept aside, by label</span>`
        : html`<span class="hint">${current === defaultReport() ? "its report split" : current.startsWith("data:") ? "it must ask the same questions" : "another split of it"}</span>`}
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
      <span class="mt"><b>${title}</b>${method.auto && html`<span class="pill-tag">Recommended</span>`}</span>
      <p>${method.description}</p>
      <span class="est num">
        <span>Memory at the peak <b>about ${size(method.peak_bytes)}</b></span>
        ${method.disk_bytes > 0 && html`<span>On disk while it runs <b>about ${size(method.disk_bytes)}</b></span>`}
        <span class=${`fit ${method.fits ? "ok" : "no"}`}>${method.fits ? "Fits this Mac" : "More than this Mac's GPU working set"}</span>
      </span>
    </button>
  `;
}

// LoRA for the families that take it: an action of its own beside the
// methods, with what it does, its estimate, its options and its own start.
function LoraPanel({ lora }) {
  const reason = notReadyLora.value;
  return html`
    <div class="lora-panel" aria-label="LoRA">
      <span class="mt"><b>Adapt the network itself (LoRA)</b></span>
      <p>${lora.description}</p>
      <span class="est num">
        <span>Memory at the peak <b>about ${size(lora.peak_bytes)}</b></span>
        <span class=${`fit ${lora.fits ? "ok" : "no"}`}>${lora.fits ? "Fits this Mac" : "More than this Mac's GPU working set"}</span>
      </span>
      <details class="adv lora-options"><summary>Options <span class="hint num">epochs ${loraEpochs.value} · learning rate ${loraRate.value} · rank ${loraRank.value}</span></summary>
        <div class="form">
          <div class="row"><label for="lora-epochs">Epochs</label><div class="line">
            <input id="lora-epochs" type="number" min="1" max="20" value=${loraEpochs.value} onInput=${(event) => (loraEpochs.value = event.target.value)} />
            <span class="hint">the best is kept; one or two are usually enough</span>
          </div></div>
          <div class="row"><label for="lora-rate">Learning rate</label><div class="line">
            <input id="lora-rate" class="mono" value=${loraRate.value} spellcheck=${false} onInput=${(event) => (loraRate.value = event.target.value.trim())} />
            <span class="hint">3e-5 to 1e-4; above 3e-4 the fit breaks down</span>
          </div></div>
          <div class="row"><label for="lora-rank">Rank</label><div class="line">
            <select id="lora-rank" value=${String(loraRank.value)} onChange=${(event) => (loraRank.value = Number(event.target.value))}>
              ${RANKS.map((rank) => html`<option value=${String(rank)}>${rank}</option>`)}
            </select>
            <span class="hint">the adapters' size; the learning rate stays right for each</span>
          </div></div>
        </div>
      </details>
      <div class="lora-start">
        <span class="note">${reason && reason !== notReady.value ? reason : ""}</span>
        <button class="btn" disabled=${Boolean(reason) || starting.value} title=${reason ?? "Start LoRA Fit"} onClick=${startLoraFit}>Start LoRA Fit</button>
      </div>
    </div>
  `;
}

function gpuNote() {
  const loaded = [...loadedNames.value];
  const unloaded = loaded.length ? `${loaded.join(", ")} ${loaded.length === 1 ? "is" : "are"} unloaded, and ` : "";
  return `While it runs, the fit has the GPU to itself: ${unloaded}the Playground and the API answer busy. `;
}

export function FitForm() {
  const data = fitData.value;
  const summary = useSummary(data, trainSplit.value);
  const files = dataFiles.value ?? [];
  const result = notReadyData.value ? null : planned.value;
  const list = models.value ?? [];
  const method = picked.value;
  return html`
    <div class="fit-form">
      <section class="gsec"><div class="gsec-head"><h2>Data</h2></div>
        <div class="group form">
          <div class="row"><label for="fit-data">Dataset</label><div class="line">
            <select id="fit-data" value=${data ?? ""} onChange=${(event) => (chosenData.value = event.target.value)}>
              ${files.map((item) => html`<option value=${item.name}>${item.name}${item.missing?.length || item.error ? " (not ready)" : ""}</option>`)}
            </select>
            ${result && result.data === data && html`<span class="hint num">${dataItem.value?.kind === "folder" ? `its ${result.split} split: ` : ""}${plural(result.splits.examples, "row")}</span>`}
            ${data && html`<a class="hint-link" href=${href("datasets", data)}>Show in Datasets</a>`}
            <button class="btn sm" title="Pick several datasets into a mix, for one fit on all of them"
                    onClick=${() => openMix({ first: data, onSaved: (made) => (chosenData.value = made) })}>Mix Datasets…</button>
          </div></div>
          ${dataItem.value?.mix &&
          html`<div class="row"><label></label><span class="hint">A mix of ${dataItem.value.mix.sources.join(", ")}${
            dataItem.value.mix.held_out.length ? html`; held out: ${dataItem.value.mix.held_out.join(", ")}` : ""
          }. The report shows each dataset.</span></div>`}
          ${notReadyData.value &&
          html`<p class="warn-text">${notReadyData.value}${
            dataItem.value?.kind === "folder" && !dataItem.value?.error
              ? html` <button class="btn sm" onClick=${() => navigate("playground", `data:${data}`)}>${dataItem.value.questions?.length ? "Edit Questions" : "Write Questions"}</button>`
              : ""
          }</p>`}
          ${data && !notReadyData.value && dataItem.value?.kind === "folder" && dataItem.value.splits.length > 1 &&
          html`<div class="row"><label for="fit-split">Train on</label><div class="line">
            <select id="fit-split" value=${trainSplit.value ?? ""} onChange=${(event) => (trainChoice.value = { data, split: event.target.value })}>
              ${dataItem.value.splits.map((split) => html`<option value=${split.name}>${splitOption(split)}</option>`)}
            </select>
          </div></div>`}
          ${data && !notReadyData.value && html`<${ReportOn} data=${data} />`}
          ${result && html`<div class="row"><label>Split</label><${Split} splits=${result.splits}
            report=${result.report == null ? null : result.report === result.data ? `${result.report_split}` : result.report} /></div>`}
          ${result && html`<div class="row"><label>Labels</label><div><${Labels} labels=${result.labels} summary=${summary?.name === data ? summary : null} /></div></div>`}
          ${result?.warnings.map((warning) => html`<p class="warn-text">${warning}</p>`)}
          ${planError.value && html`<p class="warn-text">${planError.value}</p>`}
        </div></section>

      <section class="gsec"><div class="gsec-head"><h2>Base model</h2></div>
        <div class="group form">
          <div class="row"><label for="fit-base">Model</label><div class="line">
            ${list.length === 0
              ? html`<${NoModels} />`
              : html`<select id="fit-base" value=${baseName.value ?? ""} onChange=${(event) => (base.value = event.target.value)}>
                  ${list.map((model) => html`<option value=${model.name}>${model.name}${model.fitted ? ` (fitted on ${model.base})` : ""}</option>`)}
                </select>`}
            ${result && html`<span class="hint">${result.family}${result.base !== baseName.value ? ` · its base ${result.base}; the fit is replaced` : ""}</span>`}
          </div></div>
        </div></section>

      ${result &&
      html`<section class="gsec"><div class="gsec-head"><h2>Method</h2><span class="tag">What ${result.base} can learn</span></div>
        <div class="group">
          <div class="methods" role="radiogroup" aria-label="Method">${result.methods.map((m) => html`<${MethodCard} key=${key(m)} method=${m} />`)}</div>
          ${result.lora && html`<${LoraPanel} lora=${result.lora} />`}
          <p class="method-foot">Memory is an estimate, checked before the fit starts; the time shows once it runs.</p>
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
              <label class="check"><input type="checkbox" checked=${earlyStop.value} onChange=${(event) => (earlyStop.value = event.target.checked)} /> stop early when the tuning rows get worse</label>
              <span class="hint">the best epoch is kept</span>
            </div></div>`}
            <div class="row"><label for="fit-calibration">Calibration</label><div class="line">
              <select id="fit-calibration" value=${method?.method === "calibrate" && calibration.value === "none" ? "auto" : calibration.value}
                      onChange=${(event) => (calibration.value = event.target.value)}>
                ${Object.keys(CALIBRATIONS)
                  .filter((choice) => choice !== "none" || method?.method !== "calibrate")
                  .map((choice) => html`<option value=${choice}>${choice}</option>`)}
              </select>
              <span class="hint">${CALIBRATIONS[method?.method === "calibrate" && calibration.value === "none" ? "auto" : calibration.value]}</span>
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
