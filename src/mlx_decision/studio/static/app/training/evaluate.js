// The Evaluate form: a model, a model to compare it with (a fitted model's base
// by default) and a dataset (one kept for Evaluate by default, or the one the
// Datasets page's "Evaluate on It…" picked). The plan says what the
// data asks and warns when a fitted model was fitted on this very file. Start
// Evaluation runs `training eval` as a job, which holds the GPU like a fit.
import { computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { plural } from "../format.js";
import { href, navigate } from "../router.js";
import { dataFiles } from "../store/data.js";
import { gpuJob } from "../store/jobs.js";
import { remembered } from "../store/prefs.js";
import { loadedNames, models, status } from "../store/status.js";
import { toast } from "../store/ui.js";

const chosenModel = remembered("training.evalModel", null);
const chosenData = remembered("training.evalData", null);
const compareChoice = signal(null); // {model, compare}: picked for that model
const starting = signal(false);

export const evalPlan = signal(null);
const planError = signal(null);

const listed = () => models.value ?? [];
const has = (name) => listed().some((model) => model.name === name);

// The model: the one chosen while it is listed, else the newest fitted one,
// else the default, else the first.
export const evalModel = computed(() => {
  const list = listed();
  if (chosenModel.value && has(chosenModel.value)) return chosenModel.value;
  const fitted = list.filter((model) => model.fitted);
  if (fitted.length) return fitted[fitted.length - 1].name;
  const fallback = status.value?.default_model;
  return fallback && has(fallback) ? fallback : (list[0]?.name ?? null);
});

// What it is compared with: picked for this model, else a fitted model's base, else none.
export const evalCompare = computed(() => {
  const model = evalModel.value;
  const choice = compareChoice.value;
  if (choice?.model === model && (choice.compare === "" || has(choice.compare))) return choice.compare;
  const entry = listed().find((item) => item.name === model);
  return entry?.fitted && entry.base && has(entry.base) ? entry.base : "";
});

// The dataset: the one chosen while it is listed, else one kept for Evaluate, else the first.
export const evalData = computed(() => {
  const files = dataFiles.value ?? [];
  if (chosenData.value && files.some((item) => item.name === chosenData.value)) return chosenData.value;
  return (files.find((item) => item.record?.role === "evaluate") ?? files[0])?.name ?? null;
});

// "Evaluate on It…" (the Datasets page): the form, with the dataset picked.
export function evalOn(name) {
  chosenData.value = name;
  navigate("training", "evaluate");
}

let timer = null;
let asked = 0;
effect(() => {
  const model = evalModel.value;
  const compare = evalCompare.value;
  const data = evalData.value;
  clearTimeout(timer);
  if (!model || !data) return;
  timer = setTimeout(() => plan(model, compare, data), 150);
});

async function plan(model, compare, data) {
  const ask = ++asked;
  const body = { model, data };
  if (compare) body.compare = compare;
  try {
    const result = await api("/eval/plan", { method: "POST", body });
    if (ask !== asked) return;
    evalPlan.value = result;
    planError.value = null;
  } catch (error) {
    if (ask !== asked) return;
    evalPlan.value = null;
    planError.value = error.message;
  }
}

export const evalNotReady = computed(() => {
  const job = gpuJob.value;
  if (job) return `${job.title} is running; one job at a time can use the GPU`;
  if (!evalModel.value) return "No model to evaluate";
  if (!evalData.value) return "Get or add a dataset first, on the Datasets page";
  if (planError.value) return planError.value;
  const current = evalPlan.value;
  if (!current || current.model !== evalModel.value || (current.compare ?? "") !== evalCompare.value || current.data !== evalData.value)
    return "Reading the data…";
  return null;
});

export async function startEval() {
  if (evalNotReady.value || starting.value) return;
  const body = { kind: "eval", model: evalModel.value, data: evalData.value };
  if (evalCompare.value) body.compare = evalCompare.value;
  starting.value = true;
  try {
    const { job } = await api("/jobs", { method: "POST", body });
    navigate("training", job.id); // its progress, then its report
  } catch (error) {
    toast(`Could not start the evaluation: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

const label = (model) => `${model.name}${model.fitted ? ` (fitted on ${model.base})` : ""}`;

function gpuNote() {
  const loaded = [...loadedNames.value];
  const unloaded = loaded.length ? `${loaded.join(", ")} ${loaded.length === 1 ? "is" : "are"} unloaded, and ` : "";
  return `While it runs, the evaluation has the GPU to itself: ${unloaded}the Playground and the API answer busy. The models load one at a time.`;
}

export function EvalForm() {
  const list = listed();
  const files = dataFiles.value ?? [];
  const result = evalPlan.value;
  const model = evalModel.value;
  const data = evalData.value;
  const roleOf = (name) => files.find((item) => item.name === name)?.record?.role;
  const questions = result ? Object.entries(result.questions) : [];
  return html`
    <div class="fit-form eval-form">
      <section class="gsec"><div class="gsec-head"><h2>Models</h2><span class="tag">Scored one after the other on the same examples</span></div>
        <div class="group form">
          <div class="row"><label for="eval-model">Model</label><div class="line">
            <select id="eval-model" value=${model ?? ""} onChange=${(event) => (chosenModel.value = event.target.value)}>
              ${list.map((item) => html`<option value=${item.name}>${label(item)}</option>`)}
            </select>
          </div></div>
          <div class="row"><label for="eval-compare">Compare with</label><div class="line">
            <select id="eval-compare" value=${evalCompare.value}
                    onChange=${(event) => (compareChoice.value = { model, compare: event.target.value })}>
              <option value="">No comparison</option>
              ${list.filter((item) => item.name !== model).map((item) => html`<option value=${item.name}>${label(item)}</option>`)}
            </select>
            <span class="hint">${list.find((item) => item.name === model)?.fitted && evalCompare.value ? "its base, to see what the fit changed" : "a second model, scored after the first"}</span>
          </div></div>
        </div></section>

      <section class="gsec"><div class="gsec-head"><h2>Data</h2><span class="tag">Labelled examples to score them on</span></div>
        <div class="group form">
          <div class="row"><label for="eval-data">Dataset</label><div class="line">
            <select id="eval-data" value=${data ?? ""} onChange=${(event) => (chosenData.value = event.target.value)}>
              ${files.map((item) => html`<option value=${item.name}>${item.name}${roleOf(item.name) ? ` (${roleOf(item.name)})` : ""}</option>`)}
            </select>
            ${result && result.data === data && html`<span class="hint num">${plural(result.examples, "example")}</span>`}
            ${data && html`<a class="hint-link" href=${href("datasets", data)}>Show in Datasets</a>`}
          </div></div>
          ${questions.length > 0 &&
          html`<div class="row"><label>Scores</label><div class="line">${questions.map(
            ([id, kind]) => html`<span class="q-chip"><code>${id}</code> ${kind}</span>`,
          )}</div></div>`}
          ${result?.warnings.map((warning) => html`<p class="warn-text">${warning}</p>`)}
          ${planError.value && html`<p class="warn-text">${planError.value}</p>`}
        </div></section>

      <div class="start-bar">
        <button class="btn primary" disabled=${Boolean(evalNotReady.value) || starting.value} title=${evalNotReady.value ?? "Start Evaluation (⌘↩)"}
                onClick=${startEval}>Start Evaluation</button>
        <p class="note">${evalNotReady.value && !gpuJob.value ? evalNotReady.value : gpuNote()}</p>
      </div>
    </div>
  `;
}
