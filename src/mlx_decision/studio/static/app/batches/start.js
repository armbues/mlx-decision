// Starting a batch (New Batch on the Batches page, Run as Batch… on the
// Datasets page) and saving a batch's answers as a dataset: two sheets.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { splitOption } from "../datasets/splits.js";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { navigate } from "../router.js";
import { batches, loadBatches, runningBatch } from "../store/batches.js";
import { dataChanged, dataFiles, datasetOf, describe, described, freeName, nameTaken, summaryKey } from "../store/data.js";
import { uploadData } from "../datasets/files.js";
import { openGetDataset } from "../datasets/get.js";
import { gpuJob } from "../store/jobs.js";
import { reloadSets, sets } from "../store/sets.js";
import { NO_MODELS, NoModels } from "../components/no-models.js";
import { models, status } from "../store/status.js";
import { toast } from "../store/ui.js";

const OWN = "";
const NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$/;
const form = signal(null);
const error = signal(null);
const busy = signal(false);

// The split a batch reads by default: the one asked for, else the evaluate
// split, the report split, the first.
function defaultSplit(item, wanted = null) {
  const splits = item?.splits ?? [];
  if (wanted && splits.some((split) => split.name === wanted)) return wanted;
  return (splits.find((split) => split.role === "evaluate") ?? splits.find((split) => split.role === "report") ?? splits[0])?.name ?? null;
}

function defaultModel() {
  const loaded = status.value?.loaded ?? [];
  const names = (models.value ?? []).map((item) => item.name);
  return loaded.find((name) => names.includes(name)) ?? names[0] ?? null;
}

const usable = () => (dataFiles.value ?? []).filter((item) => !item.error);

// Whether the dataset asks questions of its own: its dataset.json's, or those
// on its lines (the list reads them from the first line).
const asks = (item) => (item?.questions ?? []).length > 0;

// A batch's default name, as the Studio makes it: dataset-model, then -2, -3…
// when taken.
function defaultName(dataset, model) {
  const base = `${dataset}-${model}`.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+|[-.]+$/g, "").slice(0, 70) || "batch";
  const taken = new Set((batches.value ?? []).map((item) => item.name));
  let name = base;
  for (let n = 2; taken.has(name); n += 1) name = `${base}-${n}`;
  return name;
}

// The set's questions (by name), read when the set is picked: {name, questions} or null.
const setQuestions = signal(null);

let reading = null; // the set being read

async function readSet(name) {
  if (!name || setQuestions.value?.name === name || reading === name) return;
  reading = name;
  try {
    const { body } = await api(`/sets/${encodeURIComponent(name)}`);
    setQuestions.value = { name, questions: body?.questions ?? {} };
  } catch {
    setQuestions.value = { name, questions: {} };
  } finally {
    reading = null;
  }
}

// Whether a set's question is asked as the dataset's labels answer it: the same
// id, type and options (as the Studio decides it).
function sameAs(asked, labelled) {
  if (!asked || !labelled || (asked.type ?? "noul") !== (labelled.type ?? "noul")) return false;
  const options = (question) =>
    question.type === "choice" ? Object.keys(question.criteria ?? {}).sort().join("\u0000") : question.type === "score" ? String((question.criteria ?? []).length) : "noul";
  const theirs = labelled.type === "choice" ? [...(labelled.options ?? [])].sort().join("\u0000") : labelled.type === "score" ? String((labelled.options ?? []).length) : "noul";
  return options(asked) === theirs;
}

// What a labelled question gets in the batch's metrics.
const SCORED = "accuracy, F1, calibration and a confusion matrix";
const ids = (list) => list.map((id, index) => html`${index ? ", " : ""}<code>${id}</code>`);

// What the batch's labels will count for, in words.
function LabelsHint({ item, current }) {
  const summary = described.value[summaryKey(current.dataset, current.split)];
  if (!summary || summary.error) return null;
  const labelled = (summary.questions ?? []).filter((question) => question.labelled > 0);
  if (current.questions === OWN) {
    if (!labelled.length) return html`Its rows have no labels: answers only.`;
    return html`Its labels score ${ids(labelled.map((question) => question.id))}: ${SCORED}.`;
  }
  const set = setQuestions.value?.name === current.questions ? setQuestions.value.questions : null;
  if (!set) return null;
  const byId = Object.fromEntries(labelled.map((question) => [question.id, question]));
  const counted = Object.keys(set).filter((id) => sameAs(set[id], byId[id]));
  const others = Object.keys(set).filter((id) => !counted.includes(id));
  if (!counted.length) return html`Answers only: ${item?.name ?? "the dataset"} has no labels for these questions.`;
  return html`Its labels score ${ids(counted)}: ${SCORED}${
    others.length ? html`; answers only for ${ids(others)} (not asked as the dataset asks them)` : ""
  }.`;
}

function choose(dataset, split = null) {
  const item = datasetOf(dataset);
  const own = asks(item);
  const current = form.value ?? {};
  form.value = {
    ...current,
    dataset,
    split: defaultSplit(item, split),
    questions: own ? OWN : (current.questions || sets.value?.[0]?.name) ?? OWN,
  };
}

export function openNewBatch({ dataset = null, split = null } = {}) {
  reloadSets();
  const first = dataset ?? usable()[0]?.name ?? null;
  form.value = { dataset: null, split: null, questions: OWN, model: defaultModel(), sample: false, size: 100, seed: 0, name: "" };
  error.value = null;
  busy.value = false;
  if (first) choose(first, split);
  openSheet({
    title: dataset ? "Run as Batch" : "New Batch",
    purpose: "Answer a dataset's rows with a model: all of them or a sample, with the dataset's questions or a question set's. With labels you also see how often the model is right.",
    Body: NewBatch,
    wide: true,
    actions: () => [{ label: busy.value ? "Starting…" : "Start", primary: true, disabled: Boolean(whyNot()) || busy.value, onSelect: start }],
  });
}

function rowsOf(item, split) {
  return item?.splits?.find((entry) => entry.name === split)?.examples ?? null;
}

// Why the batch cannot start, or null.
function whyNot() {
  const current = form.value;
  if (!current) return "";
  const item = datasetOf(current.dataset);
  if (!item) return "Pick a dataset";
  if (!current.split) return "Pick a split";
  if (current.questions === OWN && !asks(item)) return `${item.name} has no questions: pick a question set, or write them in the Playground`;
  if (models.value && models.value.length === 0) return NO_MODELS;
  if (!current.model) return "Pick a model";
  if (current.sample && !(current.size >= 1)) return "A sample is at least one row";
  if (current.name && !NAME.test(current.name)) return "A name is letters, digits, dots, dashes and underscores";
  if (current.name && (batches.value ?? []).some((item) => item.name === current.name)) return `A batch named ${current.name} exists already`;
  if (runningBatch.value) return `The batch ${runningBatch.value.name} is running; one at a time`;
  if (gpuJob.value) return `${gpuJob.value.title} is using the GPU`;
  return null;
}

async function start() {
  const current = form.value;
  if (whyNot()) return;
  busy.value = true;
  error.value = null;
  try {
    const body = {
      dataset: current.dataset,
      split: current.split,
      model: current.model,
      questions: current.questions === OWN ? null : current.questions,
      sample: current.sample ? Number(current.size) : null,
      seed: current.sample ? Number(current.seed) || 0 : null,
      name: current.name.trim() || null,
    };
    const { batch } = await api("/batches", { method: "POST", body });
    closeSheet();
    await loadBatches();
    navigate("batches", batch.name);
  } catch (failure) {
    error.value = failure.message;
  } finally {
    busy.value = false;
  }
}

const set = (fields) => (form.value = { ...form.value, ...fields });

function NewBatch() {
  const current = form.value;
  if (!current) return null;
  const list = usable();
  const item = datasetOf(current.dataset);
  const own = asks(item);
  const count = (item?.questions ?? []).length;
  const rows = rowsOf(item, current.split);
  const placeholder = current.dataset && current.model ? defaultName(current.dataset, current.model) : "";
  if (current.dataset && current.split && !described.value[summaryKey(current.dataset, current.split)]) describe(current.dataset, current.split);
  if (current.questions !== OWN) readSet(current.questions);
  if (dataFiles.value && list.length === 0)
    return html`<div class="empty no-datasets">
      <b>No datasets yet</b><span>A batch answers a dataset's rows.</span>
      <span class="empty-actions">
        <button class="btn primary" onClick=${() => { closeSheet(); openGetDataset(); }}>Download a Dataset…</button>
        <button class="btn" onClick=${() => { closeSheet(); uploadData(); }}>Add a Dataset…</button>
      </span>
    </div>`;
  return html`
    <div class="form-grid batch-form">
      <label for="batch-data">Dataset</label>
      <select id="batch-data" value=${current.dataset ?? ""} onChange=${(event) => choose(event.target.value)}>
        ${list.length === 0 && html`<option value="">No datasets yet</option>`}
        ${list.map((entry) => html`<option value=${entry.name}>${entry.name}</option>`)}
      </select>

      <label for="batch-split">Split</label>
      <select id="batch-split" value=${current.split ?? ""} onChange=${(event) => set({ split: event.target.value })}>
        ${(item?.splits ?? []).map(
          (split) => html`<option value=${split.name}>${splitOption(split)}</option>`,
        )}
      </select>
      ${(item?.splits ?? []).find((split) => split.name === current.split)?.role === "train" &&
      html`<span></span><small class="hint split-hint">A fit learns from these rows: score a fitted model on another split.</small>`}

      <span class="label">Rows</span>
      <div class="radio-col" role="radiogroup" aria-label="Rows">
        <label><input type="radio" name="batch-rows" checked=${!current.sample} onChange=${() => set({ sample: false })} />
          ${rows != null ? `All ${rows.toLocaleString("en")} rows` : "All rows"}</label>
        <label class="sample-row"><input type="radio" name="batch-rows" checked=${current.sample} onChange=${() => set({ sample: true })} />
          A sample of <input id="batch-size" type="number" min="1" aria-label="Rows in the sample" value=${current.size}
            onInput=${(event) => set({ sample: true, size: event.target.valueAsNumber })} /> rows, seed
          <input id="batch-seed" type="number" min="0" aria-label="Seed" value=${current.seed} onInput=${(event) => set({ seed: event.target.valueAsNumber })} /></label>
      </div>

      <label for="batch-questions">Questions</label>
      <div>
        <select id="batch-questions" value=${current.questions} onChange=${(event) => set({ questions: event.target.value })}>
          ${own && html`<option value=${OWN}>Its own (${plural(count, "question")})</option>`}
          ${(sets.value ?? []).map((entry) => html`<option value=${entry.name}>The set ${entry.name}</option>`)}
          ${!own && !(sets.value ?? []).length && html`<option value=${OWN}>No questions yet</option>`}
        </select>
        <small class="hint">${
          current.questions === OWN && !own
            ? html`${current.dataset} has no questions yet. <a href=${`#/playground/data:${encodeURIComponent(current.dataset ?? "")}`} onClick=${closeSheet}>Write them in the Playground</a>, or pick a question set.`
            : html`<${LabelsHint} item=${item} current=${current} />`
        }</small>
      </div>

      <label for="batch-model">Model</label>
      ${models.value && models.value.length === 0
        ? html`<${NoModels} />`
        : html`<select id="batch-model" value=${current.model ?? ""} onChange=${(event) => set({ model: event.target.value })}>
            ${(models.value ?? []).map((entry) => html`<option value=${entry.name}>${entry.name}${entry.fitted ? ` (fitted on ${entry.base})` : ""}</option>`)}
          </select>`}

      <label for="batch-name">Name</label>
      <input id="batch-name" type="text" spellcheck=${false} placeholder=${placeholder} value=${current.name}
        onInput=${(event) => set({ name: event.target.value })} />
    </div>
    ${(error.value || whyNot()) && html`<p class="warn-text batch-why">${error.value ?? whyNot()}</p>`}
  `;
}

// ---------- Save Answers as a Dataset ----------

const saving = signal(null); // {batch, name, keep, threshold}

function saveWhyNot() {
  const current = saving.value;
  if (!current) return "";
  const name = current.name.trim();
  if (!name) return "Name the dataset";
  if (!NAME.test(name)) return "A name is letters, digits, dots, dashes and underscores";
  if (nameTaken(name)) return nameTaken(name);
  if (current.keep && !(current.threshold >= 0 && current.threshold <= 1)) return "A confidence is between 0 and 1";
  return null;
}

export function openSaveDataset(batch) {
  const base = `${batch.dataset}-by-${batch.model}`.replace(/[^A-Za-z0-9._-]+/g, "-").slice(0, 70);
  saving.value = { batch, name: freeName(base), keep: false, threshold: 0.8 };
  error.value = null;
  openSheet({
    title: "Save Answers as a Dataset",
    purpose: `${batch.model}'s answers to ${plural(batch.answered, "row")} of ${batch.dataset} become the labels of a new dataset, ready for a fit: another model learns them.`,
    Body: SaveDataset,
    actions: () => [{ label: busy.value ? "Saving…" : "Save", primary: true, disabled: Boolean(saveWhyNot()) || busy.value, onSelect: save }],
  });
}

async function save() {
  const current = saving.value;
  if (saveWhyNot()) return;
  busy.value = true;
  error.value = null;
  try {
    const name = current.name.trim();
    const body = { name, threshold: current.keep ? current.threshold : null };
    const { dataset } = await api(`/batches/${encodeURIComponent(current.batch.name)}/dataset`, { method: "POST", body });
    closeSheet();
    dataChanged();
    toast(`Saved ${plural(dataset.examples, "row")} as ${dataset.name}${dataset.left_out ? `; ${plural(dataset.left_out, "row")} left out` : ""}`);
    navigate("datasets", dataset.name);
  } catch (failure) {
    error.value = failure.message;
  } finally {
    busy.value = false;
  }
}

function SaveDataset() {
  const current = saving.value;
  if (!current) return null;
  const set = (fields) => (saving.value = { ...saving.value, ...fields });
  const why = saveWhyNot();
  return html`
    <div class="form-grid batch-form">
      <label for="save-name">Name</label>
      <input id="save-name" type="text" spellcheck=${false} value=${current.name} onInput=${(event) => set({ name: event.target.value })} />
      <span class="label">Labels</span>
      <div class="radio-col">
        <label><input type="checkbox" checked=${current.keep} onChange=${(event) => set({ keep: event.target.checked })} />
          Only answers with a confidence of at least
          <input id="save-threshold" type="number" min="0" max="1" step="0.05" aria-label="Lowest confidence" value=${current.threshold}
            onInput=${(event) => set({ keep: true, threshold: event.target.valueAsNumber })} /></label>
        <small class="hint">A row with no answer kept is left out. One split, <code>${current.batch.split}</code>, with the role train.</small>
      </div>
    </div>
    ${(error.value || (why && current.name.trim())) && html`<p class="warn-text">${error.value ?? why}</p>`}
  `;
}
