// Starting a batch (New Batch on the Batches page, Run as Batch… on the
// Datasets page) and saving a batch's answers as a dataset: two sheets.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { plural } from "../format.js";
import { navigate } from "../router.js";
import { loadBatches, runningBatch } from "../store/batches.js";
import { dataChanged, dataFiles, datasetOf, freeName, nameTaken } from "../store/data.js";
import { gpuJob } from "../store/jobs.js";
import { reloadSets, sets } from "../store/sets.js";
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

// Whether the dataset asks questions of its own: a folder's dataset.json, or
// a plain file's lines (training data carries them on every line).
const asks = (item) => item?.kind === "file" || (item?.questions ?? []).length > 0;

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
    title: "New Batch",
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
  if (!current.model) return "Pick a model";
  if (current.sample && !(current.size >= 1)) return "A sample is at least one row";
  if (current.name && !NAME.test(current.name)) return "A name is letters, digits, dots, dashes and underscores";
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
  const placeholder = current.dataset && current.model ? `${current.dataset}-${current.model}` : "";
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
          (split) => html`<option value=${split.name}>${split.name}${split.examples != null ? ` (${split.examples.toLocaleString("en")})` : ""}${split.role ? ` · ${split.role}` : ""}</option>`,
        )}
      </select>

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
          ${own && html`<option value=${OWN}>${count ? `Its own (${plural(count, "question")})` : "Its own (on its lines)"}</option>`}
          ${(sets.value ?? []).map((entry) => html`<option value=${entry.name}>The set ${entry.name}</option>`)}
          ${!own && !(sets.value ?? []).length && html`<option value=${OWN}>No questions yet</option>`}
        </select>
        <small class="hint">${
          current.questions === OWN
            ? own
              ? "Its labels give accuracy and the rest."
              : html`${current.dataset} has no questions yet. <a href=${`#/playground/data:${encodeURIComponent(current.dataset ?? "")}`} onClick=${closeSheet}>Write them in the Playground</a>, or pick a question set.`
            : own
              ? "The dataset's labels count for the set's questions asked as the dataset asks them (the same id, type and options); the others get answers only."
              : "Answers only: the dataset has no labels for these questions."
        }</small>
      </div>

      <label for="batch-model">Model</label>
      <select id="batch-model" value=${current.model ?? ""} onChange=${(event) => set({ model: event.target.value })}>
        ${(models.value ?? []).map((entry) => html`<option value=${entry.name}>${entry.name}</option>`)}
      </select>

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
  if (current.keep && !(current.threshold >= 0 && current.threshold <= 100)) return "The confidence is between 0 and 100%";
  return null;
}

export function openSaveDataset(batch) {
  const base = `${batch.dataset}-by-${batch.model}`.replace(/[^A-Za-z0-9._-]+/g, "-").slice(0, 70);
  saving.value = { batch, name: freeName(base), keep: false, threshold: 80 };
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
    const body = { name, threshold: current.keep ? current.threshold / 100 : null };
    const { dataset } = await api(`/batches/${encodeURIComponent(current.batch.name)}/dataset`, { method: "POST", body });
    closeSheet();
    dataChanged();
    toast(`Saved ${plural(dataset.examples, "example")} as ${dataset.name}${dataset.left_out ? `; ${plural(dataset.left_out, "row")} left out` : ""}`);
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
          Only answers at least
          <input id="save-threshold" type="number" min="0" max="100" aria-label="Confidence in percent" value=${current.threshold}
            onInput=${(event) => set({ keep: true, threshold: event.target.valueAsNumber })} />% sure</label>
        <small class="hint">A row with no answer kept is left out. One split, <code>${current.batch.split}</code>, with the role train.</small>
      </div>
    </div>
    ${(error.value || (why && current.name.trim())) && html`<p class="warn-text">${error.value ?? why}</p>`}
  `;
}
