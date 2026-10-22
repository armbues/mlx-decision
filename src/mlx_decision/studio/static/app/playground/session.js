// The Playground's state: which set and model, the set's draft, the live
// check, the runs and the one in flight.
import { batch, computed, effect, signal } from "@preact/signals";
import { api } from "../api.js";
import { showTab } from "../components/inspector.js";
import { route } from "../router.js";
import { remembered } from "../store/prefs.js";
import { sets } from "../store/sets.js";
import { busy, loadedNames, models, overBudget, status } from "../store/status.js";
import { toast } from "../store/ui.js";
import { answersFor, changedSince, currentExample, fromBody, locate, toBody, toFile } from "./draft.js";

const CHECK_DELAY_MS = 150;

const chosenModel = remembered("playground.model", null);
const lastSet = remembered("playground.set", null);

// The model Run uses: the one picked here, else the Studio's default, else the first.
export const modelName = computed(() => {
  const names = (models.value ?? []).map((model) => model.name);
  if (names.includes(chosenModel.value)) return chosenModel.value;
  const fallback = status.value?.default_model;
  return names.includes(fallback) ? fallback : (names[0] ?? null);
});

export function chooseModel(name) {
  chosenModel.value = name;
}

// The model picked here when it is gone (deleted, its folder moved): Run uses
// another until one is picked, and the status field says so.
export const goneModel = computed(() => {
  const list = models.value;
  if (!list || !chosenModel.value) return null;
  return list.some((model) => model.name === chosenModel.value) ? null : chosenModel.value;
});

export const model = computed(
  () => (models.value ?? []).find((entry) => entry.name === modelName.value) ?? null,
);

// A dataset opened in the Playground goes by "data:NAME" where a set goes by
// its name (no set's name holds a colon).
export const DATASET_DOCUMENT = "data:";
export const isDataset = (name) => typeof name === "string" && name.startsWith(DATASET_DOCUMENT);
export const datasetOfDocument = (name) => (isDataset(name) ? name.slice(DATASET_DOCUMENT.length) : null);

// The document shown: the address's (a set, or a dataset), else the last set
// opened, else the first.
export const setName = computed(() => {
  const item = route.value.page === "playground" ? route.value.item : null;
  if (item) return item;
  const names = (sets.value ?? []).map((entry) => entry.name);
  return names.includes(lastSet.value) ? lastSet.value : (names[0] ?? null);
});

// Set name -> the set as loaded and edited in this tab: {sha256, modified,
// baseline, draft, onDisk}, or {error}. `baseline` is the request the loaded
// file makes (as the draft writes it), so an edit is any difference from it;
// `onDisk` is "changed" or "deleted" when the file moved on behind an edited
// draft (an unedited draft just loads the new file).
export const drafts = signal({});

// The file a draft writes: what "Edited" compares (switching examples is not an
// edit; for a dataset only its questions count, its rows being a sample).
export const bodyText = (current, dataset = false) =>
  JSON.stringify(dataset ? [toFile(current).body.questions, answersFor(current)] : toFile(current).body);

export function loadedEntry(loaded, modified = null) {
  const loadedDraft = fromBody(loaded.body);
  return {
    sha256: loaded.sha256,
    modified,
    baseline: bodyText(loadedDraft),
    draft: loadedDraft,
    onDisk: null,
  };
}

export function setEntry(name, value) {
  const next = { ...drafts.value };
  if (value === null) delete next[name];
  else next[name] = value;
  drafts.value = next;
}

const modifiedOf = (name) => (sets.peek() ?? []).find((item) => item.name === name)?.modified ?? null;

// The dataset documents' loader (playground/dataset.js), set there.
export const loaders = { dataset: null };

// Load the set (or dataset) from disk, replacing any draft of it.
export async function loadSet(name) {
  if (isDataset(name)) {
    if (!loaders.dataset) await import("./dataset.js"); // sets the loader
    return loaders.dataset(name);
  }
  try {
    const loaded = await api(`/sets/${encodeURIComponent(name)}`);
    setEntry(name, loadedEntry(loaded, modifiedOf(name)));
  } catch (error) {
    setEntry(name, { error: error.message, status: error.status });
  }
}

effect(() => {
  const name = setName.value;
  if (name === null) return;
  if (!isDataset(name)) lastSet.value = name;
  if (!drafts.peek()[name]) loadSet(name);
});

export const isEdited = (value) => !!value?.draft && bodyText(value.draft, !!value.dataset) !== value.baseline;

// When the sets list shows a file changed or gone (the CLI, an editor, another
// tab): reload an unedited draft, mark an edited one.
effect(() => {
  const list = sets.value;
  if (!list) return;
  const onDisk = new Map(list.map((item) => [item.name, item.modified]));
  for (const [name, value] of Object.entries(drafts.peek())) {
    if (!value.draft || value.dataset) continue; // a dataset's own changes come from Datasets
    if (!onDisk.has(name)) {
      if (value.modified === null) continue; // created here, not in this list yet
      if (isEdited(value)) setEntry(name, { ...value, onDisk: "deleted" });
      else loadSet(name); // shows that it is gone
      continue;
    }
    const modified = onDisk.get(name);
    if (modified === value.modified) continue;
    api(`/sets/${encodeURIComponent(name)}`)
      .then((loaded) => {
        const current = drafts.peek()[name];
        if (!current?.draft) return;
        if (loaded.sha256 === current.sha256) setEntry(name, { ...current, modified, onDisk: null });
        else if (isEdited(current)) setEntry(name, { ...current, modified, onDisk: "changed" });
        else setEntry(name, loadedEntry(loaded, modified));
      })
      .catch(() => {});
  }
});

export const entry = computed(() => (setName.value ? (drafts.value[setName.value] ?? null) : null));
// The dataset open (its document's facts: split, splits, columns, answers,
// missing, …), or null for a set.
export const dataset = computed(() => entry.value?.dataset ?? null);
export const draft = computed(() => entry.value?.draft ?? null);
export const edited = computed(() => isEdited(entry.value));
export const editedNames = computed(
  () => new Set(Object.entries(drafts.value).filter(([, value]) => isEdited(value)).map(([name]) => name)),
);

export function updateDraft(change) {
  const name = setName.value;
  const current = drafts.value[name];
  if (!current?.draft) return;
  setEntry(name, { ...current, draft: change(current.draft) });
}

// The selected example, and the request it makes with the problems that need
// no server; the file's problems (every example's) for saving and Run All.
export const example = computed(() => (draft.value ? currentExample(draft.value) : null));
export const built = computed(() => (draft.value ? toBody(draft.value) : null));
export const fileProblems = computed(() => (draft.value ? toFile(draft.value).problems : []));
const requestKey = computed(() =>
  built.value ? JSON.stringify([modelName.value, built.value.body]) : null,
);

// The live check: the API's own errors for this request and model, asked for
// 150 ms after the last edit. Only a result for the request now shown counts.
const checked = signal({ key: null, errors: [] });
let checkTimer = null;
let checkController = null;
effect(() => {
  const key = requestKey.value;
  const current = built.value;
  const name = modelName.value;
  clearTimeout(checkTimer);
  checkController?.abort();
  if (!key || !name || current.problems.length) return;
  checkTimer = setTimeout(() => {
    checkController = new AbortController();
    api("/check", {
      method: "POST",
      body: { request: current.body, model: name },
      signal: checkController.signal,
    })
      .then((result) => {
        checked.value = { key, errors: result.errors };
      })
      .catch(() => {}); // aborted, or the Studio is gone (the status field says so)
  }, CHECK_DELAY_MS);
});

// A run's refusal, for the request it was about.
const refused = signal({ key: null, errors: [] });

// Every open problem, each {where, message}: the draft's own, then the
// check's (or the run's refusal) for the request shown.
export const problems = computed(() => {
  const current = built.value;
  if (!current) return [];
  if (current.problems.length) return current.problems;
  const located = (errors) =>
    errors.map((error) => ({ where: locate(draft.value, error.param), message: error.message }));
  const key = requestKey.value;
  if (refused.value.key === key && refused.value.errors.length) return located(refused.value.errors);
  return checked.value.key === key ? located(checked.value.errors) : [];
});

// Runs: the latest per set, model and example, {request, response, model,
// example, latency_ms, ...}.
export const runs = signal({});
export const runKey = (set, name, exampleName) => `${set}\u0000${name}\u0000${exampleName}`;
export const lastRun = computed(() =>
  setName.value && modelName.value && example.value
    ? (runs.value[runKey(setName.value, modelName.value, example.value.name)] ?? null)
    : null,
);

export function rememberRun(set, name, exampleName, result) {
  runs.value = { ...runs.value, [runKey(set, name, exampleName)]: result };
}

// A renamed example's answers and Compares in this tab follow it.
export function renameExampleRuns(set, from, to) {
  const suffix = `\u0000${from}`;
  const move = (map) =>
    Object.fromEntries(
      Object.entries(map).map(([key, value]) =>
        key.startsWith(`${set}\u0000`) && key.endsWith(suffix)
          ? [`${key.slice(0, -from.length)}${to}`, value.example === from ? { ...value, example: to } : value]
          : [key, value],
      ),
    );
  batch(() => {
    runs.value = move(runs.value);
    compares.value = move(compares.value);
  });
}

// Run All: the latest batch per set and model, {id, time, examples: [name],
// results: {name: run | {error}}} (filled as the answers come).
export const batches = signal({});
const batchKey = (set, name) => `${set}\u0000${name}`;
export const lastBatch = computed(() =>
  setName.value && modelName.value ? (batches.value[batchKey(setName.value, modelName.value)] ?? null) : null,
);
export function rememberBatch(set, name, value) {
  batches.value = { ...batches.value, [batchKey(set, name)]: value };
}

// Compare: the latest per set and example, {id, time, example, models,
// request, results: {model: run | {error, param}}, stopped?: the model being
// asked when Stop came} (filled as the answers come).
export const compares = signal({});
export const compareKey = (set, exampleName) => `${set}\u0000${exampleName}`;
export function rememberCompare(set, exampleName, value) {
  compares.value = { ...compares.value, [compareKey(set, exampleName)]: value };
}

// A renamed set's answers in this tab follow it.
export function moveRuns(from, to) {
  const move = (map) =>
    Object.fromEntries(
      Object.entries(map).map(([key, value]) =>
        key.startsWith(`${from}\u0000`) ? [`${to}${key.slice(from.length)}`, value] : [key, value],
      ),
    );
  batch(() => {
    runs.value = move(runs.value);
    batches.value = move(batches.value);
    compares.value = move(compares.value);
  });
}

// Drop a set's runs, batches and Compares (its history was cleared).
export function forgetRuns(set) {
  const keep = (map) => Object.fromEntries(Object.entries(map).filter(([key]) => !key.startsWith(`${set}\u0000`)));
  batch(() => {
    runs.value = keep(runs.value);
    batches.value = keep(batches.value);
    compares.value = keep(compares.value);
  });
}

// Answers this tab remembers from runs deleted from the History are dropped.
export function forgetRunIds(ids) {
  const keep = (map) => Object.fromEntries(Object.entries(map).filter(([, value]) => !ids.has(value?.id)));
  runs.value = keep(runs.value);
}

// The run in flight: {model, loads, controller, example, and for a Run All
// all: {done, total}, for a Compare compare: {id, done, total}}.
export const running = signal(null);

// True for a few seconds after a run is answered: the status field's check.
export const justAnswered = signal(false);
const ANSWERED_MS = 4000;
let answeredTimer = null;

// What changed since the last run: {all, ids}.
export const stale = computed(() => {
  const run = lastRun.value;
  if (!run || !built.value) return { all: false, ids: new Set() };
  return changedSince(built.value.body, run.request);
});

// Nothing runs while a job holds the GPU (the API would answer 503 busy).
// A dataset's search found no row: nothing to ask.
const noRow = computed(() => !!example.value?.row?.empty);

// The picked model is over the memory budget: it cannot load (why, or null).
export const modelOverBudget = computed(() => (modelName.value ? overBudget(modelName.value) : null));

export const canRun = computed(
  () =>
    !modelOverBudget.value &&
    !noRow.value &&
    !running.value &&
    !busy.value &&
    !!built.value &&
    !!modelName.value &&
    problems.value.length === 0 &&
    Object.keys(built.value.body.questions).length > 0,
);

function answered() {
  justAnswered.value = true;
  clearTimeout(answeredTimer);
  answeredTimer = setTimeout(() => {
    justAnswered.value = false;
  }, ANSWERED_MS);
}

export async function run() {
  if (!canRun.value) return;
  answersView.value = "one";
  const set = setName.value;
  const name = modelName.value;
  const exampleName = example.value.name;
  const body = built.value.body;
  const key = requestKey.value;
  const controller = new AbortController();
  running.value = { model: name, loads: !loadedNames.value.has(name), controller, example: exampleName };
  try {
    const result = await api("/run", {
      method: "POST",
      body: { request: body, model: name, set, example: exampleName },
      signal: controller.signal,
    });
    batch(() => {
      rememberRun(set, name, exampleName, { ...result, request: body });
      refused.value = { key: null, errors: [] };
    });
    answered();
  } catch (error) {
    if (error.name === "AbortError") return;
    if (error.status === 422) refused.value = { key, errors: [error] };
    else if (error.status === 503) toast(error.message);
    else toast(`The run failed: ${error.message}`);
  } finally {
    running.value = null;
  }
}

// Run All: each example in turn on the model, as one batch. The Studio
// records each run with the batch's id; a refused example shows in its row
// and the batch goes on; Stop ends it after the request in flight.
// With several examples Run All is the main action (⌘↩, the toolbar's
// button); Run asks the selected example only (⌥⌘↩, its chip's menu).
export const several = computed(() => (draft.value?.examples.length ?? 1) > 1);

// The Answers tab: "one" (the selected example's cards), "all" (the Run All
// grid, with several examples) or "compare" (the Compare table, which makes
// Compare the main action); each run shows its own.
export const answersView = signal("all");
export const comparing = computed(() => answersView.value === "compare");

export const canRunAll = computed(
  () =>
    !modelOverBudget.value &&
    !running.value &&
    !busy.value &&
    !!draft.value &&
    draft.value.examples.length > 1 &&
    !!modelName.value &&
    fileProblems.value.length === 0 &&
    problems.value.length === 0 &&
    Object.keys(built.value?.body.questions ?? {}).length > 0,
);

const newBatchId = () => Math.random().toString(36).slice(2, 10);

export const canRunMain = computed(() =>
  comparing.value ? canCompare.value : several.value ? canRunAll.value : canRun.value,
);

export function runMain() {
  if (comparing.value) return compare();
  return several.value ? runAll() : run();
}

export async function runAll() {
  if (!canRunAll.value) return;
  answersView.value = "all";
  const set = setName.value;
  const name = modelName.value;
  const current = draft.value;
  const id = newBatchId();
  const examples = current.examples.map((item) => ({ name: item.name, body: toBody(current, item.uid).body }));
  const controller = new AbortController();
  const value = { id, time: new Date().toISOString(), examples: examples.map((item) => item.name), results: {} };
  rememberBatch(set, name, value);
  try {
    for (const [index, item] of examples.entries()) {
      if (controller.signal.aborted) break;
      running.value = {
        model: name,
        loads: !loadedNames.value.has(name),
        controller,
        example: item.name,
        all: { done: index, total: examples.length },
      };
      try {
        const result = await api("/run", {
          method: "POST",
          body: { request: item.body, model: name, set, example: item.name, batch: id },
          signal: controller.signal,
        });
        const recorded = { ...result, request: item.body };
        batch(() => {
          rememberRun(set, name, item.name, recorded);
          value.results = { ...value.results, [item.name]: recorded };
          rememberBatch(set, name, { ...value });
        });
      } catch (error) {
        if (error.name === "AbortError") break;
        if (error.status === 503) {
          toast(error.message);
          break;
        }
        value.results = { ...value.results, [item.name]: { error: error.message, param: error.param } };
        rememberBatch(set, name, { ...value });
      }
    }
    if (!controller.signal.aborted) answered();
  } finally {
    running.value = null;
  }
}

// Compare (PG-7): the selected example on the model and up to two more, one
// after another; the pool loads each in turn when they do not fit together.
// Each answer is recorded with the Compare's id; a model's refusal shows in
// its column and the others go on; Stop sends no more.
export const MAX_COMPARED = 3;
const comparedOthers = remembered("playground.compareWith", []);

// The other models picked (those that exist, not the model itself).
export const compareWith = computed(() => {
  const names = new Set((models.value ?? []).map((item) => item.name));
  return comparedOthers.value
    .filter((name) => names.has(name) && name !== modelName.value)
    .slice(0, MAX_COMPARED - 1);
});
export const compareModels = computed(() => (modelName.value ? [modelName.value, ...compareWith.value] : []));

// Compare exactly these other models (a fit's base), shown in the Compare view.
export function compareOnly(names) {
  comparedOthers.value = names.slice(0, MAX_COMPARED - 1);
  answersView.value = "compare";
}

// Pick or drop a model to compare with; picking one shows the Compare view.
export function toggleCompared(name) {
  const now = compareWith.value;
  if (now.includes(name)) {
    comparedOthers.value = now.filter((item) => item !== name);
    return;
  }
  if (now.length >= MAX_COMPARED - 1) return;
  comparedOthers.value = [...now, name];
  answersView.value = "compare";
  showTab("playground", "answers");
}

export const canCompare = computed(
  () =>
    !noRow.value &&
    !running.value &&
    !busy.value &&
    !!built.value &&
    compareModels.value.length > 1 &&
    problems.value.length === 0 &&
    Object.keys(built.value.body.questions).length > 0,
);

export async function compare() {
  if (!canCompare.value) return;
  answersView.value = "compare";
  const set = setName.value;
  const exampleName = example.value.name;
  const body = built.value.body;
  const names = [...compareModels.value];
  const id = newBatchId();
  const controller = new AbortController();
  const value = { id, time: new Date().toISOString(), example: exampleName, models: names, request: body, results: {} };
  rememberCompare(set, exampleName, value);
  try {
    for (const [index, name] of names.entries()) {
      if (controller.signal.aborted) break;
      running.value = {
        model: name,
        loads: !loadedNames.value.has(name),
        controller,
        example: exampleName,
        compare: { id, done: index, total: names.length },
      };
      try {
        const result = await api("/run", {
          method: "POST",
          body: { request: body, model: name, set, example: exampleName, compare: id },
          signal: controller.signal,
        });
        const recorded = { ...result, request: body };
        batch(() => {
          rememberRun(set, name, exampleName, recorded);
          value.results = { ...value.results, [name]: recorded };
          rememberCompare(set, exampleName, { ...value });
        });
      } catch (error) {
        if (error.name === "AbortError") {
          rememberCompare(set, exampleName, { ...value, stopped: name }); // its answer is not shown
          break;
        }
        if (error.status === 503) {
          toast(error.message);
          break;
        }
        value.results = { ...value.results, [name]: { error: error.message, param: error.param } };
        rememberCompare(set, exampleName, { ...value });
      }
    }
    if (!controller.signal.aborted) answered();
  } finally {
    running.value = null;
  }
}

// Stop waiting for the run; the Studio finishes it, and its answer is not shown.
export function stop() {
  running.value?.controller.abort();
  running.value = null;
}

// The question whose group and answer card are highlighted, by id.
export const linked = signal(null);
