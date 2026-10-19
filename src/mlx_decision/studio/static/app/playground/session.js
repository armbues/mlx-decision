// The Playground's state: which set and model, the set's draft, the live
// check, the runs and the one in flight.
import { batch, computed, effect, signal } from "@preact/signals";
import { api } from "../api.js";
import { route } from "../router.js";
import { remembered } from "../store/prefs.js";
import { sets } from "../store/sets.js";
import { loadedNames, models, status } from "../store/status.js";
import { toast } from "../store/ui.js";
import { changedSince, currentExample, fromBody, locate, toBody, toFile } from "./draft.js";

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

export const model = computed(
  () => (models.value ?? []).find((entry) => entry.name === modelName.value) ?? null,
);

// The set shown: the address's, else the last one opened, else the first.
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

// The file a draft writes: what "Edited" compares (switching examples is not an edit).
const bodyText = (current) => JSON.stringify(toFile(current).body);

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

// Load the set from disk, replacing any draft of it.
export async function loadSet(name) {
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
  lastSet.value = name;
  if (!drafts.peek()[name]) loadSet(name);
});

export const isEdited = (value) => !!value?.draft && bodyText(value.draft) !== value.baseline;

// When the sets list shows a file changed or gone (the CLI, an editor, another
// tab): reload an unedited draft, mark an edited one.
effect(() => {
  const list = sets.value;
  if (!list) return;
  const onDisk = new Map(list.map((item) => [item.name, item.modified]));
  for (const [name, value] of Object.entries(drafts.peek())) {
    if (!value.draft) continue;
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

// A renamed example's answers in this tab follow it.
export function renameExampleRuns(set, from, to) {
  const suffix = `\u0000${from}`;
  runs.value = Object.fromEntries(
    Object.entries(runs.value).map(([key, value]) =>
      key.startsWith(`${set}\u0000`) && key.endsWith(suffix) ? [`${key.slice(0, -from.length)}${to}`, value] : [key, value],
    ),
  );
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
  });
}

// Drop a set's runs and batches (its history was cleared).
export function forgetRuns(set) {
  const keep = (map) => Object.fromEntries(Object.entries(map).filter(([key]) => !key.startsWith(`${set}\u0000`)));
  batch(() => {
    runs.value = keep(runs.value);
    batches.value = keep(batches.value);
  });
}

// The run in flight: {model, loads, controller, example, all: {done, total}?}.
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

export const canRun = computed(
  () =>
    !running.value &&
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

// The Answers tab with several examples: "all" (the Run All grid) or "one"
// (the selected example's cards); each run shows its own.
export const answersView = signal("all");

export const canRunAll = computed(
  () =>
    !running.value &&
    !!draft.value &&
    draft.value.examples.length > 1 &&
    !!modelName.value &&
    fileProblems.value.length === 0 &&
    Object.keys(built.value?.body.questions ?? {}).length > 0,
);

const newBatchId = () => Math.random().toString(36).slice(2, 10);

export const canRunMain = computed(() => (several.value ? canRunAll.value : canRun.value));

export function runMain() {
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

// Stop waiting for the run; the Studio finishes it, and its answer is not shown.
export function stop() {
  running.value?.controller.abort();
  running.value = null;
}

// The question whose group and answer card are highlighted, by id.
export const linked = signal(null);
