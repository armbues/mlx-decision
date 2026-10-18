// The Playground's state: which set and model, the set's draft, the live
// check, the runs and the one in flight.
import { batch, computed, effect, signal } from "@preact/signals";
import { api } from "../api.js";
import { route } from "../router.js";
import { remembered } from "../store/prefs.js";
import { sets } from "../store/sets.js";
import { loadedNames, models, status } from "../store/status.js";
import { toast } from "../store/ui.js";
import { changedSince, fromBody, locate, toBody } from "./draft.js";

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

const bodyText = (current) => JSON.stringify(toBody(current).body);

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

// The request the draft makes, with the problems that need no server.
export const built = computed(() => (draft.value ? toBody(draft.value) : null));
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

// Runs: the latest per set and model, {request, response, model, latency_ms, ...}.
export const runs = signal({});
const runKey = (set, name) => `${set}\u0000${name}`;
export const lastRun = computed(() =>
  setName.value && modelName.value ? (runs.value[runKey(setName.value, modelName.value)] ?? null) : null,
);

export const running = signal(null); // {model, loads, controller}

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

export async function run() {
  if (!canRun.value) return;
  const set = setName.value;
  const name = modelName.value;
  const body = built.value.body;
  const key = requestKey.value;
  const controller = new AbortController();
  running.value = { model: name, loads: !loadedNames.value.has(name), controller };
  try {
    const result = await api("/run", {
      method: "POST",
      body: { request: body, model: name, set },
      signal: controller.signal,
    });
    batch(() => {
      runs.value = { ...runs.value, [runKey(set, name)]: { ...result, request: body } };
      refused.value = { key: null, errors: [] };
    });
  } catch (error) {
    if (error.name === "AbortError") return;
    if (error.status === 422) refused.value = { key, errors: [error] };
    else if (error.status === 503) toast(error.message);
    else toast(`The run failed: ${error.message}`);
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
