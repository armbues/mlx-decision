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

// Set name -> {sha256, draft} as loaded and edited in this tab, or {error}.
export const drafts = signal({});

effect(() => {
  const name = setName.value;
  if (name === null) return;
  lastSet.value = name;
  if (drafts.peek()[name]) return;
  api(`/sets/${encodeURIComponent(name)}`)
    .then((loaded) => {
      drafts.value = {
        ...drafts.value,
        [name]: { sha256: loaded.sha256, draft: fromBody(loaded.body) },
      };
    })
    .catch((error) => {
      drafts.value = { ...drafts.value, [name]: { error: error.message } };
    });
});

export const entry = computed(() => (setName.value ? (drafts.value[setName.value] ?? null) : null));
export const draft = computed(() => entry.value?.draft ?? null);

export function updateDraft(change) {
  const name = setName.value;
  const current = drafts.value[name];
  if (!current?.draft) return;
  drafts.value = { ...drafts.value, [name]: { ...current, draft: change(current.draft) } };
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
