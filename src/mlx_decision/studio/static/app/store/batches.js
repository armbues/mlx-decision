// Batches: the workspace's batches/ folder as the Batches page lists them, and
// the running one as the stream reports it (`batch` on each change of state,
// `batch_progress` as rows are answered). A batch that ends while the tab is
// open says so in a toast.
import { computed, signal } from "@preact/signals";
import { api } from "../api.js";
import { navigate } from "../router.js";
import { toast } from "./ui.js";

const listed = signal(null); // GET /batches: summaries, the running one first
const live = signal({}); // name -> the stream's latest summary
const progressOf = signal({}); // name -> {answered, total, errors, seconds_left}

export async function loadBatches() {
  try {
    listed.value = (await api("/batches")).batches;
  } catch {
    // The status field reports a lost connection; keep what is shown.
  }
}

// Every batch, the stream's summaries over the list's: running first, then newest.
export const batches = computed(() => {
  if (listed.value == null) return null;
  const all = new Map(listed.value.map((item) => [item.name, item]));
  for (const [name, item] of Object.entries(live.value)) if (all.has(name) || item.state === "running") all.set(name, item);
  return [...all.values()]
    .map((item) => {
      const step = progressOf.value[item.name];
      return step && item.state === "running" ? { ...item, ...step } : item;
    })
    .sort((a, b) => Number(b.state === "running") - Number(a.state === "running") || (b.created ?? "").localeCompare(a.created ?? ""));
});

export const batchByName = (name) => batches.value?.find((item) => item.name === name) ?? null;
export const runningBatch = computed(() => batches.value?.find((item) => item.state === "running") ?? null);

// Bumped when a batch's results change, so the open one reads them again.
export const resultsChanged = signal(0);

export function applyBatchSnapshot(batch) {
  live.value = batch ? { [batch.name]: batch } : {};
  progressOf.value = {};
  loadBatches();
}

export function applyBatch(summary) {
  const before = live.value[summary.name] ?? listed.value?.find((item) => item.name === summary.name);
  live.value = { ...live.value, [summary.name]: summary };
  if (summary.state !== "running") {
    const { [summary.name]: _, ...rest } = progressOf.value;
    progressOf.value = rest;
    resultsChanged.value += 1;
  }
  if (!before || (before.state !== "running" && summary.state === "running")) loadBatches();
  if (before?.state === "running" && summary.state !== "running") toast(endedText(summary));
}

// A batch deleted (here or in another tab): gone from the list.
export function batchesChanged(change) {
  if (change.action === "deleted") {
    const { [change.name]: _, ...rest } = live.value;
    live.value = rest;
  }
  loadBatches();
}

export function applyBatchProgress({ type, name, ...fields }) {
  progressOf.value = { ...progressOf.value, [name]: fields };
}

function endedText(batch) {
  if (batch.state === "done") return `The batch ${batch.name} is done`;
  if (batch.state === "stopped") return `The batch ${batch.name} stopped${batch.message ? `: ${batch.message}` : ""}`;
  return `The batch ${batch.name} failed${batch.message ? `: ${batch.message}` : ""}`;
}

// What a batch's row in a list says: how far it got.
export function batchProgressText(batch) {
  const done = `${batch.answered.toLocaleString("en")} of ${batch.total.toLocaleString("en")}`;
  if (batch.state === "running") {
    const percent = batch.total ? `${Math.floor((batch.answered / batch.total) * 100)}%` : "";
    return [done, percent, secondsLeft(batch.seconds_left)].filter(Boolean).join(" · ");
  }
  if (batch.state === "ready") return "Not started";
  if (batch.state === "stopped") return `Stopped at ${done}`;
  if (batch.state === "failed") return "Failed";
  return `${batch.total.toLocaleString("en")} answered`;
}

export function secondsLeft(seconds) {
  if (seconds == null) return null;
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s left`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min left`;
  return `${(seconds / 3600).toFixed(1)} h left`;
}

const encoded = (name) => encodeURIComponent(name);

async function act(path, what, method = "POST") {
  try {
    const answer = await api(path, { method });
    if (answer?.batch) applyBatch(answer.batch);
    return answer;
  } catch (error) {
    toast(`Could not ${what}: ${error.message}`);
    return null;
  }
}

export const stopBatch = (name) => act(`/batches/${encoded(name)}/stop`, `stop ${name}`);
export const runRest = (name) => act(`/batches/${encoded(name)}/run`, `run the rest of ${name}`);

export async function runAgain(name) {
  const answer = await act(`/batches/${encoded(name)}/again`, `run ${name} again`);
  if (answer?.batch) {
    await loadBatches();
    navigate("batches", answer.batch.name);
  }
}

export async function deleteBatch(name) {
  const answer = await act(`/batches/${encoded(name)}`, `delete ${name}`, "DELETE");
  if (answer) {
    const { [name]: _, ...rest } = live.value;
    live.value = rest;
    await loadBatches();
    toast(`Deleted the batch ${name}`);
    navigate("batches");
  }
}
