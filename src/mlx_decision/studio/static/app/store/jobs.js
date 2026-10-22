// Jobs (downloads, conversions, fits, evaluations) as the stream reports them.
// The snapshot brings this process's jobs, each with its latest event; `job`
// messages replace a job's summary, `event` messages its latest event. A job
// that ends while the tab is open says so in a toast.
import { computed, signal } from "@preact/signals";
import { api } from "../api.js";
import { gigabytes } from "../format.js";
import { remembered } from "./prefs.js";
import { toast } from "./ui.js";

export const jobs = signal([]); // summaries, newest first
// Ended jobs the person closed in the list (per browser).
const dismissed = remembered("jobs.dismissed", []);

// The job that has the GPU or waits for it.
export const gpuJob = computed(() => jobs.value.find((job) => job.holds_gpu && job.state !== "ended") ?? null);

// What the sidebar lists: every job that has not ended, and the ended ones not
// closed yet.
export const listed = computed(() => {
  const closed = new Set(dismissed.value);
  return jobs.value.filter((job) => job.state !== "ended" || !closed.has(job.id));
});

export function applySnapshot(list) {
  jobs.value = list ?? [];
  // Forget closed jobs this Studio no longer has.
  const ids = new Set(jobs.value.map((job) => job.id));
  if (dismissed.value.some((id) => !ids.has(id))) dismissed.value = dismissed.value.filter((id) => ids.has(id));
}

export function applyJob(summary) {
  // The summary's latest event is as new as any event message before it.
  const before = jobs.value.find((job) => job.id === summary.id);
  if (before?.state === "ended" && summary.state !== "ended") return; // a late answer to Stop
  jobs.value = before
    ? jobs.value.map((item) => (item.id === summary.id ? summary : item))
    : [summary, ...jobs.value];
  if (before && before.state !== "ended" && summary.state === "ended") toast(endedText(summary));
}

// Who wants each event of a job as it comes (the Training page's open fit).
const listeners = new Set();
export function onJobEvent(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function applyEvent({ job: id, index, event }) {
  jobs.value = jobs.value.map((job) => (job.id === id ? { ...job, last: event } : job));
  for (const listener of listeners) listener(id, index, event);
}

export async function stopJob(id) {
  try {
    const { job } = await api(`/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" });
    applyJob(job);
  } catch (error) {
    toast(`Could not stop the job: ${error.message}`);
  }
}

export function dismissJob(id) {
  if (!dismissed.value.includes(id)) dismissed.value = [...dismissed.value, id];
}

// A job deleted (from Training's History, here or in another tab): gone from the list.
export function forgetJob(id) {
  if (jobs.value.some((job) => job.id === id)) jobs.value = jobs.value.filter((job) => job.id !== id);
}

// Close every ended job the list shows; running and waiting ones stay.
export function dismissEnded() {
  const ended = listed.value.filter((job) => job.state === "ended").map((job) => job.id);
  if (ended.length) dismissed.value = [...dismissed.value, ...ended];
}

// How a stage reads in the list.
const STAGES = {
  check: "Checking",
  read: "Reading the dataset",
  load: "Loading",
  download: "Downloading",
  convert: "Converting",
  before: "Scoring before",
  features: "Reading the rows",
  train: "Training",
  calibrate: "Calibrating",
  after: "Scoring after",
  score: "Scoring",
  write: "Writing",
};

const OUTCOMES = {
  done: "Done",
  cancelled: "Stopped",
  error: "Failed",
  failed: "Failed",
  interrupted: "Interrupted",
};

// Seconds left as people read them: "20 s left", "3 min left".
function left(seconds) {
  if (seconds == null) return null;
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s left`;
  return `${Math.round(seconds / 60)} min left`;
}

// A job's progress: `fraction` (0 to 1, or null when its stage counts no
// steps) and `text`, the line under its title.
export function progress(job) {
  if (job.state === "ended") {
    const outcome = OUTCOMES[job.outcome] ?? job.outcome;
    const failed = job.outcome === "error" || job.outcome === "failed";
    const why = failed && job.message ? lastLine(job.message) : null;
    return { fraction: job.outcome === "done" ? 1 : null, text: why ? `${outcome} · ${why}` : outcome };
  }
  if (job.stop_requested) return { fraction: null, text: "Stopping…" };
  if (job.state === "waiting") return { fraction: null, text: job.holds_gpu ? "Waiting for the GPU" : "Starting" };
  const last = job.last;
  if (!last) return { fraction: null, text: "Starting" };
  const stage = STAGES[last.stage] ?? last.stage;
  if (last.step == null || !last.total) return { fraction: null, text: stage };
  const fraction = Math.min(1, last.step / last.total);
  const percent = `${Math.floor(fraction * 100)}%`;
  const amount = last.stage === "download" ? `${gigabytes(last.step).replace(/ GB$/, "")} of ${gigabytes(last.total)}` : stage;
  return { fraction, text: [amount, percent, left(last.eta_s)].filter(Boolean).join(" · ") };
}

function endedText(job) {
  if (job.outcome === "done") return `${job.title} is done`;
  if (job.outcome === "cancelled") return `${job.title} stopped`;
  return `${job.title} failed${job.message ? `: ${lastLine(job.message)}` : ""}`;
}

const lastLine = (text) => text.trim().split("\n").pop();
