// The fits and evaluations in the workspace's jobs/ folder (those of this run
// and earlier ones), for the Training sidebar's History; and the events of the
// one open in the centre, loaded once and then taken from the stream.
import { computed, effect, signal } from "@preact/signals";
import { api } from "../api.js";
import { workspaceChange } from "../stream.js";
import { forgetJob, jobs, onJobEvent } from "./jobs.js";

const KINDS = new Set(["fit", "eval"]);
const listed = signal([]); // GET /jobs, filtered

export async function loadFits() {
  try {
    listed.value = (await api("/jobs")).jobs.filter((job) => KINDS.has(job.kind));
  } catch {
    // The status field reports a lost connection; keep what is shown.
  }
}

// Every fit, the running one on top, then newest first; this run's jobs as
// the stream has them.
export const fits = computed(() => {
  const live = new Map(jobs.value.filter((job) => KINDS.has(job.kind)).map((job) => [job.id, job]));
  const all = new Map(listed.value.map((job) => [job.id, live.get(job.id) ?? job]));
  for (const [id, job] of live) all.set(id, job);
  return [...all.values()].sort(
    (a, b) => Number(b.state !== "ended") - Number(a.state !== "ended") || (b.created ?? "").localeCompare(a.created ?? ""),
  );
});

export const fitById = (id) => fits.value.find((job) => job.id === id) ?? null;

// A fit that ends (or starts) in this run: the list again, so earlier runs'
// summaries stay as the stream left them.
let known = "";
effect(() => {
  const now = jobs.value
    .filter((job) => KINDS.has(job.kind))
    .map((job) => `${job.id}:${job.state}`)
    .join(",");
  if (now !== known) {
    known = now;
    loadFits();
  }
});

// A fit or evaluation deleted in another tab.
effect(() => {
  const change = workspaceChange.value;
  if (change?.folder !== "jobs" || change.action !== "deleted") return;
  forgetJob(change.name);
  loadFits();
});

// The open fit's events: {id, events (by index), loaded}.
export const opened = signal({ id: null, events: [], loaded: false });
let pending = []; // events from the stream before the load answered

export async function openFit(id) {
  if (opened.value.id === id) return;
  opened.value = { id, events: [], loaded: false };
  pending = [];
  try {
    const { events } = await api(`/jobs/${encodeURIComponent(id)}`);
    if (opened.value.id !== id) return;
    const all = [...events];
    for (const [index, event] of pending) if (index >= all.length) all[index] = event;
    pending = [];
    opened.value = { id, events: all, loaded: true };
  } catch {
    if (opened.value.id === id) opened.value = { id, events: [], loaded: true };
  }
}

onJobEvent((id, index, event) => {
  const current = opened.value;
  if (current.id !== id) return;
  if (!current.loaded) {
    pending.push([index, event]);
    return;
  }
  const events = [...current.events];
  events[index ?? events.length] = event;
  opened.value = { ...current, events };
});
