// The datasets in the workspace's data/ folder (a dataset folder with its
// splits, or a JSON-lines file), and each split's summary (questions, labels,
// values, counts) once read.
import { computed, signal } from "@preact/signals";
import { api } from "../api.js";
import { jobs } from "./jobs.js";

export const dataFiles = signal(null); // null until loaded
// The data/ folder ({folder, display}) and the most an upload can hold.
export const dataFolder = signal(null);

export async function reloadData() {
  try {
    const { files, ...folder } = await api("/data");
    dataFiles.value = files;
    dataFolder.value = folder;
  } catch {
    // The status field reports a lost connection; keep what is shown.
  }
}

// Several workspace messages in a row reload the list once.
let pending = null;
export function dataChanged() {
  clearTimeout(pending);
  pending = setTimeout(reloadData, 50);
}

// "NAME\u0000SPLIT" -> a split's summary (GET /data/NAME?split=SPLIT), or
// {error}; SPLIT empty for the dataset's default split.
export const described = signal({});
export const summaryKey = (name, split = null) => `${name}\u0000${split ?? ""}`;

export async function describe(name, split = null) {
  const key = summaryKey(name, split);
  try {
    const query = split ? `?split=${encodeURIComponent(split)}` : "";
    const summary = await api(`/data/${encodeURIComponent(name)}${query}`);
    described.value = { ...described.value, [key]: summary };
  } catch (error) {
    described.value = { ...described.value, [key]: { error: error.message } };
  }
}

export function forget(name) {
  described.value = Object.fromEntries(Object.entries(described.value).filter(([key]) => !key.startsWith(`${name}\u0000`)));
}

// A dataset as the list has it ({name, kind, splits, missing, ...}), or null.
export const datasetOf = (name) => (dataFiles.value ?? []).find((item) => item.name === name) ?? null;

// The split a run reads by default: the one with ``role`` (evaluate falls back
// to report), else the only one; null when there is none to pick.
export function splitFor(item, role) {
  const splits = item?.splits ?? [];
  const order = role === "evaluate" ? ["evaluate", "report"] : [role];
  for (const wanted of order) {
    const found = splits.find((split) => split.role === wanted);
    if (found) return found.name;
  }
  return splits.length === 1 ? splits[0].name : null;
}

// When the file was last changed, as the list says (a summary reads again then).
export const modifiedOf = (name) => (dataFiles.value ?? []).find((item) => item.name === name)?.modified;

// The names a new dataset cannot take: the datasets there, and the ones a
// download that has not ended is writing.
export const writing = computed(
  () => new Set(jobs.value.filter((job) => job.kind === "data" && job.state !== "ended" && job.details?.name).map((job) => job.details.name)),
);

// Why a new dataset cannot be named ``name``, or null.
export function nameTaken(name) {
  if ((dataFiles.value ?? []).some((item) => item.name === name)) return `A dataset named ${name} exists already`;
  if (writing.value.has(name)) return `A download that is running writes ${name}`;
  return null;
}

// The first free name like ``base``, ``base-2``, ``base-3``….
export function freeName(base) {
  let candidate = base;
  for (let n = 2; nameTaken(candidate); n += 1) candidate = `${base}-${n}`;
  return candidate;
}
