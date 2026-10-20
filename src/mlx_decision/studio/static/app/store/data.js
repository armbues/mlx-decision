// The datasets: the JSON-lines files of labelled examples in the workspace's
// data/ folder, and each one's summary (questions, labels, counts) once read.
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

// name -> the file's summary (GET /data/NAME), or {error}.
export const described = signal({});

export async function describe(name) {
  try {
    const summary = await api(`/data/${encodeURIComponent(name)}`);
    described.value = { ...described.value, [name]: summary };
  } catch (error) {
    described.value = { ...described.value, [name]: { error: error.message } };
  }
}

export function forget(name) {
  const { [name]: _, ...rest } = described.value;
  described.value = rest;
}

// When the file was last changed, as the list says (a summary reads again then).
export const modifiedOf = (name) => (dataFiles.value ?? []).find((item) => item.name === name)?.modified;

// The names a new dataset cannot take: the files there, and the ones a Get
// that has not ended is writing.
export const writing = computed(
  () => new Set(jobs.value.filter((job) => job.kind === "data" && job.state !== "ended" && job.details?.name).map((job) => job.details.name)),
);

// Why a new dataset cannot be named ``name``, or null.
export function nameTaken(name) {
  if ((dataFiles.value ?? []).some((item) => item.name === name)) return `A data file named ${name} exists already`;
  if (writing.value.has(name)) return `A Get that is running writes ${name}`;
  return null;
}

// The first free name like ``base``, ``base-2``, ``base-3``….
export function freeName(base) {
  let candidate = base;
  for (let n = 2; nameTaken(candidate); n += 1) candidate = `${base}-${n}`;
  return candidate;
}
