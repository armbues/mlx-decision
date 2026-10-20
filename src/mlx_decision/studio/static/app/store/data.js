// The datasets: the JSON-lines files of labelled examples in the workspace's
// data/ folder, and each one's summary (questions, labels, counts) once read.
import { signal } from "@preact/signals";
import { api } from "../api.js";

export const dataFiles = signal(null); // null until loaded

export async function reloadData() {
  try {
    dataFiles.value = (await api("/data")).files;
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
