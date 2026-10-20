// The training data files in the workspace's data/ folder.
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
