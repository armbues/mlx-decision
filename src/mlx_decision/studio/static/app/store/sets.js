// The question sets in the workspace's sets/ folder.
import { signal } from "@preact/signals";
import { api } from "../api.js";

export const sets = signal(null); // null until loaded

export async function reloadSets() {
  try {
    sets.value = (await api("/sets")).sets;
  } catch {
    // The status field reports a lost connection; keep what is shown.
  }
}

// Several workspace messages in a row reload the list once.
let pending = null;
export function setsChanged() {
  clearTimeout(pending);
  pending = setTimeout(reloadSets, 50);
}
