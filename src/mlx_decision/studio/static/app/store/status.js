// What the Studio serves: its status (address, version, GPU, memory), the
// models and the connection of the stream that keeps them current.
import { computed, signal } from "@preact/signals";
import { api } from "../api.js";

export const status = signal(null);
export const models = signal(null); // null until loaded
export const connection = signal("connecting"); // "connecting", "open", "down" or "restarted" (stream.js)
// The page cannot reach the Studio: stopped, or restarted with a new session.
export const offline = computed(() => connection.value === "down" || connection.value === "restarted");
export const loadError = signal(null);

export const loadedNames = computed(() => new Set(status.value?.loaded ?? []));
export const busy = computed(() => status.value?.busy ?? null);

// Why a model cannot load (more than the memory budget), or null.
export function overBudget(name) {
  const model = (models.value ?? []).find((item) => item.name === name);
  if (!model?.memory?.over_budget || loadedNames.value.has(name)) return null;
  const budget = status.value?.memory?.budget;
  const needs = `about ${(model.memory.bytes / 1e9).toFixed(1)} GB`;
  return `${name} needs ${needs}, more than the ${budget ? `${(budget / 1e9).toFixed(1)} GB ` : ""}budget: raise it in Settings`;
}

export function applyStatus(fields) {
  status.value = { ...(status.value ?? {}), ...fields };
}

export async function loadStatus() {
  try {
    applyStatus(await api("/status"));
    loadError.value = null;
  } catch (error) {
    loadError.value = error.message;
  }
}

export async function loadModels() {
  try {
    models.value = (await api("/models")).models;
  } catch (error) {
    loadError.value = error.message;
  }
}

// Look for models that came or went (the CLI may have downloaded or deleted
// one); the catalog message that follows a change reloads the list in every tab.
export async function rescanModels() {
  try {
    await api("/models/rescan", { method: "POST" });
  } catch {
    // The list below is still worth reloading.
  }
  await loadModels();
}
