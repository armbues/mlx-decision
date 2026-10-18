// What the Studio serves: its status (address, version, GPU, memory), the
// models and the connection of the stream that keeps them current.
import { computed, signal } from "@preact/signals";
import { api } from "../api.js";

export const status = signal(null);
export const models = signal(null); // null until loaded
export const connection = signal("connecting"); // "connecting", "open" or "lost"
export const loadError = signal(null);

export const loadedNames = computed(() => new Set(status.value?.loaded ?? []));
export const busy = computed(() => status.value?.busy ?? null);

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
