// Per-browser conveniences kept in localStorage; everything works without it.
import { effect, signal } from "@preact/signals";

const PREFIX = "mlxd.";

function read(key, fallback) {
  try {
    const stored = localStorage.getItem(PREFIX + key);
    return stored === null ? fallback : JSON.parse(stored);
  } catch {
    return fallback;
  }
}

// A signal whose value survives a reload of this tab and others like it.
export function remembered(key, fallback) {
  const value = signal(read(key, fallback));
  effect(() => {
    const current = value.value;
    try {
      localStorage.setItem(PREFIX + key, JSON.stringify(current));
    } catch {
      // Storage off (a private window): the value lasts until the reload.
    }
  });
  return value;
}
