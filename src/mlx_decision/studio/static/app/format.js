// Numbers and words as the Studio shows them.

export function gigabytes(bytes) {
  if (bytes == null) return "";
  const value = bytes / 1e9;
  return `${value === 0 ? "0" : value >= 100 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

export function plural(count, one, many = `${one}s`) {
  return `${count} ${count === 1 ? one : many}`;
}

// "127.0.0.1:8000" from "http://127.0.0.1:8000".
export function hostPort(address) {
  return (address || "").replace(/^https?:\/\//, "");
}

// A latency as people read it: "0.4 ms", "212 ms", "1.6 s".
export function duration(ms) {
  if (ms == null) return "";
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

// A run's time of day, from its ISO time.
export const clock = (iso) =>
  iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
