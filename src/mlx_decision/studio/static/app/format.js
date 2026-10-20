// Numbers and words as the Studio shows them.

export function gigabytes(bytes) {
  if (bytes == null) return "";
  const value = bytes / 1e9;
  return `${value === 0 ? "0" : value >= 100 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

// A file size as the CLI writes it: "1.2 GB", or "84 MB" below 0.1 GB.
export function size(bytes) {
  if (bytes == null) return "";
  return bytes >= 1e8 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}

// A parameter count: "421M", "0.6B", "3.1B", "27B".
export function parameters(count) {
  if (!count) return "";
  if (count < 1e9) return `${Math.round(count / 1e6)}M`;
  const billions = count / 1e9;
  return `${billions < 10 ? billions.toFixed(1).replace(/\.0$/, "") : Math.round(billions)}B`;
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

// A word as a shell command line needs it: as it is when safe, else in single quotes.
export function shellWord(word) {
  const text = String(word);
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text;
  return `'${text.replace(/'/g, "'\\''")}'`;
}
