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
