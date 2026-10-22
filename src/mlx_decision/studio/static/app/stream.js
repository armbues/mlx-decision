// One stream per tab: the Studio's messages, applied once per animation frame.
// EventSource reconnects by itself and sends the last id, so the Studio
// replays what was missed.
import { batch, signal } from "@preact/signals";
import { applyBatch, applyBatchProgress, applyBatchSnapshot, batchesChanged } from "./store/batches.js";
import { applyEvent, applyJob, applySnapshot } from "./store/jobs.js";
import { applyStatus, connection, loadModels, loadStatus } from "./store/status.js";
import { dataChanged } from "./store/data.js";
import { setsChanged } from "./store/sets.js";

// The latest change to the workspace ({folder, name, action, ...}), for pages
// that show a set or its history.
export const workspaceChange = signal(null);
// The latest change to the config file ({key}), for the Settings page.
export const settingsChange = signal(null);

let queue = [];
let scheduled = false;

function apply(message) {
  switch (message.type) {
    case "snapshot":
      applyStatus(message.status);
      applySnapshot(message.jobs);
      applyBatchSnapshot(message.batch);
      setsChanged();
      dataChanged();
      loadModels(); // catalog messages may have been missed
      break;
    case "catalog": // models came or went
      loadModels();
      break;
    case "settings": // the config file changed (Settings, in this tab or another)
      settingsChange.value = message;
      loadStatus();
      break;
    case "job":
      applyJob(message.job);
      break;
    case "event":
      applyEvent(message);
      break;
    case "batch":
      applyBatch(message.batch);
      break;
    case "batch_progress":
      applyBatchProgress(message);
      break;
    case "status": {
      const { type, ...fields } = message;
      applyStatus(fields);
      break;
    }
    case "workspace":
      if (message.folder === "sets") setsChanged();
      if (message.folder === "data") dataChanged();
      if (message.folder === "batches") batchesChanged(message);
      workspaceChange.value = message;
      break;
    case "reload": // web --dev: a page file changed
      location.reload();
      break;
    default:
      break;
  }
}

function flush() {
  scheduled = false;
  const messages = queue;
  queue = [];
  batch(() => messages.forEach(apply));
}

// Hidden tabs get no animation frames: catch up when shown again.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && queue.length) flush();
});

// The connection's states: "connecting" (the browser retries), "down" (no
// answer for DOWN_AFTER_MS: the Studio stopped), "restarted" (the stream was
// refused: a new Studio with a new session), "open". After a refusal the
// page asks for the status every POLL_MS and opens a new stream once it
// answers (another tab opened the address the new Studio printed).
const DOWN_AFTER_MS = 2000;
const POLL_MS = 3000;
let downTimer = null;

export function connect() {
  const source = new EventSource("/studio/api/stream");
  source.onopen = () => {
    clearTimeout(downTimer);
    downTimer = null;
    connection.value = "open";
  };
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      source.close();
      waitForStudio();
      return;
    }
    if (connection.value === "open") connection.value = "connecting";
    if (!downTimer)
      downTimer = setTimeout(() => {
        downTimer = null;
        if (connection.value === "connecting") connection.value = "down";
      }, DOWN_AFTER_MS);
  };
  source.onmessage = (event) => {
    queue.push(JSON.parse(event.data));
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(flush);
    }
  };
  return source;
}

async function waitForStudio() {
  clearTimeout(downTimer);
  downTimer = null;
  try {
    const response = await fetch("/studio/api/status", { headers: { Accept: "application/json" } });
    if (response.ok) {
      connect(); // its snapshot brings the page up to date
      return;
    }
    connection.value = response.status === 401 ? "restarted" : "down";
  } catch {
    connection.value = "down";
  }
  setTimeout(waitForStudio, POLL_MS);
}
