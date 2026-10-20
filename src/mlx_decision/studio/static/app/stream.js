// One stream per tab: the Studio's messages, applied once per animation frame.
// EventSource reconnects by itself and sends the last id, so the Studio
// replays what was missed.
import { batch, signal } from "@preact/signals";
import { applyEvent, applyJob, applySnapshot } from "./store/jobs.js";
import { applyStatus, connection } from "./store/status.js";
import { setsChanged } from "./store/sets.js";

// The latest change to the workspace ({folder, name, action, ...}), for pages
// that show a set or its history.
export const workspaceChange = signal(null);

let queue = [];
let scheduled = false;

function apply(message) {
  switch (message.type) {
    case "snapshot":
      applyStatus(message.status);
      applySnapshot(message.jobs);
      setsChanged();
      break;
    case "job":
      applyJob(message.job);
      break;
    case "event":
      applyEvent(message);
      break;
    case "status": {
      const { type, ...fields } = message;
      applyStatus(fields);
      break;
    }
    case "workspace":
      if (message.folder === "sets") setsChanged();
      workspaceChange.value = message;
      break;
    case "reload": // web --dev: a page file changed
      location.reload();
      break;
    default:
      break; // the models catalog arrives with its page
  }
}

function flush() {
  scheduled = false;
  const messages = queue;
  queue = [];
  batch(() => messages.forEach(apply));
}

export function connect() {
  const source = new EventSource("/studio/api/stream");
  source.onopen = () => {
    connection.value = "open";
  };
  source.onerror = () => {
    // CLOSED: refused (no session); otherwise the browser retries.
    connection.value = source.readyState === EventSource.CLOSED ? "lost" : "connecting";
  };
  source.onmessage = (event) => {
    queue.push(JSON.parse(event.data));
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(flush);
    }
  };
  // Hidden tabs get no animation frames: catch up when shown again.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && queue.length) flush();
  });
  return source;
}
