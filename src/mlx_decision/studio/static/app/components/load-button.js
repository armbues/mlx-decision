// Load / Unload for one model: what it does now, disabled while a job holds the GPU.
import { html } from "htm/preact";
import { loadModel, pending, unloadModel } from "../store/loading.js";
import { busy, loadedNames, overBudget } from "../store/status.js";

export function LoadButton({ name, className = "btn" }) {
  const loaded = loadedNames.value.has(name);
  const doing = pending.value[name];
  const holder = busy.value;
  const over = overBudget(name);
  const label = doing === "load" ? "Loading…" : doing === "unload" ? "Unloading…" : loaded ? "Unload" : "Load";
  const title = holder
    ? `${holder} has the GPU: Load and Unload are off until it is done`
    : over
      ? over
    : loaded
      ? `Free the memory ${name} takes (it loads again on its next request)`
      : `Load ${name} now, unloading others if it needs the room`;
  const click = (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (loaded) unloadModel(name);
    else loadModel(name);
  };
  return html`
    <button class=${className} disabled=${Boolean(doing || holder || over)} title=${title}
            aria-busy=${doing ? "true" : undefined} onClick=${click}>${label}</button>
  `;
}

// The sidebar's small unload button next to a loaded model's name.
export function UnloadMark({ name, label = name }) {
  const doing = pending.value[name];
  const holder = busy.value;
  return html`
    <button class="mem-x" disabled=${Boolean(doing || holder)}
            title=${holder ? `${holder} has the GPU` : `Unload ${label}`}
            aria-label=${`Unload ${label}`} onClick=${() => unloadModel(name)}>
      <svg viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true">
        <path d="M2.5 2.5l5 5M7.5 2.5l-5 5" />
      </svg>
    </button>
  `;
}
