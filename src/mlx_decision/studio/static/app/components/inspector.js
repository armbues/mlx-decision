// The inspector: details of what is selected, its tabs as a segmented control.
import { html } from "htm/preact";
import { signal } from "@preact/signals";
import { inspector, toggleInspector } from "../store/ui.js";
import * as icons from "./icons.js";

const chosenTab = signal({}); // page id -> tab id

export function showTab(pageId, tabId) {
  chosenTab.value = { ...chosenTab.value, [pageId]: tabId };
}

export function Inspector({ page }) {
  const tabs = page.tabs ?? [];
  const { mode, shown, over } = inspector.value;
  // Hidden in the layout (no width): nothing in it. Over the centre it slides out with its content.
  if (tabs.length === 0 || mode === "none" || (!shown && !over)) return html`<aside class="inspector" aria-label="Inspector"></aside>`;
  const current = tabs.find((tab) => tab.id === chosenTab.value[page.id]) ?? tabs[0];
  const choose = (id) => {
    chosenTab.value = { ...chosenTab.value, [page.id]: id };
  };
  const Body = current.Body;
  return html`
    <aside class="inspector" aria-label="Inspector">
      <div class="insp-tabs">
        ${tabs.length > 1 &&
        html`<div class="seg" role="tablist" aria-label="Inspector">
          ${tabs.map(
            (tab) => html`<button
              role="tab"
              aria-pressed=${tab.id === current.id}
              aria-selected=${tab.id === current.id}
              onClick=${() => choose(tab.id)}
            >
              ${tab.label}
            </button>`,
          )}
        </div>`}
        ${tabs.length === 1 && html`<h2 class="insp-title">${current.label}</h2>`}
        ${mode === "pick" &&
        html`<button class="icon-btn insp-close" aria-label=${`Close the ${current.label}`} title="Close (Esc)" onClick=${toggleInspector}><${icons.Close} /></button>`}
      </div>
      <div class="insp-body"><${Body} /></div>
    </aside>
  `;
}
