// The inspector: details of what is selected, its tabs as a segmented control.
import { html } from "htm/preact";
import { signal } from "@preact/signals";

const chosenTab = signal({}); // page id -> tab id

export function showTab(pageId, tabId) {
  chosenTab.value = { ...chosenTab.value, [pageId]: tabId };
}

export function Inspector({ page }) {
  const tabs = page.tabs ?? [];
  if (tabs.length === 0) return html`<aside class="inspector" aria-label="Inspector"></aside>`;
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
      </div>
      <div class="insp-body"><${Body} /></div>
    </aside>
  `;
}
