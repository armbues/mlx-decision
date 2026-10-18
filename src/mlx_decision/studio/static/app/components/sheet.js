// Sheets: a panel that drops from under the toolbar over a scrim, for a
// choice or a short form. One at a time; Esc or Cancel closes it.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";

// {title, purpose, Body, actions: [{label, primary, danger, disabled, onSelect}], note}
export const sheet = signal(null);

export function openSheet(value) {
  sheet.value = value;
}

export function closeSheet() {
  sheet.value = null;
}

export function Sheet() {
  const current = sheet.value;
  const panel = useRef(null);
  useEffect(() => {
    if (current) panel.current?.querySelector("input, button.primary, button")?.focus();
  }, [current?.title]);
  if (!current) return null;
  const { title, purpose, Body, actions = [], note } = current;
  const primary = actions.find((action) => action.primary && !action.disabled);
  const onKeyDown = (event) => {
    if (event.key === "Enter" && event.target.tagName === "INPUT" && primary) {
      event.preventDefault();
      primary.onSelect();
    }
  };
  return html`
    <div class="scrim" onClick=${closeSheet}></div>
    <div class="sheet" role="dialog" aria-modal="true" aria-label=${title} ref=${panel} onKeyDown=${onKeyDown}>
      <div class="sheet-head">
        <h2>${title}</h2>
        ${purpose && html`<p>${purpose}</p>`}
      </div>
      <div class="sheet-body">${Body && html`<${Body} />`}</div>
      <div class="sheet-foot">
        ${note && html`<span class="note">${note}</span>`}
        <button class="btn" onClick=${closeSheet}>Cancel</button>
        ${actions.map(
          (action) => html`<button class=${`btn${action.primary ? " primary" : ""}${action.danger ? " danger" : ""}`}
            disabled=${action.disabled} onClick=${action.onSelect}>${action.label}</button>`,
        )}
      </div>
    </div>
  `;
}
