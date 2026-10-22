// Sheets: a panel that drops from under the toolbar over a scrim, for a
// choice or a short form. One at a time; Esc or Cancel closes it.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";

// {title, purpose, Body, actions: [{label, primary, danger, disabled, onSelect}]
// (or a function returning them, read as the sheet renders, so they follow
// signals), note, Foot (a component at the footer's start, in place of the
// note), done: a label for a sheet that only shows something (its one button,
// in place of Cancel), wide}
export const sheet = signal(null);

export function openSheet(value) {
  sheet.value = value;
}

export function closeSheet() {
  sheet.value = null;
}

// Why the sheet's action was refused, shown in the sheet above its buttons
// (it stays open); null takes it away.
export function sheetError(message) {
  if (sheet.value) sheet.value = { ...sheet.value, error: message };
}

export function Sheet() {
  const current = sheet.value;
  const panel = useRef(null);
  useEffect(() => {
    if (current) panel.current?.querySelector("input, button.primary, button")?.focus();
  }, [current?.title]);
  if (!current) return null;
  const { title, purpose, Body, note, Foot, done, wide } = current;
  const actions = (typeof current.actions === "function" ? current.actions() : current.actions) ?? [];
  const primary = actions.find((action) => action.primary && !action.disabled);
  const onKeyDown = (event) => {
    if (event.key === "Enter" && event.target.tagName === "INPUT" && primary) {
      event.preventDefault();
      primary.onSelect();
    }
  };
  return html`
    <div class="scrim" onClick=${closeSheet}></div>
    <div class=${`sheet${wide ? " wide" : ""}`} role="dialog" aria-modal="true" aria-label=${title} ref=${panel} onKeyDown=${onKeyDown}>
      <div class="sheet-head">
        <h2>${title}</h2>
        ${purpose && html`<p>${purpose}</p>`}
      </div>
      <div class="sheet-body">${Body && html`<${Body} />`}</div>
      ${current.error && html`<p class="warn-text sheet-error" role="alert">${current.error}</p>`}
      <div class="sheet-foot">
        ${Foot ? html`<${Foot} />` : note && html`<span class="note">${note}</span>`}
        ${done
          ? html`<button class="btn primary" title=${`${done} (Esc)`} onClick=${closeSheet}>${done}</button>`
          : html`<button class="btn" title="Cancel (Esc)" onClick=${closeSheet}>Cancel</button>`}
        ${actions.map(
          (action) => html`<button class=${`btn${action.primary ? " primary" : ""}${action.danger ? " danger" : ""}`}
            disabled=${action.disabled} onClick=${action.onSelect}>${action.label}</button>`,
        )}
      </div>
    </div>
  `;
}
