// One-line notes after an action, bottom centre.
import { html } from "htm/preact";
import { toasts } from "../store/ui.js";

export function Toasts() {
  return html`
    <div class="toasts" aria-live="polite">
      ${toasts.value.map((item) => html`<div class="toast" key=${item.id}>${item.text}</div>`)}
    </div>
  `;
}
