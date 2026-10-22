// One-line notes after an action, bottom centre; failures with a warning
// icon and a close button (they stay longer, store/ui.js).
import { html } from "htm/preact";
import { dismissToast, holdToast, toasts } from "../store/ui.js";
import { Close, StatusIcon } from "./icons.js";

export function Toasts() {
  return html`
    <div class="toasts" aria-live="polite">
      ${toasts.value.map((item) =>
        item.error
          ? html`<div class="toast error" key=${item.id} role="alert"
                onMouseEnter=${() => holdToast(item.id, true)} onMouseLeave=${() => holdToast(item.id, false)}>
              <${StatusIcon} kind="warn" /><span>${item.text}</span>
              <button class="toast-close" aria-label="Close" title="Close" onClick=${() => dismissToast(item.id)}><${Close} /></button>
            </div>`
          : html`<div class="toast" key=${item.id}>${item.text}</div>`,
      )}
    </div>
  `;
}
