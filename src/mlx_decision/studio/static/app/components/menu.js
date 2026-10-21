// Menus: a button that opens a solid popover; one open at a time. Esc or
// the button again closes it (its tooltip says so while it is open).
import { html } from "htm/preact";
import { useEffect, useRef } from "preact/hooks";
import { openMenu } from "../store/ui.js";
import { Tick } from "./icons.js";

export function MenuButton({ id, label, className, title, menuStyle, ariaLabel, children }) {
  const open = openMenu.value === id;
  const menu = useRef(null);
  useEffect(() => {
    if (open) menu.current?.querySelector(".mi:not([disabled])")?.focus();
  }, [open]);
  const toggle = (event) => {
    event.stopPropagation();
    openMenu.value = open ? null : id;
  };
  return html`
    <div class="menu-wrap">
      <button
        class=${className}
        aria-haspopup="menu"
        aria-expanded=${open}
        aria-label=${ariaLabel}
        title=${open ? "Close (Esc)" : title}
        onClick=${toggle}
      >
        ${label}
      </button>
      ${open &&
      html`<div class="menu" role="menu" ref=${menu} style=${menuStyle} onClick=${(e) => e.stopPropagation()}>
        ${children}
      </div>`}
    </div>
  `;
}

export function MenuItem({ onSelect, ticked = false, right = "", sub = null, disabled = false, children }) {
  const select = () => {
    openMenu.value = null;
    onSelect?.();
  };
  return html`
    <button class="mi" role="menuitem" disabled=${disabled} onClick=${select}>
      <span>${ticked && html`<${Tick} />`}</span>
      <span>${children}${sub && html`<span class="sub">${sub}</span>`}</span>
      <span class="right">${right}</span>
    </button>
  `;
}

export const MenuHeader = ({ children }) => html`<div class="mh">${children}</div>`;
export const MenuSeparator = () => html`<div class="sep"></div>`;
export const MenuNote = ({ children }) => html`<div class="foot">${children}</div>`;
