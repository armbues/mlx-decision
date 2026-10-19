// The toolbar: sidebar toggle, the page's main action and its target, the
// status field, then the page's tools, the theme and the inspector toggle.
import { html } from "htm/preact";
import { SHORTCUTS } from "../keys.js";
import { busy, connection } from "../store/status.js";
import { inspectorShown, sidebarShown, theme, toggleInspector } from "../store/ui.js";
import * as icons from "./icons.js";
import { MenuButton, MenuItem } from "./menu.js";

// What the status field says: the Studio's own state first, then the page's.
// ``icon`` only when it means something: "spin" while working, "warn" when
// something needs attention, "ok" right after a run is answered; else none.
function fieldContent(page) {
  if (connection.value === "lost")
    return { icon: "warn", text: html`Not connected to the Studio · reload the page` };
  if (busy.value)
    return { icon: "warn", text: html`<b>${busy.value}</b> has the GPU · nothing else runs until it is done` };
  return page.status?.value ?? null;
}

function StatusField({ page }) {
  const content = fieldContent(page);
  return html`
    <div class="activity" role="status" aria-live="polite">
      ${content?.icon && html`<${icons.StatusIcon} kind=${content.icon} />`}
      ${content && html`<span class="txt">${content.text}</span>`}
      ${content?.progress != null &&
      html`<span class="prog" style=${`width:${(content.progress * 100).toFixed(1)}%`}></span>`}
    </div>
  `;
}

function MainAction({ action }) {
  if (!action) return null;
  const title = action.title ?? `${action.label} (${SHORTCUTS.run})`;
  return html`
    <div class="run-group">
      <button class="run-btn" disabled=${action.disabled} title=${title} onClick=${() => action.run?.()}>
        ${action.icon !== false && html`<${icons.Play} />`}${action.label}
        ${action.badge && html`<span class="badge"></span>`}
      </button>
      ${action.stop !== undefined &&
      html`<button
        class="icon-btn stop-btn"
        aria-label="Stop"
        title="Stop"
        disabled=${!action.running}
        onClick=${() => action.stop?.()}
      >
        <${icons.Stop} />
      </button>`}
    </div>
  `;
}

const THEMES = [
  ["system", "Follow the System"],
  ["light", "Light"],
  ["dark", "Dark"],
];

function ThemeMenu() {
  return html`
    <${MenuButton}
      id="theme"
      className="icon-btn"
      ariaLabel="Appearance"
      title="Appearance: follow the system, light or dark"
      label=${html`<${icons.Theme} />`}
      menuStyle="top:34px; right:0; width:200px"
    >
      ${THEMES.map(
        ([value, label]) => html`<${MenuItem}
          ticked=${theme.value === value}
          onSelect=${() => {
            theme.value = value;
          }}
          >${label}<//
        >`,
      )}
    <//>
  `;
}

export function Toolbar({ page }) {
  const Target = page.Target;
  const Tools = page.Tools;
  const sidebarLabel = sidebarShown.value ? "Hide Sidebar" : "Show Sidebar";
  const inspectorLabel = inspectorShown.value ? "Hide Inspector" : "Show Inspector";
  return html`
    <header class="toolbar">
      <div class="tb-left">
      <button
        class="icon-btn"
        aria-label=${sidebarLabel}
        title=${sidebarLabel}
        aria-pressed=${!sidebarShown.value}
        onClick=${() => {
          sidebarShown.value = !sidebarShown.value;
        }}
      >
        <${icons.SidebarToggle} />
      </button>
        <${MainAction} action=${page.action?.value} />
        ${Target && html`<${Target} />`}
      </div>
      <${StatusField} page=${page} />
      <div class="tb-right">
        ${Tools && html`<${Tools} />`}
        <${ThemeMenu} />
        <button class="icon-btn" aria-label=${inspectorLabel} title=${inspectorLabel} onClick=${toggleInspector}>
          <${icons.InspectorToggle} />
        </button>
      </div>
    </header>
  `;
}
