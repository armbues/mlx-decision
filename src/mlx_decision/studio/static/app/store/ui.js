// The window: theme, sidebar and inspector, the open menu, toasts.
import { computed, effect, signal } from "@preact/signals";
import { route } from "../router.js";
import { remembered } from "./prefs.js";

export const theme = remembered("theme", "system"); // "system", "light" or "dark"
export const sidebarShown = remembered("sidebar", true);
// Shown or hidden, per page that always has an inspector (once one setting
// for all pages: its value is the default).
const inspectorPinned = remembered("inspector", true);
const pinnedPages = remembered("inspector.pages", {});

// Below 1,100 px the inspector leaves the layout and opens over the centre.
const narrowQuery = matchMedia("(max-width: 1099px)");
export const narrow = signal(narrowQuery.matches);
const inspectorOverlay = signal(false);
narrowQuery.addEventListener("change", (event) => {
  narrow.value = event.matches;
  inspectorOverlay.value = false;
});

// Below 1,280 px an inspector shown on a pick opens over the centre.
const compactQuery = matchMedia("(max-width: 1279px)");
const compact = signal(compactQuery.matches);
compactQuery.addEventListener("change", (event) => {
  compact.value = event.matches;
});

// How each page uses the inspector, as the page registers it: "always" (the
// default), "pick" ({picked: a signal, close}: shown while something is
// picked; closing it unpicks) or "none".
const modes = {};
export function inspectorMode(pageId, spec) {
  modes[pageId] = spec;
}
const modeOf = (pageId) => modes[pageId] ?? { mode: "always" };

// The current page's inspector: {mode, shown, over} (over: over the centre, not in the layout).
export const inspector = computed(() => {
  const id = route.value.page;
  const spec = modeOf(id);
  if (spec.mode === "none") return { mode: "none", shown: false, over: false };
  if (spec.mode === "pick") return { mode: "pick", shown: Boolean(spec.picked.value), over: narrow.value || compact.value };
  const shown = narrow.value ? inspectorOverlay.value : (pinnedPages.value[id] ?? inspectorPinned.value);
  return { mode: "always", shown, over: narrow.value };
});

export const inspectorShown = computed(() => inspector.value.shown);

export function toggleInspector() {
  const id = route.value.page;
  const spec = modeOf(id);
  if (spec.mode === "none") return;
  if (spec.mode === "pick") {
    if (spec.picked.value) spec.close();
    return;
  }
  if (narrow.value) inspectorOverlay.value = !inspectorOverlay.value;
  else pinnedPages.value = { ...pinnedPages.value, [id]: !(pinnedPages.value[id] ?? inspectorPinned.value) };
}

effect(() => {
  const root = document.documentElement;
  if (theme.value === "light" || theme.value === "dark") root.dataset.theme = theme.value;
  else delete root.dataset.theme;
});

// One menu open at a time, named by its id; Esc and a click elsewhere close it.
export const openMenu = signal(null);

export const toasts = signal([]);
let toastId = 0;
export function toast(text) {
  const id = ++toastId;
  toasts.value = [...toasts.value, { id, text }];
  setTimeout(() => {
    toasts.value = toasts.value.filter((item) => item.id !== id);
  }, 3500);
}
