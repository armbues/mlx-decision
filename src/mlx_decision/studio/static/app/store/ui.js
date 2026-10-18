// The window: theme, sidebar and inspector, the open menu, toasts.
import { computed, effect, signal } from "@preact/signals";
import { remembered } from "./prefs.js";

export const theme = remembered("theme", "system"); // "system", "light" or "dark"
export const sidebarShown = remembered("sidebar", true);
const inspectorPinned = remembered("inspector", true);

// Below 1,100 px the inspector leaves the layout and opens over the centre.
const narrowQuery = matchMedia("(max-width: 1099px)");
export const narrow = signal(narrowQuery.matches);
const inspectorOverlay = signal(false);
narrowQuery.addEventListener("change", (event) => {
  narrow.value = event.matches;
  inspectorOverlay.value = false;
});

export const inspectorShown = computed(() =>
  narrow.value ? inspectorOverlay.value : inspectorPinned.value,
);

export function toggleInspector() {
  if (narrow.value) inspectorOverlay.value = !inspectorOverlay.value;
  else inspectorPinned.value = !inspectorPinned.value;
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
