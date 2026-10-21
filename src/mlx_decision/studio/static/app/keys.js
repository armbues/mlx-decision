// Keyboard shortcuts. Pages register what an action does while they are shown;
// a key whose action nobody handles is left to the browser.
import { closePreview, preview } from "./components/lightbox.js";
import { closeSheet, sheet } from "./components/sheet.js";
import { inspector, openMenu, toggleInspector } from "./store/ui.js";

export const SHORTCUTS = {
  run: "⌘↩",
  runExample: "⌥⌘↩", // with several examples; ⌘↩ is then Run All
  download: "⌘D",
  newSet: "⌥⌘N", // ⌘N belongs to Chrome
  import: "⌘O",
  save: "⌘S",
  saveAs: "⇧⌘S",
  escape: "Esc",
};

const handlers = new Map(); // action -> [handler, ...], the latest first

export function onKey(action, handler) {
  handlers.set(action, [handler, ...(handlers.get(action) ?? [])]);
  return () => handlers.set(action, (handlers.get(action) ?? []).filter((h) => h !== handler));
}

export function actionFor(event) {
  if (event.key === "Escape") return "escape";
  if (!event.metaKey || event.ctrlKey) return null;
  const { shiftKey: shift, altKey: alt } = event;
  if (event.key === "Enter" && !shift) return alt ? "runExample" : "run";
  if (event.code === "KeyD" && !shift && !alt) return "download";
  if (event.code === "KeyN" && alt && !shift) return "newSet";
  if (event.code === "KeyO" && !shift && !alt) return "import";
  if (event.code === "KeyS" && !alt) return shift ? "saveAs" : "save";
  return null;
}

// Runs the action's handler; true when one took it.
export function trigger(action, event = null) {
  const [handler] = handlers.get(action) ?? [];
  if (!handler) return false;
  return handler(event) !== false;
}

// Esc closes the image preview, else the open menu, else the sheet, else an
// inspector opened over the centre or on a pick (which unpicks).
onKey("escape", () => {
  if (preview.value !== null) closePreview();
  else if (openMenu.value !== null) openMenu.value = null;
  else if (sheet.value !== null) closeSheet();
  else if (inspector.value.shown && (inspector.value.over || inspector.value.mode === "pick")) toggleInspector();
  else return false;
});

addEventListener("keydown", (event) => {
  const action = actionFor(event);
  if (!action) return;
  if (action !== "escape" && (sheet.value !== null || preview.value !== null)) {
    // Off while a sheet or the preview is open, but not passed on to Chrome
    // either (its Save Page As for cmd-S, say) when a page handles the key.
    if (handlers.get(action)?.length) event.preventDefault();
    return;
  }
  if (trigger(action, event)) event.preventDefault();
});
