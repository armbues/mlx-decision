// Keyboard shortcuts. Pages register what an action does while they are shown;
// a key whose action nobody handles is left to the browser.
import { closeSheet, sheet } from "./components/sheet.js";
import { inspectorShown, narrow, openMenu, toggleInspector } from "./store/ui.js";

export const SHORTCUTS = {
  run: "⌘↩",
  runAll: "⌥⌘↩",
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
  if (event.key === "Enter" && !shift) return alt ? "runAll" : "run";
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

// Esc closes the open menu, else the sheet, else the inspector opened over a
// narrow window.
onKey("escape", () => {
  if (openMenu.value !== null) openMenu.value = null;
  else if (sheet.value !== null) closeSheet();
  else if (narrow.value && inspectorShown.value) toggleInspector();
  else return false;
});

addEventListener("keydown", (event) => {
  const action = actionFor(event);
  if (action && action !== "escape" && sheet.value !== null) return;
  if (action && trigger(action, event)) event.preventDefault();
});
