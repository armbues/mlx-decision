// Load and Unload by hand: the model's button state and what the toast says.
// The status message that follows the change updates the lists in every tab.
import { signal } from "@preact/signals";
import { api } from "../api.js";
import { duration } from "../format.js";
import { toast } from "./ui.js";

export const pending = signal({}); // model name -> "load" or "unload"

function setPending(name, action) {
  const next = { ...pending.value };
  if (action) next[name] = action;
  else delete next[name];
  pending.value = next;
}

const names = (list) => list.join(", ");

export async function loadModel(name) {
  if (pending.value[name]) return;
  setPending(name, "load");
  try {
    const done = await api("/models/load", { method: "POST", body: { model: name } });
    if (done.load_ms == null) toast(`${name} is already loaded`);
    else {
      const room = done.unloaded.length ? ` · unloaded ${names(done.unloaded)} to make room` : "";
      toast(`Loaded ${name} in ${duration(done.load_ms)}${room}`);
    }
  } catch (error) {
    toast(`Could not load ${name}: ${error.message}`);
  } finally {
    setPending(name, null);
  }
}

export async function unloadModel(name) {
  if (pending.value[name]) return;
  setPending(name, "unload");
  try {
    const done = await api("/models/unload", { method: "POST", body: { model: name } });
    toast(done.unloaded.length ? `Unloaded ${names(done.unloaded)}` : `${name} was not loaded`);
  } catch (error) {
    toast(`Could not unload ${name}: ${error.message}`);
  } finally {
    setPending(name, null);
  }
}
