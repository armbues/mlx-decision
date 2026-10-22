// Delete a model the Studio made (converted or fitted, in the workspace's
// models/): asked first, unloaded by the Studio before its folder goes.
// Downloaded models are not deleted here (other tools may use them).
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet, sheetError } from "../components/sheet.js";
import { gigabytes } from "../format.js";
import { navigate, route } from "../router.js";
import { loadModels, models } from "../store/status.js";
import { toast } from "../store/ui.js";

// Whether the page offers Delete for it: only the Studio's own models.
export const deletable = (model) => model?.place === "studio";

// Why it cannot go now, or null.
export function deleteWhyNot(model) {
  if (model.default) return `${model.name} is the default model: choose another default in Settings first`;
  const fits = (models.value ?? []).filter((other) => other.fitted && other.base === model.name).map((other) => other.name);
  if (fits.length) return `Fitted models are made from it (${fits.join(", ")}): delete them first`;
  return null;
}

export function confirmDeleteModel(model) {
  const size = model.disk_bytes != null ? ` (${gigabytes(model.disk_bytes)})` : "";
  openSheet({
    title: `Delete the model ${model.name}?`,
    purpose: html`Its folder <code>${model.path}</code>${size} is deleted, after it is unloaded. This cannot be undone.`,
    actions: [
      {
        label: "Delete",
        primary: true,
        danger: true,
        onSelect: async () => {
          try {
            await api(`/models/${encodeURIComponent(model.name)}`, { method: "DELETE" });
          } catch (error) {
            sheetError(`Not deleted: ${error.message}`);
            return;
          }
          closeSheet();
          toast(`Deleted the model ${model.name}`);
          if (route.value.page === "models" && route.value.item === model.name) navigate("models");
          loadModels();
        },
      },
    ],
  });
}
