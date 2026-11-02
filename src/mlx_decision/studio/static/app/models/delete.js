// Delete a model: one the Studio made (in the workspace's models/), a repo in
// the Hugging Face cache (every revision) or a folder in your models folder
// (models_dir; a link there loses only the link). Asked first, unloaded by
// the Studio before it goes. The Studio says which may go (deletable): never
// a model named with -m or in the config's models.
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet, sheetError } from "../components/sheet.js";
import { gigabytes } from "../format.js";
import { navigate, route } from "../router.js";
import { loadModels, models } from "../store/status.js";
import { toast } from "../store/ui.js";

// Whether the page offers Delete for it.
export const deletable = (model) => Boolean(model?.deletable);

// The Delete button's title: what it deletes.
export function deleteTitle(model) {
  if (model.place === "cache") return `Delete ${model.repo} from the Hugging Face cache`;
  if (model.link) return `Delete the link ${model.name}`;
  return `Delete the folder of ${model.name}`;
}

// What goes, in words, for the confirmation.
function whatGoes(model) {
  const size = model.freed_bytes != null ? ` (${gigabytes(model.freed_bytes)})` : "";
  if (model.place === "cache")
    return html`The Hugging Face cache repo <code>${model.repo}</code>${size}, every revision of it, is deleted after the model is
      unloaded. Other tools share the cache: they download it again when they need it. This cannot be undone.`;
  if (model.link)
    return html`The link <code>${model.path}</code> in your models folder is deleted after the model is unloaded. The folder it
      points to stays.`;
  if (model.place === "models")
    return html`Its folder <code>${model.path}</code>${size} is deleted from disk after it is unloaded. This cannot be undone.`;
  return html`Its folder <code>${model.path}</code>${size} is deleted, after it is unloaded. This cannot be undone.`;
}

// Why it cannot go now, or null.
export function deleteWhyNot(model) {
  if (model.default) return `${model.name} is the default model: choose another default in Settings first`;
  const fits = (models.value ?? []).filter((other) => other.fitted && other.base === model.name).map((other) => other.name);
  if (fits.length) return `Fitted models are made from it (${fits.join(", ")}): delete them first`;
  return null;
}

export function confirmDeleteModel(model) {
  openSheet({
    title: `Delete the model ${model.name}?`,
    purpose: whatGoes(model),
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
