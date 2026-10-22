// Delete an ended fit or evaluation from Training's History: its entry in the
// workspace's jobs/ folder; a fit's model too when asked (it is then unloaded
// and its folder in models/ deleted).
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { gigabytes } from "../format.js";
import { navigate, route } from "../router.js";
import { loadFits } from "../store/fits.js";
import { forgetJob } from "../store/jobs.js";
import { models } from "../store/status.js";
import { toast } from "../store/ui.js";

const withModel = signal(false);

// The model a fit wrote, while the Studio lists it.
const modelOf = (job) =>
  job.kind === "fit" && job.outcome === "done" && job.output ? ((models.value ?? []).find((model) => model.path === job.output) ?? null) : null;

export function confirmDeleteRun(job) {
  const model = modelOf(job);
  withModel.value = false;
  const what = job.kind === "eval" ? "evaluation" : "fit";
  openSheet({
    title: `Delete ${job.title}?`,
    purpose: `The ${what}'s entry goes from the History, with its progress and report.${model ? "" : " What it wrote stays."}`,
    Body: model
      ? () => html`<label class="check">
          <input type="checkbox" checked=${withModel.value} onChange=${(event) => (withModel.value = event.target.checked)} />
          Also delete its model <code>${model.name}</code>${model.disk_bytes != null ? ` (${gigabytes(model.disk_bytes)})` : ""}
        </label>`
      : null,
    actions: () => [
      {
        label: "Delete",
        primary: true,
        danger: true,
        onSelect: async () => {
          const both = Boolean(model && withModel.value);
          closeSheet();
          try {
            await api(`/jobs/${encodeURIComponent(job.id)}${both ? "?model=true" : ""}`, { method: "DELETE" });
            forgetJob(job.id);
            if (route.value.page === "training" && route.value.item === job.id) navigate("training");
            loadFits();
            toast(both ? `Deleted ${job.title} and ${model.name}` : `Deleted ${job.title}`);
          } catch (error) {
            toast(`Not deleted: ${error.message}`);
          }
        },
      },
    ],
  });
}
