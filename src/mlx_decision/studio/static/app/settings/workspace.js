// The workspace's folders on the Settings page: what each holds, its size,
// and Clear… (everything in it deleted, as deleting each entry would; asked
// first). Only from a browser on this Mac.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet, sheetError } from "../components/sheet.js";
import { fileSize, plural } from "../format.js";
import { runningBatch } from "../store/batches.js";
import { jobs } from "../store/jobs.js";
import { toast } from "../store/ui.js";

const listed = signal(null); // {folders, editable}

export async function loadFolders() {
  try {
    listed.value = await api("/workspace/folders");
  } catch (error) {
    toast(`Could not read the workspace's folders: ${error.message}`);
  }
}

// What a folder's entries are called, one and many.
const ENTRIES = {
  sets: ["set", "sets"],
  history: ["history", "histories"],
  data: ["dataset", "datasets"],
  batches: ["batch", "batches"],
  models: ["model", "models"],
  jobs: ["job", "jobs"],
  speed: ["record", "records"],
};
const count = (folder) => plural(folder.count, ...ENTRIES[folder.folder]);

function confirmClear(folder) {
  openSheet({
    title: `Clear ${folder.folder}/?`,
    purpose: html`Deletes everything in <code>${folder.display}</code> (${count(folder)}, ${fileSize(folder.bytes)}): ${folder.what.charAt(0).toLowerCase()}${folder.what.slice(1)}. This cannot be undone.${
      folder.folder === "jobs" ? " Jobs that have not ended stay." : ""
    }`,
    actions: [
      {
        label: "Clear",
        primary: true,
        danger: true,
        onSelect: async () => {
          let result;
          try {
            result = await api(`/workspace/folders/${folder.folder}/clear`, { method: "POST" });
          } catch (error) {
            sheetError(`Not cleared: ${error.message}`);
            return;
          }
          closeSheet();
          const kept = result.kept ? `; ${plural(result.kept, "job")} still running stayed` : "";
          toast(`Cleared ${folder.folder}/: ${plural(result.cleared, ...ENTRIES[folder.folder])} deleted${kept}`);
          loadFolders();
        },
      },
    ],
  });
}

// Why a folder cannot be cleared now (a job reads a dataset, a batch runs), or null.
function busyWith(folder) {
  if (folder === "data") {
    const reader = jobs.value.find((job) => job.state !== "ended" && (job.details?.data || job.details?.report));
    if (reader) return `${reader.title} reads a dataset: stop it or wait until it ends`;
  }
  if (folder === "batches" && runningBatch.value) return `The batch ${runningBatch.value.name} is running: stop it first`;
  if (folder === "models") {
    const job = jobs.value.find((item) => item.state !== "ended" && item.holds_gpu);
    if (job) return `${job.title} is running: wait until it ends`;
  }
  return null;
}

export function WorkspaceFolders() {
  const current = listed.value;
  if (!current) return null;
  const { folders, editable } = current;
  return html`
    <section class="igroup config-group workspace-group" aria-label="Workspace folders">
      <h4>Workspace</h4>
      <dl>
        ${folders.map(
          (folder) => html`
            <div class="kv folder" data-folder=${folder.folder}>
              <dt><code>${folder.folder}/</code></dt>
              <dd>${folder.what}<small class="num">${count(folder)} · ${fileSize(folder.bytes)}</small></dd>
              <button class="btn sm" disabled=${!editable || folder.count === 0 || Boolean(busyWith(folder.folder))}
                      title=${!editable ? "Only from a browser on this Mac" : folder.count === 0 ? "Nothing in it" : (busyWith(folder.folder) ?? `Delete everything in ${folder.display}`)}
                      onClick=${() => confirmClear(folder)}>Clear…</button>
            </div>
          `,
        )}
      </dl>
    </section>
  `;
}
