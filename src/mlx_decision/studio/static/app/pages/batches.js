// Batches: a dataset's rows answered by a model, kept in the workspace's
// batches/ folder. The sidebar has New Batch and the batches (the running one
// on top, then newest first); the centre shows the one picked (#/batches/NAME):
// what it ran, its progress, its answers and metrics, its rows; the inspector
// the row picked.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { openNewBatch } from "../batches/start.js";
import { BatchView, confirmDelete, pickedRow, RowInspector } from "../batches/view.js";
import { uploadData } from "../datasets/files.js";
import { openGetDataset } from "../datasets/get.js";
import * as icons from "../components/icons.js";
import { plural } from "../format.js";
import { href, route } from "../router.js";
import { batchByName, batches, batchProgressText, loadBatches } from "../store/batches.js";
import { dataFiles } from "../store/data.js";
import { inspectorMode } from "../store/ui.js";

const shown = () => (route.value.page === "batches" ? route.value.item : null);

const status = computed(() => {
  const name = shown();
  const batch = name ? batchByName(name) : null;
  if (batch) return { text: html`<b>${batch.name}</b> · ${batchProgressText(batch)}` };
  const list = batches.value;
  if (!list) return null;
  if (list.length === 0) return { text: "No batches yet" };
  return { text: html`<b>${plural(list.length, "batch", "batches")}</b> in batches/` };
});

const action = computed(() => ({
  label: "New Batch…",
  icon: false,
  title: "New Batch… (a dataset's rows answered by a model)",
  run: () => openNewBatch(),
}));

function Empty() {
  const none = dataFiles.value && dataFiles.value.length === 0;
  return html`
    <div class="pane-head"><div><h1>Batches</h1></div></div>
    <div class="empty batch-empty">
      <b>No batches yet</b>
      <span>A batch answers a dataset's rows with a model and, with labels, shows how often it is right.</span>
      <span class="empty-actions">${
        none
          ? html`<button class="btn primary" onClick=${openGetDataset}>Download a Dataset…</button>
              <button class="btn" onClick=${uploadData}>Add a Dataset…</button>`
          : html`<button class="btn primary" onClick=${() => openNewBatch()}>New Batch…</button>`
      }</span>
    </div>
  `;
}

function Centre() {
  useEffect(() => {
    loadBatches();
  }, []);
  const list = batches.value;
  const name = shown();
  let body;
  if (!list) body = null;
  else if (name) body = html`<${BatchView} key=${name} name=${name} />`;
  else if (list.length === 0) body = html`<${Empty} />`;
  else
    body = html`
      <div class="pane-head"><div><h1>Batches</h1><p>${plural(list.length, "batch", "batches")} in the workspace's batches/ folder</p></div></div>
      <div class="empty"><b>Pick a batch</b><span>Pick one on the left to see its answers, or start a new one.</span></div>`;
  return html`<div class="centre-inner">${body}</div>`;
}

const STATE_ICON = { done: "ok", failed: "warn" };

function BatchList() {
  const list = batches.value ?? [];
  const name = shown();
  return html`
    <div class="sb-sec">
      <ul class="sb-list new-list">
        <li><button class="sb-item" onClick=${() => openNewBatch()}><${icons.Plus} /><span class="label">New Batch…</span></button></li>
      </ul>
    </div>
    <div class="sb-sec">
      <h3>Batches</h3>
      <ul class="sb-list batch-list">
        ${list.map((batch) => {
          const running = batch.state === "running";
          const kind = running ? "spin" : STATE_ICON[batch.state];
          const short = running
            ? `${batch.total ? Math.floor((batch.answered / batch.total) * 100) : 0}%`
            : { stopped: "Stopped", failed: "Failed", ready: "Not started" }[batch.state] ?? batch.answered.toLocaleString("en");
          return html`<li key=${batch.name} class=${`batch-entry${running ? " running" : ""}`}>
            <a class="sb-item" href=${href("batches", batch.name)} aria-current=${batch.name === name ? "true" : undefined}
               title=${`${batch.name}: ${batch.dataset} on ${batch.model}`}>
              ${kind ? html`<${icons.StatusIcon} kind=${kind} />` : html`<${icons.Batches} />`}
              <span class="label">${batch.name}<small>${batch.dataset} · ${batch.model}</small></span>
              <span class="count num">${short}</span>
            </a>
            ${!running &&
            html`<button class="sb-delete" aria-label=${`Delete ${batch.name}…`} title="Delete…" onClick=${() => confirmDelete(batch)}>
              <${icons.Trash} />
            </button>`}
          </li>`;
        })}
      </ul>
      ${list.length === 0 && html`<p class="sb-empty">No batches yet</p>`}
    </div>
  `;
}

const Row = () => html`<${RowInspector} name=${shown()} />`;

// The inspector only while a row is picked (closing it unpicks the row).
inspectorMode("batches", {
  mode: "pick",
  picked: computed(() => pickedRow.value != null && pickedRow.value.name === shown()),
  close: () => (pickedRow.value = null),
});

export default {
  id: "batches",
  title: "Batches",
  action,
  status,
  Centre,
  SidebarList: BatchList,
  tabs: [{ id: "row", label: "Row", Body: Row }],
};

