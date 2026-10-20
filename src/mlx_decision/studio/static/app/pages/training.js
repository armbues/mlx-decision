// Training: fit a model on labelled data and evaluate it. The data comes from
// the workspace's data/ folder: the sidebar lists it, "+" adds a file, the
// inspector describes the one picked, and the centre is the fit form.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { onKey, SHORTCUTS } from "../keys.js";
import { dataFiles } from "../store/data.js";
import { DataInspector, DataList, pickedData, uploadData } from "../training/data.js";
import { openDatasets } from "../training/datasets.js";
import { FitForm, notReady, planned, startFit } from "../training/fit.js";

const status = computed(() => {
  const data = pickedData();
  if (!data) return dataFiles.value?.length ? { text: "Pick training data on the left" } : null;
  const plan = planned.value;
  if (!plan) return { text: html`Fit on <b>${data}</b>` };
  return { text: html`Fit <b>${plan.base}</b> on <b>${data}</b> · ${plan.splits.examples} examples` };
});

const action = computed(() => ({
  label: "Start Fit",
  icon: false,
  disabled: Boolean(notReady.value),
  title: notReady.value ?? `Start Fit (${SHORTCUTS.run})`,
  run: startFit,
}));

function Centre() {
  useEffect(() => onKey("run", () => startFit()), []);
  const picked = pickedData();
  return html`
    <div class="centre-inner">
      <div class="pane-head"><div><h1>New Fit</h1><p>Teach a model your task from labelled examples. The result is a new model next to its base.</p></div></div>
      ${picked
        ? html`<${FitForm} />`
        : html`<div class="empty">
            <b>${dataFiles.value?.length ? "Pick training data" : "No training data yet"}</b>
            <span>A fit learns from a JSON-lines file of labelled examples in the workspace's data/ folder: pick one on the left, get a public dataset, or add a file of your own.</span>
            <span class="empty-actions">
              <button class="btn" onClick=${openDatasets}>Get a Dataset…</button>
              <button class="btn" onClick=${uploadData}>Add Training Data…</button>
            </span>
          </div>`}
    </div>
  `;
}

export default {
  id: "training",
  title: "Training",
  action,
  status,
  Centre,
  SidebarList: DataList,
  tabs: [{ id: "data", label: "Data", Body: DataInspector }],
};
