// Training: fit a model on a dataset and evaluate it. The centre is the fit
// form, which picks its dataset (from data/, the Datasets page's files); the
// sidebar lists the fits; the inspector shows the form's dataset.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import * as icons from "../components/icons.js";
import { uploadData } from "../datasets/files.js";
import { openGetDataset } from "../datasets/get.js";
import { DatasetSummary } from "../datasets/summary.js";
import { plural } from "../format.js";
import { onKey, SHORTCUTS } from "../keys.js";
import { href } from "../router.js";
import { dataFiles } from "../store/data.js";
import { FitForm, fitData, notReady, planned, startFit } from "../training/fit.js";

const status = computed(() => {
  const data = fitData.value;
  if (!data) return dataFiles.value ? { text: "No datasets yet · get or add one on the Datasets page" } : null;
  const plan = planned.value;
  if (!plan || plan.data !== data) return { text: html`Fit on <b>${data}</b>` };
  return { text: html`Fit <b>${plan.base}</b> on <b>${data}</b> · ${plural(plan.splits.examples, "example")}` };
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
  const list = dataFiles.value;
  return html`
    <div class="centre-inner">
      <div class="pane-head"><div><h1>New Fit</h1><p>Teach a model your task from labelled examples. The result is a new model next to its base.</p></div></div>
      ${list?.length
        ? html`<${FitForm} />`
        : list &&
          html`<div class="empty">
            <b>No datasets yet</b>
            <span>A fit learns from a dataset: labelled examples in the workspace's data/ folder. Get a public dataset or add a file of your own; both are on the <a href=${href("datasets")}>Datasets</a> page.</span>
            <span class="empty-actions">
              <button class="btn" onClick=${openGetDataset}>Get a Dataset…</button>
              <button class="btn" onClick=${uploadData}>Add…</button>
            </span>
          </div>`}
    </div>
  `;
}

// The fits: New Fit, then the fits and evaluations once they are listed here.
function FitList() {
  return html`
    <div class="sb-sec">
      <h3>Fits</h3>
      <ul class="sb-list fit-list">
        <li><a class="sb-item" href=${href("training")} aria-current="true"><${icons.Plus} /><span class="label">New Fit</span></a></li>
      </ul>
      <p class="sb-empty">No fits yet</p>
    </div>
  `;
}

const Dataset = () => {
  const name = fitData.value;
  if (!name) return html`<div class="empty"><b>No dataset</b><span>The form's dataset shows here.</span></div>`;
  return html`<${DatasetSummary} name=${name} />`;
};

export default {
  id: "training",
  title: "Training",
  action,
  status,
  Centre,
  SidebarList: FitList,
  tabs: [{ id: "dataset", label: "Dataset", Body: Dataset }],
};
