// Training: fit a model on labelled data and evaluate it. The data comes from
// the workspace's data/ folder: the sidebar lists it, "+" adds a file, the
// inspector describes the one picked.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { dataFiles } from "../store/data.js";
import { DataInspector, DataList, pickedData, uploadData } from "../training/data.js";

const status = computed(() => {
  const count = dataFiles.value?.length;
  if (count == null) return null;
  return { text: html`Fit from the command line: <code>mlx-decision training fit</code>` };
});

const action = computed(() => ({ label: "Start Fit", icon: false, disabled: true }));

function Centre() {
  const picked = pickedData();
  return html`
    <div class="centre-inner">
      <div class="pane-head"><div><h1>Training</h1></div></div>
      <div class="empty">
        <b>${picked ? `Fit on ${picked}: not in the Studio yet` : "Not in the Studio yet"}</b>
        <span>Fit a model on a JSON-lines file of labelled examples with <code>mlx-decision training fit -m MODEL --data FILE.jsonl</code>; the fitted model appears under Models. Add the file here with + under Training Data to see its questions and labels.</span>
        ${!picked && html`<button class="btn" onClick=${uploadData}>Add Training Data…</button>`}
      </div>
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
