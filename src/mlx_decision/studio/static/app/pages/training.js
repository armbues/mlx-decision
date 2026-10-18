// Training: fit a model on labelled data and evaluate it.
import { computed } from "@preact/signals";
import { html } from "htm/preact";

const status = computed(() => ({
  icon: "info",
  text: html`Fit from the command line: <code>mlx-decision training fit</code>`,
}));

const action = computed(() => ({ label: "Start Fit", icon: false, disabled: true }));

function Centre() {
  return html`
    <div class="centre-inner">
      <div class="pane-head"><div><h1>Training</h1></div></div>
      <div class="empty">
        <b>Not in the Studio yet</b>
        <span>
          Fit a model on a JSON-lines file of labelled examples with${" "}
          <code>mlx-decision training fit -m MODEL --data FILE.jsonl</code>; the fitted model
          appears under Models.
        </span>
      </div>
    </div>
  `;
}

const Data = () => html`<div class="empty"><b>No training data</b></div>`;

export default {
  id: "training",
  title: "Training",
  action,
  status,
  Centre,
  tabs: [{ id: "data", label: "Data", Body: Data }],
};
