// Training: fit a model on a dataset, and evaluate models on one. The sidebar
// has New Fit and New Evaluation, then the History of fits and evaluations
// (the running one on top); the centre is the fit form (#/training), the
// Evaluate form (#/training/evaluate) or a run's progress and report
// (#/training/JOB). No inspector: the forms show the dataset's labels.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import * as icons from "../components/icons.js";
import { uploadData } from "../datasets/files.js";
import { openGetDataset } from "../datasets/get.js";
import { plural } from "../format.js";
import { onKey, SHORTCUTS } from "../keys.js";
import { href, navigate, route } from "../router.js";
import { dataFiles } from "../store/data.js";
import { fitById, fits } from "../store/fits.js";
import { progress } from "../store/jobs.js";
import { inspectorMode } from "../store/ui.js";
import { confirmDeleteRun } from "../training/delete.js";
import { EvalForm, evalData, evalModel, evalNotReady, startEval } from "../training/evaluate.js";
import { FitForm, fitData, notReady, planned, startFit } from "../training/fit.js";
import { FitView } from "../training/run.js";

const EVALUATE = "evaluate";
// The run shown (#/training/JOB), or null for a form.
const shown = () => {
  const item = route.value.page === "training" ? route.value.item : null;
  return item === EVALUATE ? null : item;
};
const evaluating = () => route.value.page === "training" && route.value.item === EVALUATE;

const status = computed(() => {
  const id = shown();
  if (id) {
    const job = fitById(id);
    return job ? { text: html`<b>${job.title}</b> · ${progress(job).text}` } : null;
  }
  if (evaluating()) {
    const data = evalData.value;
    return evalModel.value && data ? { text: html`Evaluate <b>${evalModel.value}</b> on <b>${data}</b>` } : null;
  }
  const data = fitData.value;
  if (!data) return dataFiles.value ? { text: "No datasets yet · download or add one on the Datasets page" } : null;
  const plan = planned.value;
  if (!plan || plan.data !== data) return { text: html`Fit on <b>${data}</b>` };
  return { text: html`Fit <b>${plan.base}</b> on <b>${data}</b> · ${plural(plan.splits.examples, "example")}` };
});

const action = computed(() =>
  shown()
    ? { label: "New Fit", icon: false, title: "A new fit: the form", run: () => navigate("training") }
    : evaluating()
      ? {
          label: "Start Evaluation",
          icon: false,
          disabled: Boolean(evalNotReady.value),
          title: evalNotReady.value ?? `Start Evaluation (${SHORTCUTS.run})`,
          run: startEval,
        }
      : {
        label: "Start Fit",
        icon: false,
        disabled: Boolean(notReady.value),
        title: notReady.value ?? `Start Fit (${SHORTCUTS.run})`,
        run: startFit,
      },
);

function Form() {
  useEffect(() => onKey("run", () => startFit()), []);
  const list = dataFiles.value;
  return html`
    <div class="pane-head"><div><h1>New Fit</h1><p>Teach a model your task from labelled examples. The result is a new model next to its base.</p></div></div>
    ${list?.length
      ? html`<${FitForm} />`
      : list &&
        html`<div class="empty">
          <b>No datasets yet</b>
          <span>A fit learns from a dataset: labelled examples in the workspace's data/ folder. Get a public dataset or add a file of your own; both are on the <a href=${href("datasets")}>Datasets</a> page.</span>
          <span class="empty-actions">
            <button class="btn" onClick=${openGetDataset}>Download a Dataset…</button>
            <button class="btn" onClick=${uploadData}>Add a Dataset…</button>
          </span>
        </div>`}
  `;
}

function Evaluate() {
  useEffect(() => onKey("run", () => startEval()), []);
  return html`
    <div class="pane-head"><div><h1>New Evaluation</h1><p>Score a model on labelled examples, and another beside it: a fitted model against its base, or two bases.</p></div></div>
    ${dataFiles.value?.length
      ? html`<${EvalForm} />`
      : dataFiles.value &&
        html`<div class="empty"><b>No datasets yet</b><span>An evaluation scores models on a dataset; get one or add a file of your own on the <a href=${href("datasets")}>Datasets</a> page.</span></div>`}
  `;
}

function Centre() {
  const id = shown();
  let body = html`<${Form} />`;
  if (id) body = html`<${FitView} key=${id} id=${id} />`;
  else if (evaluating()) body = html`<${Evaluate} />`;
  return html`<div class="centre-inner">${body}</div>`;
}

// A fit's few words in the list: how far it is, or how it ended (none when done).
function rowText(job) {
  if (job.state !== "ended") {
    const { fraction } = progress(job);
    return job.stop_requested ? "Stopping" : fraction == null ? "Running" : `${Math.floor(fraction * 100)}%`;
  }
  return { done: "", cancelled: "Stopped", interrupted: "Interrupted" }[job.outcome] ?? "Failed";
}

const STATE_ICON = { done: "ok", cancelled: null, error: "warn", failed: "warn", interrupted: "warn" };

// New Fit and New Evaluation, then the History: the running one on top, then newest first.
function FitList() {
  const id = shown();
  const list = fits.value;
  const form = evaluating() ? EVALUATE : id ? null : "fit";
  return html`
    <div class="sb-sec">
      <ul class="sb-list new-list">
        <li><a class="sb-item" href=${href("training")} aria-current=${form === "fit" ? "true" : undefined}><${icons.Plus} /><span class="label">New Fit</span></a></li>
        <li><a class="sb-item" href=${href("training", EVALUATE)} aria-current=${form === EVALUATE ? "true" : undefined}><${icons.Checklist} /><span class="label">New Evaluation</span></a></li>
      </ul>
    </div>
    <div class="sb-sec">
      <h3>History</h3>
      <ul class="sb-list fit-list">
        ${list.map((job) => {
          const running = job.state !== "ended";
          const kind = running ? "spin" : STATE_ICON[job.outcome];
          return html`<li key=${job.id} class=${`fit-row${running ? " running" : ""}`}>
            <a class="sb-item" href=${href("training", job.id)} aria-current=${job.id === id ? "true" : undefined} title=${job.title}>
              ${kind ? html`<${icons.StatusIcon} kind=${kind} />` : html`<${icons.Training} />`}
              ${job.kind === "eval" && html`<span class="run-kind" title="An evaluation">eval</span>`}
              <span class="label">${job.kind === "eval" ? job.title.replace(/^Evaluate /, "") : (job.details?.name ?? job.title)}</span>
              <span class="count">${rowText(job)}</span>
            </a>
            ${!running &&
            html`<button class="sb-delete" aria-label=${`Delete ${job.title}…`} title="Delete…" onClick=${() => confirmDeleteRun(job)}>
              <${icons.Trash} />
            </button>`}
          </li>`;
        })}
      </ul>
      ${list.length === 0 && html`<p class="sb-empty">No fits or evaluations yet</p>`}
    </div>
  `;
}

inspectorMode("training", { mode: "none" });

export default {
  id: "training",
  title: "Training",
  action,
  status,
  Centre,
  SidebarList: FitList,
};
