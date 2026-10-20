// While a job holds the GPU, every page says so, with Stop.
import { html } from "htm/preact";
import { gpuJob, stopJob } from "../store/jobs.js";
import { busy } from "../store/status.js";
import { StatusIcon } from "./icons.js";

// What the job means for each page.
const MEANS = {
  playground: "Run is off until it is done",
  models: "Load and Unload are off until it is done",
};

export function BusyBanner({ page }) {
  const holder = busy.value;
  if (!holder) return null;
  // The GPU's holder is the job's title.
  const job = gpuJob.value?.title === holder ? gpuJob.value : null;
  return html`
    <div class="banner" role="status">
      <${StatusIcon} kind="warn" />
      <span>
        <b>${holder}</b> has the GPU. ${MEANS[page] ?? "Nothing runs on it until it is done"},
        and the API answers 503 busy meanwhile.
      </span>
      ${job &&
      html`<button class="btn" disabled=${job.stop_requested} onClick=${() => stopJob(job.id)}>
        ${job.stop_requested ? "Stopping…" : "Stop"}
      </button>`}
    </div>
  `;
}
