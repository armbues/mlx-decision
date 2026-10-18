// While a job holds the GPU, every page says so.
import { html } from "htm/preact";
import { jobs } from "../store/jobs.js";
import { busy } from "../store/status.js";
import { StatusIcon } from "./icons.js";

export function BusyBanner() {
  const holder = busy.value;
  if (!holder) return null;
  const job = jobs.value.find((item) => item.id === holder);
  return html`
    <div class="banner" role="status">
      <${StatusIcon} kind="warn" />
      <span>
        <b>${job?.title ?? holder}</b> has the GPU. Runs here wait until it is done, and the
        API answers 503 busy meanwhile.
      </span>
      ${job?.stop && html`<button class="btn" onClick=${job.stop}>Stop</button>`}
    </div>
  `;
}
