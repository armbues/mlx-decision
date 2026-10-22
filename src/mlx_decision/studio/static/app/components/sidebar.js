// The sidebar: brand, pages, the page's own list, jobs, memory and the API.
import { html } from "htm/preact";
import { gigabytes, hostPort } from "../format.js";
import { href } from "../router.js";
import { batchProgressText, runningBatch, stopBatch } from "../store/batches.js";
import { dismissEnded, dismissJob, listed, progress, stopJob } from "../store/jobs.js";
import { groupColour, groupLabel } from "../families.js";
import { dataFiles } from "../store/data.js";
import { busy, models, offline, status } from "../store/status.js";
import * as icons from "./icons.js";
import { UnloadMark } from "./load-button.js";

const PAGE_LINKS = [
  { id: "playground", label: "Playground", Icon: icons.Playground },
  { id: "models", label: "Models", Icon: icons.Models },
  { id: "datasets", label: "Datasets", Icon: icons.Datasets },
  { id: "batches", label: "Batches", Icon: icons.Batches },
  { id: "training", label: "Training", Icon: icons.Training },
  { id: "settings", label: "Settings", Icon: icons.Settings },
];

function Pages({ current }) {
  const count = { models: models.value?.length, datasets: dataFiles.value?.length };
  return html`
    <div class="sb-sec">
      <ul class="sb-list">
        ${PAGE_LINKS.map(
          ({ id, label, Icon }) => html`
            <li>
              <a class="sb-item" href=${href(id)} aria-current=${id === current ? "page" : undefined}>
                <${Icon} /><span class="label">${label}</span>
                ${count[id] != null && html`<span class="count num">${count[id]}</span>`}
              </a>
            </li>
          `,
        )}
      </ul>
    </div>
  `;
}

const JOB_ICONS = { download: icons.Download, convert: icons.Models, data: icons.Datasets, fit: icons.Training, eval: icons.Training };

function Job({ job }) {
  const Icon = JOB_ICONS[job.kind] ?? icons.Training;
  const { fraction, text } = progress(job);
  const ended = job.state === "ended";
  const action = ended ? `Clear ${job.title}` : `Stop ${job.title}`;
  const failed = ended && job.outcome !== "done" && job.outcome !== "cancelled";
  return html`
    <li class=${`job ${ended ? `ended ${job.outcome}` : job.state}`} data-job=${job.id}>
      <div class="job-top">
        <${Icon} /><span class="job-title" title=${job.title}>${job.title}</span>
        <button aria-label=${action} title=${ended ? "Clear" : "Stop"}
                disabled=${!ended && job.stop_requested}
                onClick=${() => (ended ? dismissJob(job.id) : stopJob(job.id))}>
          ${ended ? html`<${icons.Close} />` : html`<${icons.Stop} />`}
        </button>
      </div>
      ${!ended && html`<div class="bar"><i style=${`width:${((fraction ?? 0) * 100).toFixed(1)}%`}></i></div>`}
      <small class="num" title=${failed && job.message ? job.message : undefined}>${text}</small>
    </li>
  `;
}

// The running batch, listed with the jobs: its progress and Stop.
function BatchJob({ batch }) {
  const fraction = batch.total ? batch.answered / batch.total : 0;
  return html`
    <li class="job running batch-job" data-batch=${batch.name}>
      <div class="job-top">
        <${icons.Batches} /><a class="job-title" href=${href("batches", batch.name)} title=${`Batch ${batch.name}`}>Batch ${batch.name}</a>
        <button aria-label=${`Stop the batch ${batch.name}`} title="Stop" disabled=${batch.stop_requested} onClick=${() => stopBatch(batch.name)}><${icons.Stop} /></button>
      </div>
      <div class="bar"><i style=${`width:${(fraction * 100).toFixed(1)}%`}></i></div>
      <small class="num">${batch.stop_requested ? "Stopping…" : batchProgressText(batch)}</small>
    </li>
  `;
}

function Jobs() {
  const shown = listed.value;
  const batch = runningBatch.value;
  const ended = shown.filter((job) => job.state === "ended").length;
  return html`
    <div class="sb-sec">
      <h3>
        Jobs
        ${ended > 0 &&
        html`<button class="jobs-clear" aria-label="Clear ended jobs" title=${ended === 1 ? "Clear the ended job" : `Clear the ${ended} ended jobs`} onClick=${dismissEnded}>
          <${icons.ClearAll} />
        </button>`}
      </h3>
      ${shown.length === 0 && !batch
        ? html`<p class="sb-empty">Nothing running</p>`
        : html`<ul class="jobs">${batch && html`<${BatchJob} batch=${batch} />`}${shown.map((job) => html`<${Job} key=${job.id} job=${job} />`)}</ul>`}
    </div>
  `;
}

function Memory() {
  const memory = status.value?.memory;
  if (!memory) return null;
  const groups = status.value?.groups ?? [];
  const budget = memory.budget || 1;
  const width = (bytes) => `${Math.min(100, (bytes / budget) * 100).toFixed(1)}%`;
  return html`
    <div class="mem" title="What the loaded models count against the memory budget (the memory check's estimates; --memory-budget sets it)">
      <div class="mem-head">
        <span>Memory budget</span>
        <span><b class="num">${gigabytes(memory.used)}</b> of ${gigabytes(memory.budget)}</span>
      </div>
      <div class="membar">
        ${groups.map((group) => html`<i key=${group.root} style=${`width:${width(group.bytes)}; background:${groupColour(group, groups)}`}></i>`)}
      </div>
      <div class="mem-legend">
        ${groups.length === 0
          ? html`<div>No model loaded</div>`
          : groups.map(
              (group) => html`<div key=${group.root}>
                <s style=${`background:${groupColour(group, groups)}`}></s><span class="mono-s">${groupLabel(group)}</span>
                <${UnloadMark} name=${group.members[group.members.length - 1]} label=${groupLabel(group)} />
              </div>`,
            )}
      </div>
    </div>
  `;
}

function Serving() {
  const current = status.value;
  if (!current) return null;
  const state = offline.value ? "lost" : busy.value ? "busy" : "ready";
  const title = {
    ready: "The Jev-compatible API answers here, from the same models",
    busy: `${busy.value} has the GPU: the API answers 503 until it is done`,
    lost: "The page lost its connection to the Studio",
  }[state];
  return html`
    <div class=${`serving ${state}`} title=${title}>
      <span class="dot"></span>
      ${state === "busy" ? "API busy" : state === "lost" ? "Not connected" : "API at"}
      <code>${hostPort(current.address)}</code>
      <span class="ver">${current.version}</span>
    </div>
  `;
}

export function Sidebar({ page, current }) {
  const List = page.SidebarList;
  return html`
    <nav class="sidebar" aria-label="Studio">
      <div class="brand">
        <${icons.Brand} />
        <div class="wordmark"><b>mlx-decision</b><span>Studio</span></div>
      </div>
      <div class="sb-scroll">
        <${Pages} current=${current} />
        ${List && html`<${List} />`}
        <${Jobs} />
      </div>
      <div class="sb-foot">
        <${Memory} />
        <${Serving} />
      </div>
    </nav>
  `;
}
