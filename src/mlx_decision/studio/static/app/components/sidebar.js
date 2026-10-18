// The sidebar: brand, pages, the page's own list, jobs, memory and the API.
import { html } from "htm/preact";
import { gigabytes, hostPort } from "../format.js";
import { href } from "../router.js";
import { jobs } from "../store/jobs.js";
import { busy, connection, loadedNames, models, status } from "../store/status.js";
import * as icons from "./icons.js";

const PAGE_LINKS = [
  { id: "playground", label: "Playground", Icon: icons.Playground },
  { id: "models", label: "Models", Icon: icons.Models },
  { id: "training", label: "Training", Icon: icons.Training },
];

function Pages({ current }) {
  const count = { models: models.value?.length };
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

function Jobs() {
  return html`
    <div class="sb-sec">
      <h3>Jobs</h3>
      ${jobs.value.length === 0 && html`<p class="sb-empty">Nothing running</p>`}
    </div>
  `;
}

function Memory() {
  const memory = status.value?.memory;
  if (!memory) return null;
  const loaded = [...loadedNames.value];
  const share = memory.budget ? Math.min(memory.used / memory.budget, 1) : 0;
  return html`
    <div class="mem" title="What the loaded models count against the memory budget (the memory check's estimates; --memory-budget sets it)">
      <div class="mem-head">
        <span>Memory budget</span>
        <span><b class="num">${gigabytes(memory.used)}</b> of ${gigabytes(memory.budget)}</span>
      </div>
      <div class="membar"><i style=${`width:${(share * 100).toFixed(1)}%; background: var(--accent)`}></i></div>
      <div class="mem-legend">
        ${loaded.length === 0
          ? html`<div>No model loaded</div>`
          : loaded.map((name) => html`<div><s style="background: var(--accent)"></s><span class="mono-s">${name}</span></div>`)}
      </div>
    </div>
  `;
}

function Serving() {
  const current = status.value;
  if (!current) return null;
  const state = connection.value === "lost" ? "lost" : busy.value ? "busy" : "ready";
  const title = {
    ready: "The Jev-compatible API answers here, from the same models",
    busy: `A job holds the GPU: the API answers 503 until it is done`,
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
