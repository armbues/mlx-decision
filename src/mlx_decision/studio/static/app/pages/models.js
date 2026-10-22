// Models: what the Studio serves, what each needs and reads, and what is in memory.
// A row's status pill turns into Load / Unload under the pointer or on focus;
// they are also in the inspector, the memory card and a row's context menu.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { Download, Picture } from "../components/icons.js";
import { LoadButton } from "../components/load-button.js";
import { groupColour, groupLabel, markOf } from "../families.js";
import { MenuItem, MenuSeparator } from "../components/menu.js";
import { duration, gigabytes, parameters, plural, shellWord } from "../format.js";
import { chooseModel } from "../playground/session.js";
import { SHORTCUTS, onKey } from "../keys.js";
import { openConvert } from "../models/convert.js";
import { openDownload } from "../models/download.js";
import { FirstStart } from "../models/first-start.js";
import { Places, loadPlaces } from "../settings/places.js";
import { settingsChange } from "../stream.js";
import { href, route } from "../router.js";
import { gpuJob, jobs, progress, stopJob } from "../store/jobs.js";
import { loadModel, pending, unloadModel } from "../store/loading.js";
import { remembered } from "../store/prefs.js";
import { busy, loadedNames, loadModels, models, status } from "../store/status.js";
import { openMenu, toast } from "../store/ui.js";

function Mark({ family, big = false }) {
  const [text, colour] = markOf(family);
  return html`<span class=${big ? "ficon big" : "ficon"} style=${`background:${colour}`} aria-hidden="true">${text}</span>`;
}

const tokens = (count) => (count >= 1024 ? `${Math.round(count / 1024)}k` : `${count}`);

function reads(model) {
  const caps = model.capabilities;
  if (!caps) return "";
  const limit = caps.max_input_tokens ? tokens(caps.max_input_tokens) : "no limit";
  // Every row keeps the icon's slot, so the limits line up.
  return html`${limit}<span class="pic-slot">${caps.supports_images && html`<${Picture} />`}</span>`;
}

const PLACES = { cache: "HF cache", studio: "made here", folder: "your folder" };

function subline(model) {
  if (model.fitted) return `fitted on ${model.base}`;
  return [model.maker ?? model.family_title, model.precision, PLACES[model.place]].filter(Boolean).join(" · ");
}

// "used just now", "used 4 min ago": the clock ticks so the words stay true.
const now = signal(Date.now() / 1000);
setInterval(() => {
  now.value = Date.now() / 1000;
}, 20000);

function usedWords(seconds) {
  if (seconds == null) return "in memory";
  const ago = Math.max(0, now.value - seconds);
  if (ago < 60) return "used just now";
  if (ago < 3600) return `used ${Math.round(ago / 60)} min ago`;
  return `used ${Math.round(ago / 3600)} h ago`;
}

let refreshed = false; // the list was loaded since the page opened

// What the list shows: every model, the loaded ones, or the fitted ones
// (remembered per browser).
const filter = remembered("models.filter", "all");
const FILTERS = [
  ["all", "All"],
  ["loaded", "Loaded"],
  ["fitted", "Fitted"],
];

function Filters() {
  const list = models.value ?? [];
  const counts = {
    all: list.length,
    loaded: list.filter((model) => loadedNames.value.has(model.name)).length,
    fitted: list.filter((model) => model.fitted).length,
  };
  return html`
    <div class="seg" role="group" aria-label="Show">
      ${FILTERS.map(
        ([id, label]) => html`<button aria-pressed=${filter.value === id} data-filter=${id}
          onClick=${() => (filter.value = id)}>${label} <span class="count num">${counts[id]}</span></button>`,
      )}
    </div>
  `;
}

// Speed here (PG-9): the median of the answers measured on this Mac, else a dash.
function speedCell(model) {
  const speed = model.speed;
  if (!speed) return html`<span title="Not measured yet: every answer here is timed">–</span>`;
  const tokens = speed.tokens_min == null ? "" : `, ${tokensText(speed)}`;
  return html`<span title=${`Median of ${plural(speed.count, "answer")} here${tokens}`}>${duration(speed.median_ms)}</span>`;
}

const tokens_range = (speed) =>
  speed.tokens_min === speed.tokens_max ? `${speed.tokens_min}` : `${speed.tokens_min}–${speed.tokens_max}`;
// "1 input token", "3–40 input tokens".
const tokensText = (speed) => `${tokens_range(speed)} input token${speed.tokens_max === 1 ? "" : "s"}`;

// The selected model, and the one the pointer is over (for the memory preview).
const selected = computed(() => {
  const name = route.value.page === "models" ? route.value.item : null;
  return (models.value ?? []).find((model) => model.name === name) ?? null;
});
const hovered = signal(null);
const byName = computed(() => new Map((models.value ?? []).map((model) => [model.name, model])));

// Why Convert is off for ``model``, or null.
function convertWhyNot(model) {
  if (!model) return "No model";
  if (!model.convertible) return `${model.name} cannot be converted: ${model.convert_reason}`;
  const job = gpuJob.value;
  if (job) return `${job.title} is running; one job at a time can use the GPU`;
  return null;
}

function openInPlayground(name) {
  chooseModel(name);
  location.hash = href("playground");
}

// Each base with the fitted models made from it, Loaded first (most recently used).
const units = computed(() => {
  const list = models.value ?? [];
  const names = new Set(list.map((model) => model.name));
  const fits = new Map();
  for (const model of list)
    if (model.fitted && names.has(model.base)) fits.set(model.base, [...(fits.get(model.base) ?? []), model]);
  const used = status.value?.used ?? {};
  const loaded = loadedNames.value;
  const all = list
    .filter((model) => !(model.fitted && names.has(model.base)))
    .map((model) => {
      const members = [model, ...(fits.get(model.name) ?? [])];
      const inMemory = members.filter((member) => loaded.has(member.name));
      return { models: members, loaded: inMemory.length > 0, used: Math.max(0, ...inMemory.map((m) => used[m.name] ?? 0)) };
    });
  return {
    loaded: all.filter((unit) => unit.loaded).sort((a, b) => b.used - a.used),
    rest: all.filter((unit) => !unit.loaded),
  };
});

// The row's state as a pill; while the pointer is on it (or it has focus) it
// turns into the Load / Unload button in place. States without an
// action stay a pill and say why in the tooltip.
function StatePill({ model }) {
  const name = model.name;
  const doing = pending.value[name];
  const loaded = loadedNames.value.has(name);
  const sharing = (status.value?.groups ?? []).find((group) => group.root === name);
  let kind, text, why = null;
  if (doing) [kind, text] = ["busy", doing === "load" ? "Loading…" : "Unloading…"];
  else if (loaded) [kind, text] = ["on", usedWords(status.value?.used?.[name])];
  else if (sharing) {
    [kind, text] = ["on", `with ${sharing.members.join(", ")}`];
  } else if (model.memory?.over_budget) {
    [kind, text] = ["over", "Over budget"];
    why = `${name} needs more than the memory budget, so it cannot load`;
  } else if (model.error) [kind, text, why] = ["over", "Can't read", model.error];
  else [kind, text] = ["off", "Not loaded"];
  if (doing) why = text;
  else if (busy.value) why = `A job holds the GPU (${busy.value}): models load again when it is done`;
  const action = loaded ? "Unload" : "Load";
  const act = (event) => {
    event.stopPropagation();
    if (why) return;
    if (loaded) unloadModel(name);
    else loadModel(name);
  };
  return html`
    <button class=${`pill-act ${kind}${why ? "" : " can"}${loaded ? " unload" : " load"}`}
            aria-disabled=${why ? "true" : undefined} title=${why ?? `${action} ${name}`}
            aria-label=${`${name}: ${text}${why ? "" : `. ${action}`}`}
            onClick=${act} onDblClick=${(event) => event.stopPropagation()}>
      <span class="as-state">${kind !== "off" && html`<i></i>`}${text}</span>
      <span class="as-action" aria-hidden="true">${action}</span>
    </button>
  `;
}

function memoryCell(model) {
  const memory = model.memory;
  if (!memory) return "";
  // A fit that shares its base's weights adds only its own part to them.
  if (model.fitted && memory.added < memory.bytes) return `+${gigabytes(memory.added)}`;
  return gigabytes(memory.bytes);
}

function Row({ model, nested }) {
  const name = model.name;
  const current = selected.value?.name === name;
  const menu = (event) => {
    event.preventDefault();
    event.stopPropagation();
    contextMenu.value = { name, x: event.clientX, y: event.clientY };
    openMenu.value = "model-context";
  };
  const select = () => {
    location.hash = href("models", name);
  };
  // A list item, not a link: it holds the pill's button.
  return html`
    <div class=${`mrow${nested ? " fitted" : ""}`} data-name=${name} role="option" tabindex="0"
       aria-selected=${current ? "true" : "false"} onClick=${select} onContextMenu=${menu}
       onKeyDown=${(event) => event.key === "Enter" && event.target === event.currentTarget && select()}
       onDblClick=${() => openInPlayground(name)}
       onMouseEnter=${() => (hovered.value = name)} onMouseLeave=${() => (hovered.value = null)}>
      <span class="nm">
        <${Mark} family=${model.family} />
        <span class="nm-txt"><b class="mono">${name}</b><span>${subline(model)}</span></span>
      </span>
      <span class="cell r num c-size" title=${model.parameters ? `${parameters(model.parameters)} parameters` : null}>${model.disk_bytes != null ? gigabytes(model.disk_bytes) : parameters(model.parameters)}</span>
      <span class="cell r num"><b>${memoryCell(model)}</b></span>
      <span class="cell r c-reads">${reads(model)}</span>
      <span class="cell r num c-speed">${speedCell(model)}</span>
      <span class="st"><${StatePill} model=${model} /></span>
    </div>
  `;
}

// Downloads and conversions under way: a download in its own section, a
// conversion under the model it copies. They leave when the job ends; what
// they wrote comes with the next catalog message.
const working = computed(() =>
  jobs.value.filter((job) => job.state !== "ended" && (job.kind === "download" || job.kind === "convert")),
);
const downloads = computed(() => working.value.filter((job) => job.kind === "download"));
const conversionsOf = (name) => working.value.filter((job) => job.kind === "convert" && job.details?.model === name);

function JobPill({ job }) {
  const { fraction } = progress(job);
  const verb = job.kind === "download" ? "Downloading" : "Converting";
  let text = fraction == null ? verb : `${verb} ${Math.floor(fraction * 100)}%`;
  if (job.stop_requested) text = "Stopping…";
  else if (job.state === "waiting") text = progress(job).text;
  const stop = (event) => {
    event.stopPropagation();
    if (!job.stop_requested) stopJob(job.id);
  };
  return html`
    <button class=${`pill-act busy jp${job.stop_requested ? "" : " can"}`} title=${job.stop_requested ? text : `Stop ${job.title}`}
            aria-label=${`${job.title}: ${text}${job.stop_requested ? "" : ". Stop"}`} onClick=${stop}>
      <span class="as-state"><i></i>${text}</span>
      <span class="as-action" aria-hidden="true">Stop</span>
    </button>
  `;
}

function JobRow({ job, family = null, nested = false }) {
  const details = job.details ?? {};
  const download = job.kind === "download";
  const name = download ? details.repo?.split("/").pop() : details.name;
  const bits = details.bits ? `${details.bits}-bit` : null;
  const sub = download ? [details.repo, bits].filter(Boolean).join(" · ") : `${bits} copy of ${details.model}`;
  return html`
    <div class=${`mrow mjob${nested ? " fitted" : ""}`} data-job=${job.id} role="option" aria-selected="false">
      <span class="nm">
        ${download ? html`<span class="ficon dl" aria-hidden="true"><${Download} /></span>` : html`<${Mark} family=${family} />`}
        <span class="nm-txt"><b class="mono">${name}</b><span>${sub}</span></span>
      </span>
      <span class="cell r num c-size"></span>
      <span class="cell r num"></span>
      <span class="cell r c-reads"></span>
      <span class="cell r c-speed"></span>
      <span class="st"><${JobPill} job=${job} /></span>
    </div>
  `;
}

function Section({ title, detail, list }) {
  if (list.length === 0) return null;
  return html`
    <div class="msec">${title}${detail && html`<span>${detail}</span>`}</div>
    ${list.map((unit) => [
      ...unit.models.map((model, index) => html`<${Row} key=${model.name} model=${model} nested=${index > 0} />`),
      ...unit.models.flatMap((model) =>
        conversionsOf(model.name).map((job) => html`<${JobRow} key=${job.id} job=${job} family=${model.family} nested />`),
      ),
    ])}
  `;
}

// The sections the filter shows.
function Listed() {
  if (filter.value === "fitted") {
    const fits = (models.value ?? []).filter((model) => model.fitted);
    if (fits.length === 0)
      return html`<div class="mempty">No fitted models yet. Training fits one on labelled data.</div>`;
    return html`
      <div class="msec">Fitted models<span>fitted on top of a base model</span></div>
      ${fits.map((model) => html`<${Row} key=${model.name} model=${model} nested=${false} />`)}
    `;
  }
  if (filter.value === "loaded") {
    const loaded = loadedNames.value;
    const list = units.value.loaded.map((unit) => ({ ...unit, models: unit.models.filter((m) => loaded.has(m.name)) }));
    if (list.length === 0) return html`<div class="mempty">Nothing loaded. Models load on their first request, or with Load.</div>`;
    return html`<${Section} title="Loaded" detail=${gigabytes(status.value?.memory?.used ?? 0)} list=${list} />`;
  }
  return html`
    <${Section} title="Loaded" detail=${gigabytes(status.value?.memory?.used ?? 0)} list=${units.value.loaded} />
    <${Section} title="On this Mac" list=${units.value.rest} />
    <${Downloading} />
  `;
}

function Downloading() {
  const list = downloads.value;
  if (list.length === 0) return null;
  return html`
    <div class="msec">Downloading</div>
    ${list.map((job) => html`<${JobRow} key=${job.id} job=${job} />`)}
  `;
}

// What loading ``model`` would do: the bytes it adds and the groups that make room.
function preview(model) {
  const groups = status.value?.groups ?? [];
  const budget = status.value?.memory?.budget ?? 0;
  const memory = model.memory;
  if (!memory || loadedNames.value.has(model.name)) return null;
  if (memory.over_budget) return { model, over: true, needs: memory.bytes, leaving: [] };
  const shared = groups.some((group) => group.root === memory.group);
  const needs = shared ? memory.added : memory.bytes;
  let used = groups.reduce((sum, group) => sum + group.bytes, 0);
  const leaving = [];
  for (const group of groups) {
    if (used + needs <= budget) break;
    if (group.root === memory.group) continue;
    leaving.push(group);
    used -= group.bytes;
  }
  return { model, over: false, needs, leaving };
}


function MemoryCard() {
  const memory = status.value?.memory;
  if (!memory) return null;
  const groups = status.value?.groups ?? [];
  const budget = memory.budget || 1;
  const hover = hovered.value ? byName.value.get(hovered.value) : null;
  const plan = hover ? preview(hover) : null;
  const leaving = new Set((plan?.leaving ?? []).map((group) => group.root));
  // The bar holds the budget, or more while a preview needs it (the line marks the budget).
  const scale = Math.max(budget, memory.used + (plan && !plan.over ? plan.needs : 0));
  const share = (bytes) => `${Math.min(100, (bytes / scale) * 100).toFixed(2)}%`;
  let note;
  if (plan?.over)
    note = html`<b>${plan.model.name}</b> needs about <b>${gigabytes(plan.needs)}</b>, more than the budget: it cannot load. Change the budget in Settings (<code>memory_budget_gb</code>) or with <code>--memory-budget</code>.`;
  else if (plan)
    note = html`Loading <b>${plan.model.name}</b> (about ${gigabytes(plan.needs)})${
      plan.leaving.length
        ? html` unloads <span class="warn">${plan.leaving.map(groupLabel).join("; ")}</span> to stay within the budget.`
        : " fits beside what is loaded."
    }`;
  else if (groups.length === 0) note = "Nothing loaded. Models load on their first request, or with Load.";
  else note = html`${gigabytes(Math.max(0, memory.budget - memory.used))} free in the budget. Point at a model to see what loading it takes.`;
  return html`
    <section class="group memcard" aria-label="Memory">
      <div class="memcard-head">
        <h2>Memory</h2>
        <span class="figure num">${gigabytes(memory.used)}<small>in use of ${gigabytes(memory.budget)} budget</small></span>
        <a class="memcard-link" href=${href("settings")} title="memory_budget_gb in the config file, or --memory-budget">Change the budget</a>
        ${memory.working_set &&
        html`<span class="memcard-sub">This Mac's GPU working set is ${gigabytes(memory.working_set)}: the most a budget can use.</span>`}
      </div>
      <div class="membig">
        <div class="segs">
          ${groups.map((group) => {
            const colour = groupColour(group, groups);
            return html`<i key=${group.root} class=${leaving.has(group.root) ? "leaving" : ""}
                         style=${`width:${share(group.bytes)}; background:${colour}`} title=${`${groupLabel(group)} · ${gigabytes(group.bytes)}`}>
              <span>${groupLabel(group)}</span></i>`;
          })}
          ${plan && !plan.over && html`<i class="ghost" style=${`width:${share(plan.needs)}`}></i>`}
          ${plan?.over && html`<i class="ghost over" style="width:100%"></i>`}
        </div>
        ${scale > budget && html`<div class="budget" style=${`left:${share(budget)}`} title="The memory budget"></div>`}
      </div>
      ${groups.length > 0 &&
      html`<div class="mem-groups">
        ${groups.map((group) => {
          const colour = groupColour(group, groups);
          const name = group.members[group.members.length - 1];
          const doing = group.members.some((member) => pending.value[member]);
          return html`<span class="mem-group" key=${group.root}>
            <s style=${`background:${colour}`}></s><span class="mono">${groupLabel(group)}</span>
            <span class="num">${gigabytes(group.bytes)}</span>
            <button class="mem-x" disabled=${Boolean(doing || busy.value)} aria-label=${`Unload ${groupLabel(group)}`}
                    title=${busy.value ? `A job holds the GPU (${busy.value})` : `Unload ${groupLabel(group)}`}
                    onClick=${() => unloadModel(name)}>
              <svg viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" aria-hidden="true"><path d="M2.5 2.5l5 5M7.5 2.5l-5 5" /></svg>
            </button>
          </span>`;
        })}
      </div>`}
      <div class="memnote">${note}</div>
    </section>
  `;
}

// The row's context menu, where the pointer was.
const contextMenu = signal(null); // {name, x, y}

function ContextMenu() {
  const at = contextMenu.value;
  if (!at || openMenu.value !== "model-context") return null;
  const loaded = loadedNames.value.has(at.name);
  const blocked = Boolean(busy.value || pending.value[at.name]);
  const left = Math.min(at.x, innerWidth - 230);
  const top = Math.min(at.y, innerHeight - 170);
  const copy = () =>
    navigator.clipboard.writeText(at.name).then(
      () => toast(`Copied ${at.name}`),
      () => toast("Could not copy the name"),
    );
  return html`
    <div class="menu ctx-menu" role="menu" style=${`left:${left}px; top:${top}px`} onClick=${(e) => e.stopPropagation()}>
      <${MenuItem} onSelect=${() => openInPlayground(at.name)}>Open in Playground<//>
      <${MenuItem} disabled=${blocked} sub=${busy.value ? `${busy.value} has the GPU` : pending.value[at.name] ? "Busy with it already" : null}
        onSelect=${() => (loaded ? unloadModel(at.name) : loadModel(at.name))}>
        ${loaded ? "Unload" : "Load"}
      <//>
      <${MenuItem} disabled=${Boolean(convertWhyNot(byName.value.get(at.name)))} onSelect=${() => openConvert(at.name)}>Convert…<//>
      <${MenuSeparator} />
      <${MenuItem} onSelect=${copy}>Copy Name<//>
    </div>
  `;
}

const status_ = computed(() => {
  const list = models.value;
  if (!list) return null;
  if (list.length === 0)
    return { text: html`No models yet · Download a Model… (${SHORTCUTS.download})` };
  return {
    text: html`<b>${plural(list.length, "model")}</b> · ${loadedNames.value.size} loaded`,
  };
});

const action = computed(() => ({
  label: "Download a Model…",
  title: `Download a Model… (${SHORTCUTS.download})`,
  icon: false,
  run: openDownload,
}));

// ⌘D opens Download a Model from any page.
onKey("download", () => openDownload());

function Centre() {
  // Where models live: again when the list changes (a download, a rescan) or the config does.
  useEffect(() => {
    loadPlaces();
  }, [models.value, settingsChange.value]);
  // Speed: the list again when the page opens and, while it is open, after
  // models answered (at most every 2 s).
  const used = JSON.stringify(status.value?.used ?? {});
  useEffect(() => {
    const timer = setTimeout(loadModels, refreshed ? 2000 : 0);
    refreshed = true;
    return () => clearTimeout(timer);
  }, [used]);
  useEffect(
    () => () => {
      refreshed = false;
    },
    [],
  );
  const list = models.value;
  if (!list) return html`<div class="centre-inner"></div>`;
  const fitted = list.filter((model) => model.fitted).length;
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>Models</h1>
          <p>${plural(list.length - fitted, "model")}${fitted ? ` and ${plural(fitted, "fitted model")}` : ""}, ${loadedNames.value.size} loaded</p>
        </div>
        ${list.length > 0 && html`<div class="actions"><${Filters} /></div>`}
      </div>
      ${list.length === 0 && downloads.value.length === 0
        ? html`<${FirstStart} />`
        : html`
            <${MemoryCard} />
            <section class="group mlist" aria-label="Models">
              <div class="mhead"><span>Model</span><span class="r c-size" title="Its files on disk">Size</span><span class="r" title="Estimates: the weights plus 10%, as the memory check counts them">Memory, est.</span><span class="r c-reads">Reads</span><span class="r c-speed" title="The median time of an answer here, measured">Speed</span><span class="r">Status</span></div>
              <div role="listbox" aria-label="Models"><${Listed} /></div>
            </section>
            <${Places} />
            <${ContextMenu} />
          `}
    </div>
  `;
}

const yesNo = (value) => (value ? "Yes" : "No");

function Kv({ label, children }) {
  return html`<div class="kv"><dt>${label}</dt><dd>${children}</dd></div>`;
}

function Details() {
  const model = selected.value;
  if (!model)
    return html`<div class="empty"><b>No model selected</b><span>Pick one to see what it needs and reads.</span></div>`;
  const caps = model.capabilities;
  const memory = model.memory;
  const budget = status.value?.memory?.budget;
  const loaded = loadedNames.value.has(model.name);
  const fits = (models.value ?? []).filter((other) => other.fitted && other.base === model.name);
  const sub = [model.maker ?? model.family_title, model.family_title, parameters(model.parameters), model.precision]
    .filter(Boolean)
    .filter((part, index, parts) => parts.indexOf(part) === index)
    .join(" · ");
  return html`
    <div class="idet">
      <div class="idet-head">
        <${Mark} family=${model.family} big=${true} />
        <div><h2 class="mono">${model.name}</h2><p>${sub}</p></div>
      </div>
      <div class="idet-actions">
        <button class="btn primary" onClick=${() => openInPlayground(model.name)}>Open in Playground</button>
        <${LoadButton} name=${model.name} />
        <button class="btn" disabled=${Boolean(convertWhyNot(model))} title=${convertWhyNot(model) ?? `Make an 8-bit or 4-bit copy of ${model.name}`}
                onClick=${() => openConvert(model.name)}>Convert…</button>
      </div>
      ${model.error && html`<p class="warn-text">${model.error}</p>`}

      ${memory &&
      html`<section class="igroup"><h4>Memory</h4><dl>
        <${Kv} label="Needs">
          about <b class="num">${gigabytes(memory.bytes)}</b>${budget ? ` · ${Math.round((memory.bytes / budget) * 100)}% of the ${gigabytes(budget)} budget` : ""}
          <div class="minibar"><i style=${`width:${budget ? Math.min(100, (memory.bytes / budget) * 100) : 0}%`} class=${memory.over_budget ? "over" : ""}></i></div>
          ${model.fitted && memory.added < memory.bytes &&
          html`<small>With its base; ${gigabytes(memory.added)} more when ${memory.group} is loaded.</small>`}
          ${memory.over_budget && html`<small class="warn-text">More than the budget: it cannot load.</small>`}
          <small>An estimate: the weights plus 10%, as the memory check counts them.</small>
        <//>
        <${Kv} label="Now">${loaded ? `In memory · ${usedWords(status.value?.used?.[model.name])}` : "Not loaded"}<//>
      </dl></section>`}

      <section class="igroup speed-group"><h4>Speed here</h4><dl>
        ${model.speed
          ? html`<${Kv} label="Median">
              <b class="num">${duration(model.speed.median_ms)}</b> an answer
              <small>${plural(model.speed.count, "answer")}${model.speed.tokens_min == null ? "" : `, ${tokensText(model.speed)}`}${model.speed.chip ? `, on ${model.speed.chip}` : ""}</small>
            <//>`
          : html`<${Kv} label="Median">Not measured yet<small>Run it in the Playground or call the API: every answer here is timed.</small><//>`}
      </dl></section>

      ${caps &&
      html`<section class="igroup"><h4>Reads</h4><dl>
        <${Kv} label="Input">
          ${caps.max_input_tokens ? `${caps.max_input_tokens.toLocaleString("en")} tokens` : "Any length"}
          ${caps.max_input_tokens &&
          html`<small>${caps.truncates_input ? "Longer states are cut to fit, and the answer says so." : "Longer states are refused."}</small>`}
        <//>
        <${Kv} label="Images">${yesNo(caps.supports_images)}<//>
        <${Kv} label="Questions">${caps.question_types.join(", ")}<//>
        <${Kv} label="Options">${caps.max_choice_options ? `Up to ${caps.max_choice_options} per choice` : "No limit"}${
          caps.max_score_levels ? html`<small>Up to ${caps.max_score_levels} score levels</small>` : ""
        }<//>
        <${Kv} label="Instructions">${caps.requires_instructions ? "Required" : "Optional"}<//>
      </dl></section>`}

      <section class="igroup"><h4>Files</h4><dl>
        <${Kv} label="Location"><code>${model.path}</code><small>${{ cache: "In the Hugging Face cache", studio: "Made by the Studio, in its models folder", folder: "A folder you named" }[model.place] ?? ""}</small><//>
        ${model.disk_bytes != null && html`<${Kv} label="On disk">${gigabytes(model.disk_bytes)}<//>`}
        ${model.repo && html`<${Kv} label="From"><code>${model.repo}</code><small>Revision ${model.revision.slice(0, 7)}</small><//>`}
        ${model.converted_from && html`<${Kv} label="Converted">from <code>${model.converted_from}</code><//>`}
        ${model.licence_note && html`<${Kv} label="Licence"><span class="warn-text">${model.licence_note}</span><//>`}
      </dl></section>

      ${(model.fitted || fits.length > 0) &&
      html`<section class="igroup"><h4>${model.fitted ? "Fitted" : "Fitted from this model"}</h4><dl>
        ${model.fitted && html`<${Kv} label="Base"><a href=${href("models", model.base)}>${model.base}</a><//>`}
        ${fits.map((fit) => html`<${Kv} label="Fit"><a href=${href("models", fit.name)}>${fit.name}</a><//>`)}
      </dl></section>`}

      <section class="igroup"><h4>From code</h4><dl>
        <${Kv} label="API"><code>"model": "${model.name}"</code><//>
        <${Kv} label="CLI"><code>${(model.cli?.short ?? ["-m", model.name]).map(shellWord).join(" ")}</code><//>
      </dl></section>
    </div>
  `;
}

export default {
  id: "models",
  title: "Models",
  action,
  status: status_,
  Centre,
  tabs: [{ id: "details", label: "Model", Body: Details }],
};
