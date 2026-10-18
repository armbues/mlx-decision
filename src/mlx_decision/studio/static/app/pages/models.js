// Models: what the Studio serves, what each needs and reads, and what is in memory.
// A model's state is a dot and text, never shaped like a button; Load / Unload
// sit in the inspector, the memory card and a row's context menu (D156).
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { Picture } from "../components/icons.js";
import { LoadButton } from "../components/load-button.js";
import { groupColour, groupLabel, markOf } from "../families.js";
import { MenuItem, MenuSeparator } from "../components/menu.js";
import { gigabytes, plural } from "../format.js";
import { chooseModel } from "../playground/session.js";
import { href, route } from "../router.js";
import { loadModel, pending, unloadModel } from "../store/loading.js";
import { busy, loadedNames, models, status } from "../store/status.js";
import { openMenu, toast } from "../store/ui.js";

function Mark({ family, big = false }) {
  const [text, colour] = markOf(family);
  return html`<span class=${big ? "ficon big" : "ficon"} style=${`background:${colour}`} aria-hidden="true">${text}</span>`;
}

// "421M", "0.6B", "3.1B", "27B".
function parameters(count) {
  if (!count) return "";
  if (count < 1e9) return `${Math.round(count / 1e6)}M`;
  const billions = count / 1e9;
  return `${billions < 10 ? billions.toFixed(1).replace(/\.0$/, "") : Math.round(billions)}B`;
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

// The selected model, and the one the pointer is over (for the memory preview).
const selected = computed(() => {
  const name = route.value.page === "models" ? route.value.item : null;
  return (models.value ?? []).find((model) => model.name === name) ?? null;
});
const hovered = signal(null);
const byName = computed(() => new Map((models.value ?? []).map((model) => [model.name, model])));

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
// turns into the Load / Unload button in place (D157). States without an
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
  // A list item, not a link: it holds the pill's button (D157).
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
      <span class="cell r num c-size">${parameters(model.parameters)}</span>
      <span class="cell r num"><b>${memoryCell(model)}</b></span>
      <span class="cell r c-reads">${reads(model)}</span>
      <span class="st"><${StatePill} model=${model} /></span>
    </div>
  `;
}

function Section({ title, detail, list }) {
  if (list.length === 0) return null;
  return html`
    <div class="msec">${title}${detail && html`<span>${detail}</span>`}</div>
    ${list.map((unit) =>
      unit.models.map((model, index) => html`<${Row} key=${model.name} model=${model} nested=${index > 0} />`),
    )}
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
    note = html`<b>${plan.model.name}</b> needs <b>${gigabytes(plan.needs)}</b>, more than the budget: it cannot load. <code>--memory-budget</code> or <code>memory_budget_gb</code> sets the budget.`;
  else if (plan)
    note = html`Loading <b>${plan.model.name}</b> (${gigabytes(plan.needs)})${
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
  const top = Math.min(at.y, innerHeight - 140);
  const copy = () =>
    navigator.clipboard.writeText(at.name).then(
      () => toast(`Copied ${at.name}`),
      () => toast("Could not copy the name"),
    );
  return html`
    <div class="menu ctx-menu" role="menu" style=${`left:${left}px; top:${top}px`} onClick=${(e) => e.stopPropagation()}>
      <${MenuItem} onSelect=${() => openInPlayground(at.name)}>Open in Playground<//>
      <${MenuItem} disabled=${blocked} onSelect=${() => (loaded ? unloadModel(at.name) : loadModel(at.name))}>
        ${loaded ? "Unload" : "Load"}
      <//>
      <${MenuSeparator} />
      <${MenuItem} onSelect=${copy}>Copy Name<//>
    </div>
  `;
}

const status_ = computed(() => {
  const list = models.value;
  if (!list) return null;
  if (list.length === 0)
    return { icon: "info", text: html`No models yet · <code>mlx-decision download</code>` };
  return {
    icon: "ok",
    text: html`<b>${plural(list.length, "model")}</b> · ${loadedNames.value.size} loaded`,
  };
});

const action = computed(() => ({ label: "Download Model…", icon: false, disabled: true }));

function Centre() {
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
      </div>
      ${list.length === 0
        ? html`<div class="empty">
            <b>No models yet</b>
            <span>Download one with <code>mlx-decision download</code>; it appears here.</span>
          </div>`
        : html`
            <${MemoryCard} />
            <section class="group mlist" aria-label="Models">
              <div class="mhead"><span>Model</span><span class="r c-size">Size</span><span class="r">Memory</span><span class="r c-reads">Reads</span><span class="r">Status</span></div>
              <div role="listbox" aria-label="Models">
                <${Section} title="Loaded" detail=${gigabytes(status.value?.memory?.used ?? 0)} list=${units.value.loaded} />
                <${Section} title="On this Mac" list=${units.value.rest} />
              </div>
            </section>
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
      </div>
      ${model.error && html`<p class="warn-text">${model.error}</p>`}

      ${memory &&
      html`<section class="igroup"><h4>Memory</h4><dl>
        <${Kv} label="Needs">
          <b class="num">${gigabytes(memory.bytes)}</b>${budget ? ` · ${Math.round((memory.bytes / budget) * 100)}% of the ${gigabytes(budget)} budget` : ""}
          <div class="minibar"><i style=${`width:${budget ? Math.min(100, (memory.bytes / budget) * 100) : 0}%`} class=${memory.over_budget ? "over" : ""}></i></div>
          ${model.fitted && memory.added < memory.bytes &&
          html`<small>With its base; ${gigabytes(memory.added)} more when ${memory.group} is loaded.</small>`}
          ${memory.over_budget && html`<small class="warn-text">More than the budget: it cannot load.</small>`}
        <//>
        <${Kv} label="Now">${loaded ? `In memory · ${usedWords(status.value?.used?.[model.name])}` : "Not loaded"}<//>
      </dl></section>`}

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
        <${Kv} label="CLI"><code>-m ${model.name}</code><//>
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
