// Download Model (⌘D): the families `download` offers, then the family's
// models, then how to store the one picked (full size, 8-bit, 4-bit) with its
// size and the fit check; or a repo id typed in. Each pick is checked on the
// Hub first, as `download` checks it; the download itself is a job.
import { computed, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { StatusIcon } from "../components/icons.js";
import { markOf } from "../families.js";
import { size } from "../format.js";
import { jobs } from "../store/jobs.js";
import { toast } from "../store/ui.js";

const offered = signal(null); // {families, destination}
const familyName = signal(null); // the family shown, or null for a typed repo
const picked = signal(null); // the repo id to download
const typed = signal("");
const checks = signal({}); // repo id -> {state: "checking" | "done" | "failed", result, error}
const bitsOf = signal({}); // repo id -> the bits picked (0: full size)
const starting = signal(false);

const REPO = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

const family = computed(() => offered.value?.families.find((item) => item.name === familyName.value) ?? null);
const current = computed(() => checks.value[picked.value] ?? null);
const downloading = computed(
  () => new Set(jobs.value.filter((job) => job.kind === "download" && job.state !== "ended").map((job) => job.details?.repo)),
);

async function check(repo) {
  if (!repo || checks.value[repo]?.state === "checking" || checks.value[repo]?.state === "done") return;
  checks.value = { ...checks.value, [repo]: { state: "checking" } };
  try {
    const result = await api("/downloads/check", { method: "POST", body: { repo } });
    checks.value = { ...checks.value, [repo]: { state: "done", result } };
    if (bitsOf.value[repo] == null && result.default_bits != null) bitsOf.value = { ...bitsOf.value, [repo]: result.default_bits };
  } catch (error) {
    checks.value = { ...checks.value, [repo]: { state: "failed", error: error.message } };
  }
}

function pick(repo) {
  picked.value = repo;
  check(repo);
}

function pickFamily(name) {
  familyName.value = name;
  const models = offered.value.families.find((item) => item.name === name).models;
  pick((models.find((model) => !model.on_this_mac) ?? models[0]).repo);
}

function checkTyped() {
  const repo = typed.value.trim();
  if (!REPO.test(repo)) {
    toast("Type a repo id as org/name, e.g. vllm-sr/d3-edge");
    return;
  }
  familyName.value = null;
  // A failed check is asked again (the network may be back).
  if (checks.value[repo]?.state === "failed") {
    const { [repo]: _, ...rest } = checks.value;
    checks.value = rest;
  }
  pick(repo);
}

// What the Download button says and whether it can start one.
const verdict = computed(() => {
  const repo = picked.value;
  const state = current.value;
  if (!repo || !state || state.state === "checking") return { label: "Download", ready: false };
  if (state.state === "failed" || !state.result.family) return { label: "Download", ready: false };
  if (downloading.value.has(repo)) return { label: "Downloading", ready: false };
  if (state.result.on_this_mac) return { label: "On this Mac", ready: false };
  return { label: "Download", ready: bitsOf.value[repo] != null };
});

async function start() {
  if (!verdict.value.ready || starting.value) return;
  const repo = picked.value;
  const bits = bitsOf.value[repo] || null;
  starting.value = true;
  try {
    await api("/jobs", { method: "POST", body: { kind: "download", repo, bits } });
    closeSheet();
    toast(`Downloading ${repo}. It shows up in Models when it is done.`);
  } catch (error) {
    toast(`Could not start the download: ${error.message}`);
  } finally {
    starting.value = false;
  }
}

// ---------- the panes ----------

// "843 MB" or "16 GB" in megabytes, to order sizes.
const megabytes = (size) => parseFloat(size) * (/GB/.test(size) ? 1000 : 1);

// A family's line: its one model and size, or how many and their range.
function sizeRange(models) {
  if (models.length === 1) return `${models[0].repo.split("/")[1]} · ${models[0].size}`;
  const sizes = models.map((model) => model.size).sort((a, b) => megabytes(a) - megabytes(b));
  const [low, high] = [sizes[0], sizes[sizes.length - 1]];
  if (low === high) return `${models.length} models · ${low}`;
  // "19-55 GB", or "1.2 GB-52 GB" across units, as the CLI's menu has it.
  const unit = (size) => size.split(" ")[1];
  const range = unit(low) === unit(high) ? `${low.split(" ")[0]}-${high}` : `${low}-${high}`;
  return `${models.length} models · ${range}`;
}

function Families() {
  return html`
    <div class="fam-list" role="listbox" aria-label="Families">
      ${offered.value.families.map((item) => {
        const [text, colour] = markOf(item.name);
        return html`
          <button class="fam" role="option" aria-selected=${item.name === familyName.value ? "true" : "false"}
                  data-family=${item.name} onClick=${() => pickFamily(item.name)}>
            <span class="ficon" style=${`background:${colour}`} aria-hidden="true">${text}</span>
            <span><b>${item.title}</b><span title=${`${item.maker} · ${sizeRange(item.models)}`}>${item.maker} · ${sizeRange(item.models)}</span></span>
          </button>
        `;
      })}
    </div>
  `;
}

function Models() {
  const shown = family.value;
  if (!shown) return null;
  return html`
    <p>${shown.title}: ${shown.summary}. By ${shown.maker}.</p>
    <div class="cat" role="radiogroup" aria-label=${`${shown.title} models`}>
      ${shown.models.map(
        (model) => html`
          <button class="crow" role="radio" aria-checked=${model.repo === picked.value ? "true" : "false"}
                  data-repo=${model.repo} onClick=${() => pick(model.repo)}>
            <span class="radio"></span>
            <span><code>${model.repo}</code><small>${model.description}</small></span>
            <span class="meta num">
              ${model.on_this_mac
                ? html`<span class="have">On this Mac</span><br />`
                : downloading.value.has(model.repo) && html`<span class="getting">Downloading</span><br />`}
              ${model.size}
            </span>
          </button>
        `,
      )}
    </div>
  `;
}

const FITS = {
  ok: ["ok", "Fits the memory budget"],
  over_budget: ["mid", "Over the memory budget"],
  too_large: ["no", "Too large for this Mac"],
};

const optionTitle = (bits) => (bits ? `${bits}-bit` : "Original");
const optionLine = (bits) => (bits === 8 ? "quantized after the download" : bits === 4 ? "smallest, a little less exact" : "as published");

function Options({ result }) {
  const repo = result.repo;
  const chosen = bitsOf.value[repo];
  return html`
    <div class="opts">
      <h4>Download as</h4>
      <div class="qopts" role="radiogroup" aria-label="Precision">
        ${result.options.map((option) => {
          const [kind, words] = FITS[option.fit];
          const disk = `${option.estimate ? "about " : ""}${size(option.bytes)} on disk`;
          return html`
            <button class="qopt" role="radio" aria-checked=${option.bits === chosen ? "true" : "false"} data-bits=${option.bits}
                    onClick=${() => (bitsOf.value = { ...bitsOf.value, [repo]: option.bits })}>
              <b>${optionTitle(option.bits)}</b>
              <span>${optionLine(option.bits)}</span>
              <span class="num">${disk}</span>
              <span class="num">about ${size(option.memory_bytes)} to run</span>
              <span class=${`fit ${kind}`} title=${option.fit === "over_budget" ? "It fits this Mac's GPU, but not the Studio's memory budget (--memory-budget), so it cannot load here" : undefined}>${words}</span>
            </button>
          `;
        })}
      </div>
      ${result.options.length === 1 &&
      html`<span class="note">${result.family_title} models are not stored quantized: full size only.</span>`}
    </div>
  `;
}

function Notice({ kind = "warn", children }) {
  return html`<div class=${`notice ${kind}`}><${StatusIcon} kind=${kind} /><span>${children}</span></div>`;
}

function Checked() {
  const repo = picked.value;
  const state = current.value;
  if (!repo || !state) return null;
  const heading = familyName.value ? null : html`<p><code>${repo}</code></p>`;
  if (state.state === "checking")
    return html`${heading}<div class="checking"><${StatusIcon} kind="spin" /> Checking ${repo} on the Hugging Face Hub…</div>`;
  if (state.state === "failed") return html`${heading}<${Notice}>${state.error}<//>`;
  const result = state.result;
  if (!result.family)
    return html`${heading}<${Notice}>${repo}: ${result.reason}. The Studio cannot load it, so it is not offered.<//>`;
  return html`
    ${heading ?? null}
    ${!familyName.value && html`<p>${result.family_title} model · ${size(result.size_bytes)} to download in full</p>`}
    <${Options} result=${result} />
    ${result.licence && html`<${Notice} kind="info">${result.licence}<//>`}
  `;
}

function Body() {
  if (!offered.value) return null;
  return html`
    <div class="dl-panes">
      <${Families} />
      <div class="fam-pane">
        <${Models} />
        <${Checked} />
      </div>
    </div>
  `;
}

function Foot() {
  const onKeyDown = (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    event.stopPropagation(); // Enter here checks the repo, it does not download
    checkTyped();
  };
  return html`
    <div class="repo-field">
      <label for="repo-id">Other repo id</label>
      <input id="repo-id" class="sheet-input mono" placeholder="org/name" spellcheck="false" autocomplete="off"
             value=${typed.value} onInput=${(event) => (typed.value = event.target.value)} onKeyDown=${onKeyDown} />
      <button class="btn" disabled=${!typed.value.trim()} onClick=${checkTyped}>Check</button>
    </div>
  `;
}

function purpose(destination) {
  const where =
    destination.kind === "folder"
      ? `Into ${destination.path} (models_dir in the config file)`
      : `Into the Hugging Face cache (${destination.path}), where other tools can use it too`;
  return `${where}. It shows up in Models and the Playground when it is done; Stop is in the Jobs list.`;
}

// ``preselect``: {family, repo} to open on (the first start's picks).
export async function openDownload(preselect = null) {
  try {
    offered.value = await api("/downloads");
  } catch (error) {
    toast(`Could not list the models to download: ${error.message}`);
    return;
  }
  typed.value = "";
  checks.value = {}; // what is on this Mac may have changed since
  if (preselect) {
    familyName.value = preselect.family;
    pick(preselect.repo);
  } else if (!family.value) pickFamily(offered.value.families[0].name);
  else pickFamily(familyName.value); // the list may have changed (a model is here now)
  openSheet({
    title: "Download a Model",
    purpose: purpose(offered.value.destination),
    Body,
    Foot,
    wide: true,
    actions: () => [{ label: verdict.value.label, primary: true, disabled: !verdict.value.ready || starting.value, onSelect: start }],
  });
}
