// The config file: its path and the seven keys with the values the Studio uses,
// where each comes from, and Change… (a sheet per key). A change is written at
// once, one line in the file; the page says whether it applies now or at the
// next start (CFG-4). Read-only for a browser on another machine.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { closeSheet, openSheet } from "../components/sheet.js";
import { models } from "../store/status.js";
import { toast } from "../store/ui.js";
import { listed, loadPlaces, reveal } from "./places.js";

// What each key is called (a row's label, a sheet's title), what it does,
// the flag that overrides it (D204: the key itself shows under its name).
export const KEYS = {
  models_dir: {
    label: "Where models go",
    title: "Where Models Go",
    help: "Where downloads go, and a folder of models the Studio lists: the Hugging Face cache, or a folder of your own.",
  },
  models: {
    label: "Other models",
    title: "Other Models",
    help: "Models the Studio lists besides models_dir: folders (a model, or a folder of models) or Hugging Face repo ids, as -m takes them.",
    flag: "-m",
  },
  default_model: {
    label: "Default model",
    title: "Default Model",
    help: "The model a request gets when it names none; run and chat use it without -m.",
    flag: "--default-model",
  },
  memory_budget_gb: {
    label: "Memory budget",
    title: "Memory Budget",
    help: "How much memory the loaded models may take.",
    flag: "--memory-budget",
  },
  host: { label: "Address", title: "Address", help: "The address the Studio and its API listen on.", flag: "--host" },
  port: { label: "Port", title: "Port", help: "The port the Studio listens on.", flag: "--port" },
  workspace: {
    label: "Workspace",
    title: "Workspace",
    help: "The folder of question sets, history, training data, the Studio's models and jobs.",
    flag: "--workspace",
  },
};

const SOURCES = { flag: "from a flag", config: "from the config file", default: "built in" };

// A path under the home folder written with ~, as the config file has it.
export function tilde(path) {
  const home = listed.value?.home;
  if (!home || typeof path !== "string") return path;
  return path === home || path.startsWith(`${home}/`) ? `~${path.slice(home.length)}` : path;
}

// A setting's value as the page shows it.
export function shown(key, value, source = null) {
  if (key === "models_dir") return value ? tilde(value) : "not set: the Hugging Face cache";
  if (key === "memory_budget_gb") return `${value} GB${source === "default" ? " (the GPU's working set)" : ""}`;
  if (key === "models") return value?.length ? value.map(tilde).join(", ") : "none";
  if (value == null) return "none";
  return tilde(String(value));
}

// ---------- the sheet ----------

const draft = signal(null); // the value being edited, in the editor's terms
const problem = signal(null);
const saving = signal(false);

// The editor's value as the API takes it (null: take it out of the file).
function valueOf(key, value) {
  switch (key) {
    case "models_dir":
      return value.place === "hf-cache" ? "hf-cache" : value.path.trim();
    case "models":
      return value
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
    case "host":
      return value.choice === "local" ? "127.0.0.1" : value.choice === "all" ? "0.0.0.0" : value.other.trim() || null;
    case "memory_budget_gb":
    case "port":
      return value.trim() === "" ? null : Number(value);
    default:
      return value.trim() === "" ? null : value.trim();
  }
}

function startValue(item) {
  const value = item.in_file ?? (item.key === "models" || item.key === "models_dir" ? null : item.value);
  switch (item.key) {
    case "models_dir":
      return value && value !== "hf-cache" ? { place: "folder", path: tilde(value) } : { place: "hf-cache", path: "~/Models" };
    case "models":
      return (item.in_file ?? []).map(tilde).join("\n");
    case "host": {
      const host = String(value ?? "127.0.0.1");
      if (host === "127.0.0.1" || host === "localhost") return { choice: "local", other: "" };
      if (host === "0.0.0.0") return { choice: "all", other: "" };
      return { choice: "other", other: host };
    }
    default:
      return value == null ? "" : tilde(String(value));
  }
}

async function save(key, value) {
  if (saving.value) return;
  saving.value = true;
  problem.value = null;
  try {
    const result = await api(`/settings/${key}`, { method: "PUT", body: { value } });
    closeSheet();
    const when = result.applies === "now" ? "applies now" : "takes effect at the next start";
    const what = result.line ? `Saved ${result.line}` : `Took ${key} out of the config file`;
    toast(`${what} · ${when}${result.note ? ` (${result.note})` : ""}`);
    await loadPlaces();
  } catch (error) {
    problem.value = error.message;
  } finally {
    saving.value = false;
  }
}

function Editor({ item }) {
  const key = item.key;
  const value = draft.value;
  const set = (next) => {
    draft.value = next;
    problem.value = null;
  };
  let field;
  if (key === "models_dir") {
    field = html`
      <div class="choices" role="radiogroup" aria-label="Where models go">
        <label class="choice">
          <input type="radio" name="place" checked=${value.place === "hf-cache"} onChange=${() => set({ ...value, place: "hf-cache" })} />
          <span><b>The Hugging Face cache</b></span>
        </label>
        <label class="choice">
          <input type="radio" name="place" checked=${value.place === "folder"} onChange=${() => set({ ...value, place: "folder" })} />
          <span><b>A folder</b>
            <input class="sheet-input mono folder-path" aria-label="Folder" value=${value.path} spellcheck=${false}
                   onFocus=${() => set({ ...value, place: "folder" })} onInput=${(event) => set({ ...value, place: "folder", path: event.target.value })} />
          </span>
        </label>
      </div>`;
  } else if (key === "models") {
    field = html`<textarea class="setting-input mono" rows="5" aria-label="models" spellcheck=${false}
      placeholder=${"~/Models/laya\nCloudflare/clef-flash"} value=${value} onInput=${(event) => set(event.target.value)}></textarea>
      <small class="hint">One per line.</small>`;
  } else if (key === "default_model") {
    const names = (models.value ?? []).map((model) => model.name);
    field = html`<select class="setting-input" aria-label="default_model" value=${value} onChange=${(event) => set(event.target.value)}>
      <option value="">none</option>
      ${names.map((name) => html`<option value=${name}>${name}</option>`)}
      ${value && !names.includes(value) && html`<option value=${value}>${value}</option>`}
    </select>`;
  } else if (key === "memory_budget_gb") {
    // Up to this Mac's GPU working set: a larger budget would not load more.
    const max = listed.value?.working_set_gb ?? 16;
    const number = value === "" ? max : Math.min(Number(value), max);
    field = html`
      <div class="budget-field">
        <input type="range" class="budget-slider" aria-label="Memory budget" min="1" max=${max} step="0.5"
               value=${number} onInput=${(event) => set(event.target.value)} />
        <input class="setting-input mono budget-number" aria-label="memory_budget_gb" type="number" min="0" step="0.5"
               value=${value} placeholder=${max} onInput=${(event) => set(event.target.value)} />
        <span>GB</span>
      </div>
      <small class="hint">Up to ${max} GB, this Mac's GPU working set; empty uses that.</small>`;
  } else if (key === "host") {
    const choose = (choice) => set({ ...value, choice });
    const outside = value.choice !== "local";
    const keyNote = listed.value?.api_key
      ? "This Studio runs with an API key."
      : "This Studio runs without one, so it would not start on such an address.";
    field = html`
      <div class="choices" role="radiogroup" aria-label="Where the Studio listens">
        <label class="choice">
          <input type="radio" name="host" checked=${value.choice === "local"} onChange=${() => choose("local")} />
          <span><b>This Mac only</b><small>127.0.0.1, built in</small></span>
        </label>
        <label class="choice">
          <input type="radio" name="host" checked=${value.choice === "all"} onChange=${() => choose("all")} />
          <span><b>Every network interface</b><small>0.0.0.0: other machines on your network reach the Studio and the API</small></span>
        </label>
        <label class="choice">
          <input type="radio" name="host" checked=${value.choice === "other"} onChange=${() => choose("other")} />
          <span><b>Another address</b>
            <input class="sheet-input mono folder-path" aria-label="host" value=${value.other} spellcheck=${false} placeholder="192.168.1.20"
                   onFocus=${() => choose("other")} onInput=${(event) => set({ choice: "other", other: event.target.value })} />
          </span>
        </label>
      </div>
      <p class=${outside && !listed.value?.api_key ? "warn-text" : "note"}>Anything but this Mac needs an API key, which never goes into the config file: start the Studio with <code>--api-key KEY</code>, or set <code>MLX_DECISION_API_KEY</code>. ${keyNote}</p>`;
  } else {
    const numeric = key === "port";
    field = html`<input class="setting-input mono" aria-label=${key} type=${numeric ? "number" : "text"}
      step="1" min="0" value=${value} spellcheck=${false} onInput=${(event) => set(event.target.value)} />`;
  }
  return html`
    <div class="setting-editor" data-key=${key}>
      ${field}
      ${problem.value && html`<p class="warn-text">${problem.value}</p>`}
      ${item.source === "flag" &&
      html`<p class="note">${KEYS[key].flag} sets it for this run: the file's value is used from the next start.</p>`}
      ${!item.live && item.source !== "flag" && html`<p class="note">A change takes effect at the next start.</p>`}
    </div>
  `;
}

export function openChange(item) {
  draft.value = startValue(item);
  problem.value = null;
  const key = item.key;
  openSheet({
    title: `Change the ${KEYS[key].title}`,
    purpose: KEYS[key].help,
    Body: () => html`<${Editor} item=${item} />`,
    note: `Written to ${listed.value?.config.display ?? "the config file"} as ${key}.`,
    actions: () => [
      ...(item.in_file != null
        ? [{ label: "Take Out of the File", danger: true, disabled: saving.value, onSelect: () => save(key, null) }]
        : []),
      { label: "Save", primary: true, disabled: saving.value, onSelect: () => save(key, valueOf(key, draft.value)) },
    ],
  });
}

// ---------- the section ----------

function Setting({ item, editable }) {
  const differs = item.in_file != null && JSON.stringify(item.in_file) !== JSON.stringify(item.value);
  return html`
    <div class="kv setting" data-key=${item.key}>
      <dt>${KEYS[item.key].label}<code>${item.key}</code></dt>
      <dd>
        ${shown(item.key, item.value, item.source)}
        <small>
          ${item.source ? SOURCES[item.source] : ""}${differs ? ` · the file says ${shown(item.key, item.in_file)}` : ""}${item.editable ? "" : `${item.source ? " · " : ""}moves with ${KEYS[item.key].flag} or the config file, at the next start`}
        </small>
      </dd>
      ${item.editable
        ? editable
          ? html`<button class="btn sm" onClick=${() => openChange(item)}>Change…</button>`
          : html`<span></span>`
        : html`<button class="btn sm" disabled=${!editable} title=${editable ? "Show the workspace in Finder" : "Finder opens only for a browser on this Mac"}
                  onClick=${() => reveal(item.key)}>Show in Finder</button>`}
    </div>
  `;
}

export function ConfigFile() {
  const current = listed.value;
  if (!current) return null;
  const { config, editable } = current;
  return html`
    <section class="igroup config-group" aria-label="Config file">
      <h4>Config file</h4>
      <dl>
        <div class="kv place">
          <dt>File</dt>
          <dd><code title=${config.path}>${config.display}</code><small>${config.exists ? "Read by the Studio and the CLI; a flag wins over it" : "Not created yet: a change creates it"}</small></dd>
          <button class="btn sm" disabled=${!editable} title=${editable ? "Show it in Finder" : "Finder opens only for a browser on this Mac"}
                  onClick=${() => reveal("config")}>Show in Finder</button>
        </div>
        ${config.settings.map((item) => html`<${Setting} key=${item.key} item=${item} editable=${editable} />`)}
        ${config.warnings.map((warning) => html`<p class="warn-text">${warning}</p>`)}
        ${!editable && html`<p class="note read-only">Settings change only from a browser on this Mac.</p>`}
      </dl>
    </section>
  `;
}
