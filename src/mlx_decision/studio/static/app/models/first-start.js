// First start (no models yet): where models go, asked once when the config file
// does not say, then "Download your first model" with a small, a medium and a
// large model that fit this Mac (chosen at random among those that fit). A
// pick opens Download Model on it, so its check and licence note come first.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { api } from "../api.js";
import { StatusIcon } from "../components/icons.js";
import { markOf } from "../families.js";
import { gigabytes } from "../format.js";
import { toast } from "../store/ui.js";
import { openDownload } from "./download.js";

const first = signal(null); // GET /downloads/first
const place = signal("hf-cache"); // the answer: "hf-cache" or "folder"
const folder = signal("");
const saved = signal(null); // {line, config_path} once written
const saving = signal(false);

async function load() {
  try {
    first.value = await api("/downloads/first");
    if (!folder.value) folder.value = first.value.suggestion;
  } catch (error) {
    toast(`Could not list models to start with: ${error.message}`);
  }
}

async function answer(event) {
  event.preventDefault();
  if (saving.value) return;
  saving.value = true;
  const body = place.value === "folder" ? { place: "folder", path: folder.value } : { place: "hf-cache" };
  try {
    saved.value = await api("/downloads/destination", { method: "POST", body });
    first.value = { ...first.value, models_dir_set: true };
  } catch (error) {
    toast(error.message);
  } finally {
    saving.value = false;
  }
}

function WhereModelsGo() {
  return html`
    <form class="first-card where" onSubmit=${answer}>
      <h2>Where should models go?</h2>
      <p>Asked once: the answer goes into the config file as <code>models_dir</code>, which <code>mlx-decision download</code> uses too.</p>
      <div class="choices" role="radiogroup" aria-label="Where models go">
        <label class="choice">
          <input type="radio" name="place" value="hf-cache" checked=${place.value === "hf-cache"}
                 onChange=${() => (place.value = "hf-cache")} />
          <span><b>The Hugging Face cache</b><small>Shared with other tools that use the Hub; nothing to manage.</small></span>
        </label>
        <label class="choice">
          <input type="radio" name="place" value="folder" checked=${place.value === "folder"}
                 onChange=${() => (place.value = "folder")} />
          <span><b>A folder</b><small>Each model in a folder of its own; <code>-m</code> takes their names.</small>
            <input class="folder-path mono" aria-label="Folder" value=${folder.value} spellcheck="false"
                   onFocus=${() => (place.value = "folder")} onInput=${(event) => (folder.value = event.target.value)} />
          </span>
        </label>
      </div>
      <div class="first-actions">
        <button class="btn primary" type="submit" disabled=${saving.value}>Continue</button>
      </div>
    </form>
  `;
}

const TIERS = { small: "Small", medium: "Medium", large: "Large" };

function Pick({ pick }) {
  const [text, colour] = markOf(pick.family);
  return html`
    <button class="first-pick" data-repo=${pick.repo} onClick=${() => openDownload({ family: pick.family, repo: pick.repo })}>
      <span class="tier">${TIERS[pick.tier]}</span>
      <span class="ficon" style=${`background:${colour}`} aria-hidden="true">${text}</span>
      <b class="mono">${pick.repo.split("/")[1]}</b>
      <span class="desc">${pick.description}</span>
      <span class="num">${pick.size} download · about ${gigabytes(pick.memory_bytes)} to run</span>
    </button>
  `;
}

function Picks() {
  const picks = first.value.picks;
  return html`
    <div class="first-card">
      <h2>Download your first model</h2>
      ${saved.value &&
      html`<p class="saved">
        <${StatusIcon} kind="ok" /><span>Saved <code>${saved.value.line}</code> in <code>${saved.value.config_path}</code>; change it there.</span>
      </p>`}
      <p>${picks.length
        ? "Three that fit this Mac's memory, a small, a medium and a large one; each opens Download Model with its check and terms first."
        : "None of the models the Studio knows fits this Mac's memory budget at full size; Download Model offers them quantized."}</p>
      <div class="first-picks">${picks.map((pick) => html`<${Pick} key=${pick.repo} pick=${pick} />`)}</div>
      <div class="first-actions">
        <button class="btn" onClick=${() => openDownload()}>Download Another Model…</button>
      </div>
    </div>
  `;
}

export function FirstStart() {
  useEffect(() => {
    load();
  }, []);
  if (!first.value) return null;
  return html`
    <div class="first-start">
      ${first.value.models_dir_set ? html`<${Picks} />` : html`<${WhereModelsGo} />`}
    </div>
  `;
}
