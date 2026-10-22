// Where models live (on the Models page): the places the Studio's models come
// from, each with its path, how many listed models are in it and Show in
// Finder. The page names a place by its id; the Studio knows the path. The
// same request brings the config the Settings page shows.
import { signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import { plural } from "../format.js";
import { toast } from "../store/ui.js";

export const listed = signal(null); // GET /places: {places, config, editable}

export async function loadPlaces() {
  try {
    listed.value = await api("/places");
  } catch (error) {
    toast(`Could not read the settings: ${error.message}`);
  }
}

export async function reveal(place) {
  try {
    await api("/places/reveal", { method: "POST", body: { place } });
  } catch (error) {
    toast(`Could not show it in Finder: ${error.message}`);
  }
}

function named(settings) {
  const source = settings.find((item) => item.key === "models")?.source;
  return source === "flag" ? "Named with -m when the Studio started" : "Named in the config file's models";
}

// A place's title and what it is for.
function describe(place, settings) {
  const downloads = place.downloads ? " · downloads go here" : "";
  switch (place.kind) {
    case "models_dir":
      return ["Your models folder", `models_dir in the config file${downloads}`];
    case "cache":
      return ["Hugging Face cache", `Shared with other tools that use the Hub${downloads}`];
    case "studio":
      return ["Made here", "Converted copies and fitted models"];
    default:
      return ["Your folders", named(settings)];
  }
}

function Place({ place, settings, editable }) {
  const [title, note] = describe(place, settings);
  const why = !editable ? "Finder opens only for a browser on this Mac" : place.exists ? `Show ${place.path} in Finder` : "Not created yet";
  return html`
    <div class="kv place" data-place=${place.kind}>
      <dt>${title}</dt>
      <dd><code title=${place.path}>${place.display}</code><small>${note} · ${plural(place.models, "model")}</small></dd>
      <button class="btn sm" disabled=${!place.exists || !editable} title=${why} onClick=${() => reveal(place.id)}>Show in Finder</button>
    </div>
  `;
}

export function Places() {
  const current = listed.value;
  if (!current) return null;
  const settings = current.config.settings;
  const folders = current.places.some((place) => place.kind === "named");
  return html`
    <section class="igroup places" aria-label="Where models live">
      <h4>Where models live</h4>
      <dl>
        ${current.places.map(
          (place) => html`<${Place} key=${place.id} place=${place} settings=${settings} editable=${current.editable} />`,
        )}
        ${!folders &&
        html`<div class="kv place" data-place="named">
          <dt>Your folders</dt>
          <dd><span class="muted">None</span><small>Add folders to <code>models</code> in Settings, or start the Studio with <code>-m FOLDER</code>, to list models you keep elsewhere.</small></dd>
          <span></span>
        </div>`}
      </dl>
    </section>
  `;
}
