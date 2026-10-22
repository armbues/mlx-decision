// Where a model is picked and there is none yet: say so, with the way to get one.
import { html } from "htm/preact";
import { openDownload } from "../models/download.js";

export const NO_MODELS = "Download a model first";

export function NoModels() {
  return html`<span class="no-models">No models yet
    <button class="btn sm" type="button" onClick=${() => openDownload()}>Download a Model…</button></span>`;
}
