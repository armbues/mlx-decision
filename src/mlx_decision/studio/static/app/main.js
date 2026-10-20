// The Studio's page: the window, its stores and the stream that keeps them current.
import { render } from "preact";
import { html } from "htm/preact";
import { App } from "./components/app.js";
import "./keys.js";
import { reloadSets } from "./store/sets.js";
import { loadModels, loadStatus, rescanModels } from "./store/status.js";
import { openMenu } from "./store/ui.js";
import { connect } from "./stream.js";

connect();
loadStatus();
loadModels();
reloadSets();

// The CLI may have changed files behind the Studio's back.
addEventListener("focus", () => {
  rescanModels();
  reloadSets();
});
document.addEventListener("click", () => {
  openMenu.value = null;
});

render(html`<${App} />`, document.getElementById("studio"));
