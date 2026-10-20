// The Studio's page: the window, its stores and the stream that keeps them current.
import { render } from "preact";
import { html } from "htm/preact";
import { App } from "./components/app.js";
import "./keys.js";
import { reloadData } from "./store/data.js";
import { reloadSets } from "./store/sets.js";
import { navigate } from "./router.js";
import { loadModels, loadStatus, models, rescanModels } from "./store/status.js";
import { openMenu } from "./store/ui.js";
import { connect } from "./stream.js";

connect();
loadStatus();
// First start: with no models, the Studio opens on Models (its first-start panel).
loadModels().then(() => {
  if (!location.hash && models.value?.length === 0) navigate("models");
});
reloadSets();

// The CLI may have changed files behind the Studio's back.
addEventListener("focus", () => {
  rescanModels();
  reloadSets();
  reloadData();
});
document.addEventListener("click", () => {
  openMenu.value = null;
});

render(html`<${App} />`, document.getElementById("studio"));
