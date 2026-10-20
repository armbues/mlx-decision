// Settings: the config file, which the page changes (CFG-4). Loaded again
// when the models or the file change, in any tab. Where models live is on
// the Models page.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { ConfigFile } from "../settings/config.js";
import { listed, loadPlaces } from "../settings/places.js";
import { models } from "../store/status.js";
import { settingsChange } from "../stream.js";

const status = computed(() => {
  const current = listed.value;
  if (!current) return null;
  const where = current.config.exists ? current.config.display : "no config file yet";
  return { text: html`Config: <code>${where}</code>${current.editable ? "" : " · read-only here"}` };
});

function Centre() {
  // Again when the list changes (a download, a rescan) or the file does.
  useEffect(() => {
    loadPlaces();
  }, [models.value, settingsChange.value]);
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>Settings</h1>
          <p>The config file that the Studio and the CLI read.</p>
        </div>
      </div>
      <${ConfigFile} />
    </div>
  `;
}

const About = () => html`
  <div class="about-settings">
    <p>A flag on the command line wins over the config file, and the file wins over the built-in default. The CLI reads the same file: <code>run</code>, <code>chat</code>, <code>server</code>, <code>mcp</code>, <code>web</code>, <code>training</code> and <code>benchmark</code>, where a key applies.</p>
    <p>A change is written at once, one line in the file; comments and everything else stay as they are. <code>models_dir</code>, <code>default_model</code>, <code>memory_budget_gb</code> and models added to <code>models</code> apply now; the rest at the next start.</p>
  </div>
`;

export default {
  id: "settings",
  title: "Settings",
  status,
  Centre,
  tabs: [{ id: "about", label: "About", Body: About }],
};
