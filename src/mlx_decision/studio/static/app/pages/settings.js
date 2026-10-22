// Settings: the config file, which the page changes (CFG-4), and the
// workspace's folders, each with Clear…. Loaded again
// when the models or the file change, in any tab. Where models live is on
// the Models page.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { ConfigFile } from "../settings/config.js";
import { listed, loadPlaces } from "../settings/places.js";
import { WorkspaceFolders, loadFolders } from "../settings/workspace.js";
import { models } from "../store/status.js";
import { inspectorMode } from "../store/ui.js";
import { settingsChange, workspaceChange } from "../stream.js";

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
  // The folders' sizes again when the workspace or the models change.
  useEffect(() => {
    loadFolders();
  }, [models.value, workspaceChange.value]);
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>Settings</h1>
          <p>The config file the Studio and the CLI read; a flag on the command line wins over it. A change is written at once; the models and the budget apply now, the rest at the next start.</p>
        </div>
      </div>
      <${ConfigFile} />
      <${WorkspaceFolders} />
    </div>
  `;
}

inspectorMode("settings", { mode: "none" });

export default {
  id: "settings",
  title: "Settings",
  status,
  Centre,
};
