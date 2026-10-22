// Settings: the config file, which the page changes (CFG-4), and the
// workspace's folders, each with Clear…. Loaded again
// when the models or the file change, in any tab. Where models live is on
// the Models page.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import { navigate, route } from "../router.js";
import { ConfigFile, openChange } from "../settings/config.js";
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
  // #/settings/KEY opens that key's sheet (the memory budget, from the
  // sidebar or the Models page), once the file has been read.
  useEffect(() => {
    const key = route.value.item;
    const item = listed.value?.config.settings.find((setting) => setting.key === key);
    if (!item) return;
    navigate("settings");
    if (item.editable && listed.value.editable) openChange(item);
  }, [route.value.item, listed.value]);
  // The folders' sizes again when the workspace or the models change.
  useEffect(() => {
    loadFolders();
  }, [models.value, workspaceChange.value]);
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>Settings</h1>
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
