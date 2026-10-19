// The Playground: a question set, the model it runs on, its answers and history.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import * as icons from "../components/icons.js";
import { MenuButton, MenuHeader, MenuItem, MenuNote, MenuSeparator } from "../components/menu.js";
import { clock, duration, plural } from "../format.js";
import { SHORTCUTS, onKey } from "../keys.js";
import { href, navigate, route } from "../router.js";
import { Answers, shownBatch } from "../playground/answers.js";
import { Editor } from "../playground/editor.js";
import * as files from "../playground/files.js";
import { History } from "../playground/history.js";
import {
  built,
  canRunMain,
  chooseModel,
  edited,
  editedNames,
  entry,
  fileProblems,
  justAnswered,
  lastRun,
  modelName,
  problems,
  run,
  runMain,
  running,
  setName,
  several,
  stale,
  stop,
} from "../playground/session.js";
import { sets } from "../store/sets.js";
import { loadedNames, models } from "../store/status.js";

const title = (name) => {
  const words = name.replace(/[-_.]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};

function SetsList() {
  const list = sets.value;
  return html`
    <div class="sb-sec">
      <h3>
        Question Sets
        <button aria-label="New question set" title=${`New Question Set (${SHORTCUTS.newSet})`} onClick=${files.newSet}>
          <${icons.Plus} />
        </button>
      </h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No sets yet</p>`}
      <ul class="sb-list">
        ${(list ?? []).map(
          (item) => html`
            <li key=${item.name}>
              <a class="sb-item" href=${href("playground", item.name)}
                 aria-current=${item.name === setName.value ? "true" : undefined}>
                <${icons.File} /><span class="label">${item.name}</span>
                ${editedNames.value.has(item.name) && html`<span class="edited" title="Edited, not saved"></span>`}
              </a>
            </li>
          `,
        )}
      </ul>
    </div>
  `;
}

function Target() {
  const list = models.value ?? [];
  const name = setName.value;
  const current = modelName.value;
  const loaded = loadedNames.value;
  const modelProblem = problems.value.find((problem) => problem.where.model);
  return html`
    <div class="target" role="group" aria-label="What Run runs">
      <${MenuButton} id="target-set" className="set" title="Question set"
        label=${html`<${icons.File} className="" /><span>${name ?? "No set"}</span>`}
        menuStyle="top:34px; left:0; width:280px">
        <${MenuHeader}>Question Sets<//>
        ${(sets.value ?? []).map(
          (item) => html`<${MenuItem} ticked=${item.name === name}
            onSelect=${() => navigate("playground", item.name)}>${item.name}<//>`,
        )}
        ${(sets.value ?? []).length === 0 && html`<${MenuNote}>No question sets yet.<//>`}
      <//>
      <${icons.Crumb} />
      <${MenuButton} id="target-model" className=${`mdl ${modelProblem ? "bad" : ""}`}
        title=${modelProblem ? modelProblem.message : "Model"}
        label=${html`
          ${current && html`<span class="dot" style=${loaded.has(current) ? "" : "background: var(--fg-3)"}></span>`}
          <span class="name">${current ?? "No model"}</span>
          <${icons.Chevrons} />
        `}
        menuStyle="top:34px; left:0; width:320px">
        <${MenuHeader}>Run on<//>
        ${list.map(
          (item) => html`<${MenuItem} ticked=${item.name === current}
            right=${loaded.has(item.name) ? "Loaded" : item.family}
            sub=${item.fitted ? `fitted on ${item.base}` : null}
            onSelect=${() => chooseModel(item.name)}><span class="mono">${item.name}</span><//>`,
        )}
        ${list.length === 0 && html`<${MenuNote}>No models yet. Download one: <code>mlx-decision download</code><//>`}
      <//>
    </div>
  `;
}

const questionCount = computed(() => (built.value ? Object.keys(built.value.body.questions).length : 0));

// A problem in one of the examples (the selected one's show first, with the
// live check's): "Refund ask: Give the example a name".
function exampleProblem(problem) {
  const where = problem.where?.example;
  const named = where && entry.value?.draft?.examples.find((item) => item.uid === where);
  return named ? html`<b>${named.name}</b>: ${problem.message}` : problem.message;
}

const status = computed(() => {
  if (models.value && models.value.length === 0)
    return { text: html`No models yet · <code>mlx-decision download</code>` };
  if (sets.value && setName.value === null) return { text: "No question set yet" };
  if (entry.value?.error) return { icon: "warn", text: entry.value.error };
  if (!built.value || !modelName.value) return null;
  const flight = running.value;
  const many = several.value;
  if (flight)
    return {
      icon: "spin",
      text: flight.all
        ? html`Run All on <b>${flight.model}</b>: <b>${flight.example}</b>, ${flight.all.done + 1} of ${flight.all.total}…${flight.loads ? " · loading it first" : ""}`
        : html`Asking <b>${flight.model}</b> ${plural(questionCount.value, "question")}${many ? html` about <b>${flight.example}</b>` : ""}…${flight.loads ? " · loading it first" : ""}`,
    };
  if (problems.value.length) return { icon: "warn", text: problems.value[0].message };
  const loads = loadedNames.value.has(modelName.value) ? "" : " · loads it first";
  if (many) {
    if (fileProblems.value.length) return { icon: "warn", text: exampleProblem(fileProblems.value[0]) };
    const batch = shownBatch.value;
    if (batch?.done && batch.changed)
      return { icon: "warn", text: html`Edited since the Run All at ${clock(batch.shown.time)} · Run All to update` };
    if (batch?.done) {
      const answered = batch.rows.filter((row) => row.run);
      const total = answered.reduce((sum, row) => sum + (row.run.latency_ms ?? 0), 0);
      const count = answered.length === batch.rows.length
        ? plural(answered.length, "example")
        : `${answered.length} of ${plural(batch.rows.length, "example")}`;
      return {
        icon: justAnswered.value ? "ok" : null,
        text: html`<b>${modelName.value}</b> answered ${count} in ${duration(total)} · ${clock(batch.shown.time)}`,
      };
    }
    const count = entry.value.draft.examples.length;
    return {
      text: html`Run All sends <b>${plural(questionCount.value, "question")}</b> about <b>${count} examples</b> to <b>${modelName.value}</b>${loads}`,
    };
  }
  const result = lastRun.value;
  const changed = stale.value;
  if (result && (changed.all || changed.ids.size))
    return {
      icon: "warn",
      text: changed.all
        ? html`State changed since the run at ${clock(result.time)} · Run to update`
        : html`<b>${[...changed.ids].join(", ")}</b> edited since the run · Run to update`,
    };
  if (result)
    return {
      // The check only right after the answer came; then the line is plain.
      icon: justAnswered.value ? "ok" : null,
      text: html`<b>${result.model}</b> answered in ${duration(result.latency_ms)} · ${clock(result.time)}`,
    };
  return {
    text: html`Run sends <b>${plural(questionCount.value, "question")}</b> to <b>${modelName.value}</b>${loads}`,
  };
});

// With several examples the button is Run All (⌘↩); ⌥⌘↩ and the example's
// chip menu ask the selected example only.
const action = computed(() => {
  const many = several.value;
  const changed = stale.value;
  const batch = shownBatch.value;
  return {
    label: many ? "Run All" : "Run",
    title: many
      ? `Run All: ask every example in turn (${SHORTCUTS.run}); ${SHORTCUTS.runExample} asks the selected one`
      : undefined,
    disabled: !canRunMain.value,
    run: runMain,
    stop,
    running: !!running.value,
    badge: many ? !!batch?.done && batch.changed : !!lastRun.value && (changed.all || changed.ids.size > 0),
  };
});

function MoreMenu() {
  return html`
    <${MenuButton} id="set-more" className="btn" ariaLabel="More actions for this set"
      label=${html`<svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15"><circle cx="5" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="15" cy="10" r="1.5" /></svg>`}
      menuStyle="top:32px; right:0; width:260px">
      <${MenuItem} right=${SHORTCUTS.saveAs} onSelect=${files.saveAs}>Save As…<//>
      <${MenuItem} onSelect=${files.renameSet}>Rename…<//>
      <${MenuItem} right=${SHORTCUTS.newSet} onSelect=${files.newSet}>New Question Set…<//>
      <${MenuItem} right=${SHORTCUTS.import} sub="A -q questions file or a whole request body" onSelect=${files.importSet}>Import Questions…<//>
      <${MenuSeparator} />
      <${MenuItem} sub=${edited.value ? "The saved file, without your edits" : null} onSelect=${files.exportSet}>Export<//>
      <${MenuItem} onSelect=${files.reveal}>Show in Workspace Folder<//>
      <${MenuSeparator} />
      <${MenuItem} onSelect=${files.deleteSet}>Delete…<//>
    <//>
  `;
}

function DiskBanner() {
  const where = entry.value?.onDisk;
  if (!where) return null;
  return html`
    <div class="banner disk-banner" role="status">
      <${icons.StatusIcon} kind="warn" />
      <span>
        ${where === "deleted"
          ? html`<b>sets/${setName.value}.json</b> was deleted on disk. Your edits are only here.`
          : html`<b>sets/${setName.value}.json</b> changed on disk. Saving will not overwrite it.`}
      </span>
      ${where === "changed" &&
      html`<button class="btn" onClick=${() => files.reloadFromDisk()}>Reload</button>`}
      <button class="btn" onClick=${files.saveAs}>Save As…</button>
    </div>
  `;
}

function Centre() {
  useEffect(() => {
    const off = [
      onKey("run", () => runMain()),
      onKey("runExample", () => run()),
      onKey("save", () => files.save()),
      onKey("saveAs", () => files.saveAs()),
      onKey("newSet", () => files.newSet()),
      onKey("import", () => files.importSet()),
    ];
    return () => off.forEach((remove) => remove());
  }, []);
  const list = sets.value;
  const current = entry.value;
  if (list && list.length === 0 && !route.value.item) {
    return html`
      <div class="centre-inner">
        <div class="pane-head"><div><h1>Playground</h1></div></div>
        <div class="empty">
          <b>No question sets yet</b>
          <span>
            A question set is a request body saved in the workspace's${" "}
            <code>sets/</code> folder; <code>mlx-decision run -q sets/NAME.json</code> answers it too.
          </span>
          <div class="empty-actions">
            <button class="btn primary" onClick=${files.newSet}>New Question Set…</button>
            <button class="btn" onClick=${files.importSet}>Import Questions…</button>
          </div>
        </div>
      </div>
    `;
  }
  if (!current) return html`<div class="centre-inner"></div>`;
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>${title(setName.value)}</h1>
          <p><code>sets/${setName.value}.json</code>${edited.value && html`<span class="edited-tag">· Edited</span>`}</p>
        </div>
        ${!current.error &&
        html`<div class="actions">
          <button class="btn" disabled=${!edited.value} title=${`Save (${SHORTCUTS.save})`} onClick=${files.save}>Save</button>
          <${MoreMenu} />
        </div>`}
      </div>
      <${DiskBanner} />
      ${current.error
        ? html`<div class="empty"><b>${current.error}</b>${
            current.status === 404 &&
            html`<div class="empty-actions"><button class="btn" onClick=${files.newSet}>New Question Set…</button></div>`
          }</div>`
        : html`<${Editor} />`}
    </div>
  `;
}

export default {
  id: "playground",
  title: "Playground",
  SidebarList: SetsList,
  Target,
  action,
  status,
  Centre,
  tabs: [
    { id: "answers", label: "Answers", Body: Answers },
    { id: "history", label: "History", Body: History },
  ],
};
