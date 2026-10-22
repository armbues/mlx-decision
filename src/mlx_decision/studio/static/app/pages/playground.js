// The Playground: a question set (or a dataset's questions, "data:NAME"), the
// model it runs on, its answers and history.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import * as icons from "../components/icons.js";
import { MenuButton, MenuHeader, MenuItem, MenuNote, MenuSeparator } from "../components/menu.js";
import { openDownload } from "../models/download.js";
import { clock, duration, plural } from "../format.js";
import { SHORTCUTS, onKey } from "../keys.js";
import { href, navigate, route } from "../router.js";
import { Answers, shownBatch } from "../playground/answers.js";
import { CodeButton } from "../playground/code.js";
import { shownCompare } from "../playground/compare.js";
import { startFromSet } from "../playground/dataset.js";
import { Editor } from "../playground/editor.js";
import * as files from "../playground/files.js";
import { History } from "../playground/history.js";
import {
  MAX_COMPARED,
  built,
  canRunMain,
  chooseModel,
  compareModels,
  compareWith,
  comparing,
  dataset,
  datasetOfDocument,
  drafts,
  edited,
  editedNames,
  entry,
  example,
  fileProblems,
  isDataset,
  justAnswered,
  lastRun,
  modelName,
  modelOverBudget,
  goneModel,
  problems,
  run,
  runMain,
  running,
  setName,
  several,
  stale,
  stop,
  toggleCompared,
} from "../playground/session.js";
import { sets } from "../store/sets.js";
import { busy, loadedNames, models } from "../store/status.js";

const title = (name) => {
  const words = name.replace(/[-_.]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};

// The datasets open in the Playground (the one shown, and any with edits).
function DatasetsOpen() {
  const shown = setName.value;
  const open = Object.keys(drafts.value).filter((name) => isDataset(name) && (name === shown || editedNames.value.has(name)));
  if (!open.length) return null;
  return html`
    <div class="sb-sec">
      <h3>Dataset Questions</h3>
      <ul class="sb-list">
        ${open.map(
          (name) => html`<li key=${name} class=${name === shown ? "open" : ""}>
            <a class="sb-item" href=${href("playground", name)} aria-current=${name === shown ? "true" : undefined}
               title=${`${datasetOfDocument(name)}'s questions, saved in its dataset.json`}>
              <${icons.Datasets} /><span class="label">${datasetOfDocument(name)}</span>
              ${editedNames.value.has(name) && html`<span class="edited" title="Edited, not saved"></span>`}
            </a>
          </li>`,
        )}
      </ul>
    </div>
  `;
}

function SetsList() {
  const list = sets.value;
  return html`
    <${DatasetsOpen} />
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
          (item) => {
            const open = item.name === setName.value;
            return html`
              <li key=${item.name} class=${open ? "open" : ""}>
                <a class="sb-item" href=${href("playground", item.name)} aria-current=${open ? "true" : undefined}>
                  <${icons.File} /><span class="label">${item.name}</span>
                  ${editedNames.value.has(item.name) && html`<span class="edited" title="Edited, not saved"></span>`}
                </a>
                <button class="sb-delete" aria-label=${`Delete ${item.name}…`} title="Delete…"
                  onClick=${() => files.deleteSet(item.name)}><${icons.Trash} /></button>
              </li>
            `;
          },
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
  const others = comparing.value ? compareWith.value : [];
  const picked = compareWith.value;
  const full = picked.length >= MAX_COMPARED - 1;
  return html`
    <div class="target" role="group" aria-label="What Run runs">
      <${MenuButton} id="target-set" className="set" title=${isDataset(name) ? "A dataset's questions" : "Question set"}
        label=${isDataset(name)
          ? html`<${icons.Datasets} /><span>${datasetOfDocument(name)}</span>`
          : html`<${icons.File} className="" /><span>${name ?? "No set"}</span>`}
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
          ${others.length > 0 && html`<span class="plus num" title=${`Compared with ${others.join(", ")}`}>+ ${others.length}</span>`}
          <${icons.Chevrons} />
        `}
        menuStyle="top:34px; left:0; width:320px">
        <${MenuHeader}>Run on<//>
        ${list.map(
          (item) => html`<${MenuItem} ticked=${item.name === current}
            right=${loaded.has(item.name) ? "Loaded" : item.memory?.over_budget ? "Over the budget" : item.family}
            sub=${item.fitted ? `fitted on ${item.base}` : null}
            onSelect=${() => chooseModel(item.name)}><span class="mono">${item.name}</span><//>`,
        )}
        ${list.length === 0 &&
        html`<${MenuNote}>No models yet<//>
          <${MenuItem} right=${SHORTCUTS.download} onSelect=${() => openDownload()}>Download a Model…<//>`}
        ${list.length > 1 &&
        html`<${MenuSeparator} />
          <${MenuHeader}>Compare With<//>
          ${list
            .filter((item) => item.name !== current)
            .map(
              (item) => html`<${MenuItem} ticked=${picked.includes(item.name)}
                disabled=${!picked.includes(item.name) && full}
                right=${loaded.has(item.name) ? "Loaded" : item.family}
                onSelect=${() => toggleCompared(item.name)}><span class="mono">${item.name}</span><//>`,
            )}
          <${MenuNote}>Up to two more models. Compare asks the same questions on each in turn.<//>`}
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
    return { text: html`No models yet · Download one in Models (${SHORTCUTS.download})` };
  if (sets.value && setName.value === null) return { text: "No question set yet" };
  if (entry.value?.error) return { icon: "warn", text: entry.value.error };
  if (!built.value || !modelName.value) return null;
  const flight = running.value;
  const many = several.value;
  if (flight?.compare)
    return {
      icon: "spin",
      text: html`Compare: asking <b>${flight.model}</b>${many ? html` about <b>${flight.example}</b>` : ""}, ${flight.compare.done + 1} of ${flight.compare.total}…${flight.loads ? " · loading it first" : ""}`,
    };
  if (flight)
    return {
      icon: "spin",
      text: flight.all
        ? html`Run All on <b>${flight.model}</b>: <b>${flight.example}</b>, ${flight.all.done + 1} of ${flight.all.total}…${flight.loads ? " · loading it first" : ""}`
        : html`Asking <b>${flight.model}</b> ${plural(questionCount.value, "question")}${many ? html` about <b>${flight.example}</b>` : ""}…${flight.loads ? " · loading it first" : ""}`,
    };
  if (problems.value.length) return { icon: "warn", text: problems.value[0].message };
  if (comparing.value) return compareStatus();
  if (modelOverBudget.value) return { icon: "warn", text: modelOverBudget.value };
  if (goneModel.value) return { icon: "warn", text: html`<b>${goneModel.value}</b> is gone · Run uses <b>${modelName.value}</b>` };
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

// "a, b and c".
const listed = (names) =>
  names.length < 2 ? html`<b>${names[0]}</b>` : html`${names.slice(0, -1).map((name, i) => html`${i ? ", " : ""}<b>${name}</b>`)} and <b>${names.at(-1)}</b>`;

// The status field in the Compare view.
function compareStatus() {
  const names = compareModels.value;
  if (names.length < 2) return { text: "Compare: pick one or two more models in the model menu" };
  const state = shownCompare.value;
  const about = several.value ? html` about <b>${example.value.name}</b>` : "";
  if (state?.done && (state.changed.all || state.changed.ids.size || state.modelsChanged))
    return { icon: "warn", text: html`Edited since the Compare at ${clock(state.shown.time)} · Compare to update` };
  if (state?.done) {
    const total = state.shown.models.length;
    const answered = state.shown.models.filter((name) => state.shown.results[name]?.response).length;
    const count = answered === total ? plural(total, "model") : `${answered} of ${plural(total, "model")}`;
    return {
      icon: justAnswered.value ? "ok" : null,
      text: html`<b>${count}</b> answered${about} · ${clock(state.shown.time)}`,
    };
  }
  if (state?.stopped)
    return { icon: "warn", text: html`The Compare at ${clock(state.shown.time)} was stopped · Compare to ask them all` };
  const toLoad = names.filter((name) => !loadedNames.value.has(name)).length;
  return {
    text: html`Compare sends <b>${plural(questionCount.value, "question")}</b>${about} to ${listed(names)} in turn${
      toLoad ? ` · ${toLoad === names.length ? "all load" : `${toLoad} load`} first` : ""
    }`,
  };
}

// With several examples the button is Run All (⌘↩); ⌥⌘↩ and the example's
// chip menu ask the selected example only. In the Compare view it is Compare.
// While a job holds the GPU the main action is off, and its tooltip says why.
const heldTitle = (label) => busy.value && `${busy.value} has the GPU: ${label} is off until it is done`;

const action = computed(() => {
  const many = several.value;
  const changed = stale.value;
  const batch = shownBatch.value;
  if (comparing.value) {
    const state = shownCompare.value;
    return {
      label: "Compare",
      title:
        heldTitle("Compare") ||
        `Compare: ask the same questions on ${compareModels.value.join(", ")} in turn (${SHORTCUTS.run})`,
      disabled: !canRunMain.value,
      run: runMain,
      stop,
      running: !!running.value,
      badge: !!state && !state.inFlight && (state.changed.all || state.changed.ids.size > 0 || state.modelsChanged),
    };
  }
  return {
    label: many ? "Run All" : "Run",
    title:
      heldTitle(many ? "Run All" : "Run") ||
      (many
        ? `Run All: ask every example in turn (${SHORTCUTS.run}); ${SHORTCUTS.runExample} asks the selected one`
        : undefined),
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
      <${MenuItem} onSelect=${() => files.deleteSet()}>Delete…<//>
    <//>
  `;
}

function DatasetMenu() {
  const name = dataset.value?.name;
  return html`
    <${MenuButton} id="set-more" className="btn" ariaLabel="More actions for this dataset"
      label=${html`<svg viewBox="0 0 20 20" fill="currentColor" width="15" height="15"><circle cx="5" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="15" cy="10" r="1.5" /></svg>`}
      menuStyle="top:32px; right:0; width:280px">
      <${MenuItem} sub="Its questions replace these; save to keep them" onSelect=${startFromSet}>Start from a Question Set…<//>
      <${MenuItem} right=${SHORTCUTS.saveAs} sub="The questions and the rows shown, as a new set" onSelect=${files.saveAs}>Save as Question Set…<//>
      <${MenuSeparator} />
      <${MenuItem} onSelect=${() => navigate("datasets", name)}>Show in Datasets<//>
      <${MenuItem} sub=${edited.value ? "Your edits go" : null} onSelect=${files.reloadFromDisk}>Reload<//>
    <//>
  `;
}

// What keeps the dataset from a fit, as the last save (or load) left it.
function DatasetBanner() {
  if (entry.value?.onDisk === "deleted")
    return html`
      <div class="banner disk-banner" role="status">
        <${icons.StatusIcon} kind="warn" />
        <span><b>data/${datasetOfDocument(setName.value)}/</b> was deleted. Your questions are only here.</span>
        <button class="btn" onClick=${files.saveAs}>Save as Question Set…</button>
      </div>
    `;
  const missing = dataset.value?.missing ?? [];
  if (!missing.length) return null;
  return html`
    <div class="notice ds-banner" role="status">
      <span><b>Not ready for a fit, as saved:</b> ${missing.join("; ")}.</span>
    </div>
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
            <code>sets/</code> folder.
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
  const data = isDataset(setName.value);
  const where = data ? `data/${datasetOfDocument(setName.value)}/dataset.json` : `sets/${setName.value}.json`;
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1 class=${data ? "mono" : ""}>${data ? datasetOfDocument(setName.value) : title(setName.value)}</h1>
          <p>${data ? "The dataset's questions · " : ""}<code>${where}</code>${edited.value && html`<span class="edited-tag">· Edited</span>`}</p>
        </div>
        ${!current.error &&
        html`<div class="actions">
          <button class="btn" disabled=${!edited.value} title=${`Save (${SHORTCUTS.save})`} onClick=${files.save}>Save</button>
          ${data ? html`<${DatasetMenu} />` : html`<${MoreMenu} />`}
        </div>`}
      </div>
      ${data ? html`<${DatasetBanner} />` : html`<${DiskBanner} />`}
      ${current.error
        ? html`<div class="empty"><b>${current.error}</b>${
            current.status === 404
              ? html`<div class="empty-actions"><button class="btn" onClick=${files.newSet}>New Question Set…</button></div>`
              : !data && current.status
                ? html`<div class="empty-actions">
                    <button class="btn" onClick=${files.reveal}>Show in Finder</button>
                    <button class="btn" onClick=${() => files.deleteSet()}>Delete…</button>
                  </div>`
                : null
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
  wide: comparing,
  Tools: CodeButton,
  Centre,
  tabs: [
    { id: "answers", label: "Answers", Body: Answers },
    { id: "history", label: "History", Body: History },
  ],
};
