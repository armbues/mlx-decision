// The Playground: a question set, the model it runs on, its answers and history.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { useEffect } from "preact/hooks";
import * as icons from "../components/icons.js";
import { MenuButton, MenuHeader, MenuItem, MenuNote } from "../components/menu.js";
import { duration, plural } from "../format.js";
import { onKey } from "../keys.js";
import { href, navigate, route } from "../router.js";
import { Answers, clock } from "../playground/answers.js";
import { Editor } from "../playground/editor.js";
import {
  built,
  canRun,
  chooseModel,
  entry,
  lastRun,
  modelName,
  problems,
  run,
  running,
  setName,
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
      <h3>Question Sets</h3>
      ${list && list.length === 0 && html`<p class="sb-empty">No sets yet</p>`}
      <ul class="sb-list">
        ${(list ?? []).map(
          (item) => html`
            <li key=${item.name}>
              <a class="sb-item" href=${href("playground", item.name)}
                 aria-current=${item.name === setName.value ? "true" : undefined}>
                <${icons.File} /><span class="label">${item.name}</span>
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

const status = computed(() => {
  if (models.value && models.value.length === 0)
    return { icon: "info", text: html`No models yet · <code>mlx-decision download</code>` };
  if (sets.value && setName.value === null) return { icon: "info", text: "No question set yet" };
  if (entry.value?.error) return { icon: "warn", text: entry.value.error };
  if (!built.value || !modelName.value) return null;
  const flight = running.value;
  if (flight)
    return {
      icon: "spin",
      text: html`Asking <b>${flight.model}</b> ${plural(questionCount.value, "question")}…${flight.loads ? " · loading it first" : ""}`,
    };
  if (problems.value.length) return { icon: "warn", text: problems.value[0].message };
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
      icon: "ok",
      text: html`<b>${result.model}</b> answered in ${duration(result.latency_ms)} · ${clock(result.time)}`,
    };
  const loads = loadedNames.value.has(modelName.value) ? "" : " · loads it first";
  return {
    icon: "ok",
    text: html`Run sends <b>${plural(questionCount.value, "question")}</b> to <b>${modelName.value}</b>${loads}`,
  };
});

const action = computed(() => {
  const changed = stale.value;
  return {
    label: "Run",
    disabled: !canRun.value,
    run,
    stop,
    running: !!running.value,
    badge: !!lastRun.value && (changed.all || changed.ids.size > 0),
  };
});

function Centre() {
  useEffect(() => onKey("run", () => run()), []);
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
          <p><code>sets/${setName.value}.json</code></p>
        </div>
      </div>
      ${current.error ? html`<div class="empty"><b>${current.error}</b></div>` : html`<${Editor} />`}
    </div>
  `;
}

const History = () => html`
  <div class="empty">
    <b>Runs of this set</b>
    <span>Every run is kept in the workspace's <code>history/</code> folder until you clear it.</span>
  </div>
`;

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
