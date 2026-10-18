// The Playground: a question set, the model it runs on, its answers and history.
import { computed, effect, signal } from "@preact/signals";
import { html } from "htm/preact";
import { api } from "../api.js";
import * as icons from "../components/icons.js";
import { MenuButton, MenuHeader, MenuItem, MenuNote } from "../components/menu.js";
import { plural } from "../format.js";
import { href, navigate, route } from "../router.js";
import { remembered } from "../store/prefs.js";
import { sets } from "../store/sets.js";
import { loadedNames, models, status } from "../store/status.js";

const chosenModel = remembered("playground.model", null);
const lastSet = remembered("playground.set", null);

// The model Run uses: the one picked here, else the Studio's default, else the first.
export const modelName = computed(() => {
  const names = (models.value ?? []).map((model) => model.name);
  if (names.includes(chosenModel.value)) return chosenModel.value;
  const fallback = status.value?.default_model;
  return names.includes(fallback) ? fallback : (names[0] ?? null);
});

// The set shown: the address's, else the last one opened, else the first.
export const setName = computed(() => {
  const item = route.value.page === "playground" ? route.value.item : null;
  if (item) return item;
  const names = (sets.value ?? []).map((entry) => entry.name);
  return names.includes(lastSet.value) ? lastSet.value : (names[0] ?? null);
});

// The set as loaded: {name, body, sha256}, or {name, error}.
export const current = signal(null);

let loading = 0;
effect(() => {
  const name = setName.value;
  const ticket = ++loading;
  if (name === null) {
    current.value = null;
    return;
  }
  lastSet.value = name;
  api(`/sets/${encodeURIComponent(name)}`)
    .then((loaded) => {
      if (ticket === loading) current.value = loaded;
    })
    .catch((error) => {
      if (ticket === loading) current.value = { name, error: error.message };
    });
});

const questionsOf = (body) =>
  body && body.questions && typeof body.questions === "object" && !Array.isArray(body.questions)
    ? Object.entries(body.questions)
    : [];

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
          (entry) => html`
            <li key=${entry.name}>
              <a
                class="sb-item"
                href=${href("playground", entry.name)}
                aria-current=${entry.name === setName.value ? "true" : undefined}
              >
                <${icons.File} /><span class="label">${entry.name}</span>
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
  const model = modelName.value;
  const loaded = loadedNames.value;
  return html`
    <div class="target" role="group" aria-label="What Run runs">
      <${MenuButton}
        id="target-set"
        className="set"
        title="Question set"
        label=${html`<${icons.File} className="" /><span>${name ?? "No set"}</span>`}
        menuStyle="top:34px; left:0; width:280px"
      >
        <${MenuHeader}>Question Sets<//>
        ${(sets.value ?? []).map(
          (entry) => html`<${MenuItem}
            ticked=${entry.name === name}
            onSelect=${() => navigate("playground", entry.name)}
            >${entry.name}<//
          >`,
        )}
        ${(sets.value ?? []).length === 0 && html`<${MenuNote}>No question sets yet.<//>`}
      <//>
      <${icons.Crumb} />
      <${MenuButton}
        id="target-model"
        className="mdl"
        title="Model"
        label=${html`
          ${model && html`<span class="dot" style=${loaded.has(model) ? "" : "background: var(--fg-3)"}></span>`}
          <span class="name">${model ?? "No model"}</span>
          <${icons.Chevrons} />
        `}
        menuStyle="top:34px; left:0; width:320px"
      >
        <${MenuHeader}>Models<//>
        ${list.map(
          (entry) => html`<${MenuItem}
            ticked=${entry.name === model}
            right=${loaded.has(entry.name) ? "Loaded" : entry.family}
            sub=${entry.fitted ? `fitted on ${entry.base}` : null}
            onSelect=${() => {
              chosenModel.value = entry.name;
            }}
            ><span class="mono">${entry.name}</span><//
          >`,
        )}
        ${list.length === 0 &&
        html`<${MenuNote}>No models yet. Download one: <code>mlx-decision download</code><//>`}
      <//>
    </div>
  `;
}

const status_ = computed(() => {
  if (models.value && models.value.length === 0)
    return { icon: "info", text: html`No models yet · <code>mlx-decision download</code>` };
  if (sets.value && setName.value === null) return { icon: "info", text: "No question set yet" };
  const set = current.value;
  if (set?.error) return { icon: "warn", text: set.error };
  if (!set || !modelName.value) return null;
  const count = questionsOf(set.body).length;
  const loads = loadedNames.value.has(modelName.value) ? "" : " · loads it first";
  return {
    icon: "ok",
    text: html`Run sends <b>${plural(count, "question")}</b> to <b>${modelName.value}</b>${loads}`,
  };
});

const action = computed(() => ({
  label: "Run",
  disabled: true,
  stop: null,
  running: false,
}));

function Questions({ body }) {
  const questions = questionsOf(body);
  if (questions.length === 0) return html`<div class="empty"><b>No questions in this set</b></div>`;
  return html`
    <section class="gsec">
      <div class="gsec-head"><h2>Questions</h2><span class="tag">${questions.length}</span></div>
      <div class="group">
        ${questions.map(
          ([id, question]) => html`
            <div class="row" key=${id}>
              <span class="mono">${id}</span>
              <span>
                <span class="tag-inline">${question?.type ?? "?"}</span>
                ${question?.instructions && html` ${question.instructions}`}
              </span>
            </div>
          `,
        )}
      </div>
    </section>
  `;
}

function Centre() {
  const list = sets.value;
  const set = current.value;
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
  if (!set) return html`<div class="centre-inner"></div>`;
  return html`
    <div class="centre-inner">
      <div class="pane-head">
        <div>
          <h1>${title(set.name)}</h1>
          <p><code>sets/${set.name}.json</code></p>
        </div>
      </div>
      ${set.error ? html`<div class="empty"><b>${set.error}</b></div>` : html`<${Questions} body=${set.body} />`}
    </div>
  `;
}

const Answers = () => html`
  <div class="empty">
    <b>No run yet</b>
    <span>Run answers this set's questions here, one card per question.</span>
  </div>
`;

const History = () => html`
  <div class="empty">
    <b>No runs of this set yet</b>
    <span>Every run is kept in the workspace's <code>history/</code> folder until you clear it.</span>
  </div>
`;

export default {
  id: "playground",
  title: "Playground",
  SidebarList: SetsList,
  Target,
  action,
  status: status_,
  Centre,
  tabs: [
    { id: "answers", label: "Answers", Body: Answers },
    { id: "history", label: "History", Body: History },
  ],
};
