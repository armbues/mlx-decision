// Compare (PG-7): the selected example asked on two or three models, a table
// with a column per model and a row per question. The Answers tab shows the
// latest Compare of the example (this tab's, else the history's); History
// shows each Compare entry with the same table.
import { computed } from "@preact/signals";
import { html } from "htm/preact";
import { two } from "../charts/answers.js";
import * as icons from "../components/icons.js";
import { clock, duration, parameters, plural } from "../format.js";
import { SHORTCUTS } from "../keys.js";
import { loadedNames, models } from "../store/status.js";
import { TYPE_LABELS, changedSince } from "./draft.js";
import { focusLater } from "./editor.js";
import { compareFromHistory } from "./history.js";
import {
  MAX_COMPARED,
  built,
  canCompare,
  compare,
  compareKey,
  compareModels,
  compareWith,
  compares,
  draft,
  example,
  linked,
  modelName,
  running,
  setName,
  toggleCompared,
} from "./session.js";

const NOTHING_CHANGED = { all: false, ids: new Set() };

// The Compare shown for the selected example, with what changed since:
// {shown, done, stopped, inFlight, changed: {all, ids}, modelsChanged}.
export const shownCompare = computed(() => {
  const set = setName.value;
  const current = example.value;
  if (!set || !current) return null;
  const shown = compares.value[compareKey(set, current.name)] ?? compareFromHistory(current.name);
  if (!shown) return null;
  const inFlight = running.value?.compare?.id === shown.id;
  const done = shown.models.every((name) => shown.results[name]);
  let changed = NOTHING_CHANGED;
  if (built.value) {
    // A Compare read back from the history holds its images as references:
    // only the state and the questions can be told apart then.
    const references = (shown.request.images ?? []).some((image) => typeof image !== "string");
    const body = references ? { ...built.value.body, images: shown.request.images } : built.value.body;
    changed = changedSince(body, shown.request);
  }
  const modelsChanged = JSON.stringify(shown.models) !== JSON.stringify(compareModels.value);
  return { shown, done, stopped: !done && !inFlight, inFlight, changed, modelsChanged };
});

const facts = (name) => (models.value ?? []).find((item) => item.name === name);

// What a cell shows: the verdict, its value and how they compare across models.
function verdictOf(answer) {
  if (answer.type === "choice") return answer.choice;
  if (answer.type === "noul") return answer.noul >= 0.5 ? "yes" : "no";
  return null; // a score has no verdict to disagree on; its level moves
}

function valueOf(answer) {
  if (answer.type === "choice") return answer.probabilities?.[answer.choice] ?? 0;
  if (answer.type === "score") return answer.score;
  return answer.noul;
}

// "Verdicts differ", "Differ by 0.22" (levels for a score), or null.
export function difference(answers) {
  const given = answers.filter(Boolean);
  if (given.length < 2) return null;
  const verdicts = new Set(given.map(verdictOf));
  if (verdicts.size > 1) return "Verdicts differ";
  // A choice's spread is of the option they all chose.
  const values = given.map((answer) =>
    answer.type === "choice" ? (answer.probabilities?.[given[0].choice] ?? 0) : valueOf(answer),
  );
  const spread = Math.max(...values) - Math.min(...values);
  const threshold = given[0].type === "score" ? 0.25 : 0.1;
  if (spread < threshold) return null;
  return given[0].type === "score" ? `Differ by ${two(spread)} levels` : `Differ by ${two(spread)}`;
}

function Bars({ entries, top }) {
  return html`
    <div class="mini">
      ${entries.map(
        ([key, p]) => html`<div key=${key}>
          <span title=${key}>${key}</span>
          <span class="ptrack"><i style=${`width:${(p * 100).toFixed(1)}%${key === top ? "; background: var(--accent)" : ""}`}></i></span>
        </div>`,
      )}
    </div>
  `;
}

function Cell({ answer }) {
  if (answer.type === "choice")
    return html`
      <span class="cv">${answer.choice}</span><span class="cs num">${two(valueOf(answer))}</span>
      <${Bars} entries=${Object.entries(answer.probabilities ?? {})} top=${answer.choice} />
    `;
  if (answer.type === "score") {
    // Levels are shown from 1.
    const levels = Object.values(answer.probabilities ?? {});
    const most = Math.max(...levels);
    return html`
      <span class="cv num">${two(answer.score + 1)}</span><span class="cs">of ${levels.length}</span>
      <div class="cols" aria-hidden="true">
        ${levels.map((p, level) => html`<i key=${level} class=${p === most ? "top" : ""} style=${`height:${Math.max(2, p * 100)}%`}
          title=${`level ${level + 1}: ${two(p)}`}></i>`)}
      </div>
    `;
  }
  return html`
    <span class="cv">${answer.noul >= 0.5 ? "yes" : "no"}</span><span class="cs num">${two(answer.noul)}</span>
    <${Bars} entries=${[["yes", answer.noul]]} top="yes" />
  `;
}

// Why a model's column has no answers: refused, still waiting, or not asked.
function columnNote(name, shown, flight) {
  const result = shown.results[name];
  if (result?.error) return { kind: "err", text: result.error };
  if (result) return null;
  if (flight?.compare?.id === shown.id)
    return flight.model === name
      ? { kind: "wait", text: flight.loads ? "Loading, then asking…" : "Asking…" }
      : { kind: "wait", text: "Waiting" };
  if (shown.stopped === name) return { kind: "wait", text: "Stopped while it answered" };
  return { kind: "wait", text: "Not asked: stopped" };
}

// `shown`: a Compare ({models, results, request}); `questions`: [{id, type,
// uid?}] in the order to show (a uid links the row to its question);
// `changed`: {all, ids} edited since, shown faded.
export function CompareTable({ shown, questions, changed = NOTHING_CHANGED }) {
  const flight = running.value;
  const names = shown.models;
  const notes = names.map((name) => columnNote(name, shown, flight));
  const answered = names.map((name) => (shown.results[name]?.response ? shown.results[name] : null));
  return html`
    <div class="cmp-wrap">
      <table class="cmp">
        <thead>
          <tr>
            <th></th>
            ${names.map((name, index) => {
              const model = facts(name);
              const sub = model ? [parameters(model.parameters), model.precision].filter(Boolean).join(" · ") : "";
              const note = notes[index];
              return html`<th scope="col" key=${name}>
                <span class="mono">${name}</span>${sub && html`<small>${sub}</small>`}
                ${note && html`<small class=${note.kind}>${note.text}</small>`}
              </th>`;
            })}
          </tr>
        </thead>
        <tbody>
          ${questions.map((question) => {
            const asked = question.id in (shown.request.questions ?? {});
            const answers = answered.map((run) => (asked ? run?.response?.answers?.[question.id] : null));
            const differ = difference(answers);
            const stale = changed.all || changed.ids.has(question.id);
            const show = () => {
              linked.value = question.id;
              focusLater(`#q-${question.uid} .ins`);
            };
            return html`
              <tr key=${question.id} class=${`${stale ? "stale" : ""} ${linked.value === question.id ? "linked" : ""}`}>
                <th scope="row">
                  ${question.uid
                    ? html`<button class="qid" title="Show the question" onClick=${show}>${question.id}</button>`
                    : question.id}
                  <small>${TYPE_LABELS[question.type] ?? question.type}</small>
                  ${differ && html`<span class="spread">${differ}</span>`}
                </th>
                ${answers.map((answer, index) => html`<td key=${names[index]}>${
                  answer
                    ? html`<${Cell} answer=${answer} />`
                    : html`<span class="dim">${!asked ? "Not in this Compare" : notes[index]?.kind === "wait" ? "…" : "—"}</span>`
                }</td>`)}
              </tr>
            `;
          })}
        </tbody>
        <tfoot>
          <tr>
            <th>Latency</th>
            ${answered.map((run, index) => html`<td key=${names[index]} class="num">${
              run
                ? html`${duration(run.latency_ms)}${run.load_ms != null ? html` <span class="dim">+ ${duration(run.load_ms)} load</span>` : ""}`
                : "—"
            }</td>`)}
          </tr>
          <tr>
            <th>Input</th>
            ${answered.map((run, index) => html`<td key=${names[index]} class="num">${
              run
                ? html`${plural(run.input_tokens, "token")}${run.truncated ? html` <span class="warn-text">· cut</span>` : ""}`
                : "—"
            }</td>`)}
          </tr>
        </tfoot>
      </table>
    </div>
  `;
}

// The models to compare with, as checkboxes (up to two besides the model).
export function ComparePicks() {
  const current = modelName.value;
  const picked = compareWith.value;
  const others = (models.value ?? []).filter((item) => item.name !== current);
  const full = picked.length >= MAX_COMPARED - 1;
  if (others.length === 0)
    return html`<span class="dim">There is no other model to compare with yet.</span>`;
  return html`
    <div class="cmp-picks" role="group" aria-label="Compare with">
      ${others.map((item) => {
        const on = picked.includes(item.name);
        return html`<label key=${item.name} class=${!on && full ? "off" : ""}>
          <input type="checkbox" checked=${on} disabled=${!on && full} onChange=${() => toggleCompared(item.name)} />
          <span class="mono">${item.name}</span>
          <small>${loadedNames.value.has(item.name) ? "Loaded" : item.family}</small>
        </label>`;
      })}
    </div>
  `;
}

const names = (list) =>
  list.length < 2 ? html`<b>${list[0]}</b>` : html`${list.slice(0, -1).map((name, i) => html`${i ? ", " : ""}<b>${name}</b>`)} and <b>${list.at(-1)}</b>`;

// The Answers tab's Compare view.
export function CompareView() {
  const current = draft.value;
  const list = compareModels.value;
  const exampleName = example.value?.name;
  const many = current.examples.length > 1;
  if (list.length < 2)
    return html`
      <div class="empty">
        <b>Compare models side by side</b>
        <span>Pick one or two models to ask beside <b>${modelName.value}</b>.</span>
        <${ComparePicks} />
      </div>
    `;
  const state = shownCompare.value;
  const flight = running.value;
  if (!state)
    return html`
      <div class="empty">
        ${flight?.compare
          ? html`<b>Asking ${flight.model}…</b>`
          : html`<b>No Compare${many ? ` of ${exampleName}` : ""} yet</b>`}
        <${ComparePicks} />
        <div class="empty-actions">
          <button class="btn primary" disabled=${!canCompare.value} onClick=${() => compare()}>Compare</button>
        </div>
      </div>
    `;
  const { shown, changed, modelsChanged } = state;
  const edited = changed.all || changed.ids.size > 0;
  const questions = current.questions.map((question) => ({ id: question.id, type: question.type, uid: question.uid }));
  return html`
    <div class="run-head">
      <h3>${shown.number ? `Compare #${shown.number}` : "Compare"}<small class="num">${clock(shown.time)}${many ? ` · ${shown.example}` : ""} · ${plural(shown.models.length, "model")}</small></h3>
    </div>
    ${(edited || modelsChanged) &&
    !state.inFlight &&
    html`<div class="stale-banner">
      <${icons.StatusIcon} kind="warn" />
      <span class="msg">${
        changed.all
          ? "You changed the state or its images since this Compare. "
          : changed.ids.size
            ? html`You edited <b>${[...changed.ids].join(", ")}</b> since this Compare. `
            : ""
      }${modelsChanged ? html`It asked ${names(shown.models)}. ` : ""}</span>
      <button class="btn primary" disabled=${!canCompare.value} onClick=${() => compare()}>Compare</button>
    </div>`}
    <${CompareTable} shown=${shown} questions=${questions} changed=${changed} />
    <p class="note">Compare with:</p>
    <${ComparePicks} />
  `;
}
